"""Load the exact PI SpectrumEncoder weights without peptide-decoding dependencies."""
from types import SimpleNamespace
import torch
from PrimeNovo.components import SpectrumEncoder
from piprime_utils import load_piprime_config, process_peaks


def load_model(model_path):
    # Local model checkpoint is trusted, same source as the previous Lightning loader.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    hp = checkpoint["hyper_parameters"]
    if hp.get("custom_encoder") is not None:
        raise ValueError("A custom encoder requires its explicit architecture loader")
    defaults = dict(dim_model=512, n_head=8, dim_feedforward=1024,
                    n_layers=9, dropout=0.0, dim_intensity=None)
    kwargs = {k:hp.get(k,v) for k,v in defaults.items()}
    encoder = SpectrumEncoder(**kwargs)
    weights = {k[len("encoder."):]:v for k,v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
    if not weights:
        raise ValueError("Checkpoint contains no encoder weights")
    encoder.load_state_dict(weights, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval().to(device)
    config = load_piprime_config()
    config["dim_model"] = kwargs["dim_model"]
    return SimpleNamespace(encoder=encoder), config, device


def extract_embedding(mz_array, intensity_array, precursor_mz, precursor_charge, model, config, device):
    peaks = process_peaks(mz_array, intensity_array, precursor_mz, precursor_charge, config)
    with torch.no_grad():
        encoded, _ = model.encoder(peaks.to(device).unsqueeze(0))
        return encoded.mean(dim=1).cpu().numpy()[0]
