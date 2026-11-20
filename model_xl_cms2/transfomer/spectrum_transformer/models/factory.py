"""V20 minimal model factory."""

from __future__ import annotations

from spectrum_transformer.config import (
    AUX_FEATURE_DIM,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_FF_DIM,
    DEFAULT_SEQUENCE_CONTEXT_DIM,
    DEFAULT_USE_SEQUENCE_CONTEXT,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
)
from spectrum_transformer.models.transformer import AxialSpectrumTransformerV11


def create_model_config(**overrides):
    config = {
        "in_channels": INPUT_CHANNELS,
        "out_channels": OUTPUT_CHANNELS,
        "d_model": DEFAULT_D_MODEL,
        "n_heads": DEFAULT_N_HEADS,
        "n_layers": DEFAULT_N_LAYERS,
        "ff_dim": DEFAULT_FF_DIM,
        "dropout": DEFAULT_DROPOUT,
        "aux_dim": AUX_FEATURE_DIM,
        "use_sequence_context": DEFAULT_USE_SEQUENCE_CONTEXT,
        "sequence_context_dim": DEFAULT_SEQUENCE_CONTEXT_DIM,
        "use_packed": True,
        "model_version": "v20",
    }
    config.update(overrides)
    return config


def build_model(model_config: dict):
    return AxialSpectrumTransformerV11(
        in_channels=model_config.get("in_channels", 6),
        out_channels=model_config.get("out_channels", 6),
        d_model=model_config.get("d_model", 256),
        n_heads=model_config.get("n_heads", 8),
        n_layers=model_config.get("n_layers", 6),
        ff_dim=model_config.get("ff_dim", 1024),
        dropout=model_config.get("dropout", 0.15),
        aux_dim=model_config.get("aux_dim", 3),
        use_presence_head=True,
        use_sequence_context=model_config.get("use_sequence_context", DEFAULT_USE_SEQUENCE_CONTEXT),
        sequence_context_dim=model_config.get("sequence_context_dim", 64),
        use_packed=model_config.get("use_packed", True),
        model_version="v20",
        use_group_presence_head=False,
        use_token_gate_head=False,
        use_token_amp_head=False,
        use_export_confidence_head=False,
    )


def prepare_checkpoint_for_inference(checkpoint):
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_config = dict(create_model_config())
    model_config.update(checkpoint.get("model_config", {}))

    input_weight = model_state_dict.get("input_proj.weight")
    if input_weight is not None and getattr(input_weight, "ndim", None) == 2:
        model_config["in_channels"] = int(input_weight.shape[1])

    backbone_weight = model_state_dict.get("backbone_head.1.weight")
    if backbone_weight is not None and getattr(backbone_weight, "ndim", None) == 2:
        model_config["out_channels"] = int(backbone_weight.shape[0])

    return model_config, model_state_dict
