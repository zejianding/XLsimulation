#!/usr/bin/env python3
"""Run ATR/AQR transformer replacement evaluation with PiPrime embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_PEPTIDES = {
    "AQR": {
        "seq": "QLEEAEEEAQR",
        "mgf": "piprime_embedding/piprime_embedding/QLEEAEEEAQR_28.mgf",
        "charge": 2,
    },
    "ATR": {
        "seq": "QLEEAEEEATR",
        "mgf": "piprime_embedding/piprime_embedding/QLEEAEEEATR_32.mgf",
        "charge": 2,
    },
}


def add_local_import_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "transfomer"))
    sys.path.insert(0, str(repo_root / "piprime_embedding" / "piprime_embedding"))


def threshold_tag(threshold: float) -> str:
    return f"thr_{threshold:.2f}"


def parse_mgf_blocks(path: Path) -> list[dict]:
    blocks = []
    current: list[str] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "BEGIN IONS":
            current = [raw]
            continue
        if current is None:
            continue
        current.append(raw)
        if line == "END IONS":
            params = {}
            peak_count = 0
            for item in current:
                stripped = item.strip()
                if not stripped or stripped in {"BEGIN IONS", "END IONS"}:
                    continue
                if "=" in stripped:
                    key, value = stripped.split("=", 1)
                    params[key.upper()] = value
                else:
                    peak_count += 1
            blocks.append(
                {
                    "text": "\n".join(current),
                    "params": params,
                    "peak_count": peak_count,
                }
            )
            current = None
    return blocks


def parse_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        text = str(value).split()[0]
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else default
    except (TypeError, ValueError):
        return default


def write_blocks(path: Path, blocks: list[dict], prediction_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(block["text"].rstrip())
            handle.write("\n\n")
        handle.write(prediction_text.rstrip())
        handle.write("\n")


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = embeddings / norms
    return normalized @ normalized.T


def mean_between(sim: np.ndarray, idx_a: list[int], idx_b: list[int], same_group: bool = False):
    if not idx_a or not idx_b:
        return None
    sub = sim[np.ix_(idx_a, idx_b)]
    if same_group:
        n = len(idx_a)
        if n <= 1:
            return None
        return float((sub.sum() - np.trace(sub)) / (n * (n - 1)))
    return float(sub.mean())


def group_rows(sim: np.ndarray, groups: dict[str, list[int]]) -> list[dict]:
    rows = []
    for group_a in sorted(groups):
        for group_b in sorted(groups):
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "mean_cosine": mean_between(
                        sim, groups[group_a], groups[group_b], same_group=group_a == group_b
                    ),
                    "n_a": len(groups[group_a]),
                    "n_b": len(groups[group_b]),
                }
            )
    return rows


def load_embedding_bundle(npz_path: Path, mgf_path: Path, peptide: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    blocks = parse_mgf_blocks(mgf_path)
    embeddings = data["embeddings"].astype(np.float32)
    if len(blocks) != len(embeddings):
        raise RuntimeError(f"{mgf_path}: MGF spectra {len(blocks)} != embeddings {len(embeddings)}")

    sources = [str(x) for x in data["sources"]] if "sources" in data else [
        block["params"].get("SOURCE", "UNKNOWN") for block in blocks
    ]
    titles = [str(x) for x in data["titles"]] if "titles" in data else [
        block["params"].get("TITLE", f"index_{i}") for i, block in enumerate(blocks)
    ]
    precursor_mz = (
        data["precursor_mz"].astype(float).tolist()
        if "precursor_mz" in data
        else [parse_float(block["params"].get("PEPMASS"), 0.0) for block in blocks]
    )
    charges = (
        data["charges"].astype(int).tolist()
        if "charges" in data
        else [parse_int(block["params"].get("CHARGE"), 0) for block in blocks]
    )

    meta = pd.DataFrame(
        {
            "peptide": peptide,
            "index": np.arange(len(embeddings)),
            "source": sources,
            "seq": [block["params"].get("SEQ", "") for block in blocks],
            "title": titles,
            "precursor_mz": precursor_mz,
            "charge": charges,
            "peak_count": [block["peak_count"] for block in blocks],
        }
    )
    return {"peptide": peptide, "npz": str(npz_path), "mgf": str(mgf_path), "embeddings": embeddings, "meta": meta}


def source_indices(meta: pd.DataFrame, source: str) -> list[int]:
    return meta.index[meta["source"].str.upper() == source].tolist()


def all_real_indices(meta: pd.DataFrame) -> list[int]:
    source_upper = meta["source"].str.upper()
    return meta.index[source_upper.isin(["ORIGINAL", "FILTERED"])].tolist()


def write_same_xlsx(bundle: dict, output_path: Path) -> dict:
    emb = bundle["embeddings"]
    meta = bundle["meta"]
    sim = cosine_matrix(emb)
    labels = [
        f"{bundle['peptide']}_{int(idx)}_{source}"
        for idx, source in zip(meta["index"], meta["source"])
    ]

    groups = {
        str(source): meta.index[meta["source"] == source].tolist()
        for source in sorted(meta["source"].unique())
    }
    selected = [
        {
            "metric": "original_original_mean",
            "value": mean_between(sim, source_indices(meta, "ORIGINAL"), source_indices(meta, "ORIGINAL"), True),
        },
        {
            "metric": "filtered_filtered_mean",
            "value": mean_between(sim, source_indices(meta, "FILTERED"), source_indices(meta, "FILTERED"), True),
        },
        {
            "metric": "filtered_original_mean",
            "value": mean_between(sim, source_indices(meta, "FILTERED"), source_indices(meta, "ORIGINAL")),
        },
        {
            "metric": "all_real_all_real_mean",
            "value": mean_between(sim, all_real_indices(meta), all_real_indices(meta), True),
        },
        {
            "metric": "model_prediction_vs_original_mean",
            "value": mean_between(sim, source_indices(meta, "MODEL_PREDICTION"), source_indices(meta, "ORIGINAL")),
        },
        {
            "metric": "model_prediction_vs_filtered_mean",
            "value": mean_between(sim, source_indices(meta, "MODEL_PREDICTION"), source_indices(meta, "FILTERED")),
        },
        {
            "metric": "model_prediction_vs_all_real_mean",
            "value": mean_between(sim, source_indices(meta, "MODEL_PREDICTION"), all_real_indices(meta)),
        },
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(sim, index=labels, columns=labels).to_excel(writer, sheet_name="similarity_matrix")
        meta.to_excel(writer, sheet_name="metadata", index=False)
        pd.DataFrame(group_rows(sim, groups)).to_excel(writer, sheet_name="group_summary", index=False)
        pd.DataFrame(selected).to_excel(writer, sheet_name="selected_metrics", index=False)

    counts = Counter(meta["source"].str.upper())
    metrics = {row["metric"]: row["value"] for row in selected}
    metrics.update(
        {
            "num_embeddings": int(len(meta)),
            "original_count": int(counts.get("ORIGINAL", 0)),
            "filtered_count": int(counts.get("FILTERED", 0)),
            "all_real_count": int(counts.get("ORIGINAL", 0) + counts.get("FILTERED", 0)),
            "model_prediction_count": int(counts.get("MODEL_PREDICTION", 0)),
        }
    )
    return metrics


def cross_metrics_and_xlsx(aqr: dict, atr: dict, output_path: Path) -> dict:
    emb = np.concatenate([aqr["embeddings"], atr["embeddings"]], axis=0)
    meta = pd.concat([aqr["meta"], atr["meta"]], ignore_index=True)
    meta["group"] = meta["peptide"] + "_" + meta["source"].str.upper()
    sim = cosine_matrix(emb)
    labels = [
        f"{peptide}_{int(idx)}_{source}"
        for peptide, idx, source in zip(meta["peptide"], meta["index"], meta["source"])
    ]
    groups = {group: meta.index[meta["group"] == group].tolist() for group in sorted(meta["group"].unique())}

    def val(group_a: str, group_b: str, same: bool = False):
        return mean_between(sim, groups.get(group_a, []), groups.get(group_b, []), same)

    aqr_all = groups.get("AQR_ORIGINAL", []) + groups.get("AQR_FILTERED", [])
    atr_all = groups.get("ATR_ORIGINAL", []) + groups.get("ATR_FILTERED", [])
    aqr_model = groups.get("AQR_MODEL_PREDICTION", [])
    atr_model = groups.get("ATR_MODEL_PREDICTION", [])

    metrics = {
        "AQR_model_vs_AQR_original": val("AQR_MODEL_PREDICTION", "AQR_ORIGINAL"),
        "AQR_model_vs_AQR_filtered": val("AQR_MODEL_PREDICTION", "AQR_FILTERED"),
        "AQR_model_vs_ATR_original": val("AQR_MODEL_PREDICTION", "ATR_ORIGINAL"),
        "AQR_model_vs_ATR_filtered": val("AQR_MODEL_PREDICTION", "ATR_FILTERED"),
        "ATR_model_vs_ATR_original": val("ATR_MODEL_PREDICTION", "ATR_ORIGINAL"),
        "ATR_model_vs_ATR_filtered": val("ATR_MODEL_PREDICTION", "ATR_FILTERED"),
        "ATR_model_vs_AQR_original": val("ATR_MODEL_PREDICTION", "AQR_ORIGINAL"),
        "ATR_model_vs_AQR_filtered": val("ATR_MODEL_PREDICTION", "AQR_FILTERED"),
        "AQR_original_vs_ATR_original": val("AQR_ORIGINAL", "ATR_ORIGINAL"),
        "AQR_filtered_vs_ATR_filtered": val("AQR_FILTERED", "ATR_FILTERED"),
        "AQR_model_vs_AQR_all_real": mean_between(sim, aqr_model, aqr_all),
        "AQR_model_vs_ATR_all_real": mean_between(sim, aqr_model, atr_all),
        "ATR_model_vs_ATR_all_real": mean_between(sim, atr_model, atr_all),
        "ATR_model_vs_AQR_all_real": mean_between(sim, atr_model, aqr_all),
        "AQR_all_real_count": len(aqr_all),
        "ATR_all_real_count": len(atr_all),
    }
    metrics["AQR_same_filtered_minus_cross_filtered"] = (
        metrics["AQR_model_vs_AQR_filtered"] - metrics["AQR_model_vs_ATR_filtered"]
        if metrics["AQR_model_vs_AQR_filtered"] is not None and metrics["AQR_model_vs_ATR_filtered"] is not None
        else None
    )
    metrics["ATR_same_filtered_minus_cross_filtered"] = (
        metrics["ATR_model_vs_ATR_filtered"] - metrics["ATR_model_vs_AQR_filtered"]
        if metrics["ATR_model_vs_ATR_filtered"] is not None and metrics["ATR_model_vs_AQR_filtered"] is not None
        else None
    )
    metrics["AQR_all_real_margin"] = (
        metrics["AQR_model_vs_AQR_all_real"] - metrics["AQR_model_vs_ATR_all_real"]
        if metrics["AQR_model_vs_AQR_all_real"] is not None and metrics["AQR_model_vs_ATR_all_real"] is not None
        else None
    )
    metrics["ATR_all_real_margin"] = (
        metrics["ATR_model_vs_ATR_all_real"] - metrics["ATR_model_vs_AQR_all_real"]
        if metrics["ATR_model_vs_ATR_all_real"] is not None and metrics["ATR_model_vs_AQR_all_real"] is not None
        else None
    )

    selected_df = pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(sim, index=labels, columns=labels).to_excel(writer, sheet_name="similarity_matrix")
        meta.to_excel(writer, sheet_name="metadata", index=False)
        pd.DataFrame(group_rows(sim, groups)).to_excel(writer, sheet_name="group_summary", index=False)
        selected_df.to_excel(writer, sheet_name="selected_metrics", index=False)
    return metrics


def generate_predictions(args: argparse.Namespace, repo_root: Path) -> list[dict]:
    import predict_mgf as pred

    device = torch.device(args.device)
    model, model_config, stats, incompatible = pred.load_checkpoint(Path(args.checkpoint), device)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Checkpoint loaded with non-strict state dict: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

    out_dir = Path(args.output_dir)
    rows = []
    for peptide_tag, cfg in DEFAULT_PEPTIDES.items():
        seq = cfg["seq"]
        charge = int(cfg["charge"])
        source_mgf = repo_root / cfg["mgf"]
        blocks = parse_mgf_blocks(source_mgf)
        kept_blocks = []
        removed = 0
        for block in blocks:
            source = block["params"].get("SOURCE", "UNKNOWN").upper()
            if source == "MODEL_PREDICTION":
                removed += 1
                continue
            kept_blocks.append(block)

        real_sources = Counter(block["params"].get("SOURCE", "UNKNOWN").upper() for block in kept_blocks)
        rt_values = [
            value
            for value in (parse_float(block["params"].get("RTINSECONDS")) for block in kept_blocks)
            if value is not None
        ]
        rt_mean = float(np.mean(rt_values)) if rt_values else float(args.rt_fallback)
        pepmass = pred.calculate_pepmass(seq, charge)

        input_mz = pred.build_input_mz_matrix(seq, min_mz=args.min_mz)
        valid_mask, cleavage_count = pred.build_valid_mask(seq, charge, min_mz=args.min_mz)
        model_input = pred.prepare_model_input(
            input_mz,
            valid_mask,
            stats,
            expected_in_channels=int(model_config.get("in_channels", pred.NUM_CHANNELS)),
        )
        raw_aux = np.array([[float(charge), pepmass, rt_mean]], dtype=np.float32)
        aux_norm = (
            (raw_aux - stats["aux_feature_means"])
            / np.maximum(stats["aux_feature_stds"], 1e-6)
        ).astype(np.float32)[0]
        prediction = pred.predict_spectrum(
            model,
            model_input,
            valid_mask,
            aux_norm,
            seq,
            cleavage_count,
            device,
        )

        for threshold in args.thresholds:
            tag = threshold_tag(threshold)
            pred_dir = out_dir / "prediction_mgf" / tag
            combined_dir = out_dir / "combined_mgf" / tag
            flat_dir = out_dir / "combined_mgf_flat"
            prediction_mgf = pred_dir / f"{peptide_tag}_{seq}_{args.model_tag}_prediction_{tag}.mgf"
            combined_mgf = combined_dir / f"{peptide_tag}_{seq}_{args.model_tag}_replacement_{tag}.mgf"
            combined_flat = flat_dir / f"{peptide_tag}_{seq}_{args.model_tag}_replacement_{tag}.mgf"

            peaks = pred.prediction_to_mgf_peaks(
                input_mz,
                prediction,
                valid_mask,
                min_intensity=threshold,
                min_mz=args.min_mz,
                mz_merge_tol=args.mz_merge_tol,
            )
            pred.write_mgf(
                prediction_mgf,
                [peaks],
                seq,
                charge,
                pepmass,
                "MODEL_PREDICTION",
                [rt_mean],
            )
            prediction_text = parse_mgf_blocks(prediction_mgf)[0]["text"]
            write_blocks(combined_mgf, kept_blocks, prediction_text)
            write_blocks(combined_flat, kept_blocks, prediction_text)
            rows.append(
                {
                    "peptide": peptide_tag,
                    "seq": seq,
                    "threshold": threshold,
                    "threshold_tag": tag,
                    "prediction_peaks": int(len(peaks)),
                    "source_mgf": str(source_mgf.relative_to(repo_root)),
                    "removed_existing_model_prediction": int(removed),
                    "kept_real_spectra": int(len(kept_blocks)),
                    "kept_original": int(real_sources.get("ORIGINAL", 0)),
                    "kept_filtered": int(real_sources.get("FILTERED", 0)),
                    "rt_mean_used": rt_mean,
                    "pepmass_used": pepmass,
                    "prediction_mgf": str(prediction_mgf),
                    "combined_mgf": str(combined_mgf),
                    "combined_flat_mgf": str(combined_flat),
                }
            )
            print(f"Generated {peptide_tag} {tag}: peaks={len(peaks)}")
    return rows


def load_piprime_encoder(model_path: Path, device: torch.device):
    from PrimeNovo.components import SpectrumEncoder
    from piprime_utils import load_piprime_config

    checkpoint = torch.load(model_path, map_location=device)
    hparams = checkpoint["hyper_parameters"]
    encoder = SpectrumEncoder(
        dim_model=int(hparams["dim_model"]),
        n_head=int(hparams["n_head"]),
        dim_feedforward=int(hparams["dim_feedforward"]),
        n_layers=int(hparams["n_layers"]),
        dropout=float(hparams["dropout"]),
        dim_intensity=hparams.get("dim_intensity"),
    )
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()
    return encoder, load_piprime_config()


def parse_precursor_mz(params: dict) -> float:
    return parse_float(params.get("pepmass", params.get("PEPMASS")), 0.0) or 0.0


def parse_charge(params: dict) -> int:
    return parse_int(params.get("charge", params.get("CHARGE")), 1)


def parse_source(params: dict) -> str:
    return str(params.get("source", params.get("SOURCE", "UNKNOWN")))


def extract_embeddings_lightweight(mgf_path: Path, encoder, config: dict, device: torch.device):
    from pyteomics import mgf
    from piprime_utils import process_peaks

    embeddings = []
    titles = []
    precursor_mzs = []
    charges = []
    sources = []
    with mgf.MGF(str(mgf_path)) as reader:
        for i, spec in enumerate(reader):
            params = spec.get("params", {})
            peaks = process_peaks(
                spec.get("m/z array", np.array([], dtype=np.float32)),
                spec.get("intensity array", np.array([], dtype=np.float32)),
                parse_precursor_mz(params),
                parse_charge(params),
                config,
            ).to(device)
            with torch.inference_mode():
                encoded, _ = encoder(peaks.unsqueeze(0))
                embedding = encoded.mean(dim=1).detach().cpu().numpy()[0]
            embeddings.append(embedding)
            titles.append(str(params.get("title", params.get("TITLE", f"index_{i}"))))
            precursor_mzs.append(parse_precursor_mz(params))
            charges.append(parse_charge(params))
            sources.append(parse_source(params))

    if embeddings:
        embeddings_arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    else:
        embeddings_arr = np.zeros((0, int(config.get("dim_model", 0) or 0)), dtype=np.float32)
    return embeddings_arr, np.array(titles, dtype=np.str_), np.asarray(precursor_mzs, dtype=np.float32), np.asarray(charges, dtype=np.int32), np.array(sources, dtype=np.str_)


def save_embedding_npz(output_path: Path, embeddings, titles, precursor_mzs, charges, sources) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        titles=titles,
        precursor_mz=precursor_mzs,
        charges=charges,
        sources=sources,
    )


def run_piprime_embeddings(args: argparse.Namespace, generation_rows: list[dict]) -> dict[tuple[str, str], Path]:
    device = torch.device(args.piprime_device)
    print(f"Loading lightweight PiPrime encoder on {device}...")
    encoder, piprime_config = load_piprime_encoder(Path(args.piprime_model), device)
    print("PiPrime encoder loaded.")

    npz_paths: dict[tuple[str, str], Path] = {}
    for row in generation_rows:
        mgf_path = Path(row["combined_flat_mgf"])
        npz_path = Path(args.output_dir) / "npz" / f"{mgf_path.stem}.npz"
        arrays = extract_embeddings_lightweight(mgf_path, encoder, piprime_config, device)
        save_embedding_npz(npz_path, *arrays)
        print(f"Wrote: {npz_path} embeddings={arrays[0].shape} sources={Counter(arrays[-1])}")
        row["npz_path"] = str(npz_path)
        npz_paths[(row["peptide"], row["threshold_tag"])] = npz_path
    return npz_paths


def summarize_cosines(args: argparse.Namespace, generation_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(args.output_dir)
    xlsx_dir = out_dir / "xlsx"
    same_rows = []
    bundle_lookup = {}
    generation_lookup = {(row["peptide"], row["threshold_tag"]): row for row in generation_rows}

    for row in generation_rows:
        bundle = load_embedding_bundle(Path(row["npz_path"]), Path(row["combined_flat_mgf"]), row["peptide"])
        bundle_lookup[(row["peptide"], row["threshold_tag"])] = bundle
        same_xlsx = xlsx_dir / f"{Path(row['npz_path']).stem}_cosine.xlsx"
        metrics = write_same_xlsx(bundle, same_xlsx)
        same_rows.append(
            {
                "peptide": row["peptide"],
                "seq": row["seq"],
                "threshold": row["threshold"],
                "threshold_tag": row["threshold_tag"],
                **metrics,
                "prediction_peaks": row["prediction_peaks"],
                "npz_path": row["npz_path"],
                "xlsx": str(same_xlsx),
            }
        )

    cross_rows = []
    for threshold in args.thresholds:
        tag = threshold_tag(threshold)
        cross_xlsx = xlsx_dir / f"ATR_AQR_{args.model_tag}_replacement_{tag}_cross_cosine.xlsx"
        metrics = cross_metrics_and_xlsx(bundle_lookup[("AQR", tag)], bundle_lookup[("ATR", tag)], cross_xlsx)
        aqr_gen = generation_lookup[("AQR", tag)]
        atr_gen = generation_lookup[("ATR", tag)]
        cross_rows.append(
            {
                "threshold": threshold,
                "threshold_tag": tag,
                "AQR_prediction_peaks": aqr_gen["prediction_peaks"],
                "ATR_prediction_peaks": atr_gen["prediction_peaks"],
                "xlsx": str(cross_xlsx),
                **metrics,
            }
        )

    same_df = pd.DataFrame(same_rows)
    cross_df = pd.DataFrame(cross_rows)
    same_csv = out_dir / f"{args.model_tag}_replacement_same_peptide_summary_with_all_real.csv"
    cross_csv = out_dir / f"{args.model_tag}_replacement_cross_peptide_summary_with_all_real.csv"
    same_df.to_csv(same_csv, index=False)
    cross_df.to_csv(cross_csv, index=False)

    summary_json = out_dir / f"{args.model_tag}_replacement_threshold_summary_with_all_real.json"
    summary_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "piprime_model": str(args.piprime_model),
                "generation": generation_rows,
                "same_peptide": same_df.to_dict(orient="records"),
                "cross_peptide": cross_df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_xlsx = out_dir / f"{args.model_tag}_replacement_threshold_summary_with_all_real.xlsx"
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        same_df.to_excel(writer, sheet_name="same_peptide", index=False)
        cross_df.to_excel(writer, sheet_name="cross_peptide", index=False)
        pd.DataFrame(generation_rows).to_excel(writer, sheet_name="generation", index=False)

    pd.DataFrame(generation_rows).to_csv(out_dir / f"{args.model_tag}_prediction_generation_summary.csv", index=False)
    (out_dir / f"{args.model_tag}_prediction_generation_summary.json").write_text(
        json.dumps(generation_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved: {same_csv}")
    print(f"Saved: {cross_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_xlsx}")
    return same_df, cross_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--piprime-model",
        type=Path,
        default=Path("piprime_embedding/piprime_embedding/model.ckpt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("v8d/postprocess_replacement"))
    parser.add_argument("--model-tag", default="v8d")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10, 0.20, 0.25])
    parser.add_argument("--min-mz", type=float, default=50.0)
    parser.add_argument("--mz-merge-tol", type=float, default=0.05)
    parser.add_argument("--rt-fallback", type=float, default=1800.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--piprime-device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    add_local_import_paths(repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generation_rows = generate_predictions(args, repo_root)
    run_piprime_embeddings(args, generation_rows)
    summarize_cosines(args, generation_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
