#!/usr/bin/env python3
"""Smoke test for the 46-column token table in v20_0918.

Verifies that the token vocabulary, the dataset builder, the inference grid and
the model all agree on a (39, 46, 6) grid, and that the renumbered token
indices resolve to the intended chemistry.

Run from anywhere:
    python dataset/smoke_token46.py

Exit code 0 = all checks passed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # v20_0918/
for _p in (_ROOT, _ROOT / "transfomer", _ROOT / "dataset"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import token_table as tt  # noqa: E402

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"[{mark}] {name}{suffix}")


def info(name: str, detail: str) -> None:
    print(f"[INFO] {name}  ({detail})")


def _theoretical_peaks(seq: str, charge: int, jitter: float = 0.004):
    """Build a synthetic peak list at the theoretical b/y positions."""
    import trainingExamples as te

    peaks = []
    for ion in te.generate_by_ions_with_mods(seq, max_charge=min(3, charge)):
        peaks.append((float(ion["mz"]) + jitter, 1000.0 / float(ion["cleavage_index"])))
    return peaks


def _write_mgf(path: Path, spectra) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i, (seq, charge, peaks) in enumerate(spectra):
            fh.write("BEGIN IONS\n")
            fh.write(f"TITLE=smoke_{i}\n")
            fh.write(f"SEQ={seq}\n")
            fh.write(f"CHARGE={charge}\n")
            fh.write("PEPMASS=800.0\n")
            fh.write("RTINSECONDS=60.0\n")
            for mz, intensity in peaks:
                fh.write(f"{mz:.6f} {intensity:.6f}\n")
            fh.write("END IONS\n\n")


def main() -> int:
    # ---------------------------------------------------------------- 1. table
    check("NUM_TOKENS == 46", tt.NUM_TOKENS == 46, f"got {tt.NUM_TOKENS}")
    check(
        "each charge table has 46 offsets",
        all(len(v) == 46 for v in tt.CHARGE_TOKENS.values()),
        str({z: len(v) for z, v in tt.CHARGE_TOKENS.items()}),
    )
    c1 = tt.CHARGE_TOKENS[1]
    check("main offset at index 24 is 0.0", c1[24] == 0.0, f"got {c1[24]}")
    check(
        "Q/K are distinct columns",
        c1[7] == -128.094963 and c1[8] == -128.058578,
        f"idx7={c1[7]} idx8={c1[8]}",
    )
    check("MAIN_TOKEN_IDS == (24,)", tt.MAIN_TOKEN_IDS == (24,), str(tt.MAIN_TOKEN_IDS))
    check(
        "SATELLITE_TOKEN_IDS == (19..23, 25, 26)",
        tt.SATELLITE_TOKEN_IDS == (19, 20, 21, 22, 23, 25, 26),
        str(tt.SATELLITE_TOKEN_IDS),
    )
    check(
        "38 residue_event columns",
        len(tt.RESIDUE_EVENT_TOKEN_IDS) == 38,
        f"got {len(tt.RESIDUE_EVENT_TOKEN_IDS)}",
    )
    check(
        "TOKEN_NAMES chemistry",
        tt.TOKEN_NAMES[21] == "CO"
        and tt.TOKEN_NAMES[22] == "H2O"
        and tt.TOKEN_NAMES[23] == "NH3"
        and tt.TOKEN_NAMES[25] == "13C"
        and tt.TOKEN_NAMES[26] == "2x13C",
        str(tt.TOKEN_NAMES),
    )

    # ------------------------------------------------- 2. train == infer grid
    import trainingExamples as te
    import predict_mgf as pm

    seq = "PEPTIDEK"
    grid_train = te.build_input_mz_matrix(seq)
    grid_infer = pm.build_input_mz_matrix(seq, min_mz=0.0)
    check("training grid is (39,46,6)", grid_train.shape == (39, 46, 6), str(grid_train.shape))
    # Inference clamps non-positive m/z to 0; compare on physically valid cells.
    valid = np.where(grid_train > 0, grid_train, 0.0)
    check(
        "training grid == inference grid on valid cells",
        np.array_equal(grid_infer, valid),
        f"max|diff|={float(np.abs(grid_infer - valid).max()):.2e}",
    )

    # ------------------------------------------------- 3. NPZ builder end-to-end
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mgf = tmp_path / "smoke_averaged.mgf"
        _write_mgf(
            mgf,
            [
                (seq, 2, _theoretical_peaks(seq, 2)),
                ("M+15.995EPHSATAQFFINVANNK", 3, _theoretical_peaks("M+15.995EPHSATAQFFINVANNK", 3)),
            ],
        )

        import NPZBuilder_chunked as nb

        shard_dir = tmp_path / "shards"
        nb.build_npz_chunked(str(mgf), str(shard_dir), tol=0.05, num_workers=1)

        shards = sorted(shard_dir.glob("shard_*.npz"))
        check("NPZ builder wrote a shard", len(shards) == 1, f"{len(shards)} shard(s)")
        if shards:
            with np.load(shards[0], allow_pickle=True) as data:
                check(
                    "input_mz_matrices is (N,39,46,6)",
                    data["input_mz_matrices"].shape[1:] == (39, 46, 6),
                    str(data["input_mz_matrices"].shape),
                )
                check(
                    "target_intensity_matrices is (N,39,46,6)",
                    data["target_intensity_matrices"].shape[1:] == (39, 46, 6),
                    str(data["target_intensity_matrices"].shape),
                )
                check(
                    "tokens_charge1 stored with 46 values",
                    data["tokens_charge1"].shape == (46,),
                    str(data["tokens_charge1"].shape),
                )
                check(
                    "stored tokens round-trip from token_table",
                    np.allclose(data["tokens_charge1"], np.asarray(tt.CHARGE_TOKENS[1], np.float32)),
                )

    # ------------------------------------------- 4. model forward at 46 tokens
    import torch
    from spectrum_transformer.config import AUX_FEATURE_DIM
    from spectrum_transformer.models.factory import build_model, create_model_config

    model = build_model(create_model_config()).eval()
    check(
        "model token_family_index has 46 entries",
        model.token_family_index.shape == torch.Size([46]),
        str(tuple(model.token_family_index.shape)),
    )
    check(
        "family_index marks main at 24",
        int(model.token_family_index[24]) == 2,
        f"kind={int(model.token_family_index[24])}",
    )
    with torch.no_grad():
        out = model(
            torch.zeros(2, 39, 46, 6),
            torch.zeros(2, AUX_FEATURE_DIM),
            valid_mask=torch.ones(2, 39, 46, 6),
        )
    check(
        "model predicts (B,39,46,6)",
        tuple(out["prediction"].shape) == (2, 39, 46, 6),
        str(tuple(out["prediction"].shape)),
    )

    # ---------------------------------------- 5. checkpoint token-axis guard
    good = {k: v for k, v in model.state_dict().items()}
    pm._assert_token_axis_matches(model, good)
    bad = dict(good)
    bad["packed_pos_w.weight"] = torch.zeros(44, bad["packed_pos_w.weight"].shape[1])
    try:
        pm._assert_token_axis_matches(model, bad)
        check("44-column checkpoint is rejected", False, "no error raised")
    except ValueError as exc:
        check("44-column checkpoint is rejected", True, str(exc)[:70] + "...")

    # ------------------------------------------- 6. cluster_postprocess scenes
    import cluster_postprocess as cp

    check("cluster_postprocess MAIN_TOKEN == 24", cp.MAIN_TOKEN == 24, str(cp.MAIN_TOKEN))
    check(
        "cluster_postprocess SATELLITE_TOKENS",
        cp.SATELLITE_TOKENS == [21, 22, 23, 25, 26],
        str(cp.SATELLITE_TOKENS),
    )
    check(
        "cluster_postprocess CORE8 is the 8-column base block",
        sorted(cp.CORE8_TOKENS) == [19, 20, 21, 22, 23, 24, 25, 26],
        str(sorted(cp.CORE8_TOKENS)),
    )
    bad_scenes = {
        name: toks
        for name, toks in cp._EXPORT_TOKENS.items()
        if toks and (min(toks) < 0 or max(toks) >= cp.NUM_TOKENS)
    }
    check("all export scenes in range", not bad_scenes, str(list(bad_scenes))[:80])
    check(
        '"search" scene exports the 6 satellite/main columns',
        sorted(cp._EXPORT_TOKENS["search"]) == [21, 22, 23, 24, 25, 26],
        str(cp._EXPORT_TOKENS["search"]),
    )
    check(
        "TOKEN_KIND has exactly 1 main / 2 isotope / 3 neutral_loss / 2 combined_loss",
        sum(v == "main" for v in cp.TOKEN_KIND.values()) == 1
        and sum(v == "isotope" for v in cp.TOKEN_KIND.values()) == 2
        and sum(v == "neutral_loss" for v in cp.TOKEN_KIND.values()) == 3
        and sum(v == "combined_loss" for v in cp.TOKEN_KIND.values()) == 2,
    )

    # ------------------------------------------------------ 7. Q/K projection
    # Informational: at tol=0.05 a peak at the +Q offset STILL lands in the +K
    # column, because |Q-K| = 0.036 Da < tol.  Separating them needs tol < ~0.018.
    q_idx, k_idx = 37, 38
    ions = te.generate_by_ions_with_mods("PEPTIDEK", max_charge=1)
    b1_mz = float(ions[0]["mz"])
    for label, idx in (("Q (37)", q_idx), ("K (38)", k_idx)):
        target = te.build_target_intensity_matrix(
            "PEPTIDEK", 1, [(b1_mz + tt.CHARGE_TOKENS[1][idx], 500.0)], tol=0.05
        )
        hits = [t for t in range(tt.NUM_TOKENS) if target[0, t, 0] > 0]
        info(f"peak at b1 + token{idx} {label} lands in columns", str(hits))
        # 加序列约束后应只剩正确那一列
        import sequence_token_map as stm
        tgt_sc = te.build_target_intensity_matrix(
            "PEPTIDEK", 1, [(b1_mz + tt.CHARGE_TOKENS[1][idx], 500.0)],
            tol=0.05, sequence_constrained="qk",
        )
        hits_sc = [t for t in range(tt.NUM_TOKENS) if tgt_sc[0, t, 0] > 0]
        info(f"  ...with sequence_constrained='qk'", str(hits_sc))

    # ------------------------------------ 7. sequence-constrained target mask
    import sequence_token_map as stm

    seq_m = "AQKDLQK"
    m_qk = stm.allowed_target_mask(seq_m, mode="qk")
    m_full = stm.allowed_target_mask(seq_m, mode="full")
    check("mask shape is (39,46,6)", m_qk.shape == (39, 46, 6), str(m_qk.shape))
    check("qk mask: +Q col allowed where next residue is Q, +K not",
          bool(m_qk[0, 37, 0]) and not bool(m_qk[0, 38, 0]))
    check("qk mask: +K col allowed where next residue is K, +Q not",
          bool(m_qk[1, 38, 0]) and not bool(m_qk[1, 37, 0]))
    base_idx = list(tt.MAIN_TOKEN_IDS) + list(tt.SATELLITE_TOKEN_IDS)
    check("mask never touches base columns (main + 7 satellites)",
          bool(m_qk[0, base_idx, 0].all()) and bool(m_full[0, base_idx, 0].all()))
    check("full mask is stricter than qk",
          int(m_full.sum()) <= int(m_qk.sum()),
          f"full={int(m_full.sum())} qk={int(m_qk.sum())}")
    # On every column other than the 4 Q/K ones, the qk mask must be exactly
    # "an ion exists at this (row, channel)" — i.e. it gates nothing else.
    ion_exists = np.zeros_like(m_qk)
    for ion in te.generate_by_ions_with_mods(seq_m, max_charge=3):
        rr = ion["cleavage_index"] - 1
        if rr >= 39:
            continue
        cc = (ion["charge"] - 1) if ion["ion_type"] == "B" else (ion["charge"] + 2)
        ion_exists[rr, :, cc] = True
    non_qk = [t for t in range(tt.NUM_TOKENS) if t not in (37, 38, 7, 8)]
    check("qk mask gates nothing but the 4 Q/K columns",
          np.array_equal(m_qk[:, non_qk, :], ion_exists[:, non_qk, :]))

    failed = [n for n, ok, _ in _RESULTS if not ok]
    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(_RESULTS)}): " + ", ".join(failed))
        return 1
    print(f"OK — all {len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
