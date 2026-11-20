"""V20 clean-baseline training wrapper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V20 clean-baseline training wrapper.")
    parser.add_argument("--v20-output-dir", required=True)
    parser.add_argument("--v20-train-npz", required=True)
    parser.add_argument("--v20-val-npz", required=True)
    parser.add_argument("--v20-test-npz", required=True)
    parser.add_argument("--v20-gpu", default="0")
    parser.add_argument("--v20-epochs", default="20")
    parser.add_argument("--v20-batch-size", default="64")
    parser.add_argument("--v20-lr", default="0.001")
    parser.add_argument("--v20-weight-decay", default="0.0003")
    parser.add_argument("--v20-d-model", default="256")
    parser.add_argument("--v20-n-heads", default="8")
    parser.add_argument("--v20-n-layers", default="6")
    parser.add_argument("--v20-ff-dim", default="1024")
    parser.add_argument("--v20-dropout", default="0.15")
    parser.add_argument("--v20-num-workers", default="4")
    parser.add_argument("--v20-no-packed", action="store_true")
    parser.add_argument("--v20-backbone-weight", default="1.0")
    parser.add_argument("--v20-strict-presence-weight", default="0.30")
    parser.add_argument("--v20-group-family-focus-weight", default="0.40")
    return parser


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.v20_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).resolve().parent / "train.py"
    cmd = [
        "python", str(train_script),
        "--train-npz", args.v20_train_npz,
        "--val-npz", args.v20_val_npz,
        "--test-npz", args.v20_test_npz,
        "--output-dir", args.v20_output_dir,
        "--epochs", str(args.v20_epochs),
        "--batch-size", str(args.v20_batch_size),
        "--lr", str(args.v20_lr),
        "--weight-decay", str(args.v20_weight_decay),
        "--gpu", str(args.v20_gpu),
        "--d-model", str(args.v20_d_model),
        "--n-heads", str(args.v20_n_heads),
        "--n-layers", str(args.v20_n_layers),
        "--ff-dim", str(args.v20_ff_dim),
        "--dropout", str(args.v20_dropout),
        "--num-workers", str(args.v20_num_workers),
        "--v20-backbone-weight", str(args.v20_backbone_weight),
        "--v20-strict-presence-weight", str(args.v20_strict_presence_weight),
        "--v20-group-family-focus-weight", str(args.v20_group_family_focus_weight),
    ]
    if args.v20_no_packed:
        cmd.append("--no-packed")
    (out_dir / "v20_compiled_train_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (out_dir / "v20_training_config.json").write_text(
        json.dumps(
            {
                "v20_identity": {"version": "v20", "phase": "z23_amplitude_first_dual_focus"},
                "v20_runtime": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raise SystemExit(subprocess.call(cmd, env=dict(os.environ)))


if __name__ == "__main__":
    main()
