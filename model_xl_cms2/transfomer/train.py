"""Top-level entry point: train the Spectrum Transformer.

Usage:
    python train.py
    python train.py --npz-path datasets/spectrum_prediction_examples_len40.npz --epochs 50
    python train.py --d-model 128 --n-layers 3 --batch-size 64
"""

from spectrum_transformer.warnings import configure_v20_warning_filters
from spectrum_transformer.cli.train import main

configure_v20_warning_filters()

if __name__ == "__main__":
    main()
