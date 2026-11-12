"""V20 native configuration defaults.

This file intentionally defines a compact V20 baseline instead of inheriting
historical multi-generation defaults.
"""

import sys
from pathlib import Path

# The token vocabulary lives in <project_root>/token_table.py so that the
# dataset builders and the inference/scoring code share exactly one table.
_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from token_table import (  # noqa: E402
    CHANNEL_NAMES,
    MAX_CLEAVAGE_LEN,
    MAX_PEPTIDE_LEN,
    MAIN_TOKEN_IDS,
    NUM_CHANNELS,
    NUM_TOKENS,
    RESIDUE_EVENT_TOKEN_IDS,
    SATELLITE_TOKEN_IDS,
)

AUX_FEATURE_NAMES = ("charge", "pepmass", "rtinseconds")
AUX_FEATURE_DIM = len(AUX_FEATURE_NAMES)
SEQUENCE_CONTEXT_RESIDUE_VOCAB_SIZE = 27
SEQUENCE_CONTEXT_SCALAR_DIM = 7

INPUT_CHANNELS = 6
OUTPUT_CHANNELS = 6

TOKEN_FAMILY_NAMES = ("residue_event", "satellite", "main")

DEFAULT_NPZ_PATH = Path("datasets/all_species_average_training_examples_len40_sqrtclip50000.npz")
DEFAULT_OUTPUT_DIR = Path("runs/v20_clean")

DEFAULT_D_MODEL = 256
DEFAULT_N_HEADS = 8
DEFAULT_N_LAYERS = 6
DEFAULT_FF_DIM = 1024
DEFAULT_DROPOUT = 0.15
DEFAULT_USE_SEQUENCE_CONTEXT = True
DEFAULT_SEQUENCE_CONTEXT_DIM = 64

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 3e-4
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
DEFAULT_SEED = 42
DEFAULT_MIN_VALID_MZ = 100.0

# V20 clean baseline: keep only the terms that matter to the current mission.
DEFAULT_BACKBONE_WEIGHT = 1.0
DEFAULT_CALIBRATION_WEIGHT = 1.0

DEFAULT_POSITIVE_WEIGHT = 6.0
DEFAULT_SCHEDULER = "cosine"
DEFAULT_LR_MIN = 1e-6

DEFAULT_V20_PROFILE = "native_clean_minimal"
