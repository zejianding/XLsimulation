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

CMS2_SCHEMA_VERSION = 4
CMS2_CHANNEL_NAMES = tuple(f"{state}_{ion}" for state in ("plain_or_short", "long") for ion in CHANNEL_NAMES)
CMS2_NUM_CHANNELS = len(CMS2_CHANNEL_NAMES)
CMS2_FRAGMENT_CHARGES = (1, 2, 3, 1, 2, 3) * 2
CMS2_LINKER_IDS = {"DSSO": 0, "DSBU": 1}
# The optional last row represents the intact chain at the offsets enabled by its labels.
CMS2_INTACT_ROW = MAX_CLEAVAGE_LEN
CMS2_INTACT_CHANNELS = (3, 4, 5, 9, 10, 11)  # short 1+/2+/3+, then long 1+/2+/3+
CMS2_INTACT_OFFSET_TOKENS = {"0": MAIN_TOKEN_IDS[0], "M+1": 25, "M+2": 26, "-H2O": 22, "-NH3": 23}
_INTACT_AXES = {
    "row": CMS2_INTACT_ROW, "channels": CMS2_INTACT_CHANNELS,
    "states": ("short", "long"), "charges": (1, 2, 3),
}
# Preserve the on-disk layout of existing zero-offset checkpoints (including 122606).
CMS2_INTACT_LAYOUT = {**_INTACT_AXES, "token": MAIN_TOKEN_IDS[0], "head": "main_head"}
CMS2_INTACT_LAYOUTS = {
    MAIN_TOKEN_IDS: CMS2_INTACT_LAYOUT,
    tuple(CMS2_INTACT_OFFSET_TOKENS.values()): {
        **_INTACT_AXES, "offset_names": tuple(CMS2_INTACT_OFFSET_TOKENS),
        "offset_token_ids": tuple(CMS2_INTACT_OFFSET_TOKENS.values()),
        "heads": tuple("main_head" if t in MAIN_TOKEN_IDS else "satellite_head"
                       for t in CMS2_INTACT_OFFSET_TOKENS.values()),
    },
}

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
