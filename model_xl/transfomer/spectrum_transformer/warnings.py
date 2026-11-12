"""Centralized warning filters for V20 entrypoints."""

from __future__ import annotations

import warnings


_NESTED_TENSOR_MSG = r"The PyTorch API of nested tensors is in prototype stage.*"


def configure_v20_warning_filters():
    warnings.filterwarnings(
        "ignore",
        message=_NESTED_TENSOR_MSG,
        category=UserWarning,
    )
