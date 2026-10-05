"""Frozen seed20 protocol and U-bin contract for P2-C0 conflict diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import numpy as np


TRAINING_SEED = 20
CORRUPTION_SEED = 20
EPOCHS = 20
CARRIER_DIR = Path("outputs/bone_p2/p2_a0_msrc_seed20")
PARENT_TRUE_U_DIR = Path("outputs/bone_p2/p2_a1_seed20_formal/TRUE_U")
DEFAULT_OUTPUT_DIR = Path("outputs/bone_p2/p2_c0_seed20")
BIN_NAMES = ("ZERO", "LOW", "MID", "HIGH")


def build_u_bins(u_cycle: np.ndarray) -> Dict[str, np.ndarray]:
    """Freeze deterministic U bins using value then flattened-index stable ordering only."""
    values = np.asarray(u_cycle, dtype=np.float64)
    if values.shape != (196, 6) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("U_cycle must be finite [196,6] values in [0,inf)")
    flat = values.reshape(-1)
    indices = np.arange(flat.size, dtype=np.int64)
    zero = indices[flat == 0]
    positive = indices[flat > 0]
    ordered_positive = positive[np.lexsort((positive, flat[positive]))]
    low, mid, high = (part.astype(np.int64, copy=False) for part in np.array_split(ordered_positive, 3))
    return {"ZERO": zero, "LOW": low, "MID": mid, "HIGH": high}


def bins_sha256(bins: Dict[str, np.ndarray]) -> str:
    """Hash bin membership and names in registered order for artifact/audit parity."""
    if tuple(bins) != BIN_NAMES:
        raise ValueError("bin names/order must equal the frozen contract")
    digest = hashlib.sha256()
    for name in BIN_NAMES:
        values = np.asarray(bins[name], dtype="<i8")
        digest.update(name.encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()
