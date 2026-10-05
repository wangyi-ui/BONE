"""Generic, pre-GT P3-B0 U/T carrier mechanics.

This module is intentionally an experiment-layer adapter.  It never changes
the native dataset classes, BONE, AMV, losses, action protocol, or sparse-label
protocol.  Raw row IDs are carried explicitly through the native seed-42
permutation so frozen P3-A3 labels and P3-A4 masks retain their RAW_FILE_ROW_ID
meaning after preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol


RAW_ID_SEMANTICS = "RAW_FILE_ROW_ID"
NATIVE_SHUFFLE_SEED = 42
HISTORICAL_CORRUPTION_SEED = 20
HISTORICAL_SNR_DB = 2.5


def native_shared_permutation(n: int, seed: int = NATIVE_SHUFFLE_SEED) -> np.ndarray:
    """Return the exact permutation produced by ``MVDataset.shuffle(seed)``.

    ``MVDataset.shuffle`` first calls ``np.random.seed(seed)`` and then
    ``np.random.permutation(n)``.  An isolated ``RandomState`` reproduces that
    result without mutating global RNG state.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return np.random.RandomState(seed).permutation(n).astype(np.int64, copy=False)


@dataclass(frozen=True)
class SampleIdAdapter:
    """Training-position to raw-row identity map for one shared permutation."""

    sample_ids: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.sample_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("sample_ids must be a one-dimensional integer array")
        if ids.size == 0 or np.unique(ids).size != ids.size or np.any(ids < 0):
            raise ValueError("sample_ids must be unique nonnegative raw row IDs")
        object.__setattr__(self, "sample_ids", np.ascontiguousarray(ids, dtype=np.int64))

    @classmethod
    def from_native_shuffle(cls, n: int, seed: int = NATIVE_SHUFFLE_SEED) -> "SampleIdAdapter":
        return cls(native_shared_permutation(n, seed))

    def row_positions(self, raw_ids: np.ndarray | Iterable[int]) -> np.ndarray:
        """Map raw IDs to unique tensor rows, failing on missing or duplicate IDs."""
        raw = np.asarray(tuple(raw_ids), dtype=np.int64)
        if raw.ndim != 1 or np.unique(raw).size != raw.size:
            raise ValueError("requested raw IDs must be one-dimensional and unique")
        inverse = {int(raw_id): position for position, raw_id in enumerate(self.sample_ids)}
        missing = [int(raw_id) for raw_id in raw if int(raw_id) not in inverse]
        if missing:
            raise ValueError(f"raw IDs are absent from the carrier sample order: {missing}")
        return np.asarray([inverse[int(raw_id)] for raw_id in raw], dtype=np.int64)


class RawIdDataset(Dataset):
    """Attach RAW_FILE_ROW_ID to native-format tensors without changing native classes."""

    def __init__(self, stored_native_views: Sequence[np.ndarray], adapter: SampleIdAdapter) -> None:
        if not stored_native_views:
            raise ValueError("at least one view is required")
        n = adapter.sample_ids.size
        self._views = []
        for view in stored_native_views:
            values = np.asarray(view)
            if values.ndim != 2 or values.shape[1] != n:
                raise ValueError("native stored views must have shape [feature_dim, N]")
            self._views.append(torch.from_numpy(np.ascontiguousarray(values.T, dtype=np.float32)))
        self._sample_ids = torch.from_numpy(adapter.sample_ids.copy())

    def __len__(self) -> int:
        return int(self._sample_ids.numel())

    def __getitem__(self, position: int):
        return [view[position] for view in self._views], self._sample_ids[position]


def _validate_raw_views(raw_views: Sequence[np.ndarray]) -> int:
    if not raw_views:
        raise ValueError("at least one raw view is required")
    n = int(np.asarray(raw_views[0]).shape[0])
    if n < 1:
        raise ValueError("raw views must contain at least one row")
    for view in raw_views:
        array = np.asarray(view)
        if array.ndim != 2 or array.shape[0] != n or not np.isfinite(array).all():
            raise ValueError("raw views must be finite [N, feature_dim] arrays with shared N")
    return n


def apply_historical_noise_formula(
    standardized_views: Sequence[np.ndarray], training_order_mask: np.ndarray,
    corruption_seed: int = HISTORICAL_CORRUPTION_SEED,
) -> list[np.ndarray]:
    """Exact generic equivalent of P1's SNR-calibrated Gaussian corruption."""
    if corruption_seed != HISTORICAL_CORRUPTION_SEED:
        raise ValueError("corruption seed must remain frozen at 20")
    n = _validate_raw_views(standardized_views)
    v = len(standardized_views)
    mask = np.asarray(training_order_mask)
    if mask.dtype != np.bool_ or mask.shape != (n, v):
        raise ValueError("training-order mask must be bool [N, V]")
    noise_rng = np.random.RandomState(corruption_seed)
    snr_scale = 10.0 ** (HISTORICAL_SNR_DB / 10.0)
    corrupted: list[np.ndarray] = []
    for view_index, clean_values in enumerate(standardized_views):
        clean = np.ascontiguousarray(clean_values)
        omega = mask[:, view_index]
        if not np.any(omega):
            raise ValueError("every view must contain at least one corrupted row")
        noise = noise_rng.normal(0.0, 1.0, size=clean.shape).astype(clean.dtype, copy=False)
        signal_energy = float(np.sum(np.square(clean[omega]), dtype=np.float64))
        noise_energy = float(np.sum(np.square(noise[omega]), dtype=np.float64))
        if not np.isfinite(signal_energy) or not np.isfinite(noise_energy) or noise_energy <= 0.0:
            raise ValueError("historical corruption energy is invalid")
        alpha = float(np.sqrt(signal_energy / (noise_energy * snr_scale)))
        corrupt = clean.copy()
        corrupt[omega] += alpha * noise[omega]
        if not np.isfinite(corrupt).all() or not np.array_equal(clean[~omega], corrupt[~omega]):
            raise ValueError("historical corruption changed an unmasked location")
        corrupted.append(corrupt)
    return corrupted


def preprocess_with_raw_ids(
    raw_views: Sequence[np.ndarray], raw_mask: np.ndarray, shuffle_seed: int = NATIVE_SHUFFLE_SEED,
) -> tuple[list[np.ndarray], SampleIdAdapter]:
    """Native-order shuffle/standardization, followed by raw-ID indexed corruption.

    Inputs are raw file row order ``[N, D_v]``.  Returned views remain in
    training tensor order ``[N, D_v]`` and are deliberately not standardized a
    second time after the frozen corruption is applied.
    """
    n = _validate_raw_views(raw_views)
    v = len(raw_views)
    mask = np.asarray(raw_mask)
    if mask.dtype != np.bool_ or mask.shape != (n, v):
        raise ValueError("raw mask must be bool [N, V]")
    adapter = SampleIdAdapter.from_native_shuffle(n, shuffle_seed)
    standardized = [
        np.ascontiguousarray(StandardScaler().fit_transform(np.asarray(view, dtype=np.float32)[adapter.sample_ids]))
        for view in raw_views
    ]
    return apply_historical_noise_formula(standardized, mask[adapter.sample_ids]), adapter


def extract_q_views(model: torch.nn.Module, training_views: Sequence[np.ndarray], device: torch.device | str = "cpu") -> np.ndarray:
    """Capture per-view ``label_fc`` posteriors in the supplied sample order."""
    if not hasattr(model, "label_fc"):
        raise ValueError("BONE carrier model must expose label_fc for q extraction")
    n = _validate_raw_views(training_views)
    observed: list[np.ndarray] = []

    def capture(_module, _inputs, output) -> None:
        observed.append(output.detach().cpu().numpy().copy())

    handle = model.label_fc.register_forward_hook(capture)
    try:
        tensors = [torch.from_numpy(np.ascontiguousarray(view)).to(device=device, dtype=torch.float32) for view in training_views]
        with torch.no_grad():
            result = model(tensors, test=True)
    finally:
        handle.remove()
    if not isinstance(result, dict) or "label" not in result or len(observed) != len(training_views):
        raise ValueError("label_fc hook did not observe exactly one posterior per view")
    q_views = np.stack(observed, axis=1)
    validate_q_views(q_views, n=n, v=len(training_views))
    return q_views


def validate_q_views(q_views: np.ndarray, n: int | None = None, v: int | None = None, k: int | None = None) -> None:
    """Fail closed unless per-view BONE posteriors are finite distributions."""
    values = np.asarray(q_views)
    if values.ndim != 3 or (n is not None and values.shape[0] != n) or (v is not None and values.shape[1] != v) or (k is not None and values.shape[2] != k):
        raise ValueError("q_views must have shape [N, V, K]")
    if values.shape[2] < 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("q_views must be finite nonnegative distributions")
    if not np.allclose(values.sum(axis=2), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("q_views posterior rows must sum to one")


def action_arrays(view_num: int) -> tuple[tuple[action_protocol.BalancedDirectedAction, ...], np.ndarray, np.ndarray]:
    """Import, rather than duplicate, P3-A1's frozen action ordering."""
    actions = action_protocol.build_balanced_directed_actions(view_num)
    generator_width = max(action.generator_size for action in actions)
    verifier_width = max(action.verifier_size for action in actions)
    generator = np.full((len(actions), generator_width), -1, dtype=np.int64)
    verifier = np.full((len(actions), verifier_width), -1, dtype=np.int64)
    for action in actions:
        generator[action.action_id, :action.generator_size] = action.generator_views
        verifier[action.action_id, :action.verifier_size] = action.verifier_views
    return actions, generator, verifier


def _chunk_ranges(n: int, query_chunk_size: int | None) -> tuple[tuple[int, int], ...]:
    if query_chunk_size is None:
        return ((0, n),)
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be None or a positive integer")
    return tuple((start, min(start + query_chunk_size, n)) for start in range(0, n, query_chunk_size))


def _posteriors_for_chunk(q_chunk: np.ndarray, actions: Sequence[action_protocol.BalancedDirectedAction]) -> tuple[np.ndarray, np.ndarray]:
    generated = np.stack([q_chunk[:, action.generator_views, :].mean(axis=1) for action in actions], axis=1)
    verified = np.stack([q_chunk[:, action.verifier_views, :].mean(axis=1) for action in actions], axis=1)
    return generated, verified


def compute_u_cycle(p_gen: np.ndarray, p_ver: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frozen U_cycle: directional closure times geometric confidence/support."""
    if p_gen.shape != p_ver.shape or p_gen.ndim != 3:
        raise ValueError("p_gen and p_ver must have equal shape [N, S, K]")
    y_gen = np.argmax(p_gen, axis=2).astype(np.int64)
    conf_gen = np.take_along_axis(p_gen, y_gen[..., None], axis=2)[..., 0]
    support_ver = np.take_along_axis(p_ver, y_gen[..., None], axis=2)[..., 0]
    closure = (np.argmax(p_ver, axis=2) == y_gen).astype(np.int8)
    u_cycle = closure * np.sqrt(conf_gen * support_ver)
    if not np.isfinite(u_cycle).all() or np.any(u_cycle < 0) or np.any(u_cycle > 1):
        raise ValueError("U_cycle must be finite and in [0, 1]")
    return y_gen, conf_gen, support_ver, closure, u_cycle


def _class_signature(p_gen: np.ndarray, anchor_rows: np.ndarray, labeled_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = p_gen.shape[2]
    anchor_signature = p_gen[anchor_rows].mean(axis=1)
    class_signature = np.empty((k, k), dtype=p_gen.dtype)
    for class_id in range(k):
        members = anchor_signature[labeled_targets == class_id]
        if members.shape != (2, k):
            raise ValueError("every class requires exactly two sparse anchors")
        class_signature[class_id] = members.mean(axis=0)
    return anchor_signature, class_signature


def build_generic_carrier(
    q_views: np.ndarray,
    sample_ids: np.ndarray,
    labeled_ids: np.ndarray,
    labeled_targets: np.ndarray,
    query_chunk_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Build frozen U/T tensors from q and raw-ID sparse supervision only.

    Query chunks change only allocation/iteration; each result has the exact
    same action and sample order as full mode.  The returned arrays are NumPy
    artifacts, hence detached and non-trainable by construction.
    """
    q = np.ascontiguousarray(q_views)
    validate_q_views(q)
    n, v, k = q.shape
    adapter = SampleIdAdapter(sample_ids)
    if adapter.sample_ids.size != n:
        raise ValueError("sample_ids length must equal q_views N")
    labels = np.asarray(labeled_ids, dtype=np.int64)
    targets = np.asarray(labeled_targets, dtype=np.int64)
    if labels.ndim != 1 or targets.shape != labels.shape or labels.size != 2 * k:
        raise ValueError("labeled IDs/targets must contain exactly two anchors per class")
    if np.any(targets < 0) or np.any(targets >= k) or not np.array_equal(np.bincount(targets, minlength=k), np.full(k, 2)):
        raise ValueError("labeled targets must have exactly two anchors for every class")
    anchor_rows = adapter.row_positions(labels)
    anchor_set = set(int(row) for row in anchor_rows)
    query_rows = np.asarray([row for row in range(n) if row not in anchor_set], dtype=np.int64)
    actions, generator_actions, verifier_actions = action_arrays(v)
    s = len(actions)
    p_gen = np.empty((n, s, k), dtype=q.dtype)
    p_ver = np.empty_like(p_gen)
    y_gen = np.empty((n, s), dtype=np.int64)
    conf_gen = np.empty((n, s), dtype=q.dtype)
    support_ver = np.empty((n, s), dtype=q.dtype)
    closure = np.empty((n, s), dtype=np.int8)
    u_cycle = np.empty((n, s), dtype=q.dtype)
    for start, stop in _chunk_ranges(n, query_chunk_size):
        p_gen[start:stop], p_ver[start:stop] = _posteriors_for_chunk(q[start:stop], actions)
        values = compute_u_cycle(p_gen[start:stop], p_ver[start:stop])
        y_gen[start:stop], conf_gen[start:stop], support_ver[start:stop], closure[start:stop], u_cycle[start:stop] = values
    anchor_signature, class_signature = _class_signature(p_gen, anchor_rows, targets)
    pi = np.empty((query_rows.size, s, k), dtype=q.dtype)
    t_rel = np.empty((query_rows.size, labels.size, s), dtype=q.dtype)
    for start, stop in _chunk_ranges(query_rows.size, query_chunk_size):
        query = query_rows[start:stop]
        energy = np.einsum("usk,ck->usc", p_gen[query], class_signature)
        denominator = energy.sum(axis=2)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("T_rel denominator must be finite and strictly positive")
        pi[start:stop] = energy / denominator[..., None]
        t_rel[start:stop] = np.take(pi[start:stop], targets, axis=2).transpose(0, 2, 1)
    if not np.isfinite(pi).all() or not np.isfinite(t_rel).all() or not np.allclose(pi.sum(axis=2), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("T_rel construction produced invalid semantic distributions")
    return {
        "sample_ids": adapter.sample_ids.copy(),
        "query_sample_ids": adapter.sample_ids[query_rows].copy(),
        "anchor_sample_ids": labels.copy(),
        "labeled_ids": labels.copy(),
        "labeled_targets": targets.copy(),
        "query_rows": query_rows,
        "anchor_rows": anchor_rows,
        "generator_actions": generator_actions,
        "verifier_actions": verifier_actions,
        "generator_action_sizes": np.asarray([action.generator_size for action in actions], dtype=np.int64),
        "verifier_action_sizes": np.asarray([action.verifier_size for action in actions], dtype=np.int64),
        "q_views": q.copy(), "p_gen": p_gen, "p_ver": p_ver,
        "y_gen": y_gen, "conf_gen": conf_gen, "support_ver": support_ver, "closure": closure,
        "U_cycle": u_cycle, "query_U_cycle": u_cycle[query_rows].copy(),
        "anchor_signature": anchor_signature, "class_signature": class_signature,
        "Pi": pi, "T_rel": t_rel,
    }
