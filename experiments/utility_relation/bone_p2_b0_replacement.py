"""P2-B0 Phase-B replacement: native regularization without inter-view consistency."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from experiments.utility_relation.bone_p2_a0_protocol import K, V
from experiments.utility_relation.bone_p2_a1_relation_action import semantic_batch_loss, semantic_epoch
from experiments.utility_relation.bone_p2_a1_protocol import array_sha256
from utils.cl_loss import con_reg_loss


def extract_phase_b_posteriors(model: torch.nn.Module, batch: Sequence[torch.Tensor]) -> torch.Tensor:
    """Capture the four differentiable BONE view posteriors from one test-mode batch forward."""
    observed: list[torch.Tensor] = []

    def capture(_module, _inputs, output: torch.Tensor) -> None:
        observed.append(output)

    handle = model.label_fc.register_forward_hook(capture)
    try:
        model(batch, test=True)
    finally:
        handle.remove()
    batch_size = batch[0].shape[0]
    if len(observed) != V or any(item.shape != (batch_size, K) for item in observed):
        raise ValueError("Phase-B hook must capture [4,B,7] posteriors")
    return torch.stack(observed, dim=0)


def reg_only_phase_b_loss(model: torch.nn.Module, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Return exactly model.reg times the original regularization loss, with no inter-view term."""
    q_views = extract_phase_b_posteriors(model, batch)
    assign_matrices = [q_views[view].T for view in range(V)]
    loss = model.reg * con_reg_loss(assign_matrices)
    return loss, q_views, assign_matrices


def replacement_native_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loader) -> str:
    """Execute one native minibatch schedule using only the original regularization loss."""
    observed_ids: list[np.ndarray] = []
    model.train()
    for batch, _dummy_labels, audit_ids in loader:
        observed_ids.append(audit_ids.detach().cpu().numpy().astype(np.int64))
        optimizer.zero_grad(set_to_none=True)
        loss, _q_views, _assign_matrices = reg_only_phase_b_loss(model, batch)
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("replacement native gradients were not cleared")
    return array_sha256(np.concatenate(observed_ids))
