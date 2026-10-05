"""Exact semantic optimizer construction and state-isolation audit."""
from __future__ import annotations
import torch
from . import protocol

def build_semantic_optimizer(model, candidate: protocol.Candidate):
    if candidate.optimizer_name == "Adam": optimizer=torch.optim.Adam(model.parameters(), lr=candidate.semantic_lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
    elif candidate.optimizer_name == "AdamW": optimizer=torch.optim.AdamW(model.parameters(), lr=candidate.semantic_lr, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)
    elif candidate.optimizer_name == "SGD_MOMENTUM": optimizer=torch.optim.SGD(model.parameters(), lr=candidate.semantic_lr, momentum=0.9, weight_decay=0.0, nesterov=False)
    else: raise ValueError("B5_UNREGISTERED_OPTIMIZER")
    group=optimizer.param_groups[0]
    return optimizer, {"optimizer_name":candidate.optimizer_name,"optimizer_class":type(optimizer).__name__,"lr":group["lr"],"betas":list(group.get("betas", ())) or None,"eps":group.get("eps"),"weight_decay":group["weight_decay"],"momentum":group.get("momentum"),"nesterov":group.get("nesterov"),"optimizer_state_entry_count_before_first_step":len(optimizer.state)}

def assert_independent(native_optimizer, semantic_optimizer) -> None:
    if native_optimizer is semantic_optimizer or id(native_optimizer.state)==id(semantic_optimizer.state): raise RuntimeError("B5_OPTIMIZER_STATE_REUSE")
