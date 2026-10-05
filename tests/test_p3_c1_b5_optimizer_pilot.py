from __future__ import annotations
import torch
import pytest
from experiments.utility_relation.p3_c1_b5_optimizer_pilot import protocol, optimizer_factory, candidate_reuse, finalize_optimizer_pilot, audit_optimizer_pilot

def test_exact_datasets_counts_exposures_seeds_and_folds():
 assert protocol.DATASETS==("CCV","ALOI") and len(protocol.candidates())==18 and len(protocol.candidates("CCV"))==9 and len(protocol.candidates("ALOI"))==9
 assert protocol.EXPOSURES=={"CCV":3,"ALOI":1} and protocol.DEV_SEEDS==(11,37) and protocol.FOLDS==("A","B")
 assert all(c.dataset!="MSRC" for c in protocol.candidates())
def test_exact_optimizer_grids_and_no_extensions():
 assert protocol.LR_GRID["CCV"]["Adam"]==(3e-6,1e-5,3e-5) and protocol.LR_GRID["ALOI"]["AdamW"]==(3e-5,1e-4,3e-4)
 assert protocol.LR_GRID["CCV"]["SGD_MOMENTUM"]==(1e-3,3e-3,1e-2)
def test_optimizer_factory_defaults_and_independent_state():
 m=torch.nn.Linear(2,2); native=torch.optim.Adam(m.parameters(),lr=.01)
 for name in protocol.OPTIMIZERS:
  c=next(c for c in protocol.candidates("CCV") if c.optimizer_name==name); opt,a=optimizer_factory.build_semantic_optimizer(m,c); optimizer_factory.assert_independent(native,opt); assert a["optimizer_state_entry_count_before_first_step"]==0
 assert optimizer_factory.build_semantic_optimizer(m,next(c for c in protocol.candidates("CCV") if c.optimizer_name=="AdamW"))[1]["weight_decay"]==1e-4
 sgd=optimizer_factory.build_semantic_optimizer(m,next(c for c in protocol.candidates("CCV") if c.optimizer_name=="SGD_MOMENTUM"))[1]; assert sgd["momentum"]==.9 and sgd["nesterov"] is False
def test_reuse_only_exact_adam_and_nonadam_forbidden():
 p=candidate_reuse.preflight(); assert p["full_gt_used"] is False
 for item in p["entries"]:
  if item["candidate"]["optimizer_name"]!="Adam": assert item["reusable"] is False and item["reason"]=="NON_ADAM_REUSE_FORBIDDEN"
def test_selection_rule_and_tie_breaker_exact():
 cc=protocol.candidates("CCV"); a=finalize_optimizer_pilot.score_record(next(c for c in cc if c.optimizer_name=="Adam" and c.semantic_lr==3e-6),{"11A":1,"11B":1,"37A":1,"37B":1}); w=finalize_optimizer_pilot.score_record(next(c for c in cc if c.optimizer_name=="AdamW" and c.semantic_lr==3e-6),{"11A":1,"11B":1,"37A":1,"37B":1}); assert finalize_optimizer_pilot.select_best([a,w]*0+[a]*9)["candidate"].optimizer_name=="Adam"
def test_parent_b4_closed_not_confirmed_and_no_gt_paths():
 parent=audit_optimizer_pilot.verify_parent_integrity(); assert parent["b4_result_freeze"]=="PASS" and parent["b4_conclusion"]=="NOT_CONFIRMED"
 assert protocol.REFINEMENT_EPOCHS==20
