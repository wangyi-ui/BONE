"""JSON-compatible, GT-free audit records for a persisted synthetic/real future study."""

import optuna

from . import protocol
from .optuna_driver import audit_study_coverage, final_selected_candidate, protocol_fingerprint


def boundary_information(best_lrs_by_dataset, selected_exposure: int) -> dict:
    """Delegate both extension and exposure-bound rules to frozen protocol."""
    return {
        "lr_extension": list(protocol.required_lr_extension(best_lrs_by_dataset)),
        "SEMANTIC_EXPOSURE_UPPER_BOUND_HIT": protocol.exposure_upper_bound_hit(selected_exposure),
    }


def study_audit(study: optuna.study.Study, storage_path: str, dataset: str, best_lrs_by_dataset=None) -> dict:
    """Export audit metadata only; it never opens datasets or ground truth."""
    coverage = audit_study_coverage(study)
    selected = None
    boundary = None
    if coverage["complete_candidate_count"]:
        item = final_selected_candidate(study)
        selected = {"semantic_lr": item.semantic_lr, "semantic_exposure": item.semantic_exposure}
        if best_lrs_by_dataset is not None:
            boundary = boundary_information(best_lrs_by_dataset, item.semantic_exposure)
    return {
        "study_name": study.study_name,
        "storage_path": str(storage_path),
        "dataset": dataset,
        "optuna_version": optuna.__version__,
        "protocol_fingerprint": protocol_fingerprint(),
        **coverage,
        "selected_candidate": selected,
        "boundary_hit_information": boundary,
    }
