"""R2 dependency binding for the frozen formal candidate evaluator.

This module deliberately contains no candidate-evaluation implementation.  It
only makes ``trial_runner_a1`` available in the frozen evaluator's module
namespace, where the frozen ``_evaluate_with_runner`` function resolves it.
"""

from __future__ import annotations

from typing import Callable, Mapping

from . import formal_candidate_evaluator as frozen_evaluator
from . import trial_runner_a1
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .study_schema import CandidateConfig, CandidateEvaluation


def _bind_trial_runner_a1() -> None:
    """Supply the single dependency omitted by the frozen evaluator."""
    frozen_evaluator.trial_runner_a1 = trial_runner_a1


# Bind eagerly for imports/introspection and again at delegated entry points so
# monkeypatched dependency tests exercise the same name-resolution path.
_bind_trial_runner_a1()


canonical_candidate_slug = frozen_evaluator.canonical_candidate_slug
candidate_directory = frozen_evaluator.candidate_directory
load_verified_candidate = frozen_evaluator.load_verified_candidate
_evaluation_payload = frozen_evaluator._evaluation_payload
_verify_candidate_payload = frozen_evaluator._verify_candidate_payload
_run_branch = frozen_evaluator._run_branch


def _evaluate_with_runner(
    context: FormalTuningContext,
    candidate: CandidateConfig,
    device: str,
    runner: Callable[
        [CarrierContext, str, CandidateConfig, str, Mapping], dict
    ] | None,
):
    _bind_trial_runner_a1()
    return frozen_evaluator._evaluate_with_runner(context, candidate, device, runner)


def evaluate_candidate(
    context: FormalTuningContext,
    candidate: CandidateConfig,
    device: str = "cuda:0",
    replace_incomplete: bool = False,
    runner: Callable[
        [CarrierContext, str, CandidateConfig, str, Mapping], dict
    ] | None = None,
) -> CandidateEvaluation:
    """Delegate unchanged evaluation semantics after binding the dependency."""
    _bind_trial_runner_a1()
    return frozen_evaluator.evaluate_candidate(
        context,
        candidate,
        device=device,
        replace_incomplete=replace_incomplete,
        runner=runner,
    )


evaluate_formal_candidate = evaluate_candidate
verify_complete_candidate = load_verified_candidate

