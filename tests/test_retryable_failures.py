from __future__ import annotations

from optuna.trial import TrialState

from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    TrialMeasurement,
)
from autotune.optuna_backend import OptunaBackend
from autotune.search_space import BranchSpace, ParameterDomain, StructureSpec
from autotune.study_storage import SearchStorage, StudyIdentity
from solution.config import ConfigSpec, portable_config


def _branch() -> BranchSpace:
    return BranchSpace(
        structure=StructureSpec.from_config(portable_config()),
        domains=(
            ParameterDomain(
                "projection_pattern",
                ("all_input",),
                default="all_input",
            ),
        ),
        scope="resident",
    )


def _study(tmp_path):
    branch = _branch()
    backend = OptunaBackend(SearchStorage(tmp_path), seed=1234)
    study = backend.create_study(
        StudyIdentity("retryable", branch.branch_id, "test", "search-v1")
    )
    return backend, study, branch


def _measurement(
    config: ConfigSpec,
    *,
    constraints: ConstraintVector | None = None,
) -> TrialMeasurement:
    resolved_constraints = constraints or ConstraintVector()
    return TrialMeasurement(
        config_id=config.config_id,
        fidelity=Fidelity.SCREEN,
        scope=EvaluationScope.RESIDENT,
        objective_ms=1.0,
        median_ms=None if not resolved_constraints.feasible else 1.0,
        constraints=resolved_constraints,
        failure_kind=(
            None if resolved_constraints.feasible else "constraint_violation"
        ),
    )


def test_deterministic_infeasibility_is_a_terminal_complete_observation(
    tmp_path,
) -> None:
    backend, study, branch = _study(tmp_path)
    trial, config = backend.ask(study, branch)

    backend.tell(
        study,
        trial,
        config,
        _measurement(config, constraints=ConstraintVector(runtime=1.0)),
    )

    frozen = study.get_trials(deepcopy=False)[0]
    assert frozen.state is TrialState.COMPLETE
    assert config.config_id in backend.terminal_config_ids(study, branch)
    assert backend.measurement_for(study, branch, config.config_id) is not None
    assert not backend.enqueue(study, branch, config, source="retry")


def test_infrastructure_failure_quarantines_only_on_the_third_attempt(
    tmp_path,
) -> None:
    backend, study, branch = _study(tmp_path)
    config = branch.default_config()

    for attempt in range(1, 4):
        backend, study, branch = _study(tmp_path)
        trial, proposed = backend.ask(study, branch)
        assert proposed == config
        backend.fail_infrastructure(
            study,
            trial,
            proposed,
            RuntimeError("worker connection lost"),
        )

        failed = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))[-1]
        assert failed.user_attrs["infrastructure_failure_attempt"] == attempt
        assert failed.user_attrs["infrastructure_quarantined"] is (attempt == 3)
        assert (config.config_id in backend.terminal_config_ids(study, branch)) is (
            attempt == 3
        )
        assert backend.enqueue(
            study,
            branch,
            config,
            source="next_process_retry",
        ) is (attempt < 3)


def test_duplicate_proposal_failure_does_not_exclude_the_configuration(
    tmp_path,
) -> None:
    backend, study, branch = _study(tmp_path)
    trial, config = backend.ask(study, branch)

    backend.reject_duplicate(study, trial, config.config_id)

    assert config.config_id not in backend.terminal_config_ids(study, branch)
    assert backend.enqueue(study, branch, config, source="real_measurement")

    retry_trial, retry_config = backend.ask(study, branch)
    assert retry_config == config
    backend.tell(study, retry_trial, retry_config, _measurement(retry_config))
    assert config.config_id in backend.terminal_config_ids(study, branch)
