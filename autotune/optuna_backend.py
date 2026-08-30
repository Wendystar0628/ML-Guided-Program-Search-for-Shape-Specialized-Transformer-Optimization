"""Small Optuna adapter for compatible, constraint-aware Level-1 studies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState

from solution.config import ConfigSpec

from .evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    TrialMeasurement,
)
from .search_space import BranchSpace
from .study_storage import SearchStorage, StudyIdentity


def _constraints_from_trial(trial: FrozenTrial) -> Sequence[float]:
    """Return finite constraints for Optuna's feasibility-aware TPE."""

    raw = trial.user_attrs.get("constraints")
    try:
        return ConstraintVector.from_value(raw).as_tuple()
    except (TypeError, ValueError):
        # A completed trial without the autotuner contract must never be treated
        # as feasible. Infrastructure failures use TrialState.FAIL and do not
        # reach this callback.
        return (1.0, 1.0, 1.0, 1.0)


def _branch_seed(seed: int, identity: StudyIdentity) -> int:
    branch_digest = identity.branch_id.removeprefix("branch-")
    return (seed ^ int(branch_digest[:8], 16)) & 0xFFFFFFFF


def startup_trial_count(
    branch: BranchSpace,
) -> int:
    """Use a stable absolute startup threshold for branch-local TPE."""

    return min(10, branch.cardinality)


def _optional_mapping(value: object, *, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object or null")
    return dict(value)


def _optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric or null")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer or null")
    return value


def measurement_from_frozen_trial(trial: FrozenTrial) -> TrialMeasurement:
    """Reconstruct the typed Level-1 observation stored in one Trial."""

    if trial.state is not TrialState.COMPLETE or trial.value is None:
        raise ValueError("trial has no complete observation")
    attrs = trial.user_attrs
    config_payload = attrs.get("config")
    config = ConfigSpec.from_dict(config_payload)
    constraints = ConstraintVector.from_value(attrs.get("constraints"))
    metrics = attrs.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("trial metrics must be an object")
    failure_kind = attrs.get("failure_kind")
    if failure_kind is not None and not isinstance(failure_kind, str):
        raise ValueError("failure_kind must be a string or null")
    return TrialMeasurement(
        config_id=config.config_id,
        fidelity=Fidelity(attrs.get("fidelity")),
        scope=EvaluationScope(attrs.get("scope")),
        objective_ms=float(trial.value),
        constraints=constraints,
        median_ms=_optional_float(attrs.get("median_ms"), field="median_ms"),
        p90_ms=_optional_float(attrs.get("p90_ms"), field="p90_ms"),
        peak_memory_bytes=_optional_int(
            attrs.get("peak_memory_bytes"),
            field="peak_memory_bytes",
        ),
        max_tolerance_ratio=_optional_float(
            attrs.get("max_tolerance_ratio"),
            field="max_tolerance_ratio",
        ),
        expected_execution_signature=_optional_mapping(
            attrs.get("expected_execution_signature"),
            field="expected_execution_signature",
        ),
        actual_execution_signature=_optional_mapping(
            attrs.get("actual_execution_signature"),
            field="actual_execution_signature",
        ),
        failure_kind=failure_kind,
        metrics=dict(metrics),
    )


@dataclass(frozen=True, slots=True)
class CompletedTrial:
    """Typed view of one Level-1 Study observation."""

    study_name: str
    branch_id: str
    number: int
    config: ConfigSpec
    measurement: TrialMeasurement

    @property
    def feasible(self) -> bool:
        return self.measurement.feasible


class OptunaBackend:
    """Own Optuna persistence and keep it out of the search algorithm."""

    def __init__(
        self,
        storage: SearchStorage,
        *,
        seed: int,
        n_startup_trials: int = 4,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if n_startup_trials <= 0:
            raise ValueError("n_startup_trials must be positive")
        self.storage = storage
        self.seed = seed
        self.n_startup_trials = n_startup_trials

    def create_study(
        self,
        identity: StudyIdentity,
        *,
        n_startup_trials: int | None = None,
    ) -> Study:
        """Create or resume exactly one compatible branch Study."""

        startup_trials = (
            self.n_startup_trials if n_startup_trials is None else n_startup_trials
        )
        if (
            isinstance(startup_trials, bool)
            or not isinstance(startup_trials, int)
            or startup_trials <= 0
        ):
            raise ValueError("n_startup_trials must be positive")
        sampler = TPESampler(
            seed=_branch_seed(self.seed, identity),
            # This is the absolute Study threshold. Optuna therefore counts
            # compatible preloaded and historical COMPLETE trials toward it.
            n_startup_trials=startup_trials,
            multivariate=True,
            group=True,
            constraints_func=_constraints_from_trial,
        )
        study = optuna.create_study(
            study_name=identity.study_name,
            storage=self.storage.database_url,
            sampler=sampler,
            direction="minimize",
            load_if_exists=True,
        )
        return study

    @classmethod
    def enqueue(
        cls,
        study: Study,
        branch: BranchSpace,
        config: ConfigSpec,
        *,
        source: str,
    ) -> bool:
        """Queue a compatible control, incumbent, or warm-start configuration."""

        parameters = branch.parameters_for(config)
        if parameters is None:
            return False
        if config.config_id in cls.trial_config_ids(study, branch):
            return False
        study.enqueue_trial(
            parameters,
            user_attrs={
                "queued_config": config.to_dict(),
                "warm_start_source": source,
            },
            skip_if_exists=True,
        )
        return True

    @staticmethod
    def ask(study: Study, branch: BranchSpace) -> tuple[Trial, ConfigSpec]:
        trial = study.ask()
        config = branch.suggest(trial)
        return trial, config

    @staticmethod
    def tell(
        study: Study,
        trial: Trial,
        config: ConfigSpec,
        measurement: TrialMeasurement,
    ) -> None:
        if measurement.fidelity is not Fidelity.SCREEN:
            raise ValueError("only Level-1 SCREEN measurements belong in TPE")
        for key, value in measurement.to_user_attrs(config).items():
            trial.set_user_attr(key, value)
        study.tell(trial, measurement.objective_ms)

    @staticmethod
    def record_completed(
        study: Study,
        branch: BranchSpace,
        config: ConfigSpec,
        measurement: TrialMeasurement,
        *,
        source: str,
    ) -> None:
        """Insert an explicitly generated Level-1 observation."""

        if measurement.fidelity is not Fidelity.SCREEN:
            raise ValueError("only Level-1 SCREEN measurements belong in TPE")
        parameters = branch.parameters_for(config)
        if parameters is None:
            raise ValueError("config does not belong to the branch")
        distributions = {
            domain.name: CategoricalDistribution(list(domain.choices))
            for domain in branch.domains
        }
        attrs = measurement.to_user_attrs(config)
        attrs["generated_source"] = source
        study.add_trial(
            optuna.trial.create_trial(
                params=parameters,
                distributions=distributions,
                value=measurement.objective_ms,
                user_attrs=attrs,
                system_attrs={"constraints": measurement.constraints.as_tuple()},
            )
        )

    @staticmethod
    def fail_infrastructure(
        study: Study,
        trial: Trial,
        config: ConfigSpec,
        error: BaseException,
    ) -> None:
        """Record an infrastructure failure without teaching TPE a fake score."""

        trial.set_user_attr("config", config.to_dict())
        trial.set_user_attr("config_id", config.config_id)
        trial.set_user_attr("infrastructure_error", type(error).__name__)
        trial.set_user_attr("infrastructure_message", str(error)[:1000])
        study.tell(trial, state=TrialState.FAIL)

    @staticmethod
    def reject_duplicate(
        study: Study,
        trial: Trial,
        config_id: str,
    ) -> None:
        """Discard a zero-information proposal without teaching it to TPE."""

        trial.set_user_attr("duplicate_config_id", config_id)
        study.tell(trial, state=TrialState.FAIL)

    @staticmethod
    def _trial_config(
        frozen: FrozenTrial,
        branch: BranchSpace,
    ) -> ConfigSpec | None:
        for attribute in ("config", "queued_config"):
            payload = frozen.user_attrs.get(attribute)
            if payload is None:
                continue
            try:
                config = ConfigSpec.from_dict(payload)
            except (TypeError, ValueError):
                continue
            if branch.parameters_for(config) is not None:
                return config
        if not frozen.params:
            return None
        try:
            return branch.build(frozen.params)
        except (TypeError, ValueError):
            return None

    @classmethod
    def trial_configs(
        cls,
        study: Study,
        branch: BranchSpace,
        *,
        states: tuple[TrialState, ...] | None = None,
    ) -> tuple[ConfigSpec, ...]:
        """Return unique compatible configurations already present in a Study."""

        values: dict[str, ConfigSpec] = {}
        for frozen in study.get_trials(deepcopy=False, states=states):
            config = cls._trial_config(frozen, branch)
            if config is not None:
                values.setdefault(config.config_id, config)
        return tuple(values.values())

    @classmethod
    def trial_config_ids(
        cls,
        study: Study,
        branch: BranchSpace,
        *,
        states: tuple[TrialState, ...] | None = None,
    ) -> frozenset[str]:
        """Return compatible configuration identities already in a Study."""

        return frozenset(
            config.config_id
            for config in cls.trial_configs(study, branch, states=states)
        )

    @classmethod
    def terminal_configs(
        cls,
        study: Study,
        branch: BranchSpace,
    ) -> tuple[ConfigSpec, ...]:
        return cls.trial_configs(
            study,
            branch,
            states=(TrialState.COMPLETE, TrialState.FAIL),
        )

    @classmethod
    def terminal_config_ids(
        cls,
        study: Study,
        branch: BranchSpace,
    ) -> frozenset[str]:
        """Return unique configurations with a COMPLETE or FAIL outcome."""

        return frozenset(
            config.config_id for config in cls.terminal_configs(study, branch)
        )

    @staticmethod
    def completed_trials(
        study: Study,
        branch: BranchSpace,
    ) -> tuple[CompletedTrial, ...]:
        values: list[CompletedTrial] = []
        for frozen in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            try:
                measurement = measurement_from_frozen_trial(frozen)
                config = ConfigSpec.from_dict(frozen.user_attrs.get("config"))
            except (TypeError, ValueError):
                continue
            if measurement.fidelity is not Fidelity.SCREEN:
                continue
            if branch.parameters_for(config) is None:
                continue
            values.append(
                CompletedTrial(
                    study_name=study.study_name,
                    branch_id=branch.branch_id,
                    number=frozen.number,
                    config=config,
                    measurement=measurement,
                )
            )
        return tuple(values)

    @classmethod
    def feasible_trials(
        cls,
        study: Study,
        branch: BranchSpace,
    ) -> tuple[CompletedTrial, ...]:
        """Return one best feasible Level-1 observation per ConfigSpec."""

        best_by_config: dict[str, CompletedTrial] = {}
        for completed in cls.completed_trials(study, branch):
            if not completed.feasible:
                continue
            previous = best_by_config.get(completed.config.config_id)
            if (
                previous is None
                or completed.measurement.objective_ms
                < previous.measurement.objective_ms
            ):
                best_by_config[completed.config.config_id] = completed
        return tuple(
            sorted(
                best_by_config.values(),
                key=lambda item: (item.measurement.objective_ms, item.number),
            )
        )

    @classmethod
    def best_feasible(
        cls,
        study: Study,
        branch: BranchSpace,
    ) -> CompletedTrial | None:
        values = cls.feasible_trials(study, branch)
        return values[0] if values else None

    @classmethod
    def measurement_for(
        cls,
        study: Study,
        branch: BranchSpace,
        config_id: str,
    ) -> TrialMeasurement | None:
        candidates = (
            item
            for item in cls.completed_trials(study, branch)
            if item.config.config_id == config_id
        )
        best = min(
            candidates,
            key=lambda item: item.measurement.objective_ms,
            default=None,
        )
        return None if best is None else best.measurement


__all__ = [
    "CompletedTrial",
    "OptunaBackend",
    "measurement_from_frozen_trial",
    "startup_trial_count",
]
