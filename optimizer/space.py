"""Programmatic, branch-structured search space for Transformer execution."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from solution.config import (
    COMPILED_FORWARD_MODES,
    AttentionBackend,
    ConfigSpec,
    InitialNormBackend,
    LinearBackend,
    ProgramConfig,
    ResidualNormBackend,
    RuntimeBackend,
    ScheduleConfig,
    TritonAttentionParams,
    TritonNormParams,
)


Scalar = str | int | float | bool


class TrialLike(Protocol):
    """Narrow Optuna Trial surface needed by the search-space module."""

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Scalar],
    ) -> Scalar: ...


class CompilationResultLike(Protocol):
    @property
    def accepted(self) -> bool: ...


class ConfigCompilerLike(Protocol):
    def evaluate(
        self,
        config: ConfigSpec,
        context: Any,
        hardware: Any | None = None,
    ) -> CompilationResultLike: ...


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Compiler inputs plus workload-level scheduling facts."""

    compilation_context: Any
    scope: str
    hardware: Any | None = None
    logical_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"resident", "streamed"}:
            raise ValueError("scope must be resident or streamed")
        if self.logical_batch_size is not None and (
            isinstance(self.logical_batch_size, bool)
            or not isinstance(self.logical_batch_size, int)
            or self.logical_batch_size <= 0
        ):
            raise ValueError("logical_batch_size must be a positive integer")

    @property
    def batch_size(self) -> int:
        explicit = self.logical_batch_size
        if explicit is not None:
            return explicit
        value = getattr(self.compilation_context, "batch_size", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("compilation context has no valid batch_size")
        return value


@dataclass(frozen=True, slots=True)
class ParameterDomain:
    """One finite conditional parameter domain exposed to TPE."""

    name: str
    choices: tuple[Scalar, ...]
    default: Scalar
    ordered: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        if not self.choices:
            raise ValueError(f"parameter {self.name!r} needs at least one choice")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"parameter {self.name!r} has duplicate choices")
        if self.default not in self.choices:
            raise ValueError(f"default for {self.name!r} is outside its domain")

    def suggest(self, trial: TrialLike) -> Scalar:
        return trial.suggest_categorical(self.name, self.choices)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Fixed high-level branch with only its active low-level knobs exposed."""

    attention: AttentionBackend
    linear: LinearBackend
    residual_norm: ResidualNormBackend
    initial_norm: InitialNormBackend
    runtime: RuntimeBackend

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention", AttentionBackend(self.attention))
        object.__setattr__(self, "linear", LinearBackend(self.linear))
        object.__setattr__(
            self,
            "residual_norm",
            ResidualNormBackend(self.residual_norm),
        )
        object.__setattr__(
            self,
            "initial_norm",
            InitialNormBackend(self.initial_norm),
        )
        object.__setattr__(self, "runtime", RuntimeBackend(self.runtime))

    def to_dict(self) -> dict[str, str]:
        return {
            "attention": self.attention.value,
            "linear": self.linear.value,
            "residual_norm": self.residual_norm.value,
            "initial_norm": self.initial_norm.value,
            "runtime": self.runtime.value,
        }

    @property
    def branch_id(self) -> str:
        return "branch-" + _digest(self.to_dict())

    @property
    def portable(self) -> bool:
        return self == StructureSpec(
            attention=AttentionBackend.REFERENCE_STREAMING,
            linear=LinearBackend.INPUT_DTYPE,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=RuntimeBackend.EAGER,
        )


def _batch_tile_choices(batch_size: int) -> tuple[int, ...]:
    values = tuple(value for value in (32, 64, 128, 256, 512, 1024) if value < batch_size)
    if values:
        return values
    return (max(1, batch_size // 2),)


def _microbatch_choices(batch_size: int) -> tuple[int, ...]:
    return tuple(
        value
        for value in (1, 2, 4, 8, 16, 32)
        if value <= batch_size and batch_size % value == 0
    )


def _attention_defaults(attention: AttentionBackend) -> tuple[int, int, int, int]:
    if attention is AttentionBackend.TRITON_DH8:
        return 32, 32, 4, 2
    return 64, 64, 4, 2


def _attention_tile_choices(sequence_length: int) -> tuple[str, ...]:
    """Encode only statically legal causal tile pairs as one parameter."""

    blocks = (16, 32, 64, 128)
    return tuple(
        f"{block_m}x{block_n}"
        for block_m in blocks
        for block_n in blocks
        if sequence_length % block_m == 0
        and sequence_length % block_n == 0
        and block_m % block_n == 0
    )


def _decode_attention_tile(value: Scalar) -> tuple[int, int]:
    if not isinstance(value, str) or "x" not in value:
        raise ValueError("attention_tile must use <block_m>x<block_n>")
    block_m, block_n = value.split("x", maxsplit=1)
    return int(block_m), int(block_n)


@dataclass(frozen=True, slots=True)
class BranchSpace:
    """A StructureSpec and the finite parameters that remain active inside it."""

    structure: StructureSpec
    domains: tuple[ParameterDomain, ...]
    scope: str

    def __post_init__(self) -> None:
        names = [domain.name for domain in self.domains]
        if len(names) != len(set(names)):
            raise ValueError("branch parameter names must be unique")
        if self.scope not in {"resident", "streamed"}:
            raise ValueError("scope must be resident or streamed")

    @property
    def branch_id(self) -> str:
        return self.structure.branch_id

    @property
    def cardinality(self) -> int:
        return math.prod(len(domain.choices) for domain in self.domains)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(domain.name for domain in self.domains)

    def default_parameters(self) -> dict[str, Scalar]:
        return {domain.name: domain.default for domain in self.domains}

    def representative_parameter_sets(self, limit: int = 3) -> tuple[dict[str, Scalar], ...]:
        """Cover defaults plus bounded one-coordinate alternatives."""

        if limit <= 0:
            raise ValueError("representative limit must be positive")
        default = self.default_parameters()
        values = [default]
        for domain in self.domains:
            for choice in domain.choices:
                if choice == domain.default:
                    continue
                alternative = dict(default)
                alternative[domain.name] = choice
                values.append(alternative)
                if len(values) >= limit:
                    return tuple(values)
        return tuple(values)

    def suggest_parameters(self, trial: TrialLike) -> dict[str, Scalar]:
        return {domain.name: domain.suggest(trial) for domain in self.domains}

    def suggest(self, trial: TrialLike) -> ConfigSpec:
        return self.build(self.suggest_parameters(trial))

    def build(self, parameters: Mapping[str, Scalar]) -> ConfigSpec:
        expected = set(self.parameter_names)
        if set(parameters) != expected:
            raise ValueError(
                "branch parameters disagree; "
                f"missing={sorted(expected - set(parameters))}; "
                f"unknown={sorted(set(parameters) - expected)}"
            )
        for domain in self.domains:
            if parameters[domain.name] not in domain.choices:
                raise ValueError(f"parameter {domain.name!r} is outside its domain")

        attention_launch = None
        if self.structure.attention in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
        }:
            block_m, block_n = _decode_attention_tile(
                parameters["attention_tile"]
            )
            attention_launch = TritonAttentionParams(
                block_m=block_m,
                block_n=block_n,
                num_warps=int(parameters["attention_num_warps"]),
                num_stages=int(parameters["attention_num_stages"]),
            )

        residual_launch = None
        if self.structure.residual_norm in {
            ResidualNormBackend.TRITON,
            ResidualNormBackend.TRITON_MIXED,
        }:
            residual_launch = TritonNormParams(
                block_rows=int(parameters["residual_block_rows"]),
                num_warps=int(parameters["residual_num_warps"]),
            )

        initial_launch = None
        if self.structure.initial_norm is InitialNormBackend.TRITON_FP16:
            initial_launch = TritonNormParams(
                block_rows=int(parameters["initial_block_rows"]),
                num_warps=int(parameters["initial_num_warps"]),
            )

        schedule_values: dict[str, Any] = {
            "runtime": self.structure.runtime,
            "attention_launch": attention_launch,
            "residual_norm_launch": residual_launch,
            "initial_norm_launch": initial_launch,
            "compile_mode": parameters.get("compile_mode"),
            "batch_tile_size": parameters.get("batch_tile_size"),
            "reuse_unchanged_input": bool(
                parameters.get("reuse_unchanged_input", False)
            ),
        }
        if "microbatch_size" in ScheduleConfig.__dataclass_fields__:
            schedule_values["microbatch_size"] = parameters.get("microbatch_size")
        elif "microbatch_size" in parameters:
            raise RuntimeError(
                "streamed search requires ScheduleConfig.microbatch_size"
            )

        return ConfigSpec(
            program=ProgramConfig(
                attention=self.structure.attention,
                linear=self.structure.linear,
                residual_norm=self.structure.residual_norm,
                initial_norm=self.structure.initial_norm,
            ),
            schedule=ScheduleConfig(**schedule_values),
        )

    def parameters_for(self, config: ConfigSpec) -> dict[str, Scalar] | None:
        if config.program != ProgramConfig(
            attention=self.structure.attention,
            linear=self.structure.linear,
            residual_norm=self.structure.residual_norm,
            initial_norm=self.structure.initial_norm,
        ):
            return None
        if config.schedule.runtime is not self.structure.runtime:
            return None
        values: dict[str, Scalar] = {}
        if config.schedule.attention_launch is not None:
            values.update(
                attention_tile=(
                    f"{config.schedule.attention_launch.block_m}x"
                    f"{config.schedule.attention_launch.block_n}"
                ),
                attention_num_warps=config.schedule.attention_launch.num_warps,
                attention_num_stages=config.schedule.attention_launch.num_stages,
            )
        if config.schedule.residual_norm_launch is not None:
            values.update(
                residual_block_rows=config.schedule.residual_norm_launch.block_rows,
                residual_num_warps=config.schedule.residual_norm_launch.num_warps,
            )
        if config.schedule.initial_norm_launch is not None:
            values.update(
                initial_block_rows=config.schedule.initial_norm_launch.block_rows,
                initial_num_warps=config.schedule.initial_norm_launch.num_warps,
            )
        if config.schedule.compile_mode is not None:
            values["compile_mode"] = config.schedule.compile_mode
        if config.schedule.batch_tile_size is not None:
            values["batch_tile_size"] = config.schedule.batch_tile_size
        if any(domain.name == "reuse_unchanged_input" for domain in self.domains):
            values["reuse_unchanged_input"] = config.schedule.reuse_unchanged_input
        if any(domain.name == "microbatch_size" for domain in self.domains):
            value = getattr(config.schedule, "microbatch_size", None)
            if value is not None:
                values["microbatch_size"] = value
        if set(values) != set(self.parameter_names):
            return None
        if any(values[domain.name] not in domain.choices for domain in self.domains):
            return None
        return values

    def representative_configs(self, limit: int = 3) -> tuple[ConfigSpec, ...]:
        return tuple(
            self.build(parameters)
            for parameters in self.representative_parameter_sets(limit)
        )

    def neighbours(self, config: ConfigSpec) -> tuple[ConfigSpec, ...]:
        """Return bounded single-coordinate neighbours for ordered parameters."""

        current = self.parameters_for(config)
        if current is None:
            return ()
        neighbours: list[ConfigSpec] = []
        seen: set[str] = set()
        for domain in self.domains:
            if not domain.ordered or len(domain.choices) <= 1:
                continue
            index = domain.choices.index(current[domain.name])
            for neighbour_index in (index - 1, index + 1):
                if not 0 <= neighbour_index < len(domain.choices):
                    continue
                parameters = dict(current)
                parameters[domain.name] = domain.choices[neighbour_index]
                candidate = self.build(parameters)
                if candidate.config_id not in seen:
                    seen.add(candidate.config_id)
                    neighbours.append(candidate)
        return tuple(neighbours)


def _domains_for_structure(
    structure: StructureSpec,
    context: SearchContext,
) -> tuple[ParameterDomain, ...]:
    domains: list[ParameterDomain] = []
    if structure.attention in {
        AttentionBackend.TRITON_SHAPE13,
        AttentionBackend.TRITON_DH8,
    }:
        block_m, block_n, warps, stages = _attention_defaults(structure.attention)
        tile_choices = _attention_tile_choices(context.compilation_context.seq_len)
        default_tile = f"{block_m}x{block_n}"
        if default_tile not in tile_choices:
            default_tile = tile_choices[0]
        domains.extend(
            (
                ParameterDomain(
                    "attention_tile",
                    tile_choices,
                    default_tile,
                    True,
                ),
                ParameterDomain(
                    "attention_num_warps",
                    (2, 4, 8),
                    warps,
                    True,
                ),
                ParameterDomain(
                    "attention_num_stages",
                    (1, 2, 3, 4),
                    stages,
                    True,
                ),
            )
        )
    if structure.residual_norm in {
        ResidualNormBackend.TRITON,
        ResidualNormBackend.TRITON_MIXED,
    }:
        domains.extend(
            (
                ParameterDomain(
                    "residual_block_rows",
                    (1, 2, 4, 8),
                    2,
                    True,
                ),
                ParameterDomain(
                    "residual_num_warps",
                    (1, 2, 4, 8),
                    2,
                    True,
                ),
            )
        )
    if structure.initial_norm is InitialNormBackend.TRITON_FP16:
        domains.extend(
            (
                ParameterDomain(
                    "initial_block_rows",
                    (1, 2, 4, 8),
                    2,
                    True,
                ),
                ParameterDomain(
                    "initial_num_warps",
                    (1, 2, 4, 8),
                    2,
                    True,
                ),
            )
        )
    if structure.runtime is RuntimeBackend.COMPILED_FORWARD:
        modes = tuple(sorted(COMPILED_FORWARD_MODES))
        domains.append(ParameterDomain("compile_mode", modes, modes[0]))
    if structure.runtime is RuntimeBackend.BATCH_TILED_CUDA_GRAPH:
        choices = _batch_tile_choices(context.batch_size)
        default = min(choices, key=lambda value: abs(value - 128))
        domains.append(ParameterDomain("batch_tile_size", choices, default, True))
    if structure.runtime is RuntimeBackend.CUDA_GRAPH:
        domains.append(
            ParameterDomain(
                "reuse_unchanged_input",
                (False, True),
                False,
            )
        )
    if context.scope == "streamed":
        choices = _microbatch_choices(context.batch_size)
        default = 2 if 2 in choices else choices[0]
        domains.append(ParameterDomain("microbatch_size", choices, default, True))
    return tuple(domains)


def _structure_specs(scope: str) -> tuple[StructureSpec, ...]:
    if scope not in {"resident", "streamed"}:
        raise ValueError("scope must be resident or streamed")
    portable = StructureSpec(
        attention=AttentionBackend.REFERENCE_STREAMING,
        linear=LinearBackend.INPUT_DTYPE,
        residual_norm=ResidualNormBackend.TORCH,
        initial_norm=InitialNormBackend.TORCH,
        runtime=RuntimeBackend.EAGER,
    )
    values = [] if scope == "streamed" else [portable]
    runtimes = (
        (RuntimeBackend.STREAMED,)
        if scope == "streamed"
        else tuple(
            runtime
            for runtime in RuntimeBackend
            if runtime is not RuntimeBackend.STREAMED
        )
    )
    for attention, linear, residual_norm, initial_norm, runtime in itertools.product(
        AttentionBackend,
        LinearBackend,
        ResidualNormBackend,
        InitialNormBackend,
        runtimes,
    ):
        structure = StructureSpec(
            attention=attention,
            linear=linear,
            residual_norm=residual_norm,
            initial_norm=initial_norm,
            runtime=runtime,
        )
        if structure.portable:
            continue
        # For resident execution, reference streaming is a single explicit
        # control. In streamed execution it remains a valid inner program.
        if scope == "resident" and attention is AttentionBackend.REFERENCE_STREAMING:
            continue
        values.append(structure)
    return tuple(values)


PrimitiveToken = tuple[str, str]
PrimitivePair = tuple[PrimitiveToken, PrimitiveToken]


def _primitive_tokens(branch: BranchSpace) -> frozenset[PrimitiveToken]:
    return frozenset(branch.structure.to_dict().items())


def _primitive_pairs(branch: BranchSpace) -> frozenset[PrimitivePair]:
    tokens = sorted(_primitive_tokens(branch))
    return frozenset(itertools.combinations(tokens, 2))


def _pairwise_cover(
    candidates: Sequence[BranchSpace],
    *,
    required: Sequence[BranchSpace],
    limit: int,
) -> tuple[tuple[BranchSpace, ...], frozenset[str]]:
    """Select a bounded deterministic covering array from legal structures."""

    by_id = {branch.branch_id: branch for branch in candidates}
    selected: list[BranchSpace] = []
    selected_ids: set[str] = set()
    for branch in required:
        candidate = by_id.get(branch.branch_id)
        if candidate is None or candidate.branch_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.branch_id)
    if len(selected) > limit:
        raise ValueError("max_branches is smaller than required branch coverage")

    universe_singletons = frozenset(
        token for branch in candidates for token in _primitive_tokens(branch)
    )
    universe_pairs = frozenset(
        pair for branch in candidates for pair in _primitive_pairs(branch)
    )
    covered_singletons = frozenset(
        token for branch in selected for token in _primitive_tokens(branch)
    )
    covered_pairs = frozenset(
        pair for branch in selected for pair in _primitive_pairs(branch)
    )

    # First guarantee one representative for every legal primitive value. The
    # resulting prefix is the mandatory structure screen.
    while covered_singletons != universe_singletons and len(selected) < limit:
        remaining = [
            branch for branch in candidates if branch.branch_id not in selected_ids
        ]
        if not remaining:
            break
        branch = max(
            remaining,
            key=lambda item: (
                len(_primitive_tokens(item) - covered_singletons),
                len(_primitive_pairs(item) - covered_pairs),
                -item.cardinality,
                item.branch_id,
            ),
        )
        selected.append(branch)
        selected_ids.add(branch.branch_id)
        covered_singletons |= _primitive_tokens(branch)
        covered_pairs |= _primitive_pairs(branch)
    mandatory_ids = frozenset(selected_ids)
    if covered_singletons != universe_singletons:
        raise ValueError(
            "max_branches is too small to cover every legal primitive value"
        )

    # Spend the remaining structure budget on legal pairwise interactions.
    while covered_pairs != universe_pairs and len(selected) < limit:
        remaining = [
            branch for branch in candidates if branch.branch_id not in selected_ids
        ]
        if not remaining:
            break
        branch = max(
            remaining,
            key=lambda item: (
                len(_primitive_pairs(item) - covered_pairs),
                -item.cardinality,
                item.branch_id,
            ),
        )
        gain = _primitive_pairs(branch) - covered_pairs
        if not gain:
            break
        selected.append(branch)
        selected_ids.add(branch.branch_id)
        covered_singletons |= _primitive_tokens(branch)
        covered_pairs |= _primitive_pairs(branch)
    return tuple(selected), mandatory_ids


class ProgramSearchSpace:
    """Generate legal branches by passing programmatic structures to Compiler."""

    def __init__(
        self,
        *,
        compiler: ConfigCompilerLike,
        context: SearchContext,
        max_branches: int = 24,
        required_configs: Iterable[ConfigSpec] = (),
    ) -> None:
        if (
            isinstance(max_branches, bool)
            or not isinstance(max_branches, int)
            or max_branches <= 0
        ):
            raise ValueError("max_branches must be a positive integer")
        self.compiler = compiler
        self.context = context
        candidates: list[BranchSpace] = []
        for structure in _structure_specs(context.scope):
            branch = BranchSpace(
                structure=structure,
                domains=_domains_for_structure(structure, context),
                scope=context.scope,
            )
            try:
                representatives = branch.representative_configs()
            except (TypeError, ValueError):
                continue
            if any(self.accepted(config) for config in representatives):
                candidates.append(branch)
        required: list[BranchSpace] = []
        if context.scope == "resident":
            portable = next(
                (
                    branch
                    for branch in candidates
                    if branch.structure.portable
                ),
                None,
            )
            if portable is not None:
                required.append(portable)
        for config in required_configs:
            structure = StructureSpec(
                attention=config.program.attention,
                linear=config.program.linear,
                residual_norm=config.program.residual_norm,
                initial_norm=config.program.initial_norm,
                runtime=config.schedule.runtime,
            )
            branch = BranchSpace(
                structure=structure,
                domains=_domains_for_structure(structure, context),
                scope=context.scope,
            )
            if (
                branch.parameters_for(config) is not None
                and self.accepted(config)
                and all(item.branch_id != branch.branch_id for item in candidates)
            ):
                candidates.append(branch)
            if (
                branch.parameters_for(config) is not None
                and self.accepted(config)
            ):
                required.append(branch)
        if not candidates:
            raise ValueError("config compiler accepted no search branches")
        branches, mandatory = _pairwise_cover(
            candidates,
            required=required,
            limit=max_branches,
        )
        self.branches = branches
        self.mandatory_branch_ids = mandatory
        self._by_id = {branch.branch_id: branch for branch in self.branches}

    def accepted(self, config: ConfigSpec) -> bool:
        result = self.compiler.evaluate(
            config,
            self.context.compilation_context,
            self.context.hardware,
        )
        return bool(result.accepted)

    def branch(self, branch_id: str) -> BranchSpace:
        try:
            return self._by_id[branch_id]
        except KeyError as exc:
            raise KeyError(f"unknown branch_id: {branch_id}") from exc

    def branch_for(self, config: ConfigSpec) -> BranchSpace | None:
        for branch in self.branches:
            if branch.parameters_for(config) is not None:
                return branch
        return None


__all__ = [
    "BranchSpace",
    "ConfigCompilerLike",
    "ParameterDomain",
    "ProgramSearchSpace",
    "SearchContext",
    "StructureSpec",
]
