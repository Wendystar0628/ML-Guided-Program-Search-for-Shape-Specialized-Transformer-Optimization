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
    AttentionOutputBridge,
    ConfigSpec,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    ProgramConfig,
    ProjectionBackend,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
    ScheduleConfig,
    TritonAttentionParams,
    TritonFFNParams,
    TritonGemmParams,
    TritonNormParams,
    TritonQKVParams,
    fp16_projection_fields,
)

Scalar = str | int | float | bool
DEFAULT_MAX_STRUCTURE_BRANCHES = 36
ProjectionTuple = tuple[
    ProjectionBackend,
    ProjectionBackend,
    ProjectionBackend,
    ProjectionBackend,
]

_PROJECTION_FIELDS = (
    "qkv_projection",
    "attention_output_projection",
    "ffn_input_projection",
    "ffn_output_projection",
)
def _projection_patterns(
    precision_plan: PrecisionPlan,
) -> tuple[tuple[str, ProjectionTuple], ...]:
    """Return every legal per-role FP16 weight implementation pattern."""

    active_fields = tuple(
        field_name
        for field_name in _PROJECTION_FIELDS
        if field_name in fp16_projection_fields(precision_plan)
    )
    if not active_fields:
        return (
            (
                "all_input",
                (
                    ProjectionBackend.INPUT_DTYPE,
                    ProjectionBackend.INPUT_DTYPE,
                    ProjectionBackend.INPUT_DTYPE,
                    ProjectionBackend.INPUT_DTYPE,
                ),
            ),
        )

    def backends(*, shadow_fields: frozenset[str]) -> ProjectionTuple:
        values = tuple(
            (
                ProjectionBackend.INPUT_DTYPE
                if field_name not in active_fields
                else ProjectionBackend.FP16_SHADOW
                if field_name in shadow_fields
                else ProjectionBackend.AUTOCAST_FP16
            )
            for field_name in _PROJECTION_FIELDS
        )
        assert len(values) == 4
        return values  # type: ignore[return-value]

    patterns: list[tuple[str, ProjectionTuple]] = []
    for shadow_count in range(len(active_fields) + 1):
        for fields in itertools.combinations(active_fields, shadow_count):
            shadow_fields = frozenset(fields)
            if not fields:
                name = "all_autocast"
            elif len(fields) == len(active_fields):
                name = "all_shadow"
            elif len(fields) == 1:
                name = f"shadow_{fields[0]}"
            else:
                name = "shadow_" + "__".join(fields)
            patterns.append((name, backends(shadow_fields=shadow_fields)))
    return tuple(patterns)


def _projection_pattern_choices(
    precision_plan: PrecisionPlan,
    *,
    require_qkv_shadow: bool = False,
    require_attention_output_shadow: bool = False,
    require_ffn_input_shadow: bool = False,
    require_ffn_output_shadow: bool = False,
) -> tuple[str, ...]:
    """Return only patterns that can execute the selected fused primitives."""

    requirements = (
        require_qkv_shadow,
        require_attention_output_shadow,
        require_ffn_input_shadow,
        require_ffn_output_shadow,
    )
    return tuple(
        name
        for name, values in _projection_patterns(precision_plan)
        if all(
            not required or values[index] is ProjectionBackend.FP16_SHADOW
            for index, required in enumerate(requirements)
        )
    )


def _projection_backends(
    precision_plan: PrecisionPlan,
    pattern: Scalar,
) -> ProjectionTuple:
    if not isinstance(pattern, str):
        raise TypeError("projection_pattern must be a string")
    for name, values in _projection_patterns(precision_plan):
        if pattern == name:
            return values
    raise ValueError(
        f"projection_pattern {pattern!r} is invalid for {precision_plan.value}"
    )


def _projection_pattern_for(
    precision_plan: PrecisionPlan,
    values: ProjectionTuple,
) -> str | None:
    for name, candidate in _projection_patterns(precision_plan):
        if candidate == values:
            return name
    return None


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


class PlanBuilderLike(Protocol):
    def evaluate(
        self,
        config: ConfigSpec,
        context: Any,
        hardware: Any | None = None,
    ) -> CompilationResultLike: ...


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Compiler inputs plus workload-level scheduling facts."""

    execution_context: Any
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
        value = getattr(self.execution_context, "batch_size", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("compilation context has no valid batch_size")
        return value


@dataclass(frozen=True, slots=True)
class ParameterDomain:
    """One finite conditional parameter domain exposed to TPE."""

    name: str
    choices: tuple[Scalar, ...]
    default: Scalar

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
    precision_plan: PrecisionPlan
    qkv_materialization: QKVMaterialization
    attention_output_bridge: AttentionOutputBridge
    ffn: FFNBackend
    residual_norm: ResidualNormBackend
    initial_norm: InitialNormBackend
    runtime: RuntimeBackend

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention", AttentionBackend(self.attention))
        object.__setattr__(
            self,
            "precision_plan",
            PrecisionPlan(self.precision_plan),
        )
        object.__setattr__(
            self,
            "qkv_materialization",
            QKVMaterialization(self.qkv_materialization),
        )
        object.__setattr__(
            self,
            "attention_output_bridge",
            AttentionOutputBridge(self.attention_output_bridge),
        )
        object.__setattr__(self, "ffn", FFNBackend(self.ffn))
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
            "precision_plan": self.precision_plan.value,
            "qkv_materialization": self.qkv_materialization.value,
            "attention_output_bridge": self.attention_output_bridge.value,
            "ffn": self.ffn.value,
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
            precision_plan=PrecisionPlan.INPUT_DTYPE,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=(AttentionOutputBridge.TORCH_BHSD_TO_BSD),
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=RuntimeBackend.EAGER,
        )

    @classmethod
    def from_config(cls, config: ConfigSpec) -> StructureSpec:
        return cls(
            attention=config.program.attention,
            precision_plan=config.program.precision_plan,
            qkv_materialization=config.program.qkv_materialization,
            attention_output_bridge=config.program.attention_output_bridge,
            ffn=config.program.ffn,
            residual_norm=config.program.residual_norm,
            initial_norm=config.program.initial_norm,
            runtime=config.schedule.runtime,
        )


def _batch_tile_choices(batch_size: int) -> tuple[int, ...]:
    if batch_size <= 1:
        return ()
    return tuple(
        value
        for exponent in range(batch_size.bit_length())
        if (value := 1 << exponent) < batch_size
    )


def _microbatch_choices(batch_size: int) -> tuple[int, ...]:
    return tuple(
        value
        for value in (1, 2, 4, 8, 16, 32)
        if value <= batch_size and batch_size % value == 0
    )


def _attention_defaults(attention: AttentionBackend) -> tuple[int, int, int, int]:
    if attention is AttentionBackend.TRITON_DH8:
        return 32, 32, 4, 2
    if attention is AttentionBackend.TRITON_STREAMING_DH64:
        return 32, 64, 4, 2
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


_GEMM_BLOCK_CHOICES = (16, 32, 64, 128)


def _gemm_tile_choices(
    *,
    output_width: int,
    input_width: int,
    supported_block_k: tuple[int, ...],
) -> tuple[str, ...]:
    """Generate implemented tiles and avoid blocks wider than small matrices."""

    block_n_choices = tuple(
        value for value in _GEMM_BLOCK_CHOICES if value <= output_width
    )
    block_k_choices = tuple(
        value for value in supported_block_k if value <= input_width
    )
    return tuple(
        f"{block_m}x{block_n}x{block_k}"
        for block_m in _GEMM_BLOCK_CHOICES
        for block_n in block_n_choices
        for block_k in block_k_choices
    )


def _decode_gemm_tile(value: Scalar) -> tuple[int, int, int]:
    if not isinstance(value, str) or value.count("x") != 2:
        raise ValueError("GEMM tile must use <block_m>x<block_n>x<block_k>")
    block_m, block_n, block_k = value.split("x")
    return int(block_m), int(block_n), int(block_k)


def _streaming_attention_tile_choices() -> tuple[str, ...]:
    """Cover Shape-14 occupancy and key-block reuse without tiny KV blocks."""

    return tuple(
        f"{block_m}x{block_n}" for block_m in (16, 32, 64) for block_n in (64, 128)
    )


def _context_supports_streaming_dh64(context: SearchContext) -> bool:
    execution = context.execution_context
    return bool(
        context.scope == "streamed"
        and execution.causal
        and not execution.has_valid_token_mask
        and execution.seq_len == 100000
        and execution.d_model == 1024
        and execution.num_heads == 16
        and execution.head_dim == 64
    )


def _context_supports_native_qkv(context: SearchContext) -> bool:
    execution = context.execution_context
    return bool(
        execution.seq_len in {32, 128, 1024}
        and execution.d_model in {32, 128, 1024}
        and execution.num_heads in {1, 2, 4, 16}
        and execution.head_dim is not None
    )


def _context_supports_fused_output_projection(context: SearchContext) -> bool:
    execution = context.execution_context
    return bool(
        execution.seq_len in {32, 128, 1024} and execution.d_model in {32, 128, 1024}
    )


def _context_supports_fused_ffn(context: SearchContext) -> bool:
    execution = context.execution_context
    return bool(
        execution.d_model in {32, 128, 1024} and execution.ffn_dim in {32, 128, 1024}
    )


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
        return "branch-" + _digest(
            {
                "structure": self.structure.to_dict(),
                "scope": self.scope,
                "domains": [
                    {
                        "name": domain.name,
                        "choices": list(domain.choices),
                        "default": domain.default,
                    }
                    for domain in self.domains
                ],
            }
        )

    @property
    def cardinality(self) -> int:
        return math.prod(len(domain.choices) for domain in self.domains)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(domain.name for domain in self.domains)

    def config_at(self, index: int) -> ConfigSpec:
        """Decode one point in the finite branch without materializing it."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.cardinality:
            raise IndexError("branch configuration index is out of range")
        parameters: dict[str, Scalar] = {}
        remainder = index
        for domain in reversed(self.domains):
            remainder, choice_index = divmod(remainder, len(domain.choices))
            parameters[domain.name] = domain.choices[choice_index]
        return self.build(parameters)

    def index_for(self, config: ConfigSpec) -> int | None:
        """Return the mixed-radix index of a branch member."""

        parameters = self.parameters_for(config)
        if parameters is None:
            return None
        index = 0
        for domain in self.domains:
            index *= len(domain.choices)
            index += domain.choices.index(parameters[domain.name])
        return index

    def default_parameters(self) -> dict[str, Scalar]:
        return {domain.name: domain.default for domain in self.domains}

    def default_config(self) -> ConfigSpec:
        """Build the reproducible starting point for this branch."""

        return self.build(self.default_parameters())

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

        (
            qkv_projection,
            attention_output_projection,
            ffn_input_projection,
            ffn_output_projection,
        ) = _projection_backends(
            self.structure.precision_plan,
            parameters["projection_pattern"],
        )

        attention_launch = None
        if self.structure.attention in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
            AttentionBackend.TRITON_STREAMING_DH64,
        }:
            block_m, block_n = _decode_attention_tile(parameters["attention_tile"])
            attention_launch = TritonAttentionParams(
                block_m=block_m,
                block_n=block_n,
                num_warps=int(parameters["attention_num_warps"]),
                num_stages=int(parameters["attention_num_stages"]),
            )

        qkv_launch = None
        if self.structure.qkv_materialization is QKVMaterialization.TRITON_NATIVE_BHSD:
            block_m, block_n, block_k = _decode_gemm_tile(parameters["qkv_gemm_tile"])
            qkv_launch = TritonQKVParams(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=int(parameters["qkv_num_warps"]),
            )

        attention_output_projection_launch = None
        if (
            self.structure.attention_output_bridge
            is AttentionOutputBridge.TRITON_BHSD_PROJECTION
        ):
            block_m, block_n, block_k = _decode_gemm_tile(
                parameters["attention_output_gemm_tile"]
            )
            attention_output_projection_launch = TritonGemmParams(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=int(parameters["attention_output_num_warps"]),
                num_stages=int(parameters["attention_output_num_stages"]),
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

        ffn_launch = None
        if self.structure.ffn is FFNBackend.TRITON_EXACT_GELU:
            ffn_launch = TritonFFNParams(
                block_size=int(parameters["ffn_block_size"]),
                num_warps=int(parameters["ffn_num_warps"]),
            )

        ffn_input_launch = None
        if self.structure.ffn is FFNBackend.TRITON_LINEAR_EXACT_GELU:
            block_m, block_n, block_k = _decode_gemm_tile(
                parameters["ffn_input_gemm_tile"]
            )
            ffn_input_launch = TritonGemmParams(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=int(parameters["ffn_input_num_warps"]),
                num_stages=int(parameters["ffn_input_num_stages"]),
            )

        schedule_values: dict[str, Any] = {
            "runtime": self.structure.runtime,
            "attention_launch": attention_launch,
            "qkv_launch": qkv_launch,
            "attention_output_projection_launch": (attention_output_projection_launch),
            "residual_norm_launch": residual_launch,
            "initial_norm_launch": initial_launch,
            "ffn_launch": ffn_launch,
            "ffn_input_launch": ffn_input_launch,
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
                qkv_projection=qkv_projection,
                attention_output_projection=attention_output_projection,
                ffn_input_projection=ffn_input_projection,
                ffn_output_projection=ffn_output_projection,
                precision_plan=self.structure.precision_plan,
                qkv_materialization=self.structure.qkv_materialization,
                attention_output_bridge=self.structure.attention_output_bridge,
                ffn=self.structure.ffn,
                residual_norm=self.structure.residual_norm,
                initial_norm=self.structure.initial_norm,
            ),
            schedule=ScheduleConfig(**schedule_values),
        )

    def parameters_for(self, config: ConfigSpec) -> dict[str, Scalar] | None:
        if StructureSpec.from_config(config) != self.structure:
            return None
        projection_pattern = _projection_pattern_for(
            self.structure.precision_plan,
            (
                config.program.qkv_projection,
                config.program.attention_output_projection,
                config.program.ffn_input_projection,
                config.program.ffn_output_projection,
            ),
        )
        if projection_pattern is None:
            return None
        values: dict[str, Scalar] = {"projection_pattern": projection_pattern}
        if config.schedule.attention_launch is not None:
            values.update(
                attention_tile=(
                    f"{config.schedule.attention_launch.block_m}x"
                    f"{config.schedule.attention_launch.block_n}"
                ),
                attention_num_warps=config.schedule.attention_launch.num_warps,
                attention_num_stages=config.schedule.attention_launch.num_stages,
            )
        if config.schedule.qkv_launch is not None:
            values.update(
                qkv_gemm_tile=(
                    f"{config.schedule.qkv_launch.block_m}x"
                    f"{config.schedule.qkv_launch.block_n}x"
                    f"{config.schedule.qkv_launch.block_k}"
                ),
                qkv_num_warps=config.schedule.qkv_launch.num_warps,
            )
        if config.schedule.attention_output_projection_launch is not None:
            launch = config.schedule.attention_output_projection_launch
            values.update(
                attention_output_gemm_tile=(
                    f"{launch.block_m}x{launch.block_n}x{launch.block_k}"
                ),
                attention_output_num_warps=launch.num_warps,
                attention_output_num_stages=launch.num_stages,
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
        if config.schedule.ffn_launch is not None:
            values.update(
                ffn_block_size=config.schedule.ffn_launch.block_size,
                ffn_num_warps=config.schedule.ffn_launch.num_warps,
            )
        if config.schedule.ffn_input_launch is not None:
            launch = config.schedule.ffn_input_launch
            values.update(
                ffn_input_gemm_tile=(
                    f"{launch.block_m}x{launch.block_n}x{launch.block_k}"
                ),
                ffn_input_num_warps=launch.num_warps,
                ffn_input_num_stages=launch.num_stages,
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


def _domains_for_structure(
    structure: StructureSpec,
    context: SearchContext,
) -> tuple[ParameterDomain, ...]:
    model_width = int(context.execution_context.d_model)
    ffn_width = int(context.execution_context.ffn_dim)
    projection_choices = _projection_pattern_choices(
        structure.precision_plan,
        require_qkv_shadow=(
            structure.qkv_materialization is QKVMaterialization.TRITON_NATIVE_BHSD
        ),
        require_attention_output_shadow=(
            structure.attention_output_bridge
            is AttentionOutputBridge.TRITON_BHSD_PROJECTION
        ),
        require_ffn_input_shadow=(
            structure.ffn is FFNBackend.TRITON_LINEAR_EXACT_GELU
            and structure.precision_plan is not PrecisionPlan.FP16_FFN_OUTPUT
        ),
        require_ffn_output_shadow=(
            structure.ffn is FFNBackend.TRITON_LINEAR_EXACT_GELU
        ),
    )
    domains: list[ParameterDomain] = [
        ParameterDomain(
            "projection_pattern",
            projection_choices,
            projection_choices[0],
        )
    ]
    if structure.attention in {
        AttentionBackend.TRITON_SHAPE13,
        AttentionBackend.TRITON_DH8,
        AttentionBackend.TRITON_STREAMING_DH64,
    }:
        block_m, block_n, warps, stages = _attention_defaults(structure.attention)
        tile_choices = (
            _streaming_attention_tile_choices()
            if structure.attention is AttentionBackend.TRITON_STREAMING_DH64
            else _attention_tile_choices(context.execution_context.seq_len)
        )
        default_tile = f"{block_m}x{block_n}"
        if default_tile not in tile_choices:
            default_tile = tile_choices[0]
        domains.extend(
            (
                ParameterDomain(
                    "attention_tile",
                    tile_choices,
                    default_tile,
                ),
                ParameterDomain(
                    "attention_num_warps",
                    (2, 4, 8),
                    warps,
                ),
                ParameterDomain(
                    "attention_num_stages",
                    (
                        (1, 2, 3)
                        if structure.attention is AttentionBackend.TRITON_STREAMING_DH64
                        else (1, 2, 3, 4)
                    ),
                    stages,
                ),
            )
        )
    if structure.qkv_materialization is QKVMaterialization.TRITON_NATIVE_BHSD:
        tile_choices = _gemm_tile_choices(
            output_width=model_width,
            input_width=model_width,
            supported_block_k=(16, 32, 64),
        )
        default_tile = "32x32x32" if model_width == 32 else "64x64x32"
        domains.extend(
            (
                ParameterDomain(
                    "qkv_gemm_tile",
                    tile_choices,
                    default_tile,
                ),
                ParameterDomain("qkv_num_warps", (2, 4, 8), 4),
            )
        )
    if (
        structure.attention_output_bridge
        is AttentionOutputBridge.TRITON_BHSD_PROJECTION
    ):
        tile_choices = _gemm_tile_choices(
            output_width=model_width,
            input_width=model_width,
            supported_block_k=(16, 32, 64),
        )
        domains.extend(
            (
                ParameterDomain(
                    "attention_output_gemm_tile",
                    tile_choices,
                    "32x32x32",
                ),
                ParameterDomain(
                    "attention_output_num_warps",
                    (2, 4, 8),
                    4,
                ),
                ParameterDomain(
                    "attention_output_num_stages",
                    (1, 2, 3, 4),
                    2,
                ),
            )
        )
    if model_width == 1024:
        norm_rows = (1, 2)
        norm_warps = (4, 8)
    elif (
        model_width == 32
        and context.execution_context.batch_size == 64
        and context.execution_context.seq_len == 128
        and structure.runtime is not RuntimeBackend.BATCH_TILED_CUDA_GRAPH
        and not context.execution_context.has_valid_token_mask
    ):
        norm_rows = (4, 8, 16)
        norm_warps = (2, 4, 8)
    elif (
        model_width == 32
        and context.execution_context.seq_len == 128
        and structure.runtime is RuntimeBackend.BATCH_TILED_CUDA_GRAPH
        and not context.execution_context.has_valid_token_mask
    ):
        # This intersection is valid for both the general inner-batch kernel
        # and a possible B64 D32 specialization.
        norm_rows = (4, 8)
        norm_warps = (2, 4, 8)
    else:
        norm_rows = (1, 2, 4, 8)
        norm_warps = (1, 2, 4, 8)
    initial_norm_rows = norm_rows
    initial_norm_warps = norm_warps
    if 16 in norm_rows:
        # The D32 specialization applies to residual boundaries only. Initial
        # LayerNorm keeps the general kernel's launch contract.
        initial_norm_rows = (1, 2, 4, 8)
        initial_norm_warps = (1, 2, 4)

    if structure.residual_norm in {
        ResidualNormBackend.TRITON,
        ResidualNormBackend.TRITON_MIXED,
    }:
        domains.extend(
            (
                ParameterDomain(
                    "residual_block_rows",
                    norm_rows,
                    (
                        4
                        if model_width == 32 and 16 in norm_rows
                        else 2
                        if 2 in norm_rows
                        else norm_rows[0]
                    ),
                ),
                ParameterDomain(
                    "residual_num_warps",
                    norm_warps,
                    4 if model_width in {32, 1024} else 2,
                ),
            )
        )
    if structure.initial_norm is InitialNormBackend.TRITON_FP16:
        domains.extend(
            (
                ParameterDomain(
                    "initial_block_rows",
                    initial_norm_rows,
                    2 if 2 in initial_norm_rows else initial_norm_rows[0],
                ),
                ParameterDomain(
                    "initial_num_warps",
                    initial_norm_warps,
                    4 if model_width == 1024 else 2,
                ),
            )
        )
    if structure.ffn is FFNBackend.TRITON_EXACT_GELU:
        domains.extend(
            (
                ParameterDomain(
                    "ffn_block_size",
                    (128, 256, 512, 1024),
                    256,
                ),
                ParameterDomain(
                    "ffn_num_warps",
                    (1, 2, 4, 8),
                    4,
                ),
            )
        )
    if structure.ffn is FFNBackend.TRITON_LINEAR_EXACT_GELU:
        tile_choices = _gemm_tile_choices(
            output_width=ffn_width,
            input_width=model_width,
            supported_block_k=(16, 32, 64, 128),
        )
        default_tile = "32x32x32" if model_width == 32 else "64x64x32"
        domains.extend(
            (
                ParameterDomain(
                    "ffn_input_gemm_tile",
                    tile_choices,
                    default_tile,
                ),
                ParameterDomain(
                    "ffn_input_num_warps",
                    (1, 2, 4, 8),
                    4,
                ),
                ParameterDomain(
                    "ffn_input_num_stages",
                    (1, 2, 3, 4),
                    2,
                ),
            )
        )
    if structure.runtime is RuntimeBackend.COMPILED_FORWARD:
        modes = tuple(sorted(COMPILED_FORWARD_MODES))
        domains.append(ParameterDomain("compile_mode", modes, modes[0]))
    if structure.runtime is RuntimeBackend.BATCH_TILED_CUDA_GRAPH:
        choices = _batch_tile_choices(context.batch_size)
        default = min(choices, key=lambda value: abs(value - 128))
        domains.append(ParameterDomain("batch_tile_size", choices, default))
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
        if structure.attention is AttentionBackend.TRITON_STREAMING_DH64:
            choices = tuple(choice for choice in choices if choice in {1, 2, 4})
        default = 2 if 2 in choices else choices[0]
        domains.append(ParameterDomain("microbatch_size", choices, default))
    return tuple(domains)


def _attention_output_bridge_choices(
    attention: AttentionBackend,
    context: SearchContext,
) -> tuple[AttentionOutputBridge, ...]:
    if attention in {
        AttentionBackend.TRITON_DH8,
        AttentionBackend.TRITON_STREAMING_DH64,
    }:
        return (AttentionOutputBridge.ATTENTION_DIRECT_BSD,)
    fused = (
        (AttentionOutputBridge.TRITON_BHSD_PROJECTION,)
        if _context_supports_fused_output_projection(context)
        else ()
    )
    if attention is AttentionBackend.TRITON_SHAPE13:
        return (
            AttentionOutputBridge.TORCH_BHSD_TO_BSD,
            AttentionOutputBridge.ATTENTION_DIRECT_BSD,
            *fused,
        )
    return (AttentionOutputBridge.TORCH_BHSD_TO_BSD, *fused)


def _structure_specs(context: SearchContext) -> tuple[StructureSpec, ...]:
    scope = context.scope
    if scope not in {"resident", "streamed"}:
        raise ValueError("scope must be resident or streamed")
    portable = StructureSpec(
        attention=AttentionBackend.REFERENCE_STREAMING,
        precision_plan=PrecisionPlan.INPUT_DTYPE,
        qkv_materialization=QKVMaterialization.VIEW,
        attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
        ffn=FFNBackend.TORCH,
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
    execution = context.execution_context
    if execution.has_valid_token_mask or context.batch_size <= 1:
        runtimes = tuple(
            runtime
            for runtime in runtimes
            if runtime is not RuntimeBackend.BATCH_TILED_CUDA_GRAPH
        )
    attentions = tuple(AttentionBackend)
    if execution.has_valid_token_mask:
        attentions = tuple(
            value
            for value in attentions
            if value
            in {
                AttentionBackend.REFERENCE_STREAMING,
                AttentionBackend.CAUSAL_SDPA,
            }
        )
    if not _context_supports_streaming_dh64(context):
        attentions = tuple(
            value
            for value in attentions
            if value is not AttentionBackend.TRITON_STREAMING_DH64
        )
    if not (
        execution.batch_size == 64
        and execution.seq_len == 1024
        and execution.d_model == 128
        and execution.num_heads == 4
    ):
        attentions = tuple(
            value
            for value in attentions
            if value is not AttentionBackend.TRITON_SHAPE13
        )
    if not (
        execution.batch_size == 64
        and execution.seq_len == 128
        and execution.head_dim == 8
        and execution.num_heads in {4, 16}
    ):
        attentions = tuple(
            value for value in attentions if value is not AttentionBackend.TRITON_DH8
        )
    if scope == "streamed" and _context_supports_streaming_dh64(context):
        attentions = tuple(
            value
            for value in attentions
            if value
            in {
                AttentionBackend.REFERENCE_STREAMING,
                AttentionBackend.CAUSAL_SDPA,
                AttentionBackend.TRITON_STREAMING_DH64,
            }
        )

    qkv_materializations = tuple(QKVMaterialization)
    if not _context_supports_native_qkv(context):
        qkv_materializations = tuple(
            value
            for value in qkv_materializations
            if value is not QKVMaterialization.TRITON_NATIVE_BHSD
        )
    ffns = tuple(FFNBackend)
    if not _context_supports_fused_ffn(context):
        ffns = tuple(
            value for value in ffns if value is not FFNBackend.TRITON_LINEAR_EXACT_GELU
        )
    for (
        attention,
        precision_plan,
        qkv_materialization,
        ffn,
        residual_norm,
        initial_norm,
        runtime,
    ) in itertools.product(
        attentions,
        PrecisionPlan,
        qkv_materializations,
        ffns,
        ResidualNormBackend,
        InitialNormBackend,
        runtimes,
    ):
        fp16_fields = fp16_projection_fields(precision_plan)
        if runtime is RuntimeBackend.COMPILED_FORWARD and (
            residual_norm is not ResidualNormBackend.TORCH
            or initial_norm is not InitialNormBackend.TORCH
        ):
            continue
        if runtime is RuntimeBackend.BATCH_TILED_CUDA_GRAPH and attention in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
        }:
            continue
        if (
            execution.has_valid_token_mask
            and residual_norm is ResidualNormBackend.COMPILED
        ):
            continue
        if attention in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
            AttentionBackend.TRITON_STREAMING_DH64,
        } and "qkv_projection" not in fp16_fields:
            continue
        if attention is AttentionBackend.TRITON_STREAMING_DH64 and (
            runtime is not RuntimeBackend.STREAMED
            or "attention_output_projection" not in fp16_fields
        ):
            continue
        if qkv_materialization is QKVMaterialization.TRITON_NATIVE_BHSD and (
            "qkv_projection" not in fp16_fields
            or runtime is RuntimeBackend.COMPILED_FORWARD
        ):
            continue
        if (
            residual_norm is ResidualNormBackend.TRITON_MIXED
            and precision_plan is not PrecisionPlan.FP16_CORE
        ):
            continue
        if initial_norm is InitialNormBackend.TRITON_FP16 and (
            precision_plan is not PrecisionPlan.FP16_CORE
            or residual_norm is not ResidualNormBackend.TRITON_MIXED
        ):
            continue
        if ffn is FFNBackend.COMPILED and runtime is RuntimeBackend.COMPILED_FORWARD:
            continue
        if ffn is FFNBackend.TRITON_EXACT_GELU and (
            precision_plan is not PrecisionPlan.FP16_FFN_OUTPUT
            or runtime is RuntimeBackend.COMPILED_FORWARD
        ):
            continue
        if ffn is FFNBackend.TRITON_LINEAR_EXACT_GELU and (
            "ffn_output_projection" not in fp16_fields
            or runtime is RuntimeBackend.COMPILED_FORWARD
        ):
            continue
        for attention_output_bridge in _attention_output_bridge_choices(
            attention,
            context,
        ):
            if (
                attention_output_bridge is AttentionOutputBridge.TRITON_BHSD_PROJECTION
                and (
                    "attention_output_projection" not in fp16_fields
                    or runtime is RuntimeBackend.COMPILED_FORWARD
                )
            ):
                continue
            structure = StructureSpec(
                attention=attention,
                precision_plan=precision_plan,
                qkv_materialization=qkv_materialization,
                attention_output_bridge=attention_output_bridge,
                ffn=ffn,
                residual_norm=residual_norm,
                initial_norm=initial_norm,
                runtime=runtime,
            )
            if structure.portable:
                continue
            # For resident execution, reference streaming is a single explicit
            # control. In streamed execution it remains a valid inner program.
            if (
                scope == "resident"
                and attention is AttentionBackend.REFERENCE_STREAMING
            ):
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
    """Generate legal branches by passing programmatic structures to a plan builder."""

    def __init__(
        self,
        *,
        plan_builder: PlanBuilderLike,
        context: SearchContext,
        max_branches: int = DEFAULT_MAX_STRUCTURE_BRANCHES,
        required_configs: Iterable[ConfigSpec] = (),
    ) -> None:
        if (
            isinstance(max_branches, bool)
            or not isinstance(max_branches, int)
            or max_branches <= 0
        ):
            raise ValueError("max_branches must be a positive integer")
        self.plan_builder = plan_builder
        self.context = context
        candidates: list[BranchSpace] = []
        for structure in _structure_specs(context):
            branch = BranchSpace(
                structure=structure,
                domains=_domains_for_structure(structure, context),
                scope=context.scope,
            )
            try:
                default = branch.default_config()
            except (TypeError, ValueError):
                continue
            if self.accepted(default):
                candidates.append(branch)
        required: list[BranchSpace] = []
        if context.scope == "resident":
            portable = next(
                (branch for branch in candidates if branch.structure.portable),
                None,
            )
            if portable is not None:
                required.append(portable)
        for config in required_configs:
            structure = StructureSpec.from_config(config)
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
            if branch.parameters_for(config) is not None and self.accepted(config):
                required.append(branch)
        if not candidates:
            raise ValueError("plan builder accepted no search branches")
        branches, mandatory = _pairwise_cover(
            candidates,
            required=required,
            limit=max_branches,
        )
        self.branches = branches
        self.mandatory_branch_ids = mandatory
        self._by_id = {branch.branch_id: branch for branch in self.branches}

    def accepted(self, config: ConfigSpec) -> bool:
        result = self.plan_builder.evaluate(
            config,
            self.context.execution_context,
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
    "DEFAULT_MAX_STRUCTURE_BRANCHES",
    "BranchSpace",
    "ParameterDomain",
    "PlanBuilderLike",
    "ProgramSearchSpace",
    "SearchContext",
    "StructureSpec",
]
