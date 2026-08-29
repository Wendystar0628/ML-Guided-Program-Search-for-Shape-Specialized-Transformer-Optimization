"""Typed, serializable configuration for Transformer program search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

CONFIG_SCHEMA_VERSION = 3


class AttentionBackend(StrEnum):
    """Implemented attention primitives, not hand-written policy combinations."""

    REFERENCE_STREAMING = "reference_streaming"
    CAUSAL_SDPA = "causal_sdpa"
    FP16_EFFICIENT_SDPA = "fp16_efficient_sdpa"
    FP16_CUDNN_SDPA = "fp16_cudnn_sdpa"
    TRITON_SHAPE13 = "triton_shape13"
    TRITON_DH8 = "triton_dh8"


class ProjectionBackend(StrEnum):
    """Per-role projection execution and weight-storage choices."""

    INPUT_DTYPE = "input_dtype"
    AUTOCAST_FP16 = "autocast_fp16"
    FP16_SHADOW = "fp16_shadow"


class PrecisionPlan(StrEnum):
    """Finite projection/cast graphs exposed to program search."""

    INPUT_DTYPE = "input_dtype"
    FP16_QKV_ATTENTION = "fp16_qkv_attention"
    FP16_ATTENTION_BRANCH = "fp16_attention_branch"
    FP16_FFN_BRANCH = "fp16_ffn_branch"
    FP16_FFN_OUTPUT = "fp16_ffn_output"
    FP16_CORE = "fp16_core"


class QKVMaterialization(StrEnum):
    """Materialization choices between packed QKV and attention."""

    VIEW = "view"
    CONTIGUOUS = "contiguous"


class AttentionOutputBridge(StrEnum):
    """Implemented bridges from attention output to the output projection."""

    TORCH_BHSD_TO_BSD = "torch_bhsd_to_bsd"
    ATTENTION_DIRECT_BSD = "attention_direct_bsd"


class FFNBackend(StrEnum):
    """Execution choices for the two-projection Exact-GELU FFN."""

    TORCH = "torch"
    COMPILED = "compiled"
    TRITON_EXACT_GELU = "triton_exact_gelu"


class ResidualNormBackend(StrEnum):
    """Residual-add plus LayerNorm primitives."""

    TORCH = "torch"
    COMPILED = "compiled"
    TRITON = "triton"
    TRITON_MIXED = "triton_mixed"


class InitialNormBackend(StrEnum):
    """Initial LayerNorm primitives."""

    TORCH = "torch"
    TRITON_FP16 = "triton_fp16"


class RuntimeBackend(StrEnum):
    """Outer execution schedules."""

    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"
    BATCH_TILED_CUDA_GRAPH = "batch_tiled_cuda_graph"
    COMPILED_FORWARD = "compiled_forward"
    STREAMED = "streamed"


class AttentionOutputLayout(StrEnum):
    """Layouts produced by the selected attention primitive."""

    BHSD = "bhsd"
    BSD = "bsd"


_PROJECTION_FIELDS = (
    "qkv_projection",
    "attention_output_projection",
    "ffn_input_projection",
    "ffn_output_projection",
)

_PRECISION_FP16_PROJECTIONS = {
    PrecisionPlan.INPUT_DTYPE: frozenset(),
    PrecisionPlan.FP16_QKV_ATTENTION: frozenset({"qkv_projection"}),
    PrecisionPlan.FP16_ATTENTION_BRANCH: frozenset(
        {"qkv_projection", "attention_output_projection"}
    ),
    PrecisionPlan.FP16_FFN_BRANCH: frozenset(
        {"ffn_input_projection", "ffn_output_projection"}
    ),
    PrecisionPlan.FP16_FFN_OUTPUT: frozenset({"ffn_output_projection"}),
    PrecisionPlan.FP16_CORE: frozenset(_PROJECTION_FIELDS),
}


COMPILED_FORWARD_MODES = frozenset(
    {
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
)
DEFAULT_COMPILED_FORWARD_MODE = "max-autotune"


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _power_of_two(value: object, *, field: str) -> int:
    normalized = _positive_int(value, field=field)
    if normalized & (normalized - 1):
        raise ValueError(f"{field} must be a power of two")
    return normalized


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"invalid {field} fields; missing={sorted(missing)}; "
            f"unknown={sorted(unknown)}"
        )


@dataclass(frozen=True, slots=True)
class TritonAttentionParams:
    """Compile-time launch parameters for a Triton attention template."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"block_m", "block_n", "num_warps", "num_stages"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "block_m",
            _power_of_two(self.block_m, field="attention_launch.block_m"),
        )
        object.__setattr__(
            self,
            "block_n",
            _power_of_two(self.block_n, field="attention_launch.block_n"),
        )
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("attention_launch.num_warps must be one of 1, 2, 4, 8")
        if (
            isinstance(self.num_stages, bool)
            or not isinstance(self.num_stages, int)
            or not 1 <= self.num_stages <= 8
        ):
            raise ValueError("attention_launch.num_stages must be in [1, 8]")

    def to_dict(self) -> dict[str, int]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TritonAttentionParams:
        value = _mapping(payload, field="attention_launch")
        _exact_fields(value, cls._FIELDS, field="attention_launch")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TritonNormParams:
    """Compile-time launch parameters for a Triton norm template."""

    block_rows: int
    num_warps: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"block_rows", "num_warps"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "block_rows",
            _power_of_two(self.block_rows, field="norm_launch.block_rows"),
        )
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("norm_launch.num_warps must be one of 1, 2, 4, 8")

    def to_dict(self) -> dict[str, int]:
        return {
            "block_rows": self.block_rows,
            "num_warps": self.num_warps,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TritonNormParams:
        value = _mapping(payload, field="norm_launch")
        _exact_fields(value, cls._FIELDS, field="norm_launch")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TritonFFNParams:
    """Launch parameters for the fused Exact-GELU and output-cast primitive."""

    block_size: int
    num_warps: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"block_size", "num_warps"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "block_size",
            _power_of_two(self.block_size, field="ffn_launch.block_size"),
        )
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("ffn_launch.num_warps must be one of 1, 2, 4, 8")

    def to_dict(self) -> dict[str, int]:
        return {
            "block_size": self.block_size,
            "num_warps": self.num_warps,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TritonFFNParams:
        value = _mapping(payload, field="ffn_launch")
        _exact_fields(value, cls._FIELDS, field="ffn_launch")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ProgramConfig:
    """High-level execution structure independently sampled from scheduling."""

    attention: AttentionBackend
    qkv_projection: ProjectionBackend
    attention_output_projection: ProjectionBackend
    ffn_input_projection: ProjectionBackend
    ffn_output_projection: ProjectionBackend
    precision_plan: PrecisionPlan
    qkv_materialization: QKVMaterialization
    attention_output_bridge: AttentionOutputBridge
    ffn: FFNBackend
    residual_norm: ResidualNormBackend
    initial_norm: InitialNormBackend = InitialNormBackend.TORCH

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "attention",
            "qkv_projection",
            "attention_output_projection",
            "ffn_input_projection",
            "ffn_output_projection",
            "precision_plan",
            "qkv_materialization",
            "attention_output_bridge",
            "ffn",
            "residual_norm",
            "initial_norm",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention", AttentionBackend(self.attention))
        for field_name in _PROJECTION_FIELDS:
            object.__setattr__(
                self,
                field_name,
                ProjectionBackend(getattr(self, field_name)),
            )
        precision_plan = PrecisionPlan(self.precision_plan)
        object.__setattr__(self, "precision_plan", precision_plan)
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
        object.__setattr__(self, "initial_norm", InitialNormBackend(self.initial_norm))

        fp16_fields = _PRECISION_FP16_PROJECTIONS[precision_plan]
        fp16_backends = {
            ProjectionBackend.AUTOCAST_FP16,
            ProjectionBackend.FP16_SHADOW,
        }
        for field_name in _PROJECTION_FIELDS:
            backend = getattr(self, field_name)
            if field_name in fp16_fields:
                if backend not in fp16_backends:
                    raise ValueError(
                        f"{field_name} must use an FP16 backend for "
                        f"precision_plan={precision_plan.value}"
                    )
            elif backend is not ProjectionBackend.INPUT_DTYPE:
                raise ValueError(
                    f"{field_name} must use input_dtype for "
                    f"precision_plan={precision_plan.value}"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "attention": self.attention.value,
            "qkv_projection": self.qkv_projection.value,
            "attention_output_projection": (self.attention_output_projection.value),
            "ffn_input_projection": self.ffn_input_projection.value,
            "ffn_output_projection": self.ffn_output_projection.value,
            "precision_plan": self.precision_plan.value,
            "qkv_materialization": self.qkv_materialization.value,
            "attention_output_bridge": self.attention_output_bridge.value,
            "ffn": self.ffn.value,
            "residual_norm": self.residual_norm.value,
            "initial_norm": self.initial_norm.value,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ProgramConfig:
        value = _mapping(payload, field="program")
        _exact_fields(value, cls._FIELDS, field="program")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """Runtime and conditional low-level parameters for one program."""

    runtime: RuntimeBackend
    attention_launch: TritonAttentionParams | None = None
    residual_norm_launch: TritonNormParams | None = None
    initial_norm_launch: TritonNormParams | None = None
    ffn_launch: TritonFFNParams | None = None
    compile_mode: str | None = None
    batch_tile_size: int | None = None
    microbatch_size: int | None = None
    reuse_unchanged_input: bool = False

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "runtime",
            "attention_launch",
            "residual_norm_launch",
            "initial_norm_launch",
            "ffn_launch",
            "compile_mode",
            "batch_tile_size",
            "microbatch_size",
            "reuse_unchanged_input",
        }
    )

    def __post_init__(self) -> None:
        runtime = RuntimeBackend(self.runtime)
        object.__setattr__(self, "runtime", runtime)
        if self.attention_launch is not None and not isinstance(
            self.attention_launch, TritonAttentionParams
        ):
            raise TypeError("attention_launch must be TritonAttentionParams or None")
        for field_name in ("residual_norm_launch", "initial_norm_launch"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, TritonNormParams):
                raise TypeError(f"{field_name} must be TritonNormParams or None")
        if self.ffn_launch is not None and not isinstance(
            self.ffn_launch, TritonFFNParams
        ):
            raise TypeError("ffn_launch must be TritonFFNParams or None")
        compile_mode = self.compile_mode
        if runtime is RuntimeBackend.COMPILED_FORWARD:
            if compile_mode is None:
                compile_mode = DEFAULT_COMPILED_FORWARD_MODE
                object.__setattr__(self, "compile_mode", compile_mode)
            if compile_mode not in COMPILED_FORWARD_MODES:
                raise ValueError(f"unsupported compile_mode: {compile_mode}")
        elif compile_mode is not None:
            raise ValueError("compile_mode is valid only for compiled_forward")
        if runtime is RuntimeBackend.BATCH_TILED_CUDA_GRAPH:
            object.__setattr__(
                self,
                "batch_tile_size",
                _positive_int(self.batch_tile_size, field="batch_tile_size"),
            )
        elif self.batch_tile_size is not None:
            raise ValueError("batch_tile_size is valid only for batch_tiled_cuda_graph")
        if runtime is RuntimeBackend.STREAMED:
            object.__setattr__(
                self,
                "microbatch_size",
                _positive_int(self.microbatch_size, field="microbatch_size"),
            )
        elif self.microbatch_size is not None:
            raise ValueError("microbatch_size is valid only for the streamed runtime")
        if not isinstance(self.reuse_unchanged_input, bool):
            raise TypeError("reuse_unchanged_input must be a bool")
        if self.reuse_unchanged_input and runtime is not RuntimeBackend.CUDA_GRAPH:
            raise ValueError(
                "reuse_unchanged_input is valid only for the cuda_graph runtime"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime.value,
            "attention_launch": (
                None
                if self.attention_launch is None
                else self.attention_launch.to_dict()
            ),
            "residual_norm_launch": (
                None
                if self.residual_norm_launch is None
                else self.residual_norm_launch.to_dict()
            ),
            "initial_norm_launch": (
                None
                if self.initial_norm_launch is None
                else self.initial_norm_launch.to_dict()
            ),
            "ffn_launch": (
                None if self.ffn_launch is None else self.ffn_launch.to_dict()
            ),
            "compile_mode": self.compile_mode,
            "batch_tile_size": self.batch_tile_size,
            "microbatch_size": self.microbatch_size,
            "reuse_unchanged_input": self.reuse_unchanged_input,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ScheduleConfig:
        value = _mapping(payload, field="schedule")
        _exact_fields(value, cls._FIELDS, field="schedule")
        return cls(
            runtime=value["runtime"],
            attention_launch=(
                None
                if value["attention_launch"] is None
                else TritonAttentionParams.from_dict(value["attention_launch"])
            ),
            residual_norm_launch=(
                None
                if value["residual_norm_launch"] is None
                else TritonNormParams.from_dict(value["residual_norm_launch"])
            ),
            initial_norm_launch=(
                None
                if value["initial_norm_launch"] is None
                else TritonNormParams.from_dict(value["initial_norm_launch"])
            ),
            ffn_launch=(
                None
                if value["ffn_launch"] is None
                else TritonFFNParams.from_dict(value["ffn_launch"])
            ),
            compile_mode=value["compile_mode"],
            batch_tile_size=value["batch_tile_size"],
            microbatch_size=value["microbatch_size"],
            reuse_unchanged_input=value["reuse_unchanged_input"],
        )


@dataclass(frozen=True, slots=True)
class ConfigSpec:
    """Canonical program plus schedule submitted to the strict compiler."""

    program: ProgramConfig
    schedule: ScheduleConfig
    schema_version: int = CONFIG_SCHEMA_VERSION

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema_version", "program", "schedule"}
    )

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        if not isinstance(self.program, ProgramConfig):
            raise TypeError("program must be ProgramConfig")
        if not isinstance(self.schedule, ScheduleConfig):
            raise TypeError("schedule must be ScheduleConfig")

        triton_attention = self.program.attention in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
        }
        if triton_attention != (self.schedule.attention_launch is not None):
            raise ValueError(
                "attention_launch must be present exactly for Triton attention"
            )
        triton_residual = self.program.residual_norm in {
            ResidualNormBackend.TRITON,
            ResidualNormBackend.TRITON_MIXED,
        }
        if triton_residual != (self.schedule.residual_norm_launch is not None):
            raise ValueError(
                "residual_norm_launch must be present exactly for Triton residual norm"
            )
        triton_initial = self.program.initial_norm is InitialNormBackend.TRITON_FP16
        if triton_initial != (self.schedule.initial_norm_launch is not None):
            raise ValueError(
                "initial_norm_launch must be present exactly for Triton initial norm"
            )
        triton_ffn = self.program.ffn is FFNBackend.TRITON_EXACT_GELU
        if triton_ffn != (self.schedule.ffn_launch is not None):
            raise ValueError("ffn_launch must be present exactly for Triton Exact-GELU")

    @property
    def attention_output_layout(self) -> AttentionOutputLayout:
        """Derive the raw attention output layout from the selected bridge."""

        if (
            self.program.attention_output_bridge
            is AttentionOutputBridge.ATTENTION_DIRECT_BSD
        ):
            return AttentionOutputLayout.BSD
        return AttentionOutputLayout.BHSD

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program": self.program.to_dict(),
            "schedule": self.schedule.to_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def config_id(self) -> str:
        return f"cfg-{self.sha256}"

    @classmethod
    def from_dict(cls, payload: object) -> ConfigSpec:
        value = _mapping(payload, field="config")
        _exact_fields(value, cls._FIELDS, field="config")
        return cls(
            schema_version=value["schema_version"],
            program=ProgramConfig.from_dict(value["program"]),
            schedule=ScheduleConfig.from_dict(value["schedule"]),
        )


def portable_config() -> ConfigSpec:
    """Return the explicit reference-order fallback program."""

    return ConfigSpec(
        program=ProgramConfig(
            attention=AttentionBackend.REFERENCE_STREAMING,
            qkv_projection=ProjectionBackend.INPUT_DTYPE,
            attention_output_projection=ProjectionBackend.INPUT_DTYPE,
            ffn_input_projection=ProjectionBackend.INPUT_DTYPE,
            ffn_output_projection=ProjectionBackend.INPUT_DTYPE,
            precision_plan=PrecisionPlan.INPUT_DTYPE,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
        ),
        schedule=ScheduleConfig(runtime=RuntimeBackend.EAGER),
    )


def portable_streamed_config(*, microbatch_size: int = 1) -> ConfigSpec:
    """Return the reference program under the outer streamed schedule."""

    return ConfigSpec(
        program=portable_config().program,
        schedule=ScheduleConfig(
            runtime=RuntimeBackend.STREAMED,
            microbatch_size=microbatch_size,
        ),
    )


__all__ = [
    "COMPILED_FORWARD_MODES",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_COMPILED_FORWARD_MODE",
    "AttentionBackend",
    "AttentionOutputBridge",
    "AttentionOutputLayout",
    "ConfigSpec",
    "FFNBackend",
    "InitialNormBackend",
    "PrecisionPlan",
    "ProgramConfig",
    "ProjectionBackend",
    "QKVMaterialization",
    "ResidualNormBackend",
    "RuntimeBackend",
    "ScheduleConfig",
    "TritonAttentionParams",
    "TritonFFNParams",
    "TritonNormParams",
    "portable_config",
    "portable_streamed_config",
]
