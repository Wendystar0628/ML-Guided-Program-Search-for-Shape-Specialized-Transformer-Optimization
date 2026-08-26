"""Small environment and device probes executed inside a worker process."""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from official import torch_transformer_benchmark as official

_ANCHOR_ROUNDS = 3
_LAUNCH_REPEATS = 128
_GRAPH_KERNEL_NODES = 8
_GRAPH_REPEATS = 64
_COPY_BYTES = 32 * 1024 * 1024
_COPY_REPEATS = 16
_GEMM_DIMENSION = 1024
_GEMM_REPEATS = 32
_SOFTMAX_ROWS = 2048
_SOFTMAX_COLUMNS = 1024
_SOFTMAX_REPEATS = 16


def _driver_version(index: int) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"exit code {completed.returncode}"
        return None, message[-1000:]
    value = completed.stdout.strip().splitlines()
    if not value or not value[0].strip():
        return None, "nvidia-smi returned no driver version"
    return value[0].strip(), None


def collect_environment(device: torch.device) -> dict[str, Any]:
    value: dict[str, Any] = {
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        driver, _ = _driver_version(index)
        value["gpu"] = properties.name
        value["compute_capability"] = f"{properties.major}.{properties.minor}"
        value["total_memory_bytes"] = properties.total_memory
        if driver is not None:
            value["driver"] = driver
    return value


def _nvidia_smi_metadata(index: int) -> tuple[dict[str, str], str | None]:
    fields = ("driver_version", "driver_model.current", "pci.bus_id")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"exit code {completed.returncode}"
        return {}, message[-1000:]
    rows = completed.stdout.strip().splitlines()
    if not rows:
        return {}, "nvidia-smi returned no device metadata"
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != len(fields):
        return {}, "nvidia-smi returned malformed device metadata"
    return dict(zip(fields, values, strict=True)), None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _architecture_family(major: int, minor: int) -> str:
    """Map NVIDIA compute capability to a useful scheduling family."""

    if major >= 12 or major == 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "ada"
    if major == 8:
        return "ampere"
    if major == 7 and minor >= 5:
        return "turing"
    if major == 7:
        return "volta"
    return "legacy_or_unknown"


def _triton_runtime() -> tuple[bool, str]:
    """Return optional Triton availability without making the probe fragile."""

    try:
        import triton
    except Exception:  # noqa: BLE001 - optional compiler capability probe.
        return False, "unavailable"
    version = getattr(triton, "__version__", None)
    return True, str(version) if version is not None else "unknown"


def _hardware_profile(device: torch.device) -> dict[str, Any]:
    """Collect static facts useful for hardware-aware candidate selection."""

    triton_available, triton_version = _triton_runtime()
    profile: dict[str, Any] = {
        "available": True,
        "device_type": device.type,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "system": {
            "logical_cpu_count": os.cpu_count(),
            "processor": platform.processor() or None,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "triton_available": triton_available,
            "triton": triton_version,
        },
    }
    if device.type != "cuda":
        profile["gpu"] = {"available": False, "reason": "cuda_required"}
        return profile

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    memory_clock_rate_khz = _optional_int(
        getattr(properties, "memory_clock_rate", None)
    )
    memory_bus_width_bits = _optional_int(getattr(properties, "memory_bus_width", None))
    theoretical_bandwidth_gbps = None
    if (
        memory_clock_rate_khz is not None
        and memory_clock_rate_khz > 0
        and memory_bus_width_bits is not None
        and memory_bus_width_bits > 0
    ):
        theoretical_bandwidth_gbps = (
            2.0
            * memory_clock_rate_khz
            * 1000.0
            * (memory_bus_width_bits / 8.0)
            / 1_000_000_000.0
        )

    metadata, metadata_error = _nvidia_smi_metadata(index)
    driver = metadata.get("driver_version")
    if not driver:
        driver, driver_error = _driver_version(index)
        if metadata_error is None:
            metadata_error = driver_error
    if driver:
        profile["software"]["driver"] = driver

    driver_model = metadata.get("driver_model.current")
    if driver_model and driver_model.upper() != "N/A":
        profile["platform"]["driver_model"] = driver_model
    pci_bus_id = metadata.get("pci.bus_id")
    if not pci_bus_id or pci_bus_id.upper() == "N/A":
        domain = _optional_int(getattr(properties, "pci_domain_id", None))
        bus = _optional_int(getattr(properties, "pci_bus_id", None))
        pci_device = _optional_int(getattr(properties, "pci_device_id", None))
        if domain is not None and bus is not None and pci_device is not None:
            pci_bus_id = f"{domain:08X}:{bus:02X}:{pci_device:02X}.0"

    profile["gpu"] = {
        "available": True,
        "index": index,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "architecture_family": _architecture_family(properties.major, properties.minor),
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_graph_available": hasattr(torch.cuda, "CUDAGraph"),
        "sm_count": _optional_int(getattr(properties, "multi_processor_count", None)),
        "l2_cache_bytes": _optional_int(getattr(properties, "L2_cache_size", None)),
        "shared_memory_per_sm_bytes": _optional_int(
            getattr(properties, "shared_memory_per_multiprocessor", None)
        ),
        "shared_memory_per_block_bytes": _optional_int(
            getattr(properties, "shared_memory_per_block", None)
        ),
        "registers_per_sm": _optional_int(
            getattr(properties, "regs_per_multiprocessor", None)
        ),
        "warp_size": _optional_int(getattr(properties, "warp_size", None)),
        "max_threads_per_sm": _optional_int(
            getattr(properties, "max_threads_per_multi_processor", None)
        ),
        "memory_bus_width_bits": memory_bus_width_bits,
        "memory_clock_rate_khz": memory_clock_rate_khz,
        "core_clock_rate_khz": _optional_int(getattr(properties, "clock_rate", None)),
        "theoretical_memory_bandwidth_gbps": theoretical_bandwidth_gbps,
        "pci_bus_id": pci_bus_id,
    }
    if metadata_error is not None:
        profile["gpu"]["nvidia_smi_note"] = metadata_error
    return profile


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason[-1000:]}


def _safe_item(
    function: Callable[..., dict[str, Any]],
    *args: object,
) -> dict[str, Any]:
    try:
        value = function(*args)
    except Exception as exc:  # noqa: BLE001 - probe items must degrade independently.
        return _unavailable(f"{type(exc).__name__}: {exc}")
    if not isinstance(value, dict):
        return _unavailable("probe_item_returned_invalid_result")
    return value


def _cuda_event_latency_ms(
    device: torch.device,
    operation: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    rounds: int = _ANCHOR_ROUNDS,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            operation()
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
        if elapsed_ms <= 0:
            raise RuntimeError("CUDA events reported a non-positive duration")
        samples.append(elapsed_ms / repeats)
    return float(statistics.median(samples))


def _eager_launch_anchor(device: torch.device) -> dict[str, Any]:
    value = torch.zeros(1, device=device, dtype=torch.float32)
    latency_ms = _cuda_event_latency_ms(
        device,
        lambda: value.add_(1.0),
        warmup=8,
        repeats=_LAUNCH_REPEATS,
    )
    return {
        "available": True,
        "method": "single_element_inplace_add",
        "repeats_per_round": _LAUNCH_REPEATS,
        "rounds": _ANCHOR_ROUNDS,
        "effective_latency_us": latency_ms * 1000.0,
    }


def _cuda_graph_anchor(device: torch.device) -> dict[str, Any]:
    if not hasattr(torch.cuda, "CUDAGraph"):
        return _unavailable("cuda_graph_api_unavailable")

    value = torch.zeros(1, device=device, dtype=torch.float32)
    capture_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            for _ in range(_GRAPH_KERNEL_NODES):
                value.add_(1.0)
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(_GRAPH_KERNEL_NODES):
            value.add_(1.0)
    torch.cuda.synchronize(device)
    replay_ms = _cuda_event_latency_ms(
        device,
        graph.replay,
        warmup=3,
        repeats=_GRAPH_REPEATS,
    )
    return {
        "available": True,
        "method": "straight_line_single_element_add_graph",
        "kernel_nodes": _GRAPH_KERNEL_NODES,
        "repeats_per_round": _GRAPH_REPEATS,
        "rounds": _ANCHOR_ROUNDS,
        "replay_latency_us": replay_ms * 1000.0,
        "effective_latency_per_node_us": (replay_ms * 1000.0 / _GRAPH_KERNEL_NODES),
    }


def _device_copy_anchor(device: torch.device) -> dict[str, Any]:
    source = torch.empty(_COPY_BYTES, device=device, dtype=torch.uint8)
    destination = torch.empty_like(source)
    source.fill_(1)
    latency_ms = _cuda_event_latency_ms(
        device,
        lambda: destination.copy_(source, non_blocking=True),
        warmup=3,
        repeats=_COPY_REPEATS,
    )
    return {
        "available": True,
        "method": "device_to_device_copy",
        "payload_bytes": _COPY_BYTES,
        "repeats_per_round": _COPY_REPEATS,
        "rounds": _ANCHOR_ROUNDS,
        "latency_ms": latency_ms,
        "effective_bandwidth_gbps": _COPY_BYTES / (latency_ms * 1_000_000.0),
    }


def _gemm_anchor(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    dtype_name = str(dtype).removeprefix("torch.")
    if dtype == torch.bfloat16:
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(bf16_supported) and not bf16_supported():
            return _unavailable("bfloat16_not_supported")

    dimension = _GEMM_DIMENSION
    left = torch.ones((dimension, dimension), device=device, dtype=dtype)
    right = torch.ones_like(left)
    output = torch.empty_like(left)

    def operation() -> object:
        return torch.mm(left, right, out=output)

    latency_ms = _cuda_event_latency_ms(
        device,
        operation,
        warmup=3,
        repeats=_GEMM_REPEATS,
    )
    flops = 2.0 * dimension**3
    result: dict[str, Any] = {
        "available": True,
        "method": "square_torch_mm",
        "dtype": dtype_name,
        "dimension": dimension,
        "repeats_per_round": _GEMM_REPEATS,
        "rounds": _ANCHOR_ROUNDS,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "latency_ms": latency_ms,
        "tflops": flops / (latency_ms * 1_000_000_000.0),
    }
    if dtype == torch.float32:
        result["tf32_allowed"] = bool(torch.backends.cuda.matmul.allow_tf32)
    return result


def _softmax_anchor(device: torch.device) -> dict[str, Any]:
    source = torch.linspace(
        -1.0,
        1.0,
        _SOFTMAX_COLUMNS,
        device=device,
        dtype=torch.float32,
    ).repeat(_SOFTMAX_ROWS, 1)
    output: torch.Tensor | None = None

    def operation() -> torch.Tensor:
        nonlocal output
        output = torch.softmax(source, dim=-1)
        return output

    latency_ms = _cuda_event_latency_ms(
        device,
        operation,
        warmup=3,
        repeats=_SOFTMAX_REPEATS,
    )
    elements = _SOFTMAX_ROWS * _SOFTMAX_COLUMNS
    return {
        "available": True,
        "method": "fp32_row_softmax",
        "rows": _SOFTMAX_ROWS,
        "columns": _SOFTMAX_COLUMNS,
        "repeats_per_round": _SOFTMAX_REPEATS,
        "rounds": _ANCHOR_ROUNDS,
        "latency_ms": latency_ms,
        "throughput_gigaelements_per_second": (elements / (latency_ms * 1_000_000.0)),
    }


def _performance_anchors(device: torch.device) -> dict[str, Any]:
    names = (
        "eager_launch",
        "cuda_graph_replay",
        "device_copy",
        "gemm_float16",
        "gemm_bfloat16",
        "gemm_float32",
        "softmax_fp32",
    )
    if device.type != "cuda":
        return {name: _unavailable("cuda_required") for name in names}

    return {
        "eager_launch": _safe_item(_eager_launch_anchor, device),
        "cuda_graph_replay": _safe_item(_cuda_graph_anchor, device),
        "device_copy": _safe_item(_device_copy_anchor, device),
        "gemm_float16": _safe_item(_gemm_anchor, device, torch.float16),
        "gemm_bfloat16": _safe_item(_gemm_anchor, device, torch.bfloat16),
        "gemm_float32": _safe_item(_gemm_anchor, device, torch.float32),
        "softmax_fp32": _safe_item(_softmax_anchor, device),
    }


def _error_code(error: str | None) -> str:
    message = str(error or "").lower()
    if "not compiled" in message:
        return "not_compiled"
    if "no available kernel" in message:
        return "no_kernel"
    if "explicit attn_mask" in message and "is_causal=true" in message:
        return "invalid_mask_combination"
    if "non-finite" in message:
        return "non_finite"
    if "disabled" in message:
        return "policy_disabled"
    return "runtime_error"


def _call_form(*, causal: bool, padding: bool) -> str:
    if causal and padding:
        return "padding_mask_plus_is_causal"
    if causal:
        return "is_causal"
    if padding:
        return "padding_mask"
    return "dense"


def _sdpa_capabilities(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"available": False, "reason": "cuda_required"}

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend_values = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }
    scenarios = (
        ("fp32_dense_head64", torch.float32, 64, False, False),
        ("fp16_dense_head64", torch.float16, 64, False, False),
        ("fp16_causal_padding_head64", torch.float16, 64, True, True),
        ("bf16_dense_head128", torch.bfloat16, 128, False, False),
    )
    test_shape = {"batch_size": 1, "num_heads": 8, "seq_len": 64}
    results: list[dict[str, Any]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    for scenario_id, dtype, head_dim, causal, padding in scenarios:
        query = torch.randn(
            test_shape["batch_size"],
            test_shape["num_heads"],
            test_shape["seq_len"],
            head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn(
            query.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        value = torch.randn(
            query.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        attention_mask = None
        if padding:
            attention_mask = torch.ones(
                1, 1, 1, query.shape[-2], device=device, dtype=torch.bool
            )
            attention_mask[..., -16:] = False
        supported_backends: list[str] = []
        unsupported_backends: dict[str, str] = {}
        for backend_name, backend in backend_values.items():
            error = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with sdpa_kernel(backends=[backend]):
                        output = F.scaled_dot_product_attention(
                            query,
                            key,
                            value,
                            attn_mask=attention_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                        )
                torch.cuda.synchronize(device)
                success = bool(torch.isfinite(output).all().item())
                if not success:
                    error = "backend returned non-finite output"
            except (RuntimeError, NotImplementedError) as exc:
                success = False
                error = f"{type(exc).__name__}: {exc}"
            if success:
                supported_backends.append(backend_name)
            else:
                unsupported_backends[backend_name] = _error_code(error)
        results.append(
            {
                "id": scenario_id,
                "dtype": str(dtype).removeprefix("torch."),
                "head_dim": head_dim,
                "call_form": _call_form(causal=causal, padding=padding),
                "supported_backends": supported_backends,
                "unsupported_backends": unsupported_backends,
            }
        )

    flash_available = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    policy_flags = {}
    for name in ("flash", "mem_efficient", "cudnn", "math"):
        getter = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
        policy_flags[name] = bool(getter()) if callable(getter) else None
    result: dict[str, Any] = {
        "available": True,
        "test_shape": test_shape,
        "global_policy_enabled": {
            "flash": policy_flags["flash"],
            "efficient": policy_flags["mem_efficient"],
            "cudnn": policy_flags["cudnn"],
            "math": policy_flags["math"],
        },
        "scenarios": results,
    }
    if callable(flash_available):
        result["flash_compiled"] = bool(flash_available())
    return result


def execute_probe(request: dict[str, Any]) -> dict[str, Any]:
    requested_device = str(request["device"])
    matmul_precision = str(request.get("matmul_precision", "high"))
    allow_tf32 = request.get("allow_tf32", True)
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(f"unsupported matmul precision: {matmul_precision}")
    if not isinstance(allow_tf32, bool):
        raise TypeError("allow_tf32 must be a boolean")
    device = official.resolve_device(requested_device)
    torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    value = torch.arange(16, device=device, dtype=torch.float32)
    observed = float((value.square() + 1).sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "outcome": "success",
        "environment": collect_environment(device),
        "probe": {
            "device_operation_passed": observed == 1256.0,
            "runtime_policy": {
                "matmul_precision": matmul_precision,
                "allow_tf32": allow_tf32,
            },
            "hardware_profile": _safe_item(_hardware_profile, device),
            "performance_anchors": _performance_anchors(device),
            "sdpa": _safe_item(_sdpa_capabilities, device),
        },
        "failure": None,
    }
