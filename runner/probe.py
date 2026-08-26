"""Small environment and device probes executed inside a worker process."""

from __future__ import annotations

import platform
import subprocess
import warnings
from typing import Any

import torch
import torch.nn.functional as F

from official import torch_transformer_benchmark as official


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


def collect_environment(device: torch.device, requested_device: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested_device,
        "resolved_device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        driver, driver_error = _driver_version(index)
        value["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        }
        value["driver"] = driver
        if driver_error is not None:
            value["driver_query_error"] = driver_error
    return value


def _sdpa_capabilities(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"supported": False, "reason": "CUDA device required"}

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
    results: list[dict[str, Any]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    for scenario_id, dtype, head_dim, causal, padding in scenarios:
        query = torch.randn(
            1,
            8,
            64,
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
            results.append(
                {
                    "scenario_id": scenario_id,
                    "dtype": str(dtype).removeprefix("torch."),
                    "head_dim": head_dim,
                    "causal": causal,
                    "padding": padding,
                    "backend": backend_name,
                    "success": success,
                    "error": error,
                }
            )

    flash_available = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    policy_flags = {}
    for name in ("flash", "mem_efficient", "cudnn", "math"):
        getter = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
        policy_flags[name] = bool(getter()) if callable(getter) else None
    return {
        "supported": True,
        "flash_compiled_available": bool(flash_available())
        if callable(flash_available)
        else None,
        "policy_enabled_flags": policy_flags,
        "actual_call_results": results,
    }


def execute_probe(request: dict[str, Any]) -> dict[str, Any]:
    requested_device = str(request["device"])
    device = official.resolve_device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    value = torch.arange(16, device=device, dtype=torch.float32)
    observed = float((value.square() + 1).sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "outcome": "success",
        "status": "success",
        "environment": collect_environment(device, requested_device),
        "probe": {
            "operation": "sum(arange(16)^2 + 1)",
            "observed": observed,
            "expected": 1256.0,
            "passed": observed == 1256.0,
            "sdpa_capabilities": _sdpa_capabilities(device),
        },
        "path": {"requested": requested_device, "resolved": str(device)},
        "failure": None,
    }
