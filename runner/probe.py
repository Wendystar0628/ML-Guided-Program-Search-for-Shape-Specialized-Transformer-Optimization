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
    device = official.resolve_device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    value = torch.arange(16, device=device, dtype=torch.float32)
    observed = float((value.square() + 1).sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "outcome": "success",
        "environment": collect_environment(device),
        "probe": {
            "device_operation_passed": observed == 1256.0,
            "sdpa": _sdpa_capabilities(device),
        },
        "failure": None,
    }
