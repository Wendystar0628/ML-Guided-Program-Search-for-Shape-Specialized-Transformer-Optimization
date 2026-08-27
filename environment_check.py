#!/usr/bin/env python3
"""Validate the active CUDA/PyTorch/Triton environment on the current GPU.

The checks are capability based: they do not require a particular GPU model,
CUDA wheel tag, or operating system. CUDA extension compilation is optional
because the benchmark can run without a locally installed CUDA toolkit.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    element_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the active PyTorch CUDA and Triton environment"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to validate (default: cuda:0)",
    )
    parser.add_argument(
        "--skip-compile", action="store_true", help="skip the torch.compile probe"
    )
    parser.add_argument(
        "--check-extension",
        action="store_true",
        help="also compile and launch a small C++/CUDA extension",
    )
    return parser.parse_args()


def find_command(name: str) -> str | None:
    return shutil.which(name)


def require_command(name: str, *, purpose: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' is required for {purpose} but is not on PATH. "
            "Use the platform's compiler environment or skip the extension probe."
        )
    return path


def resolve_cuda_device(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    if torch.version.cuda is None:
        raise RuntimeError("The installed PyTorch build has no CUDA runtime")

    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device_name!r}")

    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; "
            f"detected {torch.cuda.device_count()} device(s)"
        )
    return torch.device("cuda", index)


def detected_cuda_arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}"


def configure_cuda_arch(device: torch.device) -> str:
    """Use the selected device as the default extension compilation target."""

    arch = detected_cuda_arch(device)
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured:
        return configured
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return arch


def check_pytorch_cuda(device: torch.device) -> None:
    with torch.cuda.device(device):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        actual = left @ right
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
        torch.testing.assert_close(actual, expected)
        torch.cuda.synchronize(device)

    print("[PASS] PyTorch CUDA matrix multiplication")


def check_triton(device: torch.device) -> None:
    element_count = 4096
    with torch.cuda.device(device):
        x = torch.randn(element_count, device=device)
        y = torch.randn(element_count, device=device)
        output = torch.empty_like(x)
        grid = (triton.cdiv(element_count, 256),)
        _vector_add_kernel[grid](
            x,
            y,
            output,
            element_count=element_count,
            BLOCK_SIZE=256,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output, x + y)
    print("[PASS] Triton JIT vector-add kernel")


def check_torch_compile(device: torch.device) -> None:
    def eager_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ y)

    compiled_function = torch.compile(
        eager_function,
        backend="inductor",
        fullgraph=True,
    )
    with torch.cuda.device(device):
        x = torch.randn(128, 128, device=device)
        y = torch.randn(128, 128, device=device)
        expected = eager_function(x, y)
        actual = compiled_function(x, y)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("[PASS] torch.compile with the Inductor backend")


def check_cuda_extension(project_root: Path, device: torch.device) -> None:
    from torch.utils import cpp_extension

    require_command("nvcc", purpose="the CUDA extension probe")
    require_command("ninja", purpose="the CUDA extension probe")

    if platform.system() == "Windows":
        require_command("cl", purpose="the CUDA extension probe")
        # Keep UTF-8 MSVC diagnostics from being hidden by a decode error.
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
        extra_cflags = ["/O2", "/Zc:preprocessor"]
        extra_cuda_cflags = ["-O2", "-lineinfo", "-Xcompiler=/Zc:preprocessor"]
    else:
        require_command("c++", purpose="the CUDA extension probe")
        extra_cflags = ["-O2"]
        extra_cuda_cflags = ["-O2", "-lineinfo"]

    cpp_source = r"""
    #include <torch/extension.h>

    torch::Tensor add_one_cuda(torch::Tensor input);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("add_one", &add_one_cuda, "CUDA add-one environment probe");
    }
    """

    cuda_source = r"""
    #include <torch/extension.h>

    __global__ void add_one_kernel(const float* input, float* output, int64_t size) {
        const int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < size) {
            output[index] = input[index] + 1.0f;
        }
    }

    torch::Tensor add_one_cuda(torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

        auto output = torch::empty_like(input);
        constexpr int threads = 256;
        const int64_t blocks = (input.numel() + threads - 1) / threads;
        add_one_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel()
        );
        return output;
    }
    """

    build_directory = project_root / ".cache" / "extension_probe"
    build_directory.mkdir(parents=True, exist_ok=True)
    module = cpp_extension.load_inline(
        name="techjam_cuda_environment_probe",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=False,
    )
    with torch.cuda.device(device):
        x = torch.arange(1024, dtype=torch.float32, device=device)
        actual = module.add_one(x)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, x + 1.0)
    print("[PASS] PyTorch C++/CUDA extension compilation and launch")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    torch.set_float32_matmul_precision("high")
    device = resolve_cuda_device(args.device)
    cuda_arch_list = configure_cuda_arch(device)

    print("=== Environment ===")
    print(f"Project       : {project_root}")
    print(f"Python        : {platform.python_version()} ({sys.executable})")
    print(f"PyTorch       : {torch.__version__}")
    print(f"PyTorch CUDA  : {torch.version.cuda}")
    print(f"Triton        : {triton.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device        : {device}")
    print(f"GPU           : {torch.cuda.get_device_name(device)}")
    print(f"Capability    : {detected_cuda_arch(device)}")
    print(f"CUDA_PATH     : {os.environ.get('CUDA_PATH')}")
    print(f"CUDA arch list: {cuda_arch_list}")
    print(f"Host compiler : {find_command('cl') or find_command('c++') or 'not found'}")
    print(f"nvcc          : {find_command('nvcc') or 'not found'}")
    print(f"ninja         : {find_command('ninja') or 'not found'}")

    print("\n=== Probes ===")
    check_pytorch_cuda(device)
    check_triton(device)
    if not args.skip_compile:
        check_torch_compile(device)
    if args.check_extension:
        check_cuda_extension(project_root, device)

    print("\nALL ENVIRONMENT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
