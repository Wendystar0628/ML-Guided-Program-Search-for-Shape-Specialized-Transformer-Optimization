#!/usr/bin/env python3
"""Validate the project-local CUDA/PyTorch/Triton build environment.

This file contains only small environment probes. It is not a competition
implementation and does not modify the official Transformer baseline.
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
        description="Validate PyTorch CUDA, torch.compile, Triton and CUDA extensions"
    )
    parser.add_argument(
        "--skip-compile", action="store_true", help="skip the torch.compile probe"
    )
    parser.add_argument(
        "--skip-extension",
        action="store_true",
        help="skip the C++/CUDA extension compilation probe",
    )
    return parser.parse_args()


def check_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Required command '{name}' is not on PATH. "
            "Run this check after dot-sourcing activate_dev_env.ps1."
        )
    return path


def check_pytorch_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    if torch.version.cuda != "13.2":
        raise RuntimeError(
            f"Expected the cu132 wheel, but torch.version.cuda={torch.version.cuda!r}"
        )

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    actual = left @ right
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device="cuda")
    torch.testing.assert_close(actual, expected)
    torch.cuda.synchronize()
    print("[PASS] PyTorch CUDA matrix multiplication")


def check_triton() -> None:
    element_count = 4096
    x = torch.randn(element_count, device="cuda")
    y = torch.randn(element_count, device="cuda")
    output = torch.empty_like(x)
    grid = (triton.cdiv(element_count, 256),)
    _vector_add_kernel[grid](
        x,
        y,
        output,
        element_count=element_count,
        BLOCK_SIZE=256,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y)
    print("[PASS] Triton JIT vector-add kernel")


def check_torch_compile() -> None:
    def eager_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ y)

    compiled_function = torch.compile(
        eager_function,
        backend="inductor",
        fullgraph=True,
    )
    x = torch.randn(128, 128, device="cuda")
    y = torch.randn(128, 128, device="cuda")
    expected = eager_function(x, y)
    actual = compiled_function(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("[PASS] torch.compile with the Inductor backend")


def check_cuda_extension(project_root: Path) -> None:
    from torch.utils import cpp_extension

    # MSVC 19.44 emits UTF-8 diagnostics on this Chinese Windows system, while
    # PyTorch 2.12 assumes the active OEM code page. Keep compiler diagnostics
    # readable instead of hiding a build error behind UnicodeDecodeError.
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

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
        extra_cflags=["/O2", "/Zc:preprocessor"],
        extra_cuda_cflags=["-O2", "-lineinfo", "-Xcompiler=/Zc:preprocessor"],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=False,
    )
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    actual = module.add_one(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, x + 1.0)
    print("[PASS] PyTorch C++/CUDA extension compilation and launch")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    torch.set_float32_matmul_precision("high")

    print("=== Environment ===")
    print(f"Project       : {project_root}")
    print(f"Python        : {platform.python_version()} ({sys.executable})")
    print(f"PyTorch       : {torch.__version__}")
    print(f"PyTorch CUDA  : {torch.version.cuda}")
    print(f"Triton        : {triton.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"Capability    : {torch.cuda.get_device_capability(0)}")
    print(f"CUDA_PATH     : {os.environ.get('CUDA_PATH')}")
    print(f"CUDA arch list: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    print(f"cl.exe        : {check_command('cl')}")
    print(f"nvcc.exe      : {check_command('nvcc')}")
    print(f"ninja.exe     : {check_command('ninja')}")

    print("\n=== Probes ===")
    check_pytorch_cuda()
    check_triton()
    if not args.skip_compile:
        check_torch_compile()
    if not args.skip_extension:
        check_cuda_extension(project_root)

    print("\nALL ENVIRONMENT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
