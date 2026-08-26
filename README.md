# AI-Assisted Shape-Aware Transformer Kernel Optimizer

This project targets end-to-end GPU acceleration of a PyTorch Transformer while preserving its numerical behavior and public interface. The repository currently provides the official benchmark harness, a baseline inspection tool, and a reproducible reference environment for establishing trustworthy measurements before kernel optimization.

## Benchmark flow

The official harness:

1. Builds the baseline and candidate models with identical weights.
2. Runs randomized correctness trials across input and mask configurations.
3. Measures complete CUDA forward passes after warm-up.
4. Reports median baseline latency, candidate latency, and speedup.

A speedup is meaningful only when the candidate passes the correctness checks. The timed region covers the complete Transformer forward pass rather than an isolated kernel.

## Repository contents

| Path | Purpose |
|---|---|
| `torch_transformer_benchmark.py` | Official benchmark and correctness harness |
| `baseline_smoke_test.py` | Baseline data-flow inspection and short local timing run |
| `environment_check.py` | CUDA, PyTorch, Triton, compiler, and extension checks |
| `activate_dev_env.ps1` | PowerShell launcher for the reference Windows environment |
| `environment/sitecustomize.py` | Windows compatibility support for PyTorch extension builds |
| `requirements.txt` | Direct runtime dependencies |
| `requirements-lock.txt` | Exact reference dependency snapshot |

## Reference environment

The included environment helpers were validated for the following configuration:

| Component | Reference configuration |
|---|---|
| Operating system | Windows |
| Python | 3.12 |
| GPU | NVIDIA GeForce RTX 4080 |
| Compute capability | 8.9 |
| PyTorch | 2.12.1+cu132 |
| CUDA Toolkit | 13.2 |
| Triton | 3.7.1.post27 for Windows |
| Build tools | MSVC and Ninja |

Other CUDA systems can run the benchmark, but may need their own compiler and CUDA path configuration.

## Quick start

```powershell
git clone https://github.com/Wendystar0628/AI-Assisted-Shape-Aware-Transformer-Kernel-Optimizer.git
Set-Location .\AI-Assisted-Shape-Aware-Transformer-Kernel-Optimizer

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
. .\activate_dev_env.ps1
```

Check the environment and inspect the official baseline:

```powershell
python .\environment_check.py --skip-compile --skip-extension
python .\baseline_smoke_test.py --device cuda --dtype float32 --warmup 5 --repeats 20
```

Run the official benchmark workflow:

```powershell
python .\torch_transformer_benchmark.py --device cuda --dtype float32
```

For a different workload shape or execution mode:

```powershell
python .\torch_transformer_benchmark.py `
  --batch-size 8 `
  --seq-len 512 `
  --dtype float16 `
  --causal
```

Run `python .\environment_check.py` without skip flags when compiler, `torch.compile`, and CUDA extension validation is required.

## Reported measurements

The benchmark reports:

- Correctness pass or failure.
- Maximum absolute and relative error.
- Baseline and candidate median forward latency.
- End-to-end speedup.
- Requested and effective execution configuration.

The smoke test additionally exposes the generated input tensor, valid-token mask, randomly initialized weights, representative attention and feed-forward intermediates, final output, and per-run timing samples.

## Optimization constraints

- Preserve the Transformer computation and external interface.
- Treat correctness as a prerequisite for performance claims.
- Optimize the complete forward path, including shape-dependent behavior.
- Keep model-serving APIs and online agents outside the timed hot path.

This public repository contains runnable project artifacts only. Restricted source material, competition research, and internal development notes are intentionally excluded.
