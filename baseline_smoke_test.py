#!/usr/bin/env python3
"""直接运行并观察官方 PyTorch Transformer baseline。

这个脚本不实现任何优化，也不修改 ``torch_transformer_benchmark.py``。
它复用官方脚本中的模型、随机测试数据生成器和计时函数，目的是帮助本地
第一次运行时看清楚：

1. 输入、Mask、模型权重和输出分别是什么；
2. 第一层 Transformer 内部的主要张量如何变化；
3. baseline 的一次完整前向传播如何验证；
4. 当前机器上的短时延测试大致如何运行。

默认模型 Shape 与官方 benchmark 一致；预热和计时次数有意缩短，避免把
第一次体验变成长时间性能测试。
"""

from __future__ import annotations

import argparse
import platform
import sys
from typing import Iterable

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "未检测到 PyTorch。请先进入已经安装 PyTorch 的 Python 环境，再运行本脚本。\n"
        f"当前 Python: {sys.executable}"
    ) from exc

from torch_transformer_benchmark import (
    BaselineTransformer,
    TimingResult,
    TransformerConfig,
    benchmark_once,
    generate_random_case,
    resolve_device,
    resolve_dtype,
    warmup_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接运行官方 BaselineTransformer，并观察数据流与短时延结果"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto、cpu、cuda、cuda:0 等"
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="与官方 benchmark 相同，默认使用 high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="与官方 benchmark 相同，在 CUDA 上默认允许 TF32",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="短测试的预热次数；官方 benchmark 默认是 20",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="短测试的正式计时次数；官方 benchmark 默认每轮 100 次、共 3 轮",
    )
    parser.add_argument(
        "--preview-values", type=int, default=6, help="每个张量预览多少个元素"
    )
    parser.add_argument(
        "--trace-first-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否展开第一层 Attention 与 FFN 的数据流",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio 必须位于 [0, 1) 区间")
    if args.input_scale <= 0:
        raise ValueError("input_scale 必须为正数")
    if args.warmup < 0:
        raise ValueError("warmup 不能为负数")
    if args.repeats <= 0:
        raise ValueError("repeats 必须为正数")
    if args.preview_values < 0:
        raise ValueError("preview_values 不能为负数")
    if device.type == "cpu" and args.dtype == "float16":
        print("[提示] CPU 上的 float16 算子可能不受支持或速度很慢。")


def human_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def tensor_preview(tensor: torch.Tensor, count: int) -> list[object]:
    if count == 0 or tensor.numel() == 0:
        return []
    values: Iterable[object] = tensor.detach().reshape(-1)[:count].cpu().tolist()
    return list(values)


def print_tensor(name: str, tensor: torch.Tensor, preview_values: int) -> None:
    detached = tensor.detach()
    print(f"\n{name}")
    print(
        "  "
        f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"device={detached.device}, elements={detached.numel():,}, "
        f"memory={human_bytes(detached.numel() * detached.element_size())}"
    )
    print(
        f"  stride={detached.stride()}, contiguous={detached.is_contiguous()}"
    )

    if detached.dtype == torch.bool:
        true_count = int(detached.sum().item())
        print(
            f"  True={true_count:,}, False={detached.numel() - true_count:,}, "
            f"preview={tensor_preview(detached, preview_values)}"
        )
        return

    numeric = detached.float()
    finite = torch.isfinite(numeric)
    finite_count = int(finite.sum().item())
    if finite_count:
        finite_values = numeric[finite]
        stats = (
            f"min={finite_values.min().item():.6g}, "
            f"max={finite_values.max().item():.6g}, "
            f"mean={finite_values.mean().item():.6g}, "
            f"std={finite_values.std(unbiased=False).item():.6g}"
        )
    else:
        stats = "没有有限数值"
    print(
        f"  finite={finite_count:,}/{detached.numel():,}, {stats}, "
        f"preview={tensor_preview(detached, preview_values)}"
    )


def print_environment(
    device: torch.device,
    dtype: torch.dtype,
    matmul_precision: str,
    allow_tf32: bool,
) -> None:
    print("=== 1. 运行环境 ===")
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"Selected device   : {device}")
    print(f"Selected dtype    : {dtype}")
    print(f"Matmul precision  : {matmul_precision}")
    print(f"Allow TF32        : {allow_tf32 if device.type == 'cuda' else 'N/A on CPU'}")

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU name          : {properties.name}")
        print(f"Compute capability: {properties.major}.{properties.minor}")
        print(f"GPU memory        : {human_bytes(properties.total_memory)}")


def print_model_summary(model: BaselineTransformer) -> None:
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    state = model.state_dict()

    print("\n=== 2. 官方 Baseline 模型 ===")
    print(model)
    print(f"\n参数总量     : {parameter_count:,}")
    print(f"参数占用估算 : {human_bytes(parameter_bytes)}")
    print(f"state_dict 项: {len(state)}")
    print("前 12 个权重/偏置张量：")
    for index, (name, tensor) in enumerate(state.items()):
        if index == 12:
            remaining = len(state) - index
            print(f"  ... 其余 {remaining} 项省略")
            break
        print(f"  {name:<42} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def trace_first_block(
    model: BaselineTransformer,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    preview_values: int,
) -> None:
    """用官方第一层的原始模块逐步重放一次 forward，展示中间数据。"""

    block = model.layers[0]
    attention = block.attention
    batch, seq_len, _ = x.shape

    print("\n=== 4. 第一层数据流（仅用于观察，不计入性能测试） ===")
    print(
        "流程：x -> LayerNorm -> Q/K/V -> Attention scores -> Softmax -> "
        "Context -> 残差 -> LayerNorm -> FFN -> GELU -> 残差"
    )

    with torch.inference_mode():
        norm1 = block.norm1(x)
        q_linear = attention.q_proj(norm1)
        k_linear = attention.k_proj(norm1)
        v_linear = attention.v_proj(norm1)
        q = attention._split_heads(q_linear)
        k = attention._split_heads(k_linear)
        v = attention._split_heads(v_linear)

        scores = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
        if model.config.causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        invalid_keys = ~valid_mask[:, None, None, :]
        scores = scores.masked_fill(invalid_keys, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context_per_head = torch.matmul(probabilities, v)
        merged_context = (
            context_per_head.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, attention.d_model)
        )
        attention_output = attention.out_proj(merged_context)
        attention_output = attention_output.masked_fill(
            ~valid_mask[..., None], 0
        )

        after_attention_residual = x + attention_output
        norm2 = block.norm2(after_attention_residual)
        ffn_linear = block.ffn_in(norm2)
        ffn_activated = F.gelu(ffn_linear, approximate="none")
        ffn_output = block.ffn_out(ffn_activated)
        block_output = after_attention_residual + ffn_output
        block_output = block_output.masked_fill(~valid_mask[..., None], 0)

        official_attention_output = attention(
            norm1, valid_mask, model.config.causal
        )
        official_block_output = block(x, valid_mask, model.config.causal)

    print_tensor("4.1 LayerNorm 后的数据", norm1, preview_values)
    print_tensor("4.2 Q（拆成多个 Head 后）", q, preview_values)
    print_tensor("4.3 K（拆成多个 Head 后）", k, preview_values)
    print_tensor("4.4 V（拆成多个 Head 后）", v, preview_values)
    print_tensor("4.5 Attention scores", scores, preview_values)
    print_tensor("4.6 Softmax probabilities", probabilities, preview_values)
    print_tensor("4.7 合并多个 Head 后的 Context", merged_context, preview_values)
    print_tensor("4.8 Attention 输出", attention_output, preview_values)
    print_tensor("4.9 第一次残差连接后", after_attention_residual, preview_values)
    print_tensor("4.10 FFN 扩展到 ffn_dim", ffn_linear, preview_values)
    print_tensor("4.11 GELU 激活后", ffn_activated, preview_values)
    print_tensor("4.12 第一层最终输出", block_output, preview_values)

    attention_diff = (
        attention_output.float() - official_attention_output.float()
    ).abs().max()
    block_diff = (block_output.float() - official_block_output.float()).abs().max()
    print("\n逐步重放与官方 forward 的一致性：")
    print(f"  Attention 最大绝对差异: {attention_diff.item():.6g}")
    print(f"  Block 最大绝对差异    : {block_diff.item():.6g}")


def validate_baseline_output(
    model: BaselineTransformer,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    with torch.inference_mode():
        first_output = model(x, valid_mask)
        second_output = model(x, valid_mask)

    repeat_diff = (first_output.float() - second_output.float()).abs().max().item()
    expected_shape = (model.config.batch_size, model.config.seq_len, model.config.d_model)
    if tuple(first_output.shape) != expected_shape:
        raise AssertionError(
            f"输出 Shape 错误：expected={expected_shape}, actual={tuple(first_output.shape)}"
        )
    if not bool(torch.isfinite(first_output).all().item()):
        raise AssertionError("Baseline 输出中出现了 NaN 或 Inf")
    if bool((~valid_mask).any().item()):
        padded_values = first_output.masked_select(~valid_mask[..., None])
        if not bool((padded_values == 0).all().item()):
            raise AssertionError("Padding 位置的输出没有全部清零")
    return first_output, repeat_diff


def run_short_benchmark(
    model: BaselineTransformer,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TimingResult:
    warmup_model(model, x, valid_mask, warmup, device)
    samples = benchmark_once(model, x, valid_mask, repeats, device)
    return TimingResult(samples)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    validate_args(args, device)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    print_environment(device, dtype, args.matmul_precision, args.allow_tf32)
    print("\n本次配置：")
    print(config)
    print(f"padding_ratio={args.padding_ratio}, input_scale={args.input_scale}")
    print(f"seed={args.seed}, warmup={args.warmup}, repeats={args.repeats}")

    model = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    print_model_summary(model)

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )

    print("\n=== 3. 官方随机测试数据 ===")
    print(
        "这里没有文本数据集、Token ID 或标签。Benchmark 直接生成随机浮点输入，"
        "再用布尔 Mask 表示哪些 Token 有效。模型权重也是按 PyTorch 默认规则随机初始化。"
    )
    print_tensor("3.1 输入 x", x, args.preview_values)
    print_tensor("3.2 valid_token_mask", valid_mask, args.preview_values)
    valid_lengths = valid_mask.sum(dim=1).detach().cpu().tolist()
    print(f"\n每个 Batch 样本的有效 Token 数: {valid_lengths}")

    if args.trace_first_layer:
        trace_first_block(model, x, valid_mask, args.preview_values)

    print("\n=== 5. 完整 Baseline 前向传播 ===")
    output, repeat_diff = validate_baseline_output(model, x, valid_mask)
    print_tensor("5.1 最终输出", output, args.preview_values)
    print("\n基础检查：PASS")
    print(f"  输出 Shape 正确   : {tuple(output.shape)}")
    print("  输出全部为有限数值: True")
    print(f"  相同输入重复运行的最大绝对差异: {repeat_diff:.6g}")

    print("\n=== 6. Baseline 短性能测试 ===")
    print("随机数据生成和第一层追踪不计入时间；只计完整 model(x, mask)。")
    timing = run_short_benchmark(
        model=model,
        x=x,
        valid_mask=valid_mask,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    tokens_per_call = config.batch_size * config.seq_len
    throughput = tokens_per_call * 1000.0 / timing.median_ms
    print(f"计时样本数 : {len(timing.samples_ms)}")
    print(f"median      : {timing.median_ms:.4f} ms")
    print(f"mean        : {timing.mean_ms:.4f} ms")
    print(f"p90         : {timing.p90_ms:.4f} ms")
    print(f"min         : {timing.min_ms:.4f} ms")
    print(f"throughput  : {throughput:.2f} token/s")

    print("\n=== 7. 这次测试实际产生的数据 ===")
    print("1. config：模型 Shape 与 causal 设置")
    print("2. model.state_dict()：随机初始化的权重和偏置")
    print("3. x：随机浮点输入，Shape 为 [B, S, D]")
    print("4. valid_token_mask：有效 Token 的布尔 Mask，Shape 为 [B, S]")
    print("5. 中间张量：Q/K/V、scores、probabilities、Context、FFN 激活")
    print("6. output：最终输出，Shape 仍为 [B, S, D]")
    print("7. timing.samples_ms：每次完整 Forward 的延迟样本")
    print("\n完成：这只是官方 baseline 的本地运行观察，不包含优化实现或正式评分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
