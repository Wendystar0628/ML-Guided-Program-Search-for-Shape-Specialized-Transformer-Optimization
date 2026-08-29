from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from solution.runtimes import compiled_forward as compiled_forward_module
from solution.runtimes.compiled_forward import CompiledForward

_COMPILE_MODE = "max-autotune"


def _fake_compiler(
    calls: list[dict[str, object]],
) -> Callable[..., object]:
    def compile_function(function: object, **kwargs: object) -> object:
        calls.append({"function": function, **kwargs})
        return function

    return compile_function


def _add_mask(
    value: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if valid_mask is None:
        return value + 1
    return value + valid_mask.to(value.dtype)


def test_same_plan_and_tensor_signature_reuses_compiled_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        _fake_compiler(compile_calls),
    )
    runner = CompiledForward()

    first = runner.run(
        _add_mask,
        torch.zeros(2, 3),
        None,
        plan_key=("plan", 1),
        compile_mode=_COMPILE_MODE,
    )
    second = runner.run(
        _add_mask,
        torch.ones(2, 3),
        None,
        plan_key=("plan", 1),
        compile_mode=_COMPILE_MODE,
    )

    torch.testing.assert_close(first, torch.ones(2, 3))
    torch.testing.assert_close(second, torch.full((2, 3), 2.0))
    assert runner.cache_size == 1
    assert len(compile_calls) == 1
    assert compile_calls[0]["fullgraph"] is True
    assert compile_calls[0]["dynamic"] is False
    assert compile_calls[0]["mode"] == "max-autotune"


def test_requested_compile_mode_is_part_of_the_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        _fake_compiler(compile_calls),
    )
    runner = CompiledForward()
    value = torch.zeros(2, 3)

    for compile_mode in (
        "max-autotune",
        "max-autotune-no-cudagraphs",
        "reduce-overhead",
    ):
        runner.run(
            _add_mask,
            value,
            None,
            plan_key="same-plan",
            compile_mode=compile_mode,
        )

    assert runner.cache_size == 3
    assert [call["mode"] for call in compile_calls] == [
        "max-autotune",
        "max-autotune-no-cudagraphs",
        "reduce-overhead",
    ]


def test_plan_input_and_mask_signatures_create_separate_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        _fake_compiler(compile_calls),
    )
    runner = CompiledForward()
    contiguous = torch.zeros(2, 3)
    transposed = torch.zeros(3, 2).transpose(0, 1)
    mask = torch.ones(2, 3, dtype=torch.bool)

    runner.run(
        _add_mask, contiguous, None, plan_key="first", compile_mode=_COMPILE_MODE
    )
    runner.run(
        _add_mask, contiguous, None, plan_key="second", compile_mode=_COMPILE_MODE
    )
    runner.run(
        _add_mask, transposed, None, plan_key="first", compile_mode=_COMPILE_MODE
    )
    runner.run(
        _add_mask, contiguous, mask, plan_key="first", compile_mode=_COMPILE_MODE
    )

    assert runner.cache_size == 4
    assert len(compile_calls) == 4


def test_unhashable_plan_key_is_rejected_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        _fake_compiler(compile_calls),
    )

    with pytest.raises(TypeError, match="plan_key must be hashable"):
        CompiledForward().run(
            _add_mask,
            torch.zeros(2, 3),
            None,
            plan_key=["invalid"],  # type: ignore[arg-type]
            compile_mode=_COMPILE_MODE,
        )

    assert compile_calls == []


def test_compiler_factory_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(compiled_forward_module.torch, "compile", fail)
    runner = CompiledForward()

    with pytest.raises(RuntimeError, match="failed to create"):
        runner.run(
            _add_mask,
            torch.zeros(2, 3),
            None,
            plan_key="plan",
            compile_mode=_COMPILE_MODE,
        )

    assert runner.cache_size == 0


def test_first_execution_failure_is_explicit_and_drops_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def compile_function(_function: object, **_kwargs: object) -> object:
        def fail(_value: torch.Tensor, _mask: torch.Tensor | None) -> torch.Tensor:
            raise RuntimeError("synthetic lazy compilation failure")

        return fail

    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        compile_function,
    )
    runner = CompiledForward()

    with pytest.raises(RuntimeError, match="compilation failed"):
        runner.run(
            _add_mask,
            torch.zeros(2, 3),
            None,
            plan_key="plan",
            compile_mode=_COMPILE_MODE,
        )

    assert runner.cache_size == 0


def test_later_execution_failure_is_explicit_and_drops_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_count = 0

    def compile_function(_function: object, **_kwargs: object) -> object:
        def compiled(value: torch.Tensor, _mask: torch.Tensor | None) -> torch.Tensor:
            nonlocal execution_count
            execution_count += 1
            if execution_count == 2:
                raise RuntimeError("synthetic execution failure")
            return value

        return compiled

    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        compile_function,
    )
    runner = CompiledForward()
    value = torch.zeros(2, 3)
    runner.run(_add_mask, value, None, plan_key="plan", compile_mode=_COMPILE_MODE)

    with pytest.raises(RuntimeError, match="execution failed"):
        runner.run(_add_mask, value, None, plan_key="plan", compile_mode=_COMPILE_MODE)

    assert runner.cache_size == 0


def test_clear_and_rebuild_replace_cached_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        _fake_compiler(compile_calls),
    )
    runner = CompiledForward()
    value = torch.zeros(2, 3)

    runner.run(_add_mask, value, None, plan_key="plan", compile_mode="reduce-overhead")
    runner.rebuild(
        _add_mask, value, None, plan_key="plan", compile_mode="reduce-overhead"
    )
    assert runner.cache_size == 1
    assert len(compile_calls) == 2
    assert [call["mode"] for call in compile_calls] == [
        "reduce-overhead",
        "reduce-overhead",
    ]

    runner.clear()
    assert runner.cache_size == 0


def test_non_tensor_output_is_rejected_without_poisoning_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiled_forward_module.torch,
        "compile",
        lambda function, **_kwargs: function,
    )

    def invalid_output(
        _value: torch.Tensor,
        _mask: torch.Tensor | None,
    ) -> object:
        return "not a tensor"

    runner = CompiledForward()
    with pytest.raises(TypeError, match="must return a Tensor"):
        runner.run(
            invalid_output,  # type: ignore[arg-type]
            torch.zeros(2, 3),
            None,
            plan_key="plan",
            compile_mode=_COMPILE_MODE,
        )

    assert runner.cache_size == 0
