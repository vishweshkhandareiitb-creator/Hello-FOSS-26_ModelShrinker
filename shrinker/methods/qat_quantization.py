"""Quantization-aware training that produces a CPU INT8 model."""

from __future__ import annotations

import logging
import platform
from contextlib import contextmanager
from typing import Any, Iterator, Tuple

import torch
from torch import nn

from shrinker.registry import register

logger = logging.getLogger(__name__)


def _patch_quantized_modules() -> None:
    """Patch PyTorch's quantized module classes so unpickling restores all missing Module dicts."""
    try:
        import torch.ao.nn.quantized as nnq
        import torch.ao.nn.intrinsic.quantized as nniq

        classes = set()
        for mod in (nnq, nniq):
            for attr in dir(mod):
                try:
                    cls = getattr(mod, attr)
                    if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
                        classes.add(cls)
                except Exception:
                    pass

        for cls in classes:
            orig_setstate = getattr(cls, "__setstate__", None)
            if orig_setstate and not getattr(orig_setstate, "_is_patched", False):
                def _make_setstate(old_fn):
                    def __setstate__(self, state):
                        old_fn(self, state)
                        for attr in (
                            "_modules",
                            "_parameters",
                            "_buffers",
                            "_backward_hooks",
                            "_backward_pre_hooks",
                            "_forward_hooks",
                            "_forward_pre_hooks",
                            "_forward_hooks_with_kwargs",
                            "_forward_hooks_always_called",
                            "_state_dict_hooks",
                            "_state_dict_pre_hooks",
                            "_load_state_dict_pre_hooks",
                            "_load_state_dict_post_hooks",
                            "_non_persistent_buffers_set",
                        ):
                            if attr not in self.__dict__:
                                if attr == "_non_persistent_buffers_set":
                                    self.__dict__[attr] = set()
                                else:
                                    self.__dict__[attr] = {}
                    __setstate__._is_patched = True
                    return __setstate__
                cls.__setstate__ = _make_setstate(orig_setstate)
    except Exception:
        pass


_patch_quantized_modules()


def _as_args(inputs: Any) -> Tuple[Any, ...]:
    return inputs if isinstance(inputs, tuple) else (inputs,)


def _to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, tuple):
        return tuple(value.to(device) for value in inputs)
    return inputs.to(device)


def _select_backend(backend: str) -> str:
    available = tuple(torch.backends.quantized.supported_engines)
    if backend == "auto":
        machine = platform.machine().lower()
        candidates = ("qnnpack",) if machine.startswith(("arm", "aarch64")) else ("x86", "onednn", "fbgemm")
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise RuntimeError(f"No supported CPU quantization backend is available; PyTorch reports {available}")
    if backend not in ("x86", "fbgemm", "onednn", "qnnpack"):
        raise ValueError("backend must be 'auto', 'x86', 'fbgemm', 'onednn', or 'qnnpack'")
    if backend not in available:
        raise RuntimeError(f"Quantization backend {backend!r} is unavailable; PyTorch reports {available}")
    return backend


@contextmanager
def _quantized_engine(backend: str) -> Iterator[None]:
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = backend
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous


def _freeze_qat_statistics(module: nn.Module) -> None:
    """Stop observer and fused-BatchNorm updates once ranges have stabilized."""
    from torch.ao.quantization import disable_observer

    module.apply(disable_observer)
    try:
        from torch.ao.nn.intrinsic.qat import freeze_bn_stats
    except ImportError:
        return
    module.apply(freeze_bn_stats)


def _make_views_layout_safe(module: nn.Module) -> None:
    """Use reshape for traced ``view`` calls after quantization changes tensor layout.

    ``reshape`` has the same value and shape semantics as ``view`` but creates a
    contiguous copy when a quantized operator returns a non-contiguous tensor.
    """
    graph = getattr(module, "graph", None)
    if graph is None:
        return
    changed = False
    for node in graph.nodes:
        if node.op == "call_method" and node.target == "view":
            node.target = "reshape"
            changed = True
    if changed:
        graph.lint()
        module.recompile()


@register(kind="quantization")
def apply(
    model: nn.Module,
    loader,
    example_input: Any = None,
    epochs: int = 5,
    lr: float = 1e-4,
    backend: str = "auto",
    device: str = "cpu",
    observer_freeze_fraction: float = 0.8,
) -> nn.Module:
    """Return an INT8 CPU model after quantization-aware fine-tuning.

    The method prepares the copied model for FX graph-mode quantization-aware
    training, fine-tunes it with cross-entropy, and converts supported graph
    operations to backend-specific integer operators. Fake quantization in
    training simulates signed per-channel INT8 weights and asymmetric INT8
    activations while preserving FP32 weights for optimization.

    ``loader`` yields ``(inputs, class_labels)`` batches. ``example_input``
    supplies the input shape used to trace the graph; the first loader batch is
    used when it is omitted. ``backend='auto'`` selects an available x86 CPU
    backend on desktop machines and QNNPACK on ARM. Operations unsupported by
    the selected backend remain FP32 with quantization boundaries around them.
    """
    if loader is None:
        raise ValueError("qat_quantization requires a labeled DataLoader via loader=")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not 0.0 <= observer_freeze_fraction <= 1.0:
        raise ValueError("observer_freeze_fraction must be in [0, 1]")
    if device != "cpu":
        raise ValueError("qat_quantization currently supports device='cpu' only")

    try:
        first_batch = next(iter(loader))
        inputs, _ = first_batch
    except Exception as exc:  # noqa: BLE001 - malformed loaders are not quantizable
        logger.info("Skipping QAT quantization: loader did not yield (inputs, labels): %s", exc)
        return model

    if example_input is None:
        example_input = inputs
    example_args = _as_args(example_input)
    if not all(isinstance(value, torch.Tensor) for value in example_args):
        logger.info("Skipping QAT quantization: example_input must be a Tensor or tuple of Tensors")
        return model

    selected_backend = _select_backend(backend)
    try:
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
    except ImportError as exc:
        raise RuntimeError(
            "qat_quantization requires PyTorch FX graph-mode QAT (torch.ao.quantization)"
        ) from exc

    try:
        qconfig_mapping = get_default_qat_qconfig_mapping(selected_backend)
        with _quantized_engine(selected_backend):
            prepared = prepare_qat_fx(model.cpu().train(), qconfig_mapping, example_args)
    except Exception as exc:  # noqa: BLE001 - a model may not be FX traceable
        logger.info("Skipping QAT quantization: could not prepare this model for FX QAT: %s", exc)
        return model

    try:
        optimizer = torch.optim.Adam(prepared.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        try:
            batches_per_epoch = len(loader)
        except TypeError:
            batches_per_epoch = 0
        freeze_after = int(epochs * batches_per_epoch * observer_freeze_fraction)
        step = 0
        frozen = False

        prepared.train()
        for _ in range(epochs):
            for batch_inputs, targets in loader:
                if not frozen and step >= freeze_after:
                    _freeze_qat_statistics(prepared)
                    frozen = True
                batch_inputs = _to_device(batch_inputs, torch.device("cpu"))
                targets = targets.to("cpu")
                optimizer.zero_grad()
                output = prepared(*_as_args(batch_inputs))
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                step += 1
        if not frozen:
            _freeze_qat_statistics(prepared)
    except Exception as exc:  # noqa: BLE001 - unsupported training output/data is left unchanged
        logger.info("Skipping QAT quantization: training failed for this model or loader: %s", exc)
        return model

    try:
        with _quantized_engine(selected_backend):
            result = convert_fx(prepared.eval())
    except Exception as exc:  # noqa: BLE001 - leave models unchanged if a backend cannot lower them
        logger.info("Skipping QAT quantization: backend conversion failed: %s", exc)
        return model

    _make_views_layout_safe(result)
    quantized = sum("quantized" in type(module).__module__ for module in result.modules())
    if quantized:
        logger.info("QAT converted %d quantized modules using the %s backend", quantized, selected_backend)
    else:
        logger.info("QAT conversion retained this model in FP32 because no supported operators were found")
    return result