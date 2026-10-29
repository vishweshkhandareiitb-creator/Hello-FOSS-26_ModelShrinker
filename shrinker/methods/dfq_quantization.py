"""Data-free INT8 post-training quantization with DFQ preprocessing.

This module implements the level-1 portion of the Data-Free Quantization
(DFQ) pipeline: cross-layer equalization, optional high-bias absorption, and
analytic bias correction.  It deliberately consumes no calibration data.
``example_input`` is used only by ``torch.fx`` to establish a safe execution
graph; its values do not contribute to any quantization parameter.

The method uses real CPU quantized Conv2d/Linear kernels for the paths whose
activation statistics can be inferred from BatchNorm.  Unsupported paths are
left in FP32 and reported through the module logger.  A mixed graph is the
safe alternative to guessing static activation ranges for arbitrary inputs.
"""

from __future__ import annotations

import logging
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import torch
import torch.fx as fx
from torch import Tensor, nn

from shrinker.registry import register
from shrinker.utils import get_module, replace_module

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

_AFFINE_TYPES: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.Linear)
_BATCHNORM_TYPES: Tuple[Type[nn.Module], ...] = (nn.BatchNorm1d, nn.BatchNorm2d)


@dataclass(frozen=True)
class _Path:
    """A single, non-branching affine--BatchNorm--ReLU--affine path."""

    producer: str
    batchnorm: str
    consumer: str


@dataclass(frozen=True)
class _ActivationStats:
    """Analytic statistics and static qparams for a ReLU activation."""

    mean: Tensor
    scale: float
    zero_point: int
    maximum: float


def _as_args(example_input: Any) -> Tuple[Any, ...]:
    return example_input if isinstance(example_input, tuple) else (example_input,)


def _select_backend(backend: str) -> str:
    """Return a supported CPU quantization backend for ``backend``."""
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
    """Temporarily select a quantized CPU backend without leaking global state."""
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = backend
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous


def _only_user(node: fx.Node) -> Optional[fx.Node]:
    """Return a node's unique user, or ``None`` when it branches or terminates."""
    users = tuple(node.users)
    return users[0] if len(users) == 1 else None


def _find_paths(model: nn.Module) -> List[_Path]:
    """Find paths for which DFQ reparameterization is exactly well-defined.

    The deliberately narrow topology avoids modifying residual, concatenated,
    reused, or functional graphs.  FX tracing is structural and does not run
    or derive statistics from a sample input.
    """
    try:
        graph = fx.symbolic_trace(model).graph
    except Exception as exc:  # noqa: BLE001 - arbitrary user models may not trace
        logger.info("Skipping DFQ: model could not be traced with torch.fx: %s", exc)
        return []

    modules = dict(model.named_modules())
    paths: List[_Path] = []
    seen_consumers = set()
    for source in graph.nodes:
        if source.op != "call_module" or type(modules.get(source.target)) not in _AFFINE_TYPES:
            continue
        bn_node = _only_user(source)
        if bn_node is None or bn_node.op != "call_module" or type(modules.get(bn_node.target)) not in _BATCHNORM_TYPES:
            continue
        activation = _only_user(bn_node)
        if (
            activation is None
            or activation.op != "call_module"
            or type(modules.get(activation.target)) is not nn.ReLU
        ):
            continue
        consumer = _only_user(activation)
        if consumer is None or consumer.op != "call_module" or type(modules.get(consumer.target)) not in _AFFINE_TYPES:
            continue

        producer_layer = modules[source.target]
        bn = modules[bn_node.target]
        consumer_layer = modules[consumer.target]
        if not _compatible_path(producer_layer, bn, consumer_layer):
            logger.info("Skipping DFQ path %s -> %s: incompatible layer shapes", source.target, consumer.target)
            continue
        if consumer.target in seen_consumers:
            logger.info("Skipping DFQ path ending at %s: consumer is shared", consumer.target)
            continue
        seen_consumers.add(consumer.target)
        paths.append(_Path(source.target, bn_node.target, consumer.target))
    if not paths:
        logger.info("DFQ found no safe Conv/Linear -> BatchNorm -> ReLU -> Conv/Linear paths")
    return paths


def _compatible_path(producer: nn.Module, bn: nn.Module, consumer: nn.Module) -> bool:
    """Return whether a path has matching channels and supported affine layers."""
    if type(producer) is nn.Conv2d:
        if producer.groups != 1 or type(bn) is not nn.BatchNorm2d:
            return False
        channels = producer.out_channels
    elif type(producer) is nn.Linear:
        if type(bn) is not nn.BatchNorm1d:
            return False
        channels = producer.out_features
    else:
        return False
    if bn.num_features != channels:
        return False
    if type(consumer) is nn.Conv2d:
        return consumer.groups == 1 and consumer.in_channels == channels
    return type(consumer) is nn.Linear and consumer.in_features == channels


def _effective_output_ranges(layer: nn.Module, bn: nn.modules.batchnorm._BatchNorm) -> Tensor:
    """Return per-output-channel ranges after BatchNorm's affine transform."""
    weight = layer.weight.detach()
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    return weight.abs().flatten(1).amax(dim=1) * scale.abs()


def _input_ranges(layer: nn.Module) -> Tensor:
    """Return the range contributed by every input channel of an affine layer."""
    if type(layer) is nn.Conv2d:
        return layer.weight.detach().abs().amax(dim=(0, 2, 3))
    return layer.weight.detach().abs().amax(dim=0)


def _scale_path(producer: nn.Module, bn: nn.modules.batchnorm._BatchNorm, consumer: nn.Module) -> None:
    """Apply one exact cross-layer equalization update to a safe path."""
    first = _effective_output_ranges(producer, bn)
    second = _input_ranges(consumer)
    eps = torch.finfo(first.dtype).eps
    # s = sqrt(r_first / r_second).  Zero ranges are already exactly represented
    # by a per-tensor quantizer, so leaving them unchanged avoids invalid scales.
    scale = torch.ones_like(first)
    valid = (first > eps) & (second > eps)
    scale[valid] = torch.sqrt(first[valid] / second[valid])
    with torch.no_grad():
        bn.weight.div_(scale)
        bn.bias.div_(scale)
        if type(consumer) is nn.Conv2d:
            consumer.weight.mul_(scale.reshape(1, -1, 1, 1))
        else:
            consumer.weight.mul_(scale.reshape(1, -1))


def _ensure_bias(layer: nn.Module) -> nn.Parameter:
    """Return ``layer``'s bias, materializing a zero bias when mathematically needed."""
    if layer.bias is None:
        size = layer.out_channels if type(layer) is nn.Conv2d else layer.out_features
        layer.bias = nn.Parameter(torch.zeros(size, device=layer.weight.device, dtype=layer.weight.dtype))
    return layer.bias


def _absorb_high_bias(bn: nn.modules.batchnorm._BatchNorm, consumer: nn.Module) -> None:
    """Absorb the DFQ Gaussian high-bias estimate from a ReLU into its consumer."""
    beta = bn.bias.detach()
    gamma = bn.weight.detach().abs()
    correction = (beta - 3.0 * gamma).clamp_min(0)
    if not torch.any(correction):
        return
    with torch.no_grad():
        bn.bias.sub_(correction)
        target_bias = _ensure_bias(consumer)
        if type(consumer) is nn.Conv2d:
            target_bias.add_((consumer.weight * correction.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)))
        else:
            target_bias.add_(consumer.weight.matmul(correction))


def _relu_stats(bn: nn.modules.batchnorm._BatchNorm) -> _ActivationStats:
    """Derive ReLU expectation and an asymmetric INT8 range from BatchNorm.

    The expectation is the closed-form mean of a rectified normal random
    variable.  The range follows DFQ's six-standard-deviation heuristic.
    """
    mean = bn.bias.detach()
    std = bn.weight.detach().abs().clamp_min(torch.finfo(mean.dtype).eps)
    alpha = mean / std
    normal_pdf = torch.exp(-0.5 * alpha.square()) / (2.0 * torch.pi) ** 0.5
    normal_cdf = 0.5 * (1.0 + torch.erf(alpha / 2.0**0.5))
    expected = std * normal_pdf + mean * normal_cdf
    maximum = float((mean + 6.0 * std).clamp_min(0).amax().item())
    maximum = max(maximum, torch.finfo(mean.dtype).eps)
    return _ActivationStats(mean=expected, scale=maximum / 255.0, zero_point=0, maximum=maximum)


def _quantize_weight(weight: Tensor) -> Tensor:
    """Quantize a weight tensor with DFQ's symmetric per-tensor INT8 scheme."""
    maximum = float(weight.detach().abs().amax().item())
    scale = max(maximum / 127.0, torch.finfo(weight.dtype).eps)
    return torch.quantize_per_tensor(weight.detach().cpu(), scale, 0, torch.qint8)


def _output_qparams(layer: nn.Module, input_stats: _ActivationStats) -> Tuple[float, int]:
    """Return conservative output qparams without observing activations."""
    if type(layer) is nn.Conv2d:
        bound = layer.weight.detach().abs().sum(dim=(1, 2, 3)) * input_stats.maximum
        bias = layer.bias.detach().abs() if layer.bias is not None else torch.zeros_like(bound)
    else:
        bound = layer.weight.detach().abs().sum(dim=1) * input_stats.maximum
        bias = layer.bias.detach().abs() if layer.bias is not None else torch.zeros_like(bound)
    maximum = max(float((bound + bias).amax().item()), torch.finfo(layer.weight.dtype).eps)
    # Quantized Conv2d/Linear output is quint8.  A centered range allows both
    # signs while retaining an affine zero point accepted by CPU kernels.
    return maximum / 127.0, 128


class _QuantizedConv2d(nn.Module):
    """Float-boundary wrapper around a static, per-tensor INT8 Conv2d kernel."""

    def __init__(self, layer: nn.Conv2d, input_stats: _ActivationStats) -> None:
        super().__init__()
        output_scale, output_zero_point = _output_qparams(layer, input_stats)
        quantized = torch.ao.nn.quantized.Conv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
            layer.padding_mode,
        )
        quantized.set_weight_bias(_quantize_weight(layer.weight), None if layer.bias is None else layer.bias.detach().cpu())
        quantized.scale = output_scale
        quantized.zero_point = output_zero_point
        self.input_scale = input_stats.scale
        self.input_zero_point = input_stats.zero_point
        self.kernel = quantized

    def forward(self, x: Tensor) -> Tensor:
        quantized = torch.quantize_per_tensor(x.contiguous(), self.input_scale, self.input_zero_point, torch.quint8)
        return self.kernel(quantized).dequantize()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "kernel"):
            for attr in (
                "_modules", "_parameters", "_buffers", "_backward_hooks",
                "_backward_pre_hooks", "_forward_hooks", "_forward_pre_hooks",
                "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                "_load_state_dict_post_hooks", "_non_persistent_buffers_set"
            ):
                if attr not in self.kernel.__dict__:
                    if attr == "_non_persistent_buffers_set":
                        self.kernel.__dict__[attr] = set()
                    else:
                        self.kernel.__dict__[attr] = {}


class _QuantizedLinear(nn.Module):
    """Float-boundary wrapper around a static, per-tensor INT8 Linear kernel."""

    def __init__(self, layer: nn.Linear, input_stats: _ActivationStats) -> None:
        super().__init__()
        output_scale, output_zero_point = _output_qparams(layer, input_stats)
        quantized = torch.ao.nn.quantized.Linear(layer.in_features, layer.out_features, layer.bias is not None)
        quantized.set_weight_bias(_quantize_weight(layer.weight), None if layer.bias is None else layer.bias.detach().cpu())
        quantized.scale = output_scale
        quantized.zero_point = output_zero_point
        self.input_scale = input_stats.scale
        self.input_zero_point = input_stats.zero_point
        self.kernel = quantized

    def forward(self, x: Tensor) -> Tensor:
        quantized = torch.quantize_per_tensor(x.contiguous(), self.input_scale, self.input_zero_point, torch.quint8)
        return self.kernel(quantized).dequantize()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "kernel"):
            for attr in (
                "_modules", "_parameters", "_buffers", "_backward_hooks",
                "_backward_pre_hooks", "_forward_hooks", "_forward_pre_hooks",
                "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                "_load_state_dict_post_hooks", "_non_persistent_buffers_set"
            ):
                if attr not in self.kernel.__dict__:
                    if attr == "_non_persistent_buffers_set":
                        self.kernel.__dict__[attr] = set()
                    else:
                        self.kernel.__dict__[attr] = {}


def _bias_correct(layer: nn.Module, expected_input: Tensor) -> None:
    """Subtract analytic quantization-error bias from a Conv2d or Linear layer."""
    quantized_weight = _quantize_weight(layer.weight).dequantize().to(layer.weight.device, layer.weight.dtype)
    error = quantized_weight - layer.weight.detach()
    with torch.no_grad():
        bias = _ensure_bias(layer)
        if type(layer) is nn.Conv2d:
            shift = (error.sum(dim=(2, 3)) * expected_input.reshape(1, -1)).sum(dim=1)
        else:
            shift = error.matmul(expected_input)
        bias.sub_(shift)


def _replace_with_quantized(model: nn.Module, name: str, stats: _ActivationStats) -> bool:
    """Bias-correct and replace one supported layer with a CPU INT8 wrapper."""
    layer = get_module(model, name)
    try:
        _bias_correct(layer, stats.mean.to(layer.weight.device, layer.weight.dtype))
        if type(layer) is nn.Conv2d:
            replace_module(model, name, _QuantizedConv2d(layer, stats))
        elif type(layer) is nn.Linear:
            replace_module(model, name, _QuantizedLinear(layer, stats))
        else:
            return False
    except Exception as exc:  # noqa: BLE001 - backend support varies by PyTorch build
        logger.info("Leaving %s in FP32: DFQ INT8 conversion failed: %s", name, exc)
        return False
    return True


@register(kind="quantization", paper="Nagel et al., 2019")
def apply(
    model: nn.Module,
    example_input: Any = None,
    backend: str = "auto",
    absorb_bias: bool = True,
) -> nn.Module:
    """Convert safe BatchNorm/ReLU paths to data-free CPU INT8 inference.

    ``example_input`` is an arbitrary Tensor or tuple of Tensors used only to
    validate FX tracing and the final graph's shape.  It is not calibration
    data: no forward pass, observer update, or quantization statistic uses its
    values.  Paths without BatchNorm-derived activation statistics remain FP32.
    """
    if example_input is None:
        logger.info("Skipping DFQ quantization: example_input is required for safe FX graph tracing")
        return model
    example_args = _as_args(example_input)
    if not example_args or not all(isinstance(value, Tensor) for value in example_args):
        logger.info("Skipping DFQ quantization: example_input must be a Tensor or tuple of Tensors")
        return model
    if not isinstance(absorb_bias, bool):
        raise ValueError(f"absorb_bias must be a bool, got {absorb_bias!r}")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        logger.info("Skipping DFQ quantization: CPU INT8 kernels cannot safely replace a non-CPU model")
        return model

    selected_backend = _select_backend(backend)
    was_training = model.training
    model.eval()
    paths = _find_paths(model)
    if not paths:
        model.train(was_training)
        return model

    # Equalizing several connected pairs requires multiple passes because an
    # update to one consumer changes the effective range of its next pair.
    for _ in range(10):
        for path in paths:
            _scale_path(get_module(model, path.producer), get_module(model, path.batchnorm), get_module(model, path.consumer))

    activation_stats: Dict[str, _ActivationStats] = {}
    for path in paths:
        bn = get_module(model, path.batchnorm)
        consumer = get_module(model, path.consumer)
        if absorb_bias:
            _absorb_high_bias(bn, consumer)
        activation_stats[path.consumer] = _relu_stats(bn)

    converted = 0
    with _quantized_engine(selected_backend):
        for name, stats in activation_stats.items():
            converted += _replace_with_quantized(model, name, stats)
    model.train(was_training)
    if converted:
        logger.info("DFQ converted %d affine layers to CPU INT8 using the %s backend", converted, selected_backend)
    else:
        logger.info("DFQ left this model in FP32 because no safe path could be lowered to CPU INT8")
    return model