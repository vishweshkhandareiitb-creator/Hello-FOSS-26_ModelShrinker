from __future__ import annotations

from typing import Any
import torch
from torch import nn

try:
    import onnxruntime as ort
    import numpy as np
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _to_numpy(x: Any) -> Any:
    """convert pytorch tensors recursively to numpy arrays for onnxruntime."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (tuple, list)):
        return type(x)(_to_numpy(t) for t in x)
    return x


def _check_quantization(model: nn.Module):
    """detect quantized models or layers and raise a clear error."""
    for name, module in model.named_modules():
        mod_type = type(module).__name__.lower()
        mod_mod = module.__class__.__module__.lower()
        if "quantized" in mod_mod or "quantized" in mod_type or "qat" in mod_type:
            raise ValueError(
                f"quantization result detected in layer '{name}' ({module.__class__.__name__}). "
                "quantized models cannot be directly exported to onnx."
            )
    
    for name, p in model.named_parameters():
        if p.dtype in (torch.qint8, torch.quint8, torch.qint32, torch.quint4x2):
            raise ValueError(
                f"quantized parameter dtype {p.dtype} detected in '{name}'. "
                "quantized models are not supported for direct onnx export."
            )


def export_onnx(
    model: nn.Module,
    example_input: Any,
    filepath: str,
    opset_version: int = 11,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """export a pytorch model to onnx and verify output parity using onnxruntime."""
    if not HAS_ORT:
        raise ImportError("onnxruntime is required for export_onnx parity checks. install via pip install onnxruntime.")

    # 1. check for quantization
    _check_quantization(model)

    model.eval()

    # 2. get baseline pytorch output
    with torch.no_grad():
        if isinstance(example_input, tuple):
            torch_out = model(*example_input)
        else:
            torch_out = model(example_input)

    if isinstance(torch_out, torch.Tensor):
        torch_out_tuple = (torch_out,)
    elif isinstance(torch_out, (tuple, list)):
        torch_out_tuple = tuple(torch_out)
    else:
        raise TypeError(f"unsupported model output type: {type(torch_out)}")

    # 3. export to onnx
    try:
        torch.onnx.export(
            model,
            example_input,
            filepath,
            opset_version=opset_version,
            input_names=input_names or ["input"],
            output_names=output_names or [f"output_{i}" for i in range(len(torch_out_tuple))],
            dynamic_axes=None,
        )
    except Exception as e:
        raise RuntimeError(f"failed to export model to onnx: {e}") from e

    # 4. verify output parity using onnxruntime
    try:
        ort_session = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
        
        if isinstance(example_input, tuple):
            ort_inputs = {node.name: _to_numpy(inp) for node, inp in zip(ort_session.get_inputs(), example_input)}
        else:
            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: _to_numpy(example_input)}

        ort_outs = ort_session.run(None, ort_inputs)
    except Exception as e:
        raise RuntimeError(f"failed to run exported onnx model in onnxruntime: {e}") from e

    # 5. compare outputs using allclose
    for i, (t_out, o_out) in enumerate(zip(torch_out_tuple, ort_outs)):
        t_np = _to_numpy(t_out)
        o_np = np.array(o_out)

        if not np.allclose(t_np, o_np, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(t_np - o_np)))
            raise AssertionError(
                f"onnx export output parity check failed for output index {i}! "
                f"maximum absolute difference: {max_diff:.6e}"
            )