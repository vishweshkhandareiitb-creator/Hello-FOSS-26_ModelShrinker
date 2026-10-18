from __future__ import annotations

import contextlib
import io
import statistics
import time
from typing import Any, Dict, Optional

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """Parameter count minus zeroed entries of weight tensors (biases/BatchNorm always count)."""
    return sum(int(torch.count_nonzero(p)) if p.dim() > 1 else p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@contextlib.contextmanager
def _eval_on(model: nn.Module, device: str):
    was_training = model.training
    first = next(model.parameters(), None)
    old_device = first.device if first is not None else torch.device("cpu")
    moved = torch.device(device) != old_device
    if moved:
        model.to(device)
    model.eval()
    try:
        yield
    finally:
        if moved:
            model.to(old_device)
        model.train(was_training)


def _run(model: nn.Module, x: Any):
    return model(*x) if isinstance(x, tuple) else model(x)


def _to(x: Any, device: str):
    if isinstance(x, tuple):
        return tuple(t.to(device) for t in x)
    return x.to(device)


def measure_latency(
    model: nn.Module, example_input: Any, warmup: int = 10, runs: int = 50, device: str = "cpu"
) -> float:
    """Median wall-clock time of one forward pass, in milliseconds."""
    x = _to(example_input, device)
    cuda = torch.device(device).type == "cuda"
    with _eval_on(model, device), torch.no_grad():
        for _ in range(warmup):
            _run(model, x)
        times = []
        for _ in range(runs):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _run(model, x)
            if cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def count_flops(model: nn.Module, example_input: Any) -> int:
    """FLOPs of the Conv2d and Linear layers for one forward pass."""
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        total[0] += out.numel() * m.in_features

    handles = []
    for m in model.modules():
        if type(m) is nn.Conv2d:
            handles.append(m.register_forward_hook(conv_hook))
        elif type(m) is nn.Linear:
            handles.append(m.register_forward_hook(linear_hook))
    try:
        with _eval_on(model, "cpu"), torch.no_grad():
            _run(model, _to(example_input, "cpu"))
    finally:
        for h in handles:
            h.remove()
    return 2 * total[0]


def estimate_memory(model: nn.Module, example_input: Any, device: str = "cpu") -> Dict[str, float]:
    """estimate weights, activations, and peak memory in mb."""
    x = _to(example_input, device)
    cuda = torch.device(device).type == "cuda"
    
    # calculate weights memory taking element size and quantization types into account
    weights_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    weights_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    weights_mb = weights_bytes / 1e6

    activation_bytes = [0]

    def hook_fn(m, inp, out):
        if isinstance(out, torch.Tensor):
            activation_bytes[0] += out.nelement() * out.element_size()
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    activation_bytes[0] += o.nelement() * o.element_size()

    handles = []
    for m in model.modules():
        if list(m.children()):
            continue
        handles.append(m.register_forward_hook(hook_fn))

    try:
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        with _eval_on(model, device), torch.no_grad():
            _run(model, x)

        if cuda:
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated(device)
        else:
            # estimate peak memory on cpu as model weights plus activation memory
            peak_bytes = weights_bytes + activation_bytes[0]
    finally:
        for h in handles:
            h.remove()

    return {
        "weights_mb": weights_mb,
        "activations_mb": activation_bytes[0] / 1e6,
        "peak_memory_mb": peak_bytes / 1e6,
    }


def evaluate(model: nn.Module, loader, device: str = "cpu") -> float:
    """Top-1 accuracy (0..1)."""
    correct = total = 0
    with _eval_on(model, device), torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def measure(
    model: nn.Module, example_input: Any, val_loader=None, device: str = "cpu", warmup: int = 10, runs: int = 50
) -> Dict[str, Optional[float]]:
    try:
        flops = count_flops(model, example_input)
    except Exception:  # noqa: BLE001
        flops = None

    try:
        memory = estimate_memory(model, example_input, device)
    except Exception:  # noqa: BLE001
        sz = model_size_mb(model)
        memory = {"weights_mb": sz, "activations_mb": 0.0, "peak_memory_mb": sz}

    return {
        "params": count_parameters(model),
        "nonzero_params": count_nonzero_parameters(model),
        "size_mb": model_size_mb(model),
        "weights_mb": memory["weights_mb"],
        "activations_mb": memory["activations_mb"],
        "peak_memory_mb": memory["peak_memory_mb"],
        "latency_ms": measure_latency(model, example_input, warmup, runs, device),
        "flops": flops,
        "accuracy": evaluate(model, val_loader, device) if val_loader is not None else None,
    }


def compare(
    original: nn.Module,
    compressed: nn.Module,
    example_input: Any,
    val_loader=None,
    device: str = "cpu",
    warmup: int = 10,
    runs: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Measure both models, print a table, and return ``{"original": {...}, "compressed": {...}}``."""
    report = {
        "original": measure(original, example_input, val_loader, device, warmup, runs),
        "compressed": measure(compressed, example_input, val_loader, device, warmup, runs),
    }
    if verbose:
        print(format_report(report))
    return report


def format_report(report: Dict[str, Dict[str, Optional[float]]]) -> str:
    a, b = report["original"], report["compressed"]

    def pct(x, y):
        return f"{(y - x) / x * 100:+.1f}%" if x else "-"

    rows = [("params", f"{a['params']:,}", f"{b['params']:,}", pct(a["params"], b["params"]))]
    if a["nonzero_params"] < a["params"] or b["nonzero_params"] < b["params"]:
        rows.append(("non-zero params", f"{a['nonzero_params']:,}", f"{b['nonzero_params']:,}",
                     pct(a["nonzero_params"], b["nonzero_params"])))
    rows.append(("size (MB)", f"{a['size_mb']:.3f}", f"{b['size_mb']:.3f}", pct(a["size_mb"], b["size_mb"])))
    rows.append(("weights (MB)", f"{a['weights_mb']:.3f}", f"{b['weights_mb']:.3f}", pct(a["weights_mb"], b["weights_mb"])))
    rows.append(("activations (MB)", f"{a['activations_mb']:.3f}", f"{b['activations_mb']:.3f}", pct(a["activations_mb"], b["activations_mb"])))
    rows.append(("peak memory (MB)", f"{a['peak_memory_mb']:.3f}", f"{b['peak_memory_mb']:.3f}", pct(a["peak_memory_mb"], b["peak_memory_mb"])))
    speedup = a["latency_ms"] / b["latency_ms"] if b["latency_ms"] else float("inf")
    rows.append(("latency (ms)", f"{a['latency_ms']:.3f}", f"{b['latency_ms']:.3f}", f"{speedup:.2f}x speed"))
    if a["flops"] is not None and b["flops"] is not None:
        rows.append(("MFLOPs", f"{a['flops'] / 1e6:.2f}", f"{b['flops'] / 1e6:.2f}", pct(a["flops"], b["flops"])))
    if a["accuracy"] is not None:
        rows.append(
            (
                "accuracy",
                f"{a['accuracy'] * 100:.2f}%",
                f"{b['accuracy'] * 100:.2f}%",
                f"{(b['accuracy'] - a['accuracy']) * 100:+.2f} pts",
            )
        )
    header = ("", "original", "compressed", "change")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(4)]
    fmt = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))  # noqa: E731
    line = "-" * len(fmt(header))
    return "\n".join([fmt(header), line] + [fmt(r) for r in rows])