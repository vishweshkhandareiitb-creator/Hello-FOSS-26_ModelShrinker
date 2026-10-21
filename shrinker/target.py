from __future__ import annotations

import inspect
from typing import Any, Optional, Sequence

from torch import nn

from . import benchmark
from .core import compress
from .registry import get_method


# Target name -> key in the dict returned by benchmark.measure().
_TARGET_KEYS = {
    "size_mb": "size_mb",
    "Mflops": "flops",
    "params": "params",
    "latency_ms": "latency_ms",
    "max_accuracy_drop": "accuracy",
}

# Signature parameter names that act as a "how much to compress" knob, in priority order.
_SWEEP_KNOBS = ("ratio", "sparsity")

_DEFAULT_RATIOS = [i / 10 for i in range(1, 10)]


def _sweep_knob(method: str) -> Optional[str]:
    """Name of the method's tunable compression knob ('ratio', 'sparsity', ...) or None."""
    params = inspect.signature(get_method(method).fn).parameters
    for knob in _SWEEP_KNOBS:
        if knob in params:
            return knob
    return None


def compress_to_target(
    model: nn.Module,
    method: str,
    target: str,
    target_value: float,
    val_loader=None,
    example_input: Optional[Any] = None,
    ratios: Optional[Sequence[float]] = None,
    refine_steps: int = 4,
    **method_kwargs,
) -> nn.Module:
    """Return a compressed copy of ``model`` (the original is not modified).

    The user only states a goal -- a ``target`` (e.g. ``"latency_ms"``) and a
    ``target_value`` -- and the right compression strength is figured out
    automatically:

    * The tunable knob of ``method`` is detected from its signature
      (``ratio`` for pruning methods, ``sparsity`` for knowledge
      distillation, none for the one-shot quantization methods).
    * A coarse grid of knob values is measured, then every grid interval
      whose metric crosses the target is bisected ``refine_steps`` times, so
      the chosen knob value lands much closer to the goal than any fixed
      grid point.
    * Selection: for ``max_accuracy_drop`` the candidate with the largest
      accuracy drop that still respects the budget wins (i.e. the most
      compressed model within budget); for the other targets the mildest
      compression that meets the target wins -- the largest metric value
      still within the budget, which preserves the most accuracy
      (falling back to the closest attempt if nothing qualifies).
    * Knob-less methods (quantization) are evaluated once: returned when
      they meet the target, otherwise returned anyway with
      ``target_reached: False`` -- there is nothing to tune.

    Extra keyword arguments are forwarded to the method, e.g.
    ``loader=...`` (required by gradient_based, knowledge_distillation and
    qat_quantization).
    """
    if target not in {*_TARGET_KEYS}:
        raise ValueError(
            f"Unknown target {target!r}. Choose one of {[*_TARGET_KEYS]}."
        )
    if target_value < 0:
        raise ValueError("target_value must be non-negative")
    if target == "max_accuracy_drop" and val_loader is None:
        raise ValueError("val_loader is required for target 'max_accuracy_drop'")

    get_method(method)  # fail fast on unknown method names, before measuring

    reported_target = target_value
    # Convert MFlops to flops for comparison against the 'flops' metric.
    if target == "Mflops":
        target_value = target_value * 1e6

    baseline = benchmark.measure(model, example_input=example_input, val_loader=val_loader)
    if target == "max_accuracy_drop":
        if baseline["accuracy"] is None:
            raise RuntimeError("Could not measure baseline accuracy")
        metric_key = "accuracy"
    elif target == "Mflops":
        metric_key = "flops"
        if baseline[metric_key] is None:
            raise RuntimeError("Could not measure baseline MFlops")
    else:
        metric_key = _TARGET_KEYS[target]
        if baseline[metric_key] is None:
            raise RuntimeError(f"Could not measure baseline {target!r}")

    knob = _sweep_knob(method)

    def _evaluate(knob_value: Optional[float]) -> Optional[dict]:
        kwargs = dict(method_kwargs)
        if knob is not None:
            kwargs[knob] = knob_value
        candidate = compress(model, method, example_input=example_input, **kwargs)
        if metric_key == "accuracy":
            metrics = benchmark.measure(candidate, example_input=example_input, val_loader=val_loader)
        else:
            metrics = benchmark.measure(candidate, example_input=example_input)

        if target == "max_accuracy_drop":
            if metrics["accuracy"] is None:
                return None
            value = baseline["accuracy"] - metrics["accuracy"]  # actual accuracy drop
        else:
            value = metrics[metric_key]
            if value is None:
                return None

        return {
            "ratio": knob_value,
            "knob": knob,
            "model": candidate,
            "metrics": metrics,
            "value": value,
            "reached": value <= target_value,
            "distance": abs(value - target_value),
        }

    candidates: list = []

    if knob is None:
        # One-shot method (e.g. quantization): a single candidate, nothing to sweep.
        only = _evaluate(None)
        if only is not None:
            candidates.append(only)
    else:
        grid = sorted(set(ratios)) if ratios is not None else _DEFAULT_RATIOS
        if not grid:
            raise ValueError("ratios must not be empty")

        points, seen = [], set()
        for knob_value in grid:
            if knob_value in seen:
                continue
            seen.add(knob_value)
            point = _evaluate(knob_value)
            if point is not None:
                points.append(point)
                candidates.append(point)

        # Bisect inside every grid interval whose metric crosses the target,
        # to land closer to it than any grid point can.
        for lo, hi in zip(points, points[1:]):
            if (lo["value"] - target_value) * (hi["value"] - target_value) >= 0:
                continue  # no crossing inside this interval
            for _ in range(max(0, refine_steps)):
                mid = (lo["ratio"] + hi["ratio"]) / 2
                if mid in seen:
                    break
                seen.add(mid)
                point = _evaluate(mid)
                if point is None:
                    break
                candidates.append(point)
                if point["distance"] == 0:
                    break
                if (point["value"] - target_value) * (lo["value"] - target_value) <= 0:
                    hi = point  # point is on the hi-side of the target
                else:
                    lo = point

    if not candidates:
        raise RuntimeError(
            "No candidates could be evaluated. Check the compression method, "
            "example_input and measurements."
        )

    successful = [c for c in candidates if c["reached"]]
    tie = lambda c: c["ratio"] if c["ratio"] is not None else 0.0  # noqa: E731

    if target == "max_accuracy_drop":
        if successful:
            # All of these satisfy the budget; take the one whose actual loss is
            # closest to the loss we wanted (largest drop still within budget).
            # On equal drops, the stronger compression (larger knob) wins.
            best = max(successful, key=lambda c: (c["value"], tie(c)))
        else:
            best = min(candidates, key=lambda c: (c["distance"], tie(c)))
    else:
        if successful:
            # Mildest compression that meets the target: the largest metric
            # value still within budget, which preserves the most accuracy.
            # On equal metric values (plateau), the milder knob wins.
            best = max(successful, key=lambda c: (c["value"], -tie(c)))
        else:
            best = min(candidates, key=lambda c: (c["distance"], tie(c)))

    print(
        f"target : {target}\n"
        f"target_value: {reported_target}\n"
        f"method: {method} (knob: {knob or 'none, one-shot method'})\n"
        f"target_reached : {best['reached']}\n"
        f"selected_{knob or 'ratio'}: {best['ratio']}\n"
    )
    print(
        f"original: {baseline}\n"
        f"compressed: {best['metrics']}\n"
        f"candidates_evaluated: {len(candidates)}\n"
    )
    if knob is None and not best["reached"]:
        print(f"warning: {method!r} has no tunable knob; the target cannot be approached more closely.\n")

    return best["model"]