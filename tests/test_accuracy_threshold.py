"""CI guard: every compression method must keep accuracy above one shared threshold.

For each method in the registry: train SmallVGG on synthetic data, compress it, and (when the
result is still a float model) finetune for three epoch, then assert accuracy >= THRESHOLD.
Quantized models cannot be trained further, so they are scored straight after compression.

Run `python -m tests.test_accuracy_threshold` from the repo root to print the numbers
(5 seeds, all methods) used to choose THRESHOLD.
"""

import inspect

import pytest
import torch

import shrinker
from examples.demo import synthetic_loaders
from examples.test_models import SmallVGG
from shrinker.benchmark import evaluate
from shrinker.registry import REGISTRY

THRESHOLD = 0.95  # worst case over 5 seeds and all methods was 0.979 (dfq_quantization); leaves margin for CI numerics
BASELINE_MIN = 0.95  # guards against passing because training broke
FINETUNE_EPOCHS = 3  # 1 epoch gave seed-dependent recovery after pruning (0.80-0.97); 3 is stable (>= 0.99)

# Options per method. A method that is not listed here runs with its defaults,
# so newly added methods are checked automatically.
OPTIONS = {
    "l1_pruning": {"ratio": 0.5},
    "fpgm": {"ratio": 0.5},
    "gradient_based": {"ratio": 0.5},
}


def is_quantized(model):
    return any("quantized" in type(m).__module__ for m in model.modules())


def train_baseline(seed=0):
    torch.manual_seed(seed)
    train_loader, val_loader = synthetic_loaders(batch_size=128, n_train=2000, n_test=1000)
    model = SmallVGG()
    shrinker.finetune(model, train_loader, epochs=3, lr=1e-3, verbose=False)
    return model, train_loader, val_loader


def compress_and_score(name, model, train_loader, val_loader):
    """Return (accuracy right after compress, accuracy after the optional finetune)."""
    kwargs = dict(OPTIONS.get(name, {}))
    if "loader" in inspect.signature(REGISTRY[name].fn).parameters:
        kwargs["loader"] = train_loader
    example = torch.randn(1, 3, 32, 32)
    small = shrinker.compress(model, name, example_input=example, **kwargs)
    acc_compressed = evaluate(small, val_loader)
    if is_quantized(small):
        return acc_compressed, acc_compressed
    shrinker.finetune(small, train_loader, epochs=FINETUNE_EPOCHS, lr=5e-4, verbose=False)
    return acc_compressed, evaluate(small, val_loader)


@pytest.fixture(scope="module")
def baseline():
    return train_baseline(seed=0)


def test_baseline_is_accurate(baseline):
    model, _, val_loader = baseline
    assert evaluate(model, val_loader) >= BASELINE_MIN


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_compressed_model_meets_accuracy_threshold(name, baseline):
    model, train_loader, val_loader = baseline
    _, acc = compress_and_score(name, model, train_loader, val_loader)
    assert acc >= THRESHOLD, f"{name}: accuracy {acc:.3f} fell below threshold {THRESHOLD:.3f}"


if __name__ == "__main__":
    import time

    results = {}
    for seed in (0, 1, 2, 3, 4):
        model, train_loader, val_loader = train_baseline(seed)
        print(f"seed={seed} baseline={evaluate(model, val_loader):.3f}")
        for name in sorted(REGISTRY):
            t0 = time.time()
            try:
                compressed, final = compress_and_score(name, model, train_loader, val_loader)
            except Exception as e:  # keep going so one broken method doesn't hide the rest
                print(f"  {name:<24} FAILED: {type(e).__name__}: {e}")
                continue
            results.setdefault(name, []).append((compressed, final))
            print(f"  {name:<24} compressed={compressed:.3f} final={final:.3f} ({time.time() - t0:.0f}s)")

    print("\nworst case over seeds (final accuracy):")
    for name, vals in results.items():
        print(f"  {name:<24} {min(f for _, f in vals):.3f}")