import torch

from examples.test_models import SmallVGG
from shrinker import compare, compress
from shrinker.benchmark import count_flops, count_parameters, measure_latency, model_size_mb
from tests.helpers import make_loader

X = torch.randn(1, 3, 32, 32)


def test_compare_report():
    model = SmallVGG()
    small = compress(model, "l1_pruning", ratio=0.5)
    report = compare(model, small, X, val_loader=make_loader((3, 32, 32)), runs=5, warmup=1, verbose=False)
    a, b = report["original"], report["compressed"]
    assert set(a) == {
        "params",
        "nonzero_params",
        "size_mb",
        "weights_mb",
        "activations_mb",
        "peak_memory_mb",
        "latency_ms",
        "flops",
        "accuracy",
    }
    assert b["params"] < a["params"]
    assert b["size_mb"] < a["size_mb"]
    assert b["flops"] < a["flops"]
    assert a["latency_ms"] > 0 and b["latency_ms"] > 0
    assert 0 <= a["accuracy"] <= 1


def test_compare_prints_table(capsys):
    model = SmallVGG()
    compare(model, compress(model, "l1_pruning"), X, runs=3, warmup=1)
    out = capsys.readouterr().out
    for word in ("params", "size (MB)", "latency (ms)", "MFLOPs"):
        assert word in out
    assert "accuracy" not in out


def test_measurement_restores_training_mode():
    model = SmallVGG().train()
    measure_latency(model, X, warmup=1, runs=2)
    count_flops(model, X)
    assert model.training


def test_flops_hand_computed():
    lin = torch.nn.Linear(10, 5)
    assert count_flops(lin, torch.randn(1, 10)) == 2 * 10 * 5


def test_size_matches_params():
    model = torch.nn.Linear(1000, 1000)
    assert abs(model_size_mb(model) - count_parameters(model) * 4 / 1e6) < 0.01


def test_nonzero_row_only_when_weights_are_zeroed(capsys):
    model = SmallVGG()
    compare(model, compress(model, "l1_pruning"), X, runs=2, warmup=1)
    assert "non-zero" not in capsys.readouterr().out

    sparse = compress(model, "l1_pruning")
    with torch.no_grad():
        sparse.classifier[1].weight[:, ::2] = 0
    report = compare(model, sparse, X, runs=2, warmup=1)
    assert "non-zero params" in capsys.readouterr().out
    assert report["compressed"]["nonzero_params"] < report["compressed"]["params"]