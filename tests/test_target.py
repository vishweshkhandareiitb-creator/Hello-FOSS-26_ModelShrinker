"""Tests for ``shrinker.target.compress_to_target``.

The selection logic is tested against a deterministic fake ``benchmark.measure``
whose metrics are pure functions of the knob value that the fake compression
methods store on the model (``model.knob``). That removes all timing/eval
noise, so the exact selection rule -- and every way it can go wrong -- can be
pinned down. A few tests at the end run against the real ``measure`` to check
the plumbing end to end.

The two headline rules being pinned:

* resource targets (``size_mb``, ``Mflops``, ``params``, ``latency_ms``):
  the mildest compression that meets the target wins, i.e. the LARGEST
  metric value still within budget (extra compression only costs accuracy);
* ``max_accuracy_drop``: the strongest compression whose actual accuracy
  drop still respects the budget.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shrinker import compress, compress_to_target, register
from shrinker.benchmark import count_parameters, evaluate
from shrinker.registry import UnknownMethodError

GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def lat(r):
    """Latency curve: 10 ms uncompressed, 2.8 ms at ratio 0.9."""
    return 10 - 8 * r


def acc(r):
    """Accuracy curve: 95% uncompressed, -30 points at ratio 0.9 (drop = 0.3 r)."""
    return 0.95 - 0.3 * r


def tent(r):
    """Rises away from 0.5 with float-exact equal arms: 2.0 at 0.25 and 0.75."""
    return 1 + 4 * abs(r - 0.5)


def flaky_latency(r):
    """The same curve, except ratio 0.4 got a fluke 3.0 ms reading."""
    return 3.0 if r == 0.4 else lat(r)


def lat_fail(r):
    """Latency unmeasurable for the strongest compressions (r > 0.6)."""
    return None if r > 0.6 else lat(r)


def acc_fail(r):
    """Accuracy unmeasurable for the strongest compressions (r > 0.5)."""
    return None if r > 0.5 else acc(r)


def acc_with_spike(r):
    """The acc curve, except ratio 0.2 accidentally IMPROVES accuracy (drop < 0)."""
    return 1.0 if r == 0.2 else acc(r)


def plateau_latency(r):
    """Latency plateaus at 4.4 ms for every ratio >= 0.5."""
    return 4.4 if r >= 0.5 else 10 - 8 * r


class FakeMeasure:
    """Deterministic stand-in for ``benchmark.measure``.

    Metrics are functions of the knob stored on the model by the fake methods;
    a metric without a function reads as ``None`` ("unmeasurable"), exactly
    like the real ``measure`` reports it when inputs are missing.
    """

    def __init__(self, latency_ms=None, accuracy=None, flops=None, params=None, size_mb=None):
        self.fns = {
            "latency_ms": latency_ms,
            "accuracy": accuracy,
            "flops": flops,
            "params": params,
            "size_mb": size_mb,
        }
        self.calls = []  # (knob, metrics) for every measure() call, baseline first

    def __call__(self, model, example_input=None, val_loader=None, device="cpu", warmup=10, runs=50):
        knob = getattr(model, "knob", 0.0)
        metrics = {key: (fn(knob) if fn else None) for key, fn in self.fns.items()}
        metrics["nonzero_params"] = metrics["params"]
        self.calls.append((knob, metrics))
        return metrics

    def knobs_evaluated(self):
        """Knobs of all candidate evaluations (everything after the baseline)."""
        return [knob for knob, _ in self.calls[1:]]


class KnobModel(nn.Module):
    """Its accuracy degrades monotonically as ``strength`` shrinks.

    With ``strength=1`` it is a plain Linear model; as ``strength -> 0`` the
    logits blend toward a fixed class-1-favoring vector, so samples flip to
    class 1 in a deterministic order (smallest margin first).
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(20, 3)
        self.register_buffer("strength", torch.tensor(1.0))

    def forward(self, x):
        pull = torch.tensor([0.0, 10.0, 0.0])
        return self.strength * self.lin(x) + (1.0 - self.strength) * pull


@pytest.fixture
def fake_bench(monkeypatch):
    """Install a ``FakeMeasure`` configured per test; returns the instance."""

    def install(**metrics):
        fake = FakeMeasure(**metrics)
        monkeypatch.setattr("shrinker.benchmark.measure", fake)
        return fake

    return install


@pytest.fixture
def fake_methods(restore_registry):
    @register("knob_sweep")
    def apply(model, ratio=0.5):
        model.knob = ratio
        return model

    @register("sparsity_sweep")
    def apply(model, sparsity=0.5):
        model.knob = sparsity
        return model

    @register("both_knobs")
    def apply(model, ratio=0.5, sparsity=0.5):
        model.knob = ratio
        return model

    @register("one_shot")
    def apply(model):
        model.knob = 1.0  # a fixed amount of compression, no knob to tune
        return model

    @register("loader_sweep")
    def apply(model, ratio=0.5, loader=None):
        model.knob = ratio
        model.got_loader = loader
        return model

    @register("input_sweep")
    def apply(model, ratio=0.5, example_input=None):
        model.knob = ratio
        model.got_input = example_input
        return model

    @register("seq_pruner")
    def apply(model, ratio=0.5):
        """Structurally shrink Sequential(Linear, ReLU, Linear) to (1-ratio) hidden units."""
        lin1, lin2 = model[0], model[2]
        keep = max(1, int(lin1.out_features * (1 - ratio)))
        new1 = nn.Linear(lin1.in_features, keep)
        new1.weight.data.copy_(lin1.weight.data[:keep])
        new1.bias.data.copy_(lin1.bias.data[:keep])
        new2 = nn.Linear(keep, lin2.out_features)
        new2.weight.data.copy_(lin2.weight.data[:, :keep])
        new2.bias.data.copy_(lin2.bias.data)
        model[0], model[2] = new1, new2
        model.knob = ratio
        return model

    @register("strength_sweep")
    def apply(model, ratio=0.5):
        model.strength = torch.tensor(1.0 - ratio)
        model.knob = ratio
        return model


# ----------------------------------------------------------- validation


def test_unknown_target_rejected_before_any_measurement(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    with pytest.raises(ValueError, match="Unknown target 'speed'"):
        compress_to_target(nn.Linear(4, 2), "knob_sweep", "speed", 5.0)
    assert fake.calls == []


def test_negative_target_value_rejected(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    with pytest.raises(ValueError, match="non-negative"):
        compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", -1.0)
    assert fake.calls == []


def test_accuracy_drop_requires_val_loader(fake_bench, fake_methods):
    fake = fake_bench(accuracy=acc)
    with pytest.raises(ValueError, match="val_loader is required"):
        compress_to_target(nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.1)
    assert fake.calls == []


def test_unknown_method_fails_before_any_measurement(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    with pytest.raises(UnknownMethodError, match="nope_method"):
        compress_to_target(nn.Linear(4, 2), "nope_method", "latency_ms", 5.0)
    assert fake.calls == []


def test_empty_ratios_rejected(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    with pytest.raises(ValueError, match="ratios must not be empty"):
        compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0, ratios=[])


# ------------------ resource targets: mildest compression meeting the target


def test_latency_target_picks_mildest_compression_meeting_the_target(fake_bench, fake_methods):
    """The headline rule: among every ratio that reaches the target, the one
    with the LARGEST metric value (mildest compression, best accuracy) must
    win -- never the most over-compressed one (that would be min())."""
    fake = fake_bench(latency_ms=lat)
    original = nn.Linear(4, 2)
    model = compress_to_target(original, "knob_sweep", "latency_ms", 5.0, ratios=GRID, refine_steps=0)
    assert model.knob == 0.7  # 4.4 ms: the slowest model that still meets 5.0 ms
    measured = {knob: m["latency_ms"] for knob, m in fake.calls}
    assert measured[0.7] <= 5.0
    assert measured[0.6] > 5.0  # 0.6 missed the target, so 0.7 sits on the boundary
    assert not hasattr(original, "knob")  # the original model is never touched


@pytest.mark.parametrize(
    "target, metric, target_value",
    [
        ("size_mb", lambda r: 10 - 8 * r, 5.0),
        ("params", lambda r: 1000 - 800 * r, 500.0),
    ],
)
def test_resource_target_reads_its_own_metric(fake_bench, fake_methods, target, metric, target_value):
    fake = fake_bench(**{target: metric})
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", target, target_value, refine_steps=0)
    assert model.knob == 0.7
    measured = {knob: m for knob, m in fake.calls}
    assert measured[model.knob][target] <= target_value


def test_mflops_target_is_converted_to_raw_flops(fake_bench, fake_methods, capsys):
    fake = fake_bench(flops=lambda r: 2e6 - 1.6e6 * r)
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "Mflops", 1.0, refine_steps=0)
    # without the x1e6 conversion nothing would qualify and the fallback would
    # pick ratio 0.9 instead
    assert model.knob == 0.7
    out = capsys.readouterr().out
    assert "target_reached : True" in out
    assert "target_value: 1.0" in out  # the reported value stays in MFlops


def test_bisection_refines_toward_the_target(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    # default grid 0.1..0.9 + default refine_steps: the boundary (latency 5.0)
    # falls between grid points 0.6 and 0.7, so bisection must improve on 0.7
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0)
    assert 0.6 < model.knob < 0.7  # refined inside the bracketing interval
    assert lat(model.knob) <= 5.0  # still feasible
    assert lat(model.knob) > 4.4  # closer than the best plain grid point
    assert len(fake.knobs_evaluated()) > 9  # grid points plus bisection mids


def test_fluke_fast_measurement_does_not_fool_selection(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=flaky_latency)
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0, ratios=GRID, refine_steps=0)
    # even though ratio 0.4 measured a fluke 3.0 ms, the rule takes the
    # highest feasible value (0.7 at 4.4 ms), not the least-compressed
    # candidate or the fastest one
    assert model.knob == 0.7
    assert len(fake.knobs_evaluated()) == 9


def test_value_tie_prefers_milder_compression(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=plateau_latency)
    # grid 0.4 (6.8 ms) and 0.6 (4.4 ms) bracket the target; the bisection mid
    # 0.5 also measures 4.4 ms, tying with 0.6. Same speed, so the milder
    # compression (more accuracy headroom) must win -- not whichever candidate
    # happened to be evaluated first.
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0, ratios=[0.4, 0.6])
    assert model.knob == 0.5
    assert len(fake.knobs_evaluated()) == 6  # 0.4, 0.6 + four bisection mids


# ------------- max_accuracy_drop: strongest compression within the budget


def test_accuracy_drop_picks_strongest_compression_within_budget(fake_bench, fake_methods):
    fake = fake_bench(accuracy=acc)
    model = compress_to_target(
        nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.1, val_loader=object(), refine_steps=0
    )
    # drops: 0.1 -> 0.03, 0.2 -> 0.06, 0.3 -> 0.09, 0.4 -> 0.12 (over budget);
    # the largest drop within budget (most compression) must win, not min()
    assert model.knob == 0.3
    assert len(fake.knobs_evaluated()) == 9


def test_accuracy_drop_bisects_toward_the_budget(fake_bench, fake_methods):
    fake = fake_bench(accuracy=acc)
    # boundary (drop 0.1) falls between grid points 0.3 and 0.4
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.1, val_loader=object())
    assert 0.3 < model.knob < 0.4  # refined inside the bracketing interval
    drop = 0.95 - acc(model.knob)
    assert 0.09 < drop <= 0.1  # closer to the budget than the best grid point
    assert len(fake.knobs_evaluated()) > 9


def test_accuracy_improvement_does_not_win(fake_bench, fake_methods):
    fake = fake_bench(accuracy=acc_with_spike)
    model = compress_to_target(
        nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.1, val_loader=object(), refine_steps=0
    )
    # ratio 0.2 has negative drop (accuracy improved); it must not be picked
    # over ratio 0.3, whose drop sits closer to the budget
    assert model.knob == 0.3


# ------------------------------------------------------------- fallbacks


def test_unreachable_latency_returns_closest_candidate(fake_bench, fake_methods, capsys):
    fake = fake_bench(latency_ms=lat)
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 0.5, ratios=GRID, refine_steps=0)
    # even ratio 0.9 gives 2.8 ms > 0.5: best effort is the closest candidate
    assert model.knob == 0.9
    assert "target_reached : False" in capsys.readouterr().out


def test_unreachable_budget_returns_smallest_drop(fake_bench, fake_methods, capsys):
    fake = fake_bench(accuracy=acc)
    model = compress_to_target(
        nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.01, val_loader=object(), refine_steps=0
    )
    # even ratio 0.1 drops 0.03 > 0.01: best effort is the smallest drop
    assert model.knob == 0.1
    assert "target_reached : False" in capsys.readouterr().out


def test_distance_tie_prefers_milder_knob(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=tent)
    model = compress_to_target(
        nn.Linear(4, 2), "knob_sweep", "latency_ms", 2.5, ratios=[0.25, 0.5, 0.75], refine_steps=0
    )
    # 0.25 and 0.75 are equidistant from the unreachable target; the milder
    # compression (smaller knob) must win the tie
    assert model.knob == 0.25
    assert fake.knobs_evaluated() == [0.25, 0.5, 0.75]


# --------------------------------------------- knob detection & one-shot


def test_sparsity_knob_is_detected(fake_bench, fake_methods, capsys):
    fake = fake_bench(latency_ms=lat)
    model = compress_to_target(nn.Linear(4, 2), "sparsity_sweep", "latency_ms", 5.0, ratios=GRID, refine_steps=0)
    assert model.knob == 0.7
    out = capsys.readouterr().out
    assert "knob: sparsity" in out
    assert "selected_sparsity: 0.7" in out


def test_ratio_takes_priority_over_sparsity(fake_bench, fake_methods, capsys):
    fake = fake_bench(latency_ms=lat)
    model = compress_to_target(nn.Linear(4, 2), "both_knobs", "latency_ms", 5.0, ratios=GRID, refine_steps=0)
    assert model.knob == 0.7  # the ratio knob was swept, not sparsity
    assert "knob: ratio" in capsys.readouterr().out


def test_one_shot_method_meeting_target_is_returned(fake_bench, fake_methods, capsys):
    fake = fake_bench(latency_ms=lat)  # one_shot stores knob=1.0 -> 2.0 ms
    model = compress_to_target(nn.Linear(4, 2), "one_shot", "latency_ms", 5.0)
    assert isinstance(model, nn.Module)
    assert fake.knobs_evaluated() == [1.0]  # evaluated once, nothing to sweep
    out = capsys.readouterr().out
    assert "knob: none, one-shot method" in out
    assert "target_reached : True" in out
    assert "no tunable knob" not in out


def test_one_shot_method_missing_target_returns_with_warning(fake_bench, fake_methods, capsys):
    fake = fake_bench(latency_ms=lat)  # 2.0 ms > 1.0 ms target
    model = compress_to_target(nn.Linear(4, 2), "one_shot", "latency_ms", 1.0)
    assert isinstance(model, nn.Module)  # returned anyway, best effort
    out = capsys.readouterr().out
    assert "target_reached : False" in out
    assert "no tunable knob" in out


# ------------------------------- unmeasurable candidates & bad baselines


def test_unmeasurable_candidates_are_skipped(fake_bench, fake_methods, capsys):
    fake = fake_bench(latency_ms=lat_fail)
    model = compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0, ratios=GRID, refine_steps=0)
    # ratios 0.7+ could not be measured; nothing feasible remains, so the
    # closest measured candidate (0.6 at 5.2 ms) is returned instead
    assert model.knob == 0.6
    assert len(fake.knobs_evaluated()) == 9  # the broken ones were still attempted
    assert "target_reached : False" in capsys.readouterr().out


def test_unmeasurable_accuracy_candidates_are_skipped(fake_bench, fake_methods, capsys):
    fake = fake_bench(accuracy=acc_fail)
    model = compress_to_target(
        nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.1, val_loader=object(), refine_steps=0
    )
    # candidates above ratio 0.5 have no accuracy; selection ignores them and
    # picks the largest drop within budget among the measurable ones
    assert model.knob == 0.3
    assert "target_reached : True" in capsys.readouterr().out


def test_baseline_accuracy_none_raises(fake_bench, fake_methods):
    fake = fake_bench()  # accuracy never measurable
    with pytest.raises(RuntimeError, match="baseline accuracy"):
        compress_to_target(nn.Linear(4, 2), "knob_sweep", "max_accuracy_drop", 0.1, val_loader=object())


def test_baseline_latency_none_raises(fake_bench, fake_methods):
    fake = fake_bench()  # latency never measurable
    with pytest.raises(RuntimeError, match="baseline 'latency_ms'"):
        compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0)


# ------------------------------------------------------ sweep bookkeeping


def test_duplicate_grid_points_evaluated_once(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    compress_to_target(nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0, ratios=[0.3, 0.3, 0.3, 0.5], refine_steps=0)
    assert fake.knobs_evaluated() == [0.3, 0.5]


def test_refine_steps_zero_disables_bisection(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    model = compress_to_target(
        nn.Linear(4, 2), "knob_sweep", "latency_ms", 5.0, ratios=[0.4, 0.6, 0.8], refine_steps=0
    )
    # the (0.6, 0.8) interval crosses the target but must not be refined
    assert fake.knobs_evaluated() == [0.4, 0.6, 0.8]
    assert model.knob == 0.8  # largest feasible value (3.6 ms)


def test_extra_kwargs_reach_the_method(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    loader = {"fake": "loader"}
    model = compress_to_target(
        nn.Linear(4, 2), "loader_sweep", "latency_ms", 5.0, loader=loader, ratios=[0.7], refine_steps=0
    )
    assert model.got_loader == {"fake": "loader"}
    assert model.knob == 0.7


def test_example_input_reaches_the_method(fake_bench, fake_methods):
    fake = fake_bench(latency_ms=lat)
    x = torch.randn(2, 4)
    model = compress_to_target(
        nn.Linear(4, 2), "input_sweep", "latency_ms", 5.0, example_input=x, ratios=[0.7], refine_steps=0
    )
    assert torch.equal(model.got_input, x)


# --------------------------------- end-to-end with the real benchmark.measure


def test_original_model_is_not_modified(fake_methods):
    original = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
    snapshot = {k: v.clone() for k, v in original.state_dict().items()}
    x = torch.randn(4, 64)

    result = compress_to_target(
        original, "seq_pruner", "params", 15000, example_input=x, ratios=[0.2, 0.5, 0.8], refine_steps=0
    )

    assert result is not original
    assert count_parameters(original) == 75 * 256 + 10
    for key, value in original.state_dict().items():
        assert torch.equal(value, snapshot[key])
    # ratio 0.2 keeps 204 units -> 15310 params (over budget); ratio 0.5 keeps
    # 128 units -> 9610 params, the mildest compression meeting 15000
    assert result.knob == 0.5
    assert count_parameters(result) == 75 * 128 + 10


def test_params_target_refines_past_the_grid(fake_methods):
    original = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
    x = torch.randn(4, 64)

    result = compress_to_target(original, "seq_pruner", "params", 10000, example_input=x)

    n = count_parameters(result)
    assert 9900 <= n <= 10000  # bisection found ~132 hidden units, not a grid point
    assert n > 75 * 128 + 10  # strictly closer than the best plain grid ratio (0.5)
    assert 0.4 < result.knob < 0.5
    assert result(x).shape == (4, 10)


def test_accuracy_drop_end_to_end_with_real_loader(fake_methods, capsys):
    model = KnobModel()
    x = torch.randn(24, 20)
    with torch.no_grad():
        y = model(x).argmax(dim=1)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    assert evaluate(model, loader) == pytest.approx(1.0)  # labels are its own predictions

    # sanity: the budget is actually reachable at strong compression
    strong = compress(model, "strength_sweep", ratio=0.9)
    assert 1.0 - evaluate(strong, loader) > 0.5

    result = compress_to_target(model, "strength_sweep", "max_accuracy_drop", 0.5, val_loader=loader)

    drop = 1.0 - evaluate(result, loader)
    assert drop <= 0.5  # the accuracy budget is respected
    assert result.knob >= 0.1  # at least the mildest sweep point was needed
    assert "target_reached : True" in capsys.readouterr().out
