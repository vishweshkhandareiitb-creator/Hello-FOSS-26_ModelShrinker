# Model Shrinker

**Hello FOSS · Web & Coding Club, IIT Bombay**

Take a PyTorch model you have **already trained** and get back a **smaller, faster copy** of it, while keeping as much accuracy as possible.

Works with MLPs and CNNs (`nn.Linear`, `nn.Conv2d`, `nn.BatchNorm`). RNNs and Transformers are not supported yet. Everything runs on your own machine.

## How it works

```
 your trained model
        │
        ▼
 1. compress()   pick a method, e.g. "l1_pruning"  →  you get a smaller COPY
        │                                              (your original is never changed)
        ▼
 2. compare()    see the difference: size, speed, accuracy
        │
        ▼
 3. finetune()   train the small model a little to win back lost accuracy
        │
        ▼
 4. save it and use the small model
```

A **method** is one way of making a model smaller, for example removing the least important filters (pruning) or storing weights with fewer bits (quantization). Each method lives in its own file in `shrinker/methods/`.

If you don't know which strength to pick, skip the guesswork: [`compress_to_target()`](#compress-to-a-target) takes a goal instead — a latency, size or accuracy budget — and finds the right strength for you.

## Install

```bash
pip install git+https://github.com/wncc/Hello-FOSS-26_ModelShrinker
```

## Example

```python
import torch
import shrinker
from model import MyNet                      # your own model class

model = MyNet()
model.load_state_dict(torch.load("model.pth"))

example = torch.randn(1, 3, 32, 32)          # one input of the right shape

small = shrinker.compress(model, "l1_pruning", ratio=0.3, example_input=example)
shrinker.compare(model, small, example_input=example, val_loader=val_loader)

shrinker.finetune(small, train_loader, epochs=5)
torch.save(small, "model_small.pt")
```

`compare()` prints a table like this:

```
              original  compressed       change
-----------------------------------------------
params         666,858     167,802       -74.8%
size (MB)        2.681       0.683       -74.5%
latency (ms)    15.693       5.919  2.65x speed
MFLOPs         3804.56      979.53       -74.3%
accuracy       100.00%     100.00%    +0.00 pts
```

Speed ("latency") is measured by actually running the model, not estimated.

**Good to know**
- `ratio=0.3` means "remove about 30%". Each method has its own options.
- Save the whole model with `torch.save(small, path)` and load it with `torch.load(path, weights_only=False)`. A shrunk model no longer fits your original class, so `load_state_dict` into `MyNet()` won't work.
- `example_input` is optional. If given, the library checks the small model still gives the same output shape.
- Layers a method can't handle safely are left unchanged instead of being broken.

## Available methods

| Method | What it does | Paper |
|---|---|---|
| `l1_pruning` | Removes the filters/neurons with the smallest L1 norm | [Li et al. 2017](https://arxiv.org/abs/1608.08710) |
| `fpgm` | Removes the most redundant filters, those closest to the geometric median | [He et al. 2019](https://arxiv.org/abs/1811.00250) |
| `qat_quantization` | Quantization-aware training, gives an INT8 model for CPU | [Jacob et al. 2018](https://arxiv.org/abs/1712.05877) |
| `dfq_quantization` | Data-free INT8 quantization with weight equalization and bias correction | [Nagel et al. 2019](https://arxiv.org/abs/1906.04721) |
| `knowledge_distillation` | Trains the model to match a frozen copy of itself, then zeroes the weakest weights | [Hinton et al. 2015](https://arxiv.org/abs/1503.02531) |
| `gradient_based` | Removes filters whose removal hurts the loss least, estimated with gradients | [Molchanov et al. 2019](https://arxiv.org/abs/1906.10771) |
| HRank | Removes filters whose feature maps carry little information | [Lin et al. 2020](https://arxiv.org/abs/2002.10179) |
| Tucker / SVD decomposition | Splits big layers into smaller ones | [Kim et al. 2016](https://arxiv.org/abs/1511.06530) |

`shrinker.describe_methods()` always lists every method that's installed.

## Functions

| Function | What it does |
|---|---|
| `compress(model, method, example_input=None, **options)` | Returns a smaller **copy** of your model using the chosen method. The original is never changed. If `example_input` is given, it checks the output shape is unchanged. |
| `compress_to_target(model, method, target, target_value, val_loader=None, example_input=None, **options)` | Sweeps compression strengths and returns the model that meets your goal: the mildest compression reaching a `latency_ms`/`size_mb`/`Mflops`/`params` target, or the strongest compression within a `max_accuracy_drop` budget. Prints a report and falls back to the closest candidate if the target is unreachable. |
| `compare(original, compressed, example_input, val_loader=None, device="cpu")` | Prints params, size, latency, MFLOPs and accuracy for both models side by side, and returns the numbers as a dict. Accuracy needs `val_loader`. |
| `finetune(model, train_loader, epochs=5, lr=1e-3, device="cpu", optimizer="adam")` | Retrains the compressed model in place (cross-entropy loss) to win back accuracy. `optimizer` can be `"adam"` or `"sgd"`. |
| `analyze_sensitivity(model, loader, ratios=(0.1, 0.2, ...), score_fn=None, max_batches=None)` | Measures validation accuracy after pruning each prunable layer independently at different ratios, then reports the baseline and per-layer accuracy drop. Useful for finding which layers are most sensitive before compressing the model. |
| `list_methods()` | Returns the names of all installed methods. |
| `describe_methods()` | Returns a table of every installed method with its type and a one-line description. |


## Compress to a target

With `compress()` you pick the compression strength yourself (`ratio=0.3`) and check afterwards what it bought you. Often you know the goal instead — "it must run in under 5 ms" or "it may lose at most 2% accuracy" — and want the right strength found for you. That is what `compress_to_target()` does:

```python
small = shrinker.compress_to_target(
    model, "l1_pruning",
    target="latency_ms", target_value=5.0,       # at most 5 ms per forward pass
    example_input=example,
)

small = shrinker.compress_to_target(
    model, "l1_pruning",
    target="max_accuracy_drop", target_value=0.02,   # lose at most 2 accuracy points
    val_loader=val_loader, example_input=example,
)
```

It measures the model on a sweep of compression strengths, then returns the model sitting exactly on your goal:

- For `latency_ms`, `size_mb`, `Mflops` and `params` it picks the **mildest compression that meets the target** — a smaller/faster model that satisfies it while keeping as much accuracy as possible.
- For `max_accuracy_drop` it picks the **strongest compression that stays within the budget** — the smallest model you can afford.
- If the target can't be reached at all, it returns the closest attempt instead of failing.

It prints a report of the sweep:

```
target : latency_ms
target_value: 5.0
method: l1_pruning (knob: ratio)
target_reached : True
selected_ratio: 0.42
```

followed by the measured metrics of the original and the chosen model.

**Targets**

| Target | Meaning | Requires |
|---|---|---|
| `latency_ms` | forward pass takes at most this many milliseconds | `example_input` |
| `size_mb` | saved model is at most this many megabytes | — |
| `Mflops` | at most this many million FLOPs per forward pass | `example_input` |
| `params` | at most this many parameters | — |
| `max_accuracy_drop` | loses at most this much accuracy vs. the original (`0.02` = 2 points) | `val_loader` |


Try the full flow yourself (no download needed, under a minute):

```bash
python examples/demo.py --synthetic
```

## Contributing

Pick an open [issue](https://github.com/bandiarham07-bot/Model-Shrinker/issues) and comment that you are working on it. You can also open a new issue to suggest a method or report a bug.

1. Fork the repository and install it: `pip install -e ".[dev]"`
2. Create a branch for your issue (`git checkout -b fix/short-name`)
3. Make your changes. A new method is **one file** in `shrinker/methods/` named after the method, with one `@register` function (use [`l1_pruning.py`](shrinker/methods/l1_pruning.py) as a template)
4. Run `pytest`. Every method is automatically tested on six small models
5. Push the branch and open a pull request that mentions the issue (e.g. `Closes #12`)

For questions, open an issue or contact the maintainers.

## License

MIT

---

Created with ❤️ by WnCC
