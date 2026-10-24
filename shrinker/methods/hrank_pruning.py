from __future__ import annotations

import copy
import warnings
from typing import Any

import torch
import torch.nn as nn

from shrinker.registry import register

_TRACKED = (nn.Conv2d, nn.BatchNorm2d, nn.Linear)


@register("hrank_pruning")
def hrank_pruning(
    model: nn.Module,
    example_input: Any = None,
    amount: float = 0.3,
) -> nn.Module:
    """Prune Conv2d filters with the lowest feature-map rank (HRank, Lin et al., CVPR 2020).

    Pruned channels are propagated to the following Conv2d / BatchNorm2d / Linear layers.
    Supports sequential CNNs (no residual/branching); grouped convs are left unpruned.
    The input model is not modified.
    """
    if example_input is None:
        raise ValueError("hrank_pruning requires an example_input to trace feature maps.")
    if not (0.0 < amount < 1.0):
        raise ValueError("Pruning amount must be between 0 and 1 exclusive.")

    original_model = model
    model = copy.deepcopy(model)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    args = example_input if isinstance(example_input, (tuple, list)) else (example_input,)
    args = tuple(x.to(device) for x in args)

    # Pass 1: a single forward pass records layer execution order and HRank scores.
    order: list[nn.Module] = []
    seen: set[nn.Module] = set()
    scores: dict[nn.Module, torch.Tensor] = {}

    def hook(module, _inputs, output):
        if module in seen:
            return
        seen.add(module)
        order.append(module)
        if isinstance(module, nn.Conv2d) and module.groups == 1:
            scores[module] = _mean_rank(output)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, _TRACKED)]
    try:
        with torch.no_grad():
            model(*args)
    finally:
        for h in handles:
            h.remove()

    # Pass 2: shrink layers in execution order, carrying kept channels forward.
    keep: torch.Tensor | None = None  # kept channel indices of the tensor currently flowing
    channels: int | None = None       # original channel count of that tensor

    with torch.no_grad():
        try:
            for layer in order:
                if isinstance(layer, nn.Conv2d):
                    orig_out = layer.out_channels
                    if keep is not None and (layer.groups != 1 or layer.in_channels != channels):
                        raise NotImplementedError(
                            "hrank_pruning only supports sequential CNNs without grouped convs "
                            "or branching after a pruned layer."
                        )
                    out_keep = _select_filters(scores.get(layer), amount)
                    _shrink_conv(layer, out_keep, keep)
                    keep, channels = out_keep, orig_out

                elif isinstance(layer, nn.BatchNorm2d):
                    if keep is not None:
                        if layer.num_features != channels:
                            raise NotImplementedError("Unsupported topology around BatchNorm2d.")
                        _shrink_bn(layer, keep)

                else:  # nn.Linear
                    if keep is not None:
                        fan, rem = divmod(layer.in_features, channels)
                        if rem:
                            raise ValueError("Linear in_features is not a multiple of the channel count.")
                        cols = (keep[:, None] * fan + torch.arange(fan, device=keep.device)).flatten()
                        layer.weight = _param(layer.weight, layer.weight.detach()[:, cols])
                        layer.in_features = cols.numel()
                    keep = channels = None
                    
        except (NotImplementedError, ValueError) as e:
            # Gracefully fallback if the model topology is unsupported
            warnings.warn(f"hrank_pruning aborted: {e} Returning original model.")
            return copy.deepcopy(original_model)

    model.train(was_training)
    return model


def _mean_rank(fmap: torch.Tensor) -> torch.Tensor:
    """Average matrix rank of each channel's feature maps over the batch -> (C,)."""
    fmap = fmap.detach()
    if fmap.dim() == 3:
        fmap = fmap.unsqueeze(0)
    ranks = torch.linalg.matrix_rank(fmap.float(), atol=1e-5)  # (N, C), batched SVD
    return ranks.float().mean(dim=0)


def _select_filters(score: torch.Tensor | None, amount: float) -> torch.Tensor | None:
    """Indices (sorted) of the highest-rank filters to keep, or None if nothing is pruned."""
    if score is None:
        return None
    n = score.numel()
    n_prune = int(n * amount)
    if n_prune == 0:
        return None
    return torch.topk(score, n - n_prune).indices.sort().values


def _param(old: nn.Parameter, data: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(data, requires_grad=old.requires_grad)


def _shrink_conv(conv: nn.Conv2d, out_keep, in_keep) -> None:
    if out_keep is None and in_keep is None:
        return
    w = conv.weight.detach()
    b = None if conv.bias is None else conv.bias.detach()
    if out_keep is not None:
        w = w[out_keep]
        b = None if b is None else b[out_keep]
    if in_keep is not None:
        w = w[:, in_keep]
    conv.weight = _param(conv.weight, w)
    if b is not None:
        conv.bias = _param(conv.bias, b)
    conv.out_channels, conv.in_channels = w.shape[0], w.shape[1]


def _shrink_bn(bn: nn.BatchNorm2d, keep: torch.Tensor) -> None:
    if bn.weight is not None:
        bn.weight = _param(bn.weight, bn.weight.detach()[keep])
        bn.bias = _param(bn.bias, bn.bias.detach()[keep])
    if bn.running_mean is not None:
        bn.running_mean = bn.running_mean[keep]
        bn.running_var = bn.running_var[keep]
    bn.num_features = keep.numel()