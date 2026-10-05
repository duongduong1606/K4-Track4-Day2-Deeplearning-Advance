"""Inference utilities: TTA, ensembling, calibration, and Conv-BN fusion."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Run deterministic inference and preserve loader filename order."""
    device = torch.device(device)
    model.eval()
    names, labels, logits = [], [], []
    amp_enabled = bool(amp and device.type == "cuda")
    transform_view = view if view is not None else view_identity
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = transform_view(images.to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
            logits.append(output.float().cpu())
            labels.append(torch.as_tensor(target).cpu())
            names.extend(str(name) for name in filenames)
    if not logits:
        raise ValueError("Loader rỗng")
    return names, torch.cat(labels).numpy(), torch.cat(logits).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    """Horizontally flip an NCHW batch."""
    if x.ndim != 4:
        raise ValueError("Batch phải có dạng NCHW")
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Return top-left, top-right, bottom-left, bottom-right, and center crops."""
    if x.ndim != 4 or crop <= 0:
        raise ValueError("x phải là NCHW và crop phải dương")
    height, width = x.shape[-2:]
    if crop > height or crop > width:
        raise ValueError(f"crop={crop} lớn hơn ảnh {height}x{width}")
    top, left = 0, 0
    bottom, right = height - crop, width - crop
    center_y, center_x = (height - crop) // 2, (width - crop) // 2
    offsets = [(top, left), (top, right), (bottom, left), (bottom, right), (center_y, center_x)]
    return [x[..., y:y + crop, x0:x0 + crop] for y, x0 in offsets]


def views_multiscale(x, sizes):
    """Resize an NCHW batch to each requested square size."""
    if x.ndim != 4:
        raise ValueError("Batch phải có dạng NCHW")
    sizes = [int(size) for size in sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes phải chứa ít nhất một kích thước dương")
    return [F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                          antialias=True) for size in sizes]


def _stack_arrays(items):
    if not items:
        raise ValueError("Danh sách dự đoán rỗng")
    arrays = [np.asarray(item, dtype=np.float64) for item in items]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Mọi dự đoán phải có cùng shape")
    if arrays[0].ndim != 2:
        raise ValueError("Mỗi dự đoán phải có dạng [N,C]")
    return np.stack(arrays, axis=0)


def _softmax_numpy(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Aggregate TTA views in probability or logit space."""
    stacked = _stack_arrays(logits_per_view)
    if space == "prob":
        probs = _softmax_numpy(stacked).mean(axis=0)
    elif space == "logit":
        probs = _softmax_numpy(stacked.mean(axis=0))
    else:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    return probs / probs.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Average already-normalized probabilities from aligned models."""
    stacked = _stack_arrays(list_of_probs)
    if not np.isfinite(stacked).all() or (stacked < 0).any():
        raise ValueError("Xác suất phải hữu hạn và không âm")
    sums = stacked.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError("Mỗi hàng xác suất phải cộng bằng 1")
    result = stacked.mean(axis=0)
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Fit one positive temperature on validation NLL using LBFGS."""
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("val_logits phải [N,C] và val_labels phải [N]")
    if len(labels) == 0:
        raise ValueError("Validation rỗng")

    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp().item())


def apply_temperature(logits, T: float):
    """Apply temperature scaling and return normalized NumPy probabilities."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T phải hữu hạn và dương")
    return _softmax_numpy(np.asarray(logits, dtype=np.float64) / float(T))


def fuse_conv_bn(model):
    """Return a copied eval model with adjacent Conv2d/BatchNorm2d pairs fused."""
    fused_model = copy.deepcopy(model).eval()

    def recurse(module):
        names = list(module._modules.keys())
        index = 0
        while index < len(names) - 1:
            first_name, second_name = names[index], names[index + 1]
            first, second = module._modules[first_name], module._modules[second_name]
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d):
                module._modules[first_name] = fuse_conv_bn_eval(first, second)
                module._modules[second_name] = nn.Identity()
                index += 2
            else:
                index += 1
        for child in module.children():
            recurse(child)

    recurse(fused_model)
    return fused_model
