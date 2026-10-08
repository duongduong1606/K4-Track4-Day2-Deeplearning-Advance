"""Loss functions and batch-level Mixup/CutMix for DeepWeeds."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", **kw):
    """Construct CE, label-smoothed CE, focal, or weighted CE."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(float(kw.get("gamma", 2.0)), kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted cần tensor weight")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
    raise ValueError(f"Loss không hỗ trợ: {kind!r}")


class LabelSmoothingCE(nn.Module):
    """Cross entropy using PyTorch's uniform label smoothing implementation."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải nằm trong [0, 1)")
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Multiclass focal loss; gamma=0 is exactly weighted/unweighted CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma phải không âm")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.ndim != 1 or (alpha_tensor < 0).any():
                raise ValueError("alpha phải là vector trọng số không âm")
            self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits, target):
        if logits.ndim != 2 or target.ndim != 1 or len(logits) != len(target):
            raise ValueError("logits phải [N,C] và target phải [N]")
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError("Độ dài alpha phải bằng số lớp")
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Compute inverse-frequency or effective-number class weights, mean-normalized to one."""
    counts_tensor = torch.as_tensor(counts, dtype=torch.float64)
    if counts_tensor.ndim != 1 or counts_tensor.numel() == 0:
        raise ValueError("counts phải là vector một chiều")
    if (counts_tensor <= 0).any():
        raise ValueError("Mọi lớp phải có ít nhất một ảnh train")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta phải nằm trong [0, 1)")
    if beta == 0.0:
        weights = counts_tensor.reciprocal()
    else:
        # expm1/log form is more stable when beta is close to one.
        denominator = -torch.expm1(counts_tensor * math.log(beta))
        weights = (1.0 - beta) / denominator
    weights = weights * (weights.numel() / weights.sum())
    return weights.to(dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Apply Mixup or CutMix and return images plus ``(y_a, y_b, lambda)``."""
    if alpha <= 0:
        raise ValueError("alpha phải dương")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode phải là mixup hoặc cutmix")
    if x.ndim != 4 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x phải [N,C,H,W] và y phải [N]")

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[permutation]
    if mode == "mixup":
        mixed = x.mul(lam).add(x[permutation], alpha=1.0 - lam)
        return mixed, (y_a, y_b, lam)

    height, width = x.shape[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = int(torch.randint(0, width, (1,), device=x.device).item())
    center_y = int(torch.randint(0, height, (1,), device=x.device).item())
    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y_a, y_b, actual_lam)


def mixed_loss(criterion, logits, targets):
    """Compute lambda-weighted loss for Mixup/CutMix targets."""
    y_a, y_b, lam = targets
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lambda phải nằm trong [0, 1]")
    return float(lam) * criterion(logits, y_a) + (1.0 - float(lam)) * criterion(logits, y_b)
