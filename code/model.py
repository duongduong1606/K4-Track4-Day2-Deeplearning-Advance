"""Model construction, freezing, optimizer groups, and complexity estimates."""
from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _classifier(model: nn.Module) -> nn.Module:
    if not hasattr(model, "get_classifier"):
        raise TypeError("Model phải cung cấp get_classifier() như model của timm")
    head = model.get_classifier()
    if not isinstance(head, nn.Module):
        raise TypeError("Không xác định được classification head")
    return head


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Build a timm classifier using scratch, frozen, or fine-tuning initialization."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Cần cài timm: pip install timm") from exc

    init = init.lower()
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init phải là scratch, frozen hoặc finetune")
    use_pretrained = False if init == "scratch" else bool(pretrained)
    model = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    model.deepweeds_name = name
    model.deepweeds_init = init
    model.deepweeds_pretrained = use_pretrained
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    model.deepweeds_weight_tag = (cfg.get("tag") or cfg.get("hf_hub_id")
                                  or cfg.get("architecture") or name)
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the classification head."""
    head = _classifier(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    model.deepweeds_frozen = True
    model.deepweeds_head = head
    model.eval()
    head.train()


def set_train_mode(model: nn.Module) -> None:
    """Enter training mode while keeping a frozen backbone (including BN) in eval mode."""
    if getattr(model, "deepweeds_frozen", False):
        model.eval()
        _classifier(model).train()
    else:
        model.train()


def _parameter_ids(parameters: Iterable[nn.Parameter]) -> set[int]:
    return {id(parameter) for parameter in parameters}


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float):
    """Create backbone-decay, backbone-no-decay, and head parameter groups."""
    if lr_backbone <= 0 or lr_head <= 0 or weight_decay < 0:
        raise ValueError("Learning rate phải dương và weight_decay không âm")
    head_ids = _parameter_ids(_classifier(model).parameters())
    backbone_decay, backbone_no_decay, head_params = [], [], []

    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            head_params.append(parameter)
        elif parameter.ndim <= 1:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay,
                       "group_name": "backbone_decay"})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0,
                       "group_name": "backbone_no_decay"})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay,
                       "group_name": "head"})
    if not groups:
        raise ValueError("Model không có tham số trainable")
    return groups


def count_params(model: nn.Module) -> float:
    """Return total parameter count in millions, including frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000.0


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Return MACs for one image in billions using THOP."""
    if img_size <= 0:
        raise ValueError("img_size phải dương")
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Cần cài thop để đếm GMAC: pip install thop") from exc

    # THOP temporarily registers ``total_ops``/``total_params`` buffers on modules.
    # Profile a disposable copy so those implementation details can never leak into
    # a training checkpoint. ``no_grad`` also avoids creating inference tensors that
    # later reject in-place updates during ``load_state_dict``.
    profiled_model = copy.deepcopy(model).eval()
    parameter = next(profiled_model.parameters(), None)
    original_device = parameter.device if parameter is not None else torch.device("cpu")
    dummy = torch.zeros(1, 3, img_size, img_size, device=original_device)
    with torch.no_grad():
        macs, _ = profile(profiled_model, inputs=(dummy,), verbose=False)
    return float(macs / 1_000_000_000.0)
