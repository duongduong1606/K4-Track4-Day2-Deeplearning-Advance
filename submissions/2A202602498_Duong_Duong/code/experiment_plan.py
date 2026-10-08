"""Controlled experiment configurations; importing this file does not start training."""
from __future__ import annotations

try:
    from .train import Config
except ImportError:
    from train import Config


BACKBONES = [
    Config(exp_id="B01", backbone="resnet50"),
    Config(exp_id="B02", backbone="resnext50_32x4d"),
    Config(exp_id="B03", backbone="convnext_tiny"),
    Config(exp_id="B04", backbone="deit_small_patch16_224"),
    Config(exp_id="B05", backbone="swin_tiny_patch4_window7_224"),
    Config(exp_id="B06", backbone="efficientnet_b0"),
]

# Every T01-T09 differs from T00 in exactly one declared factor.
TRAINING = [
    Config(exp_id="T00", backbone="resnet50"),
    Config(exp_id="T01", backbone="resnet50", init="scratch"),
    Config(exp_id="T02", backbone="resnet50", init="frozen"),
    Config(exp_id="T03", backbone="resnet50", aug="color"),
    Config(exp_id="T04", backbone="resnet50", aug="randaug"),
    Config(exp_id="T05", backbone="resnet50", loss="ls", label_smoothing=0.1),
    Config(exp_id="T06", backbone="resnet50", loss="focal", focal_gamma=2.0),
    Config(exp_id="T07", backbone="resnet50", loss="ce_weighted", class_weight_beta=0.0),
    Config(exp_id="T08", backbone="resnet50", mix="cutmix", mix_alpha=1.0),
    Config(exp_id="T09", backbone="resnet50", ema_decay=0.999),
]

# Intentional combination, run only after its components win clearly on validation.
COMBINED_CANDIDATE = Config(
    exp_id="T20", backbone="resnet50", aug="color", loss="ls",
    label_smoothing=0.1, mix="cutmix", mix_alpha=1.0, ema_decay=0.999,
)

FINAL_SEEDS = (42, 123, 2026)


def final_configs(best: Config, exp_id: str = "F01", enable_test: bool = False):
    """Clone a validation-selected config across predeclared final seeds."""
    values = vars(best).copy()
    return [
        Config(**{**values, "exp_id": exp_id, "seed": seed,
                  "save_test_predictions": enable_test})
        for seed in FINAL_SEEDS
    ]
