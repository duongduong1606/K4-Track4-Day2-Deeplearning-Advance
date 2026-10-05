"""Pre-flight checks required before real DeepWeeds experiments."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from . import dataset, model as model_utils
    from .train import Config, set_seed
except ImportError:
    import dataset
    import model as model_utils
    from train import Config, set_seed


def _save_augmented_grid(images, labels, filenames, path: Path) -> None:
    count = min(12, len(images))
    mean = torch.tensor(dataset.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(dataset.IMAGENET_STD).view(3, 1, 1)
    restored = (images[:count].cpu() * std + mean).clamp(0, 1)
    columns = 4
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            axis.imshow(restored[index].permute(1, 2, 0).numpy())
            axis.set_title(f"y={int(labels[index])} | {filenames[index]}", fontsize=8)
    fig.suptitle("Ảnh sau augmentation (đã giải chuẩn hóa để hiển thị)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pipeline_checks(cfg: Config, steps: int = 200, small_batch: int = 16,
                        output_dir: str | Path = "runs/pipeline_check") -> dict:
    """Check initial CE, augmented images, and ability to overfit one fixed batch."""
    if steps <= 0 or small_batch <= 1:
        raise ValueError("steps phải dương và small_batch phải > 1")
    set_seed(cfg.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    subset = train_df.iloc[:small_batch].copy()
    loader = dataset.make_loader(
        subset, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
        small_batch, train=False, sampler=None, num_workers=0,
    )
    images, labels, filenames = next(iter(loader))
    _save_augmented_grid(images, labels, filenames, output / "augmented_samples.png")

    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    network = model_utils.build_model(cfg.backbone, pretrained=True, num_classes=dataset.NUM_CLASSES,
                                      drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    images, labels = images.to(device), labels.to(device)
    network.train()
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(network(images), labels))

    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=1e-3,
                                  weight_decay=0.0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(network(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, steps + 1), losses)
    axis.set(title="Overfit one fixed batch", xlabel="Optimization step", ylabel="CE loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "overfit_one_batch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "config": asdict(cfg),
        "expected_uniform_ce": math.log(dataset.NUM_CLASSES),
        "initial_ce": initial_loss,
        "final_ce": losses[-1],
        "overfit_steps": steps,
        "small_batch": small_batch,
        "split": split_report,
        "augmented_samples": str(output / "augmented_samples.png"),
        "overfit_curve": str(output / "overfit_one_batch.png"),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
