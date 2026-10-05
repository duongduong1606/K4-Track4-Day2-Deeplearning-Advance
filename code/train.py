"""Single configurable training pipeline for all DeepWeeds experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import compute_metrics, read_pred, save_predictions  # noqa: E402

try:  # Support both ``python code/train.py`` and ``import code.train``.
    from . import checkpointing, dataset, losses, model as model_utils
except ImportError:
    import checkpointing
    import dataset
    import losses
    import model as model_utils


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 42
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 10
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    device: str = "cuda"
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False
    # Colab/remote persistence. Local scratch stays fast; essential files are mirrored here.
    persistent_dir: str | None = None
    resume: bool = True
    sync_every_epochs: int = 1


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split phải là val hoặc test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def persistent_run_dir(cfg: Config) -> Path | None:
    if cfg.persistent_dir is None:
        return None
    return Path(cfg.persistent_dir) / "artifacts" / "runs" / cfg.exp_id / f"seed{cfg.seed}"


def persistent_pred_path(cfg: Config, split: str) -> Path | None:
    if cfg.persistent_dir is None:
        return None
    return Path(cfg.persistent_dir) / "artifacts" / "predictions" / pred_path(cfg, split).name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    groups = model_utils.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Per-step linear warmup followed by cosine decay to 1e-6 of base LR."""
    if steps_per_epoch <= 0 or cfg.epochs <= 0:
        raise ValueError("steps_per_epoch và epochs phải dương")
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(round(cfg.warmup_epochs * steps_per_epoch))
    warmup_steps = min(max(warmup_steps, 0), total_steps - 1)
    min_factor = 1e-6

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, min_factor)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


class EMA:
    """Exponential moving average including floating buffers and exact integer buffers."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay phải nằm trong (0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict()
        for name, target_value in self.module.state_dict().items():
            source_value = source[name].detach()
            if target_value.is_floating_point():
                target_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                target_value.copy_(source_value)

    def state_dict(self):
        return self.module.state_dict()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model_utils.set_train_mode(model)
    total_loss = 0.0
    total_samples = 0
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed_targets = None
        if cfg.mix is not None:
            images, mixed_targets = losses.mix_batch(images, targets, cfg.mix_alpha, cfg.mix)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = (losses.mixed_loss(criterion, logits, mixed_targets)
                    if mixed_targets is not None else criterion(logits, targets))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        batch_size = len(targets)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Train loader rỗng")
    return {
        "train_loss": total_loss / total_samples,
        "lr_backbone": float(optimizer.param_groups[0]["lr"]),
        "lr_head": float(optimizer.param_groups[-1]["lr"]),
    }


def evaluate(model, loader, criterion, device):
    """Evaluate without gradients, preserving stable filename order."""
    model.eval()
    filenames, labels, all_logits = [], [], []
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for images, targets, names in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            batch_size = len(targets)
            total_loss += float(loss) * batch_size
            total_samples += batch_size
            filenames.extend(str(name) for name in names)
            labels.append(targets.cpu())
            all_logits.append(logits.float().cpu())
    if total_samples == 0:
        raise ValueError("Evaluation loader rỗng")
    return filenames, torch.cat(labels).numpy(), torch.cat(all_logits).numpy(), total_loss / total_samples


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Write a readable loss/F1/LR curve for one seeded run."""
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("History rỗng")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], marker="o")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    axes[2].plot(epochs, [row["lr_backbone"] for row in history], label="backbone")
    axes[2].plot(epochs, [row["lr_head"] for row in history], label="head")
    axes[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR", yscale="log")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint(model, optimizer, scheduler, scaler, ema, cfg, epoch, best_f1,
                best_epoch, history, loader_generator_state):
    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "history": history,
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "rng_state": _capture_rng_state(),
        "loader_generator_state": loader_generator_state,
    }


def _torch_load(path, device):
    return checkpointing.torch_load(path, map_location=device)


def _write_status(cfg: Config, status: str, **extra) -> dict:
    payload = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "status": status,
        "updated_at": _utc_now(),
        **extra,
    }
    local_path = run_dir(cfg) / "status.json"
    checkpointing.atomic_json_dump(payload, local_path)
    persistent = persistent_run_dir(cfg)
    if persistent is not None:
        checkpointing.atomic_copy(local_path, persistent / "status.json")
    return payload


def _sync_epoch_artifacts(cfg: Config, output_dir: Path, include_best: bool) -> dict:
    persistent = persistent_run_dir(cfg)
    if persistent is None:
        return {}
    mappings = [
        (output_dir / "config.json", persistent / "config.json"),
        (output_dir / "split_report.json", persistent / "split_report.json"),
        (output_dir / "history.csv", persistent / "history.csv"),
        (output_dir / "train.log", persistent / "train.log"),
        (output_dir / "last.pt", persistent / "last.pt"),
    ]
    if include_best:
        mappings.append((output_dir / "best.pt", persistent / "best.pt"))
    return checkpointing.mirror_files(mappings)


def run(cfg: Config) -> dict:
    """Train one controlled experiment and persist all reproducibility artifacts."""
    if cfg.fold != 0:
        raise ValueError("Bài chính bắt buộc dùng fold 0; fold 1-4 chỉ dành cho điểm thưởng")
    if cfg.epochs <= 0:
        raise ValueError("epochs phải dương")
    if cfg.sync_every_epochs <= 0:
        raise ValueError("sync_every_epochs phải dương")
    persistent_output = persistent_run_dir(cfg)
    persistent_test = persistent_pred_path(cfg, "test")
    if persistent_output is not None:
        completed_summary = persistent_output / "summary.json"
        completed_status = persistent_output / "status.json"
        if completed_summary.exists() and completed_status.exists():
            status = json.loads(completed_status.read_text(encoding="utf-8"))
            test_already_done = persistent_test is not None and persistent_test.exists()
            if status.get("status") == "completed" and (not cfg.save_test_predictions or test_already_done):
                return json.loads(completed_summary.read_text(encoding="utf-8"))
    reuse_test_predictions = bool(
        cfg.save_test_predictions
        and (pred_path(cfg, "test").exists()
             or (persistent_test is not None and persistent_test.exists()))
    )

    set_seed(cfg.seed)
    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)
    checkpointing.atomic_json_dump(asdict(cfg), output_dir / "config.json")

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    checkpointing.atomic_json_dump(split_report, output_dir / "split_report.json")
    if persistent_output is not None:
        checkpointing.mirror_files([
            (output_dir / "config.json", persistent_output / "config.json"),
            (output_dir / "split_report.json", persistent_output / "split_report.json"),
        ])
    _write_status(cfg, "running", last_completed_epoch=0)
    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers,
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
        cfg.batch_size, False, None, cfg.num_workers,
    )
    test_loader = None
    if cfg.save_test_predictions and not reuse_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
            cfg.batch_size, False, None, cfg.num_workers,
        )

    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    network = model_utils.build_model(cfg.backbone, True, dataset.NUM_CLASSES, cfg.drop_rate, cfg.init)
    network.to(device)
    counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).to_numpy()
    if cfg.loss == "ce_weighted":
        weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
        criterion = losses.build_criterion("ce_weighted", weight=weight)
    elif cfg.loss == "ls":
        criterion = losses.build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        criterion = losses.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = losses.build_criterion(cfg.loss)
    criterion.to(device)
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    ema = EMA(network, cfg.ema_decay) if cfg.ema_decay is not None else None

    parameters_m = model_utils.count_params(network)
    gmac = model_utils.count_gmacs(network, cfg.img_size)
    history: list[dict] = []
    best_f1 = -math.inf
    best_epoch = None
    start_epoch = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_dirty = False

    if cfg.resume:
        if not last_path.exists() and persistent_output is not None and (persistent_output / "last.pt").exists():
            checkpointing.atomic_copy(persistent_output / "last.pt", last_path)
        if not best_path.exists() and persistent_output is not None and (persistent_output / "best.pt").exists():
            checkpointing.atomic_copy(persistent_output / "best.pt", best_path)
        if last_path.exists():
            resume_state = _torch_load(last_path, "cpu")
            previous_cfg = resume_state.get("config", {})
            critical_keys = (
                "exp_id", "seed", "fold", "backbone", "init", "drop_rate", "img_size", "aug",
                "sampler", "mix", "mix_alpha", "loss", "label_smoothing", "focal_gamma",
                "class_weight_beta", "epochs", "batch_size", "lr_backbone", "lr_head",
                "weight_decay", "warmup_epochs", "ema_decay", "amp",
            )
            for key in critical_keys:
                if previous_cfg.get(key) != getattr(cfg, key):
                    raise ValueError(
                        f"Checkpoint không khớp Config tại {key}: "
                        f"{previous_cfg.get(key)!r} != {getattr(cfg, key)!r}"
                    )
            network.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            scaler.load_state_dict(resume_state["scaler_state_dict"])
            if ema is not None and resume_state.get("ema_state_dict") is not None:
                ema.module.load_state_dict(resume_state["ema_state_dict"])
            start_epoch = int(resume_state["epoch"])
            if start_epoch > cfg.epochs:
                raise ValueError(f"Checkpoint epoch {start_epoch} vượt cfg.epochs={cfg.epochs}")
            best_f1 = float(resume_state.get("best_val_macro_f1", -math.inf))
            best_epoch = resume_state.get("best_epoch")
            if not best_path.exists() and best_epoch == start_epoch:
                checkpointing.atomic_copy(last_path, best_path)
                best_dirty = True
            history = list(resume_state.get("history", []))
            generator_state = resume_state.get("loader_generator_state")
            if generator_state is not None and train_loader.generator is not None:
                train_loader.generator.set_state(generator_state)
            _restore_rng_state(resume_state.get("rng_state"))
            _write_status(cfg, "running", last_completed_epoch=start_epoch,
                          best_epoch=best_epoch, best_val_macro_f1=best_f1, resumed=True)
            print(f"Resume {cfg.exp_id} seed={cfg.seed} từ epoch {start_epoch + 1}")

    if (persistent_output is not None and best_path.exists()
            and not (persistent_output / "best.pt").exists()):
        best_dirty = True

    last_synced_epoch = start_epoch if persistent_output is not None and start_epoch > 0 else None
    for epoch_index in range(start_epoch, cfg.epochs):
        started = time.perf_counter()
        train_stats = train_one_epoch(
            network, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        eval_model = ema.module if ema is not None else network
        _, y_val, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_probs = _softmax(val_logits)
        metrics = compute_metrics(y_val, val_probs.argmax(1), val_probs)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch_index + 1,
            **train_stats,
            "val_loss": val_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_top1": metrics["top1"],
            "val_balanced_acc": metrics["balanced_acc"],
            "val_ece": metrics["ece"],
            "epoch_seconds": elapsed,
        }
        history.append(row)
        history_path = output_dir / "history.csv"
        history_tmp = output_dir / ".history.csv.tmp"
        pd.DataFrame(history).to_csv(history_tmp, index=False)
        history_tmp.replace(history_path)

        improved = metrics["macro_f1"] > best_f1
        if improved:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch_index + 1
        loader_state = train_loader.generator.get_state() if train_loader.generator is not None else None
        state = _checkpoint(network, optimizer, scheduler, scaler, ema, cfg,
                            epoch_index + 1, best_f1, best_epoch, history, loader_state)
        checkpointing.atomic_torch_save(state, last_path)
        if improved:
            checkpointing.atomic_copy(last_path, best_path)
            best_dirty = True
        hashes = {}
        if (epoch_index + 1) % cfg.sync_every_epochs == 0 or epoch_index + 1 == cfg.epochs:
            hashes = _sync_epoch_artifacts(cfg, output_dir, include_best=best_dirty)
            best_dirty = False
            last_synced_epoch = epoch_index + 1
        _write_status(
            cfg, "running", last_completed_epoch=epoch_index + 1,
            last_synced_epoch=last_synced_epoch,
            best_epoch=best_epoch, best_val_macro_f1=best_f1,
            checkpoint_sha256=hashes.get(str(persistent_output / "last.pt"))
            if persistent_output is not None else checkpointing.sha256_file(last_path),
        )
        message = (
            f"[{cfg.exp_id} seed={cfg.seed}] epoch {epoch_index + 1:02d}/{cfg.epochs}: "
            f"train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} top1={metrics['top1']:.4f} "
            f"time={elapsed:.1f}s"
        )
        print(message)
        with (output_dir / "train.log").open("a", encoding="utf-8") as log_file:
            log_file.write(f"{_utc_now()} {message}\n")
        if persistent_output is not None:
            checkpointing.atomic_copy(output_dir / "train.log", persistent_output / "train.log")

    checkpoint = _torch_load(best_path, device)
    evaluation_state = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
    network.load_state_dict(evaluation_state)
    val_names, y_val, val_logits, val_loss = evaluate(network, val_loader, criterion, device)
    val_probs = _softmax(val_logits)
    val_metrics = compute_metrics(y_val, val_probs.argmax(1), val_probs)
    np.save(output_dir / "val_logits.npy", val_logits)
    save_predictions(pred_path(cfg, "val"), val_names, y_val, val_probs)

    test_metrics = None
    if cfg.save_test_predictions:
        local_test_prediction = pred_path(cfg, "test")
        if reuse_test_predictions:
            if not local_test_prediction.exists():
                assert persistent_test is not None
                checkpointing.atomic_copy(persistent_test, local_test_prediction)
            prediction = read_pred(str(local_test_prediction))
            test_metrics = compute_metrics(prediction.y_true, prediction.y_pred, prediction.probs)
            print(f"Tái sử dụng prediction test đã tồn tại: {local_test_prediction}")
        else:
            assert test_loader is not None
            test_names, y_test, test_logits, _ = evaluate(network, test_loader, criterion, device)
            test_probs = _softmax(test_logits)
            test_metrics = compute_metrics(y_test, test_probs.argmax(1), test_probs)
            np.save(output_dir / "test_logits.npy", test_logits)
            save_predictions(local_test_prediction, test_names, y_test, test_probs)
            # Persist the one-time test result immediately. A later restart reuses this file.
            if persistent_test is not None:
                checkpointing.atomic_copy(local_test_prediction, persistent_test)
            _write_status(
                cfg, "running", last_completed_epoch=cfg.epochs, best_epoch=best_epoch,
                best_val_macro_f1=best_f1, test_prediction_saved=True,
            )

    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    epoch_times = np.asarray([row["epoch_seconds"] for row in history])
    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_top1": val_metrics["top1"],
        "val_ece": val_metrics["ece"],
        "val_loss": val_loss,
        "params_m": parameters_m,
        "gmac": gmac,
        "weight_tag": getattr(network, "deepweeds_weight_tag", cfg.backbone),
        "epoch_seconds_mean": float(epoch_times.mean()),
        "epoch_seconds_std": float(epoch_times.std(ddof=1)) if len(epoch_times) > 1 else None,
        "best_checkpoint": str(best_path),
        "curve": str(curve_path),
    }
    if test_metrics is not None:
        summary.update({
            "test_macro_f1": test_metrics["macro_f1"],
            "test_top1": test_metrics["top1"],
            "test_ece": test_metrics["ece"],
        })
    summary_path = checkpointing.atomic_json_dump(summary, output_dir / "summary.json")
    final_mappings = []
    if persistent_output is not None:
        persistent_root = Path(cfg.persistent_dir)
        persistent_curve = persistent_root / "artifacts" / "curves" / curve_path.name
        final_mappings.extend([
            (summary_path, persistent_output / "summary.json"),
            (output_dir / "history.csv", persistent_output / "history.csv"),
            (output_dir / "train.log", persistent_output / "train.log"),
            (output_dir / "val_logits.npy", persistent_output / "val_logits.npy"),
            (best_path, persistent_output / "best.pt"),
            (last_path, persistent_output / "last.pt"),
            (curve_path, persistent_curve),
            (pred_path(cfg, "val"), persistent_pred_path(cfg, "val")),
        ])
        if cfg.save_test_predictions:
            final_mappings.append((pred_path(cfg, "test"), persistent_pred_path(cfg, "test")))
            if (output_dir / "test_logits.npy").exists():
                final_mappings.append((output_dir / "test_logits.npy", persistent_output / "test_logits.npy"))
        checkpointing.mirror_files(final_mappings)
    _write_status(
        cfg, "completed", last_completed_epoch=cfg.epochs, best_epoch=best_epoch,
        best_val_macro_f1=best_f1, completed_at=_utc_now(),
    )
    return summary


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Không ép được {value!r} thành bool")


def parse_overrides(pairs: list[str]) -> dict:
    """Parse ``KEY=VALUE`` CLI settings using the Config field defaults as schema."""
    defaults = Config()
    valid = {field.name for field in fields(Config)}
    nullable_string = {"sampler", "mix", "persistent_dir"}
    nullable_float = {"class_weight_beta", "ema_decay"}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override phải có dạng KEY=VALUE: {pair!r}")
        key, raw = pair.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if key not in valid:
            raise KeyError(f"Config không có trường {key!r}; hợp lệ: {sorted(valid)}")
        if raw.lower() in {"none", "null"}:
            if key not in nullable_string | nullable_float:
                raise ValueError(f"{key} không nhận None")
            value = None
        elif key in nullable_string:
            value = raw
        elif key in nullable_float:
            value = float(raw)
        else:
            current = getattr(defaults, key)
            if isinstance(current, bool):
                value = _parse_bool(raw)
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one controlled DeepWeeds experiment")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override Config fields, e.g. exp_id=B01 backbone=resnet50 seed=42")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    summary = run(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
