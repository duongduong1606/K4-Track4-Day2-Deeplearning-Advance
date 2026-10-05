"""Validation-only inference sweep and one-time final inference execution."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval import compute_metrics, read_pred, save_predictions  # noqa: E402

try:
    from . import benchmark, checkpointing, dataset, inference, model as model_utils
    from .train import Config
except ImportError:
    import benchmark
    import checkpointing
    import dataset
    import inference
    import model as model_utils
    from train import Config


@dataclass(frozen=True)
class InferenceSpec:
    name: str = "I00"
    img_size: int = 224
    views: tuple[str, ...] = ("identity",)
    aggregation: str = "prob"
    calibrate: bool = False
    amp: bool = False
    fuse_bn: bool = False

    def validate(self) -> None:
        if self.img_size <= 0:
            raise ValueError("img_size phải dương")
        if not self.views or any(view not in {"identity", "hflip"} for view in self.views):
            raise ValueError("views chỉ hỗ trợ identity và hflip")
        if self.aggregation not in {"prob", "logit"}:
            raise ValueError("aggregation phải là prob hoặc logit")


def spec_hash(spec: InferenceSpec) -> str:
    payload = json.dumps(asdict(spec), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_run(run_root: str | Path, device: str):
    run_root = Path(run_root)
    config = Config(**json.loads((run_root / "config.json").read_text(encoding="utf-8")))
    network = model_utils.build_model(
        config.backbone, pretrained=False, num_classes=dataset.NUM_CLASSES,
        drop_rate=config.drop_rate, init="scratch",
    )
    checkpoint = checkpointing.torch_load(run_root / "best.pt", map_location="cpu")
    state = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
    checkpointing.load_model_state_dict(network, state)
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    network.to(device_obj).eval()
    return config, network, device_obj


def _loader(frame, images_dir, img_size, batch_size, num_workers):
    return dataset.make_loader(
        frame, images_dir, dataset.build_transforms(False, img_size), batch_size,
        train=False, sampler=None, num_workers=num_workers,
    )


def _predict_spec(network, loader, device, spec: InferenceSpec):
    spec.validate()
    if spec.fuse_bn:
        network = inference.fuse_conv_bn(network).to(device)
    outputs, reference_names, reference_labels = [], None, None
    view_functions = {"identity": inference.view_identity, "hflip": inference.view_hflip}
    for view_name in spec.views:
        names, labels, logits = inference.predict_logits(
            network, loader, device, view=view_functions[view_name], amp=spec.amp
        )
        if reference_names is None:
            reference_names, reference_labels = names, labels
        elif names != reference_names or not np.array_equal(labels, reference_labels):
            raise ValueError("Thứ tự ảnh/nhãn thay đổi giữa các view")
        outputs.append(logits)
    probs = inference.aggregate_views(outputs, space=spec.aggregation)
    if spec.aggregation == "logit":
        calibration_logits = np.mean(np.stack(outputs, axis=0), axis=0)
    else:
        calibration_logits = np.log(np.clip(probs, 1e-12, 1.0))
    return reference_names, reference_labels, probs, calibration_logits


def default_sweep_specs() -> list[InferenceSpec]:
    """At least four non-baseline methods required by the rubric."""
    return [
        InferenceSpec(name="I00", img_size=224, views=("identity",), aggregation="prob"),
        InferenceSpec(name="I01", img_size=224, views=("identity", "hflip"), aggregation="prob"),
        InferenceSpec(name="I02", img_size=224, views=("identity", "hflip"), aggregation="logit"),
        InferenceSpec(name="I03", img_size=256, views=("identity",), aggregation="prob"),
        InferenceSpec(name="I04", img_size=288, views=("identity",), aggregation="prob"),
        InferenceSpec(name="I07", img_size=224, views=("identity",), aggregation="prob", calibrate=True),
        InferenceSpec(name="I09", img_size=224, views=("identity", "hflip"),
                      aggregation="prob", calibrate=True),
    ]


def run_validation_sweep(run_root: str | Path, images_dir: str | Path, labels_dir: str | Path,
                         output_dir: str | Path, device: str = "cuda", latency_iters: int = 100,
                         specs: list[InferenceSpec] | None = None) -> pd.DataFrame:
    """Evaluate inference methods on validation only and append rubric-ready CSV rows."""
    config, network, device_obj = _load_run(run_root, device)
    _, val_df, _ = dataset.load_split(labels_dir, config.fold)
    specs = specs or default_sweep_specs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, latency_rows = [], []
    loader_cache = {}

    for spec in specs:
        spec.validate()
        try:
            if spec.img_size not in loader_cache:
                loader_cache[spec.img_size] = _loader(
                    val_df, images_dir, spec.img_size, config.batch_size, config.num_workers
                )
            loader = loader_cache[spec.img_size]
            names, labels, uncal_probs, calibration_logits = _predict_spec(
                network, loader, device_obj, spec
            )
            temperature = 1.0
            probs = uncal_probs
            if spec.calibrate:
                temperature = inference.fit_temperature(calibration_logits, labels)
                probs = inference.apply_temperature(calibration_logits, temperature)
            metrics = compute_metrics(labels, probs.argmax(1), probs)
            row = {
                "exp_id": spec.name,
                "method": spec.name,
                "model_exp_id": config.exp_id,
                "seed": config.seed,
                "checkpoint": str(Path(run_root) / "best.pt"),
                "k": len(spec.views),
                "img_size": spec.img_size,
                "aggregation": spec.aggregation,
                "calibrated": spec.calibrate,
                "temperature": temperature,
                "val_macro_f1": metrics["macro_f1"],
                "val_top1": metrics["top1"],
                "val_ece": metrics["ece"],
                "spec_json": json.dumps(asdict(spec), ensure_ascii=False, sort_keys=True),
                "spec_sha256": spec_hash(spec),
                "status": "completed",
                "error": "",
            }
            save_predictions(
                output_dir / f"{config.exp_id}_seed{config.seed}_{spec.name}_val.csv",
                names, labels, probs,
            )
            if len(spec.views) > 1:
                latency = benchmark.tta_latency(
                    network, len(spec.views), batch_size=1, img_size=spec.img_size,
                    dtype="amp" if spec.amp else "fp32", device=str(device_obj), iters=latency_iters,
                )
            else:
                latency = benchmark.latency_report(
                    network, 1, spec.img_size, dtype="amp" if spec.amp else "fp32",
                    device=str(device_obj), iters=latency_iters,
                )
            row.update({
                "p50_ms": latency["p50"], "p95_ms": latency["p95"],
                "p99_ms": latency["p99"], "images_per_s": latency["images_per_s"],
            })
            latency_rows.append({
                "configuration": f"{config.exp_id}:{spec.name}", "gpu": latency["gpu"],
                "dtype": latency["dtype"], "batch": latency["batch"], "img_size": spec.img_size,
                "fused_bn": spec.fuse_bn, "p50_ms": latency["p50"], "p95_ms": latency["p95"],
                "p99_ms": latency["p99"], "images_per_s": latency["images_per_s"],
                "torch": latency["torch"],
            })
        except Exception as exc:
            row = {
                "exp_id": spec.name, "method": spec.name, "model_exp_id": config.exp_id,
                "seed": config.seed, "checkpoint": str(Path(run_root) / "best.pt"),
                "k": len(spec.views), "img_size": spec.img_size, "aggregation": spec.aggregation,
                "calibrated": spec.calibrate, "temperature": None, "val_macro_f1": None,
                "val_top1": None, "val_ece": None, "p50_ms": None, "p95_ms": None,
                "p99_ms": None, "images_per_s": None,
                "spec_json": json.dumps(asdict(spec), ensure_ascii=False, sort_keys=True),
                "spec_sha256": spec_hash(spec), "status": "failed", "error": repr(exc),
            }
        rows.append(row)

    results_path = output_dir / "inference_results.csv"
    new_results = pd.DataFrame(rows)
    if results_path.exists():
        previous = pd.read_csv(results_path)
        new_results = pd.concat([previous, new_results], ignore_index=True)
        new_results = new_results.drop_duplicates(["model_exp_id", "seed", "method"], keep="last")
    for (model_exp_id, seed), indices in new_results.groupby(["model_exp_id", "seed"]).groups.items():
        group = new_results.loc[indices]
        baseline = group.loc[group["method"] == "I00", "p50_ms"].dropna()
        if not baseline.empty and float(baseline.iloc[-1]) > 0:
            new_results.loc[indices, "relative_cost"] = (
                new_results.loc[indices, "p50_ms"] / float(baseline.iloc[-1])
            )
    _atomic_csv(new_results, results_path)
    latency_path = output_dir / "latency_results.csv"
    new_latency = pd.DataFrame(latency_rows)
    if latency_path.exists() and not new_latency.empty:
        new_latency = pd.concat([pd.read_csv(latency_path), new_latency], ignore_index=True)
        new_latency = new_latency.drop_duplicates(["configuration", "batch", "dtype"], keep="last")
    if not new_latency.empty:
        _atomic_csv(new_latency, latency_path)
    return new_results


def freeze_inference_spec(results_csv: str | Path, method: str, model_exp_id: str,
                          seed: int, output_path: str | Path, selection_note: str) -> Path:
    """Freeze one validation-selected method, its evidence, and an integrity hash."""
    if not selection_note.strip():
        raise ValueError("selection_note không được rỗng")
    frame = pd.read_csv(results_csv)
    selected = frame[
        (frame["method"] == method) & (frame["model_exp_id"] == model_exp_id)
        & (frame["seed"] == seed) & (frame["status"] == "completed")
    ]
    if len(selected) != 1:
        raise ValueError("Phải tìm thấy đúng một kết quả validation hoàn tất")
    row = selected.iloc[0]
    values = json.loads(row["spec_json"])
    values["views"] = tuple(values["views"])
    spec = InferenceSpec(**values)
    payload = {
        "spec": asdict(spec),
        "spec_sha256": spec_hash(spec),
        "selected_from": "validation_only",
        "selection_note": selection_note.strip(),
        "evidence": {
            "method": method,
            "model_exp_id": model_exp_id,
            "seed": int(seed),
            "val_macro_f1": float(row["val_macro_f1"]),
            "val_top1": float(row["val_top1"]),
            "val_ece": float(row["val_ece"]),
            "p95_ms": float(row["p95_ms"]),
        },
    }
    return checkpointing.atomic_json_dump(payload, output_path)


def load_frozen_spec(path: str | Path) -> InferenceSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload["spec"]
    values["views"] = tuple(values["views"])
    spec = InferenceSpec(**values)
    if payload.get("spec_sha256") != spec_hash(spec):
        raise ValueError("InferenceSpec đã bị thay đổi sau khi đóng băng")
    source = payload.get("selected_from")
    baseline_ok = source == "protocol_baseline" and spec == InferenceSpec(name="I00")
    if source != "validation_only" and not baseline_ok:
        raise ValueError("InferenceSpec không được chọn hoàn toàn từ validation")
    return spec


def apply_frozen_spec_once(run_root: str | Path, spec_path: str | Path,
                           images_dir: str | Path, labels_dir: str | Path,
                           local_pred_dir: str | Path, persistent_pred_dir: str | Path,
                           device: str = "cuda") -> dict:
    """Fit calibration on val, then create/reuse exactly one persistent test prediction."""
    run_root = Path(run_root)
    config, network, device_obj = _load_run(run_root, device)
    spec = load_frozen_spec(spec_path)
    train_df, val_df, test_df = dataset.load_split(labels_dir, config.fold)
    del train_df
    val_loader = _loader(val_df, images_dir, spec.img_size, config.batch_size, config.num_workers)
    val_names, y_val, val_uncal, val_cal_logits = _predict_spec(network, val_loader, device_obj, spec)
    temperature = inference.fit_temperature(val_cal_logits, y_val) if spec.calibrate else 1.0
    val_probs = (inference.apply_temperature(val_cal_logits, temperature)
                 if spec.calibrate else val_uncal)

    local_pred_dir, persistent_pred_dir = Path(local_pred_dir), Path(persistent_pred_dir)
    local_pred_dir.mkdir(parents=True, exist_ok=True)
    persistent_pred_dir.mkdir(parents=True, exist_ok=True)
    val_name = f"{config.exp_id}_seed{config.seed}_val.csv"
    local_val = save_predictions(local_pred_dir / val_name, val_names, y_val, val_probs)
    checkpointing.atomic_copy(local_val, persistent_pred_dir / val_name)

    final_name = f"{config.exp_id}_seed{config.seed}_test.csv"
    uncal_name = f"{config.exp_id}_seed{config.seed}_test_uncal.csv"
    persistent_final = persistent_pred_dir / final_name
    persistent_uncal = persistent_pred_dir / uncal_name

    if persistent_final.exists():
        final_prediction = read_pred(str(persistent_final))
        test_metrics = compute_metrics(
            final_prediction.y_true, final_prediction.y_pred, final_prediction.probs
        )
        return {
            "exp_id": config.exp_id, "seed": config.seed, "status": "reused",
            "temperature": temperature, "spec_sha256": spec_hash(spec),
            "test_macro_f1": test_metrics["macro_f1"], "test_top1": test_metrics["top1"],
            "test_ece": test_metrics["ece"], "prediction": str(persistent_final),
        }

    if persistent_uncal.exists():
        uncal_prediction = read_pred(str(persistent_uncal))
        test_names, y_test, test_uncal = (
            uncal_prediction.filenames, uncal_prediction.y_true, uncal_prediction.probs
        )
        test_cal_logits = np.log(np.clip(test_uncal, 1e-12, 1.0))
    else:
        test_loader = _loader(test_df, images_dir, spec.img_size, config.batch_size, config.num_workers)
        test_names, y_test, test_uncal, test_cal_logits = _predict_spec(
            network, test_loader, device_obj, spec
        )
        local_uncal = save_predictions(local_pred_dir / uncal_name, test_names, y_test, test_uncal)
        checkpointing.atomic_copy(local_uncal, persistent_uncal)

    test_probs = (inference.apply_temperature(test_cal_logits, temperature)
                  if spec.calibrate else test_uncal)
    local_final = save_predictions(local_pred_dir / final_name, test_names, y_test, test_probs)
    checkpointing.atomic_copy(local_final, persistent_final)
    metrics = compute_metrics(y_test, test_probs.argmax(1), test_probs)
    metadata = {
        "exp_id": config.exp_id, "seed": config.seed, "status": "completed",
        "temperature": temperature, "spec": asdict(spec), "spec_sha256": spec_hash(spec),
        "val_macro_f1": compute_metrics(y_val, val_probs.argmax(1), val_probs)["macro_f1"],
        "test_macro_f1": metrics["macro_f1"], "test_top1": metrics["top1"],
        "test_ece": metrics["ece"], "prediction": str(persistent_final),
    }
    checkpointing.atomic_json_dump(
        metadata, run_root / f"final_inference_{spec.name}.json"
    )
    return metadata
