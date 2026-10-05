"""Google Colab staging and resumable manifest execution."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import checkpointing
    from . import final_inference
    from .experiment_plan import BACKBONES, FINAL_SEEDS, TRAINING
    from .train import Config, persistent_pred_path, persistent_run_dir, run
except ImportError:
    import checkpointing
    import final_inference
    from experiment_plan import BACKBONES, FINAL_SEEDS, TRAINING
    from train import Config, persistent_pred_path, persistent_run_dir, run


IMAGES_MD5 = "b7b30f96d466fba86016aa5a26606e0f"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_hash(runs: list[dict]) -> str:
    frozen = []
    for item in runs:
        entry = {
            "run_id": item["run_id"], "config": item["config"],
            "max_attempts": item.get("max_attempts", 3),
        }
        for key in ("exp_id", "seed", "spec_path"):
            if key in item:
                entry[key] = item[key]
        frozen.append(entry)
    payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def md5_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect_images_dir(local_data: Path) -> Path:
    """Accept archives containing either images/* or flat image files."""
    nested = local_data / "images"
    if nested.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in nested.iterdir()
    ):
        return nested
    if local_data.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in local_data.iterdir()
    ):
        return local_data
    return nested


def _count_images(images_dir: Path) -> int:
    return sum(
        1 for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def stage_dataset(drive_root: str | Path, local_root: str | Path,
                  verify_md5: bool = True) -> dict:
    """Copy one archive plus labels from Drive and extract images on local Colab disk."""
    drive_root, local_root = Path(drive_root), Path(local_root)
    resolved_local = local_root.resolve()
    if resolved_local == Path(resolved_local.anchor) or len(resolved_local.parts) < 3:
        raise ValueError(f"local_root quá rộng, không an toàn để stage dữ liệu: {resolved_local}")
    source_zip = drive_root / "data" / "images.zip"
    local_data = local_root / "data"
    local_zip = local_data / "images.zip"
    local_labels = local_data / "labels"
    images_dir = _detect_images_dir(local_data)
    if not source_zip.is_file():
        raise FileNotFoundError(f"Thiếu {source_zip}")
    local_data.mkdir(parents=True, exist_ok=True)
    if not local_zip.exists() or local_zip.stat().st_size != source_zip.stat().st_size:
        checkpointing.atomic_copy(source_zip, local_zip)
    checksum = md5_file(local_zip)
    if verify_md5 and checksum != IMAGES_MD5:
        raise ValueError(f"MD5 images.zip sai: {checksum}, kỳ vọng {IMAGES_MD5}")

    required_labels = ["labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"]
    local_labels.mkdir(parents=True, exist_ok=True)
    for name in required_labels:
        source = drive_root / "data" / "labels" / name
        if not source.is_file():
            raise FileNotFoundError(f"Thiếu {source}")
        checkpointing.atomic_copy(source, local_labels / name)

    marker = local_data / ".extracted_ok"
    if not marker.exists():
        if images_dir == local_data / "images" and images_dir.exists():
            shutil.rmtree(images_dir)
        # A failed older run may already have extracted a valid flat archive.
        images_dir = _detect_images_dir(local_data)
        image_count = _count_images(images_dir) if images_dir.is_dir() else 0
        if image_count != 17_509:
            with zipfile.ZipFile(local_zip) as archive:
                extraction_root = local_data.resolve()
                for member in archive.infolist():
                    destination = (local_data / member.filename).resolve()
                    try:
                        destination.relative_to(extraction_root)
                    except ValueError as exc:
                        raise ValueError(f"ZIP chứa đường dẫn không an toàn: {member.filename}") from exc
                archive.extractall(local_data)
            images_dir = _detect_images_dir(local_data)
            image_count = _count_images(images_dir) if images_dir.is_dir() else 0
        if not images_dir.is_dir() or image_count == 0:
            raise FileNotFoundError(
                "Sau giải nén không tìm thấy ảnh trong data/images hoặc data/"
            )
        if image_count != 17_509:
            raise ValueError(f"Số ảnh sau giải nén là {image_count}, phải là 17.509")
        marker.write_text(json.dumps({
            "md5": checksum, "images": image_count,
            "layout": "nested" if images_dir.name == "images" else "flat",
        }), encoding="utf-8")
    else:
        images_dir = _detect_images_dir(local_data)
        image_count = _count_images(images_dir)

    return {
        "images_dir": str(images_dir),
        "labels_dir": str(local_labels),
        "images": image_count,
        "md5": checksum,
    }


def _manifest_item(cfg: Config) -> dict:
    values = asdict(cfg)
    for key in ("images_dir", "labels_dir", "out_dir", "pred_dir", "curves_dir",
                "persistent_dir", "save_test_predictions"):
        values.pop(key, None)
    return {
        "run_id": f"{cfg.exp_id}_seed{cfg.seed}",
        "status": "pending",
        "attempts": 0,
        "max_attempts": 3,
        "config": values,
    }


def create_screening_manifest(path: str | Path, overwrite: bool = False) -> Path:
    """Create a validation-only queue from the controlled B/T experiment plan."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        return destination
    unique = {}
    for cfg in [*BACKBONES, *TRAINING]:
        unique[(cfg.exp_id, cfg.seed)] = cfg
    manifest = {
        "version": 1,
        "kind": "screening",
        "test_access": False,
        "locked": False,
        "created_at": _now(),
        "runs": [_manifest_item(cfg) for cfg in unique.values()],
    }
    return checkpointing.atomic_json_dump(manifest, destination)


def create_final_training_manifest(path: str | Path, best_config: Config,
                                   selection_note: str,
                                   baseline_config: Config | None = None,
                                   overwrite: bool = False) -> Path:
    """Create a test-free three-seed queue after the recipe is frozen on validation."""
    if not selection_note.strip():
        raise ValueError("Phải ghi lý do chọn cấu hình từ validation")
    destination = Path(path)
    if destination.exists() and not overwrite:
        return destination
    baseline_config = baseline_config or Config(exp_id="T00")
    configs = []
    for seed in FINAL_SEEDS:
        configs.append(Config(**{**asdict(baseline_config), "exp_id": "T00", "seed": seed,
                                 "save_test_predictions": False}))
        configs.append(Config(**{**asdict(best_config), "exp_id": "F01", "seed": seed,
                                 "save_test_predictions": False}))
    manifest = {
        "version": 1,
        "kind": "final_training",
        "test_access": False,
        "locked": False,
        "selection_source": "validation_only",
        "selection_note": selection_note.strip(),
        "created_at": _now(),
        "runs": [_manifest_item(cfg) for cfg in configs],
    }
    manifest["plan_sha256"] = _plan_hash(manifest["runs"])
    return checkpointing.atomic_json_dump(manifest, destination)


def create_final_test_manifest(path: str | Path, drive_root: str | Path,
                               final_spec_path: str | Path,
                               overwrite: bool = False) -> Path:
    """Create a locked one-time test queue using a frozen validation-selected spec."""
    destination, drive_root = Path(path), Path(drive_root)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Test manifest đã tồn tại: {destination}")
    final_inference.load_frozen_spec(final_spec_path)
    baseline_spec_path = drive_root / "queue" / "baseline_I00_spec.json"
    baseline_spec = final_inference.InferenceSpec(name="I00")
    checkpointing.atomic_json_dump({
        "spec": asdict(baseline_spec),
        "spec_sha256": final_inference.spec_hash(baseline_spec),
        "selected_from": "protocol_baseline",
        "selection_note": "T00 bắt buộc dùng inference I00 làm mốc",
    }, baseline_spec_path)
    runs = []
    for seed in FINAL_SEEDS:
        runs.extend([
            {"run_id": f"T00_seed{seed}_test", "exp_id": "T00", "seed": seed,
             "spec_path": str(baseline_spec_path), "status": "pending", "attempts": 0,
             "max_attempts": 3, "config": {"exp_id": "T00", "seed": seed}},
            {"run_id": f"F01_seed{seed}_test", "exp_id": "F01", "seed": seed,
             "spec_path": str(final_spec_path), "status": "pending", "attempts": 0,
             "max_attempts": 3, "config": {"exp_id": "F01", "seed": seed}},
        ])
    manifest = {
        "version": 1, "kind": "final_test", "test_access": True, "locked": True,
        "selection_source": "validation_only", "selection_note": "", "created_at": _now(),
        "runs": runs,
    }
    return checkpointing.atomic_json_dump(manifest, destination)


def create_final_manifest(path: str | Path, best_config: Config, baseline_config: Config | None = None,
                          overwrite: bool = False) -> Path:
    """Deprecated unsafe combined flow; final training and test must remain separate."""
    del path, best_config, baseline_config, overwrite
    raise RuntimeError(
        "Dùng create_final_training_manifest(), sau đó create_final_test_manifest(). "
        "Không tạo manifest vừa train vừa test."
    )


def unlock_final_manifest(path: str | Path, selection_note: str) -> None:
    """Explicitly unlock final test after documenting the validation-only decision."""
    if not selection_note.strip():
        raise ValueError("Phải ghi selection_note giải thích lựa chọn dựa trên validation")
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "final_test" or not manifest.get("test_access"):
        raise ValueError("Chỉ final manifest mới được mở khóa test")
    manifest["selection_note"] = selection_note.strip()
    manifest["locked"] = False
    manifest["unlocked_at"] = _now()
    manifest["plan_sha256"] = _plan_hash(manifest["runs"])
    checkpointing.atomic_json_dump(manifest, manifest_path)


def _is_persistently_complete(cfg: Config) -> bool:
    run_root = persistent_run_dir(cfg)
    if run_root is None:
        return False
    status_path = run_root / "status.json"
    if not status_path.exists() or not (run_root / "summary.json").exists() or not (run_root / "best.pt").exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        return False
    if cfg.save_test_predictions:
        prediction = persistent_pred_path(cfg, "test")
        return prediction is not None and prediction.exists()
    prediction = persistent_pred_path(cfg, "val")
    return prediction is not None and prediction.exists()


def run_queue(manifest_path: str | Path, drive_root: str | Path, local_root: str | Path,
              max_runs: int | None = None, allow_test: bool = False,
              stop_on_error: bool = True) -> list[dict]:
    """Run or resume pending manifest entries sequentially on one Colab GPU."""
    manifest_path, drive_root, local_root = Path(manifest_path), Path(drive_root), Path(local_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_access = bool(manifest.get("test_access"))
    if test_access and (manifest.get("kind") != "final" or manifest.get("locked", True)):
        raise PermissionError("Final manifest vẫn bị khóa")
    if test_access and not allow_test:
        raise PermissionError("Phải truyền allow_test=True một cách tường minh để chạy final test")
    if not test_access and allow_test:
        raise PermissionError("Screening manifest không được phép truy cập test")
    if test_access and manifest.get("plan_sha256") != _plan_hash(manifest["runs"]):
        raise PermissionError("Cấu hình final manifest đã thay đổi sau khi mở khóa")
    if manifest.get("kind") == "final_training" and manifest.get("plan_sha256") != _plan_hash(manifest["runs"]):
        raise PermissionError("Cấu hình final training đã thay đổi sau khi chốt trên validation")

    data = stage_dataset(drive_root, local_root)
    local_work = local_root / "work"
    results = []
    executed = 0
    for item in manifest["runs"]:
        if max_runs is not None and executed >= max_runs:
            break
        if item.get("status") == "running":
            item["status"] = "interrupted"
            item["interrupted_at"] = _now()
        if item.get("status") == "failed":
            continue

        values = dict(item["config"])
        values.update({
            "images_dir": data["images_dir"],
            "labels_dir": data["labels_dir"],
            "out_dir": str(local_work / "runs"),
            "pred_dir": str(local_work / "predictions"),
            "curves_dir": str(local_work / "curves"),
            "persistent_dir": str(drive_root),
            "resume": True,
            "save_test_predictions": test_access,
        })
        cfg = Config(**values)
        if _is_persistently_complete(cfg):
            item["status"] = "completed"
            item["completed_at"] = _now()
            checkpointing.atomic_json_dump(manifest, manifest_path)
            continue
        if item.get("status") == "completed":
            item["status"] = "interrupted"
            item["last_error"] = "Manifest báo completed nhưng artifact persistent chưa đầy đủ"

        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["status"] = "running"
        item["started_at"] = _now()
        item["gpu"] = os.environ.get("COLAB_GPU", "runtime_gpu")
        checkpointing.atomic_json_dump(manifest, manifest_path)
        executed += 1
        try:
            summary = run(cfg)
            item["status"] = "completed"
            item["completed_at"] = _now()
            item["summary"] = summary
            results.append(summary)
        except Exception as exc:
            item["last_error"] = repr(exc)
            item["traceback"] = traceback.format_exc()[-8000:]
            if item["attempts"] >= int(item.get("max_attempts", 3)):
                item["status"] = "failed"
            else:
                item["status"] = "interrupted"
            item["failed_at"] = _now()
            run_root = persistent_run_dir(cfg)
            if run_root is not None:
                checkpointing.atomic_json_dump({
                    "exp_id": cfg.exp_id,
                    "seed": cfg.seed,
                    "status": item["status"],
                    "attempts": item["attempts"],
                    "error": repr(exc),
                    "updated_at": _now(),
                }, run_root / "status.json")
            checkpointing.atomic_json_dump(manifest, manifest_path)
            if stop_on_error:
                raise
        checkpointing.atomic_json_dump(manifest, manifest_path)
    return results


def run_final_test_queue(manifest_path: str | Path, drive_root: str | Path,
                         local_root: str | Path, max_runs: int = 1,
                         allow_test: bool = False, stop_on_error: bool = True) -> list[dict]:
    """Apply frozen inference specs to test exactly once per completed seeded model."""
    manifest_path, drive_root, local_root = Path(manifest_path), Path(drive_root), Path(local_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "final_test" or not manifest.get("test_access"):
        raise PermissionError("Manifest không phải final_test")
    if manifest.get("locked", True) or not allow_test:
        raise PermissionError("Final test vẫn khóa hoặc allow_test chưa được bật")
    if manifest.get("plan_sha256") != _plan_hash(manifest["runs"]):
        raise PermissionError("Final test plan đã bị thay đổi sau khi mở khóa")

    staged = stage_dataset(drive_root, local_root)
    persistent_pred_dir = drive_root / "artifacts" / "predictions"
    local_pred_dir = local_root / "work" / "predictions"
    completed_now, executed = [], 0
    for item in manifest["runs"]:
        if executed >= max_runs:
            break
        final_prediction = persistent_pred_dir / f"{item['exp_id']}_seed{item['seed']}_test.csv"
        if final_prediction.exists():
            item["status"] = "completed"
            item["prediction"] = str(final_prediction)
            checkpointing.atomic_json_dump(manifest, manifest_path)
            continue
        if item.get("status") == "failed":
            continue

        run_root = drive_root / "artifacts" / "runs" / item["exp_id"] / f"seed{item['seed']}"
        training_status_path = run_root / "status.json"
        if not training_status_path.exists() or not (run_root / "best.pt").exists():
            raise FileNotFoundError(f"Chưa có training artifact hoàn chỉnh: {run_root}")
        training_status = json.loads(training_status_path.read_text(encoding="utf-8"))
        if training_status.get("status") != "completed":
            raise RuntimeError(f"Training chưa completed: {run_root}")

        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["status"] = "running"
        item["started_at"] = _now()
        checkpointing.atomic_json_dump(manifest, manifest_path)
        executed += 1
        try:
            result = final_inference.apply_frozen_spec_once(
                run_root=run_root,
                spec_path=item["spec_path"],
                images_dir=staged["images_dir"],
                labels_dir=staged["labels_dir"],
                local_pred_dir=local_pred_dir,
                persistent_pred_dir=persistent_pred_dir,
            )
            item["status"] = "completed"
            item["completed_at"] = _now()
            item["prediction"] = result["prediction"]
            item["result"] = result
            completed_now.append(result)
        except Exception as exc:
            item["last_error"] = repr(exc)
            item["traceback"] = traceback.format_exc()[-8000:]
            item["status"] = ("failed" if item["attempts"] >= int(item.get("max_attempts", 3))
                              else "interrupted")
            item["failed_at"] = _now()
            checkpointing.atomic_json_dump(manifest, manifest_path)
            if stop_on_error:
                raise
        checkpointing.atomic_json_dump(manifest, manifest_path)
    return completed_now
