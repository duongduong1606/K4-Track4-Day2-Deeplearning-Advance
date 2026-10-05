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
    from .experiment_plan import BACKBONES, FINAL_SEEDS, TRAINING
    from .train import Config, persistent_pred_path, persistent_run_dir, run
except ImportError:
    import checkpointing
    from experiment_plan import BACKBONES, FINAL_SEEDS, TRAINING
    from train import Config, persistent_pred_path, persistent_run_dir, run


IMAGES_MD5 = "b7b30f96d466fba86016aa5a26606e0f"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_hash(runs: list[dict]) -> str:
    frozen = [
        {"run_id": item["run_id"], "config": item["config"],
         "max_attempts": item.get("max_attempts", 3)}
        for item in runs
    ]
    payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def md5_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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
    images_dir = local_data / "images"
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

    marker = images_dir / ".extracted_ok"
    if not marker.exists():
        if images_dir.exists():
            shutil.rmtree(images_dir)
        with zipfile.ZipFile(local_zip) as archive:
            extraction_root = local_data.resolve()
            for member in archive.infolist():
                destination = (local_data / member.filename).resolve()
                try:
                    destination.relative_to(extraction_root)
                except ValueError as exc:
                    raise ValueError(f"ZIP chứa đường dẫn không an toàn: {member.filename}") from exc
            archive.extractall(local_data)
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Sau giải nén không có {images_dir}; kiểm tra cấu trúc bên trong images.zip"
            )
        image_count = sum(1 for path in images_dir.iterdir() if path.is_file() and not path.name.startswith("."))
        if image_count != 17_509:
            raise ValueError(f"Số ảnh sau giải nén là {image_count}, phải là 17.509")
        marker.write_text(json.dumps({"md5": checksum, "images": image_count}), encoding="utf-8")
    else:
        image_count = sum(1 for path in images_dir.iterdir() if path.is_file() and not path.name.startswith("."))

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


def create_final_manifest(path: str | Path, best_config: Config, baseline_config: Config | None = None,
                          overwrite: bool = False) -> Path:
    """Create a LOCKED three-seed baseline/final queue selected from validation."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Final manifest đã tồn tại: {destination}")
    baseline_config = baseline_config or Config(exp_id="T00")
    configs = []
    for seed in FINAL_SEEDS:
        configs.append(Config(**{**asdict(baseline_config), "exp_id": "T00", "seed": seed}))
        configs.append(Config(**{**asdict(best_config), "exp_id": "F01", "seed": seed}))
    manifest = {
        "version": 1,
        "kind": "final",
        "test_access": True,
        "locked": True,
        "selection_source": "validation_only",
        "selection_note": "",
        "created_at": _now(),
        "runs": [_manifest_item(cfg) for cfg in configs],
    }
    return checkpointing.atomic_json_dump(manifest, destination)


def unlock_final_manifest(path: str | Path, selection_note: str) -> None:
    """Explicitly unlock final test after documenting the validation-only decision."""
    if not selection_note.strip():
        raise ValueError("Phải ghi selection_note giải thích lựa chọn dựa trên validation")
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "final" or not manifest.get("test_access"):
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
