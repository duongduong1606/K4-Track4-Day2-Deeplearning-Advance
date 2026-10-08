"""Validate final artifacts, run eval.py, and assemble a submission from real results."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import eval as evaluator  # noqa: E402

try:
    from . import checkpointing, reporting
except ImportError:
    import checkpointing
    import reporting


FINAL_SEEDS = (42, 123, 2026)


def _required_predictions(pred_dir: Path) -> list[Path]:
    return [pred_dir / f"{exp_id}_seed{seed}_test.csv"
            for exp_id in ("T00", "F01") for seed in FINAL_SEEDS]


def validate_final_artifacts(drive_root: str | Path, final_test_manifest: str | Path) -> dict:
    """Fail closed unless all six immutable final predictions and run artifacts exist."""
    drive_root, manifest_path = Path(drive_root), Path(final_test_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "final_test" or manifest.get("locked", True):
        raise ValueError("Final test manifest chưa hợp lệ hoặc vẫn bị khóa")
    incomplete = [item["run_id"] for item in manifest["runs"] if item.get("status") != "completed"]
    if incomplete:
        raise RuntimeError(f"Final test queue chưa hoàn tất: {incomplete}")
    pred_dir = drive_root / "artifacts" / "predictions"
    missing_predictions = [str(path) for path in _required_predictions(pred_dir) if not path.is_file()]
    if missing_predictions:
        raise FileNotFoundError(f"Thiếu prediction final: {missing_predictions}")
    missing_runs = []
    for exp_id in ("T00", "F01"):
        for seed in FINAL_SEEDS:
            run_root = drive_root / "artifacts" / "runs" / exp_id / f"seed{seed}"
            for filename in ("config.json", "history.csv", "best.pt", "summary.json", "status.json"):
                if not (run_root / filename).is_file():
                    missing_runs.append(str(run_root / filename))
    if missing_runs:
        raise FileNotFoundError(f"Thiếu run artifact: {missing_runs}")
    return {"predictions": 6, "runs": 6, "manifest": str(manifest_path)}


def _run_eval(repo_root: Path, args: list[str], output_file: Path) -> str:
    command = [sys.executable, str(repo_root / "eval.py"), *args]
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    text = completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"eval.py thất bại ({completed.returncode}); xem {output_file}\n{text}")
    return text


def _plot_confusion(group: evaluator.Group, names: list[str], output: Path) -> None:
    confusion = np.sum([metrics["confusion"] for metrics in group.metrics], axis=0)
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thật", title="F01 — confusion matrix cộng qua 3 seed")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt(summary, key: str) -> str:
    mean, std = summary[key]
    return f"{mean:.4f} ± {std:.4f}"


def _evidence_markdown(drive_root: Path) -> str:
    records = []
    for summary_path in sorted((drive_root / "artifacts" / "runs").glob("*/seed*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        records.append({**config, **summary})

    def table(prefix: str, columns: list[tuple[str, str]]) -> str:
        selected = [row for row in records if str(row.get("exp_id", "")).startswith(prefix)]
        selected.sort(key=lambda row: float(row.get("val_macro_f1", -1)), reverse=True)
        if not selected:
            return "Chưa có dữ liệu."
        header = "| " + " | ".join(label for _, label in columns) + " |"
        separator = "|" + "|".join("---" for _ in columns) + "|"
        body = []
        for row in selected:
            values = []
            for key, _ in columns:
                value = row.get(key, "")
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *body])

    backbone_table = table("B", [
        ("exp_id", "Exp"), ("backbone", "Backbone"), ("val_macro_f1", "Macro-F1 val"),
        ("val_top1", "Top-1 val"), ("params_m", "Params (M)"), ("gmac", "GMAC"),
    ])
    training_table = table("T", [
        ("exp_id", "Exp"), ("loss", "Loss"), ("aug", "Aug"), ("mix", "Mix"),
        ("val_macro_f1", "Macro-F1 val"), ("val_top1", "Top-1 val"),
    ])
    inference_path = drive_root / "artifacts" / "runs" / "inference_results.csv"
    if inference_path.exists():
        frame = __import__("pandas").read_csv(inference_path)
        complete = frame[frame["status"] == "completed"].sort_values("val_macro_f1", ascending=False)
        inference_lines = [
            "| Method | Resolution | K | Aggregation | Macro-F1 val | ECE val | p95 (ms) |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
        for row in complete.itertuples():
            inference_lines.append(
                f"| {row.method} | {row.img_size} | {row.k} | {row.aggregation} | "
                f"{row.val_macro_f1:.4f} | {row.val_ece:.4f} | {row.p95_ms:.2f} |"
            )
        inference_table = "\n".join(inference_lines)
    else:
        inference_table = "Chưa có dữ liệu inference."
    return f"""### So sánh backbone

{backbone_table}

### Ablation công thức huấn luyện

{training_table}

### Suy luận và đánh đổi latency

{inference_table}
"""


def _write_report(path: Path, final_group: evaluator.Group, baseline_group: evaluator.Group,
                  class_names: list[str], notebook_url: str, evidence_markdown: str) -> None:
    delta = final_group.summary["macro_f1"][0] - baseline_group.summary["macro_f1"][0]
    f1_mean, f1_std = final_group.summary["f1"]
    recall_mean, recall_std = final_group.summary["recall"]
    rows = [
        f"| {name} | {recall_mean[index]:.4f} ± {recall_std[index]:.4f} | "
        f"{f1_mean[index]:.4f} ± {f1_std[index]:.4f} |"
        for index, name in enumerate(class_names)
    ]
    text = f"""# Báo cáo DeepWeeds — Lab Day 2

## Tóm tắt

Cấu hình F01 đạt macro-F1 test **{_fmt(final_group.summary, 'macro_f1')}** và top-1 test
**{_fmt(final_group.summary, 'top1')}** qua 3 seed. So với mốc T00, macro-F1 thay đổi
**{delta:+.4f}**. Các số liệu được tính lại trực tiếp từ `predictions/` bằng `eval.py`.

## Dữ liệu và thiết lập

- DeepWeeds, fold 0 cố định; không gộp validation vào train.
- Validation dùng để chọn backbone, recipe, checkpoint, inference và temperature.
- Test chỉ chạy một lần cho từng seed sau khi cấu hình và inference spec đã đóng băng.
- Seed chung kết: {', '.join(map(str, FINAL_SEEDS))}.
- Notebook tái lập: {notebook_url}

## Kết quả chung kết

| Cấu hình | Macro-F1 test | Top-1 test | ECE test |
|---|---:|---:|---:|
| T00 | {_fmt(baseline_group.summary, 'macro_f1')} | {_fmt(baseline_group.summary, 'top1')} | {_fmt(baseline_group.summary, 'ece')} |
| F01 | {_fmt(final_group.summary, 'macro_f1')} | {_fmt(final_group.summary, 'top1')} | {_fmt(final_group.summary, 'ece')} |

## Kết quả thực nghiệm trên validation

{evidence_markdown}

## Chỉ số từng lớp của F01

| Lớp | Recall mean ± std | F1 mean ± std |
|---|---:|---:|
{chr(10).join(rows)}

![Confusion matrix](curves/F01_confusion_matrix.png)

## Phân tích và kết luận

Các bảng trên được sinh trực tiếp từ log chạy thật. Khi diễn giải, chỉ coi một cải thiện là rõ ràng
nếu chênh lệch lớn hơn nhiễu giữa seed. Các file dự đoán sai cần được xem trực quan, đặc biệt cặp
Chinee Apple ↔ Snake Weed, trước khi đưa ra giả thuyết sinh học hoặc điều kiện chụp ảnh.

## Hạn chế

Kết quả dùng ba seed nhưng chỉ một fold. DeepWeeds được chia ngẫu nhiên, không theo địa điểm, nên
điểm test có thể lạc quan khi triển khai tại vùng, mùa hoặc điều kiện ánh sáng mới. Cần đánh giá
thêm dưới lệch phân phối và trên phần cứng robot mục tiêu.
"""
    path.write_text(text, encoding="utf-8")


def _write_submission_readme(path: Path, notebook_url: str) -> None:
    packages = ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn",
                "Pillow", "matplotlib", "openpyxl", "thop")
    versions = []
    for package in packages:
        try:
            versions.append(f"- `{package}=={metadata.version(package)}`")
        except metadata.PackageNotFoundError:
            versions.append(f"- `{package}`: không có trong môi trường đóng gói")
    path.write_text(f"""# DeepWeeds Lab Day 2 submission

Notebook Colab: {notebook_url}

## Môi trường đã chạy

- `Python {platform.python_version()}`
{chr(10).join(versions)}

## Thứ tự tái lập

1. Cài `code/requirements.txt` và chuẩn bị DeepWeeds fold 0.
2. Chạy pipeline checks, screening queue và inference sweep chỉ trên validation.
3. Chạy final training với seed 42, 123, 2026.
4. Mở khóa final test manifest sau khi đóng băng inference spec.
5. Chạy `eval.py score`, `eval.py grade`, rồi tạo `results.xlsx` và báo cáo.

Chi tiết cấu hình từng run nằm trong `results.xlsx`, `code/experiment_plan.py` và prediction CSV.
""", encoding="utf-8")


def assemble_submission(drive_root: str | Path, final_test_manifest: str | Path,
                        repo_root: str | Path, output_dir: str | Path,
                        notebook_url: str) -> Path:
    """Run official evaluation and create the six required deliverables."""
    if not notebook_url.startswith("http"):
        raise ValueError("notebook_url phải là link Colab/Kaggle hợp lệ")
    drive_root, repo_root, output = Path(drive_root), Path(repo_root), Path(output_dir)
    validate_final_artifacts(drive_root, final_test_manifest)
    pred_dir = drive_root / "artifacts" / "predictions"
    labels_dir = drive_root / "data" / "labels"
    test_csv, labels_csv = labels_dir / "test_subset0.csv", labels_dir / "labels.csv"
    eval_out = drive_root / "artifacts" / "eval_out"

    final_pattern = str(pred_dir / "F01_seed*_test.csv")
    baseline_pattern = str(pred_dir / "T00_seed*_test.csv")
    _run_eval(repo_root, ["score", "--pred", final_pattern, "--test-csv", str(test_csv),
                          "--labels", str(labels_csv), "--tag", "F01", "--out", str(eval_out)],
              eval_out / "F01_score.txt")
    _run_eval(repo_root, ["score", "--pred", baseline_pattern, "--test-csv", str(test_csv),
                          "--labels", str(labels_csv), "--tag", "T00", "--out", str(eval_out)],
              eval_out / "T00_score.txt")
    grade_args = ["grade", "--final", final_pattern, "--baseline", baseline_pattern,
                  "--test-csv", str(test_csv), "--labels", str(labels_csv)]
    uncal_files = sorted(pred_dir.glob("F01_seed*_test_uncal.csv"))
    val_files = sorted(pred_dir.glob("F01_seed*_val.csv"))
    if len(uncal_files) == len(FINAL_SEEDS):
        grade_args.extend(["--uncal", str(pred_dir / "F01_seed*_test_uncal.csv")])
    if len(val_files) == len(FINAL_SEEDS):
        grade_args.extend(["--final-val", str(pred_dir / "F01_seed*_val.csv"),
                           "--val-csv", str(labels_dir / "val_subset0.csv")])
    test_manifest = json.loads(Path(final_test_manifest).read_text(encoding="utf-8"))
    final_item = next(item for item in test_manifest["runs"] if item["exp_id"] == "F01")
    spec_payload = json.loads(Path(final_item["spec_path"]).read_text(encoding="utf-8"))
    p95 = spec_payload.get("evidence", {}).get("p95_ms")
    if p95 is not None:
        grade_args.extend(["--latency-p95-ms", str(p95), "--latency-method", "proper"])
    grade_args.extend(["--out", str(eval_out)])
    _run_eval(repo_root, grade_args, eval_out / "grade.txt")

    workbook = reporting.build_results_workbook(
        drive_root / "results.xlsx", drive_root / "artifacts" / "runs", pred_dir
    )
    final_group = evaluator.load_group(final_pattern, str(test_csv))
    baseline_group = evaluator.load_group(baseline_pattern, str(test_csv))
    class_names = evaluator.load_names(str(labels_csv))

    if output.exists():
        raise FileExistsError(f"Thư mục bài nộp đã tồn tại: {output}")
    output.mkdir(parents=True)
    shutil.copy2(workbook, output / "results.xlsx")
    shutil.copy2(repo_root / "eval.py", output / "eval.py")
    shutil.copytree(repo_root / "code", output / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))
    shutil.copytree(drive_root / "artifacts" / "curves", output / "curves")
    shutil.copytree(pred_dir, output / "predictions")
    shutil.copytree(eval_out, output / "eval_out")
    error_frames = []
    for prediction_path in sorted(pred_dir.glob("F01_seed*_test.csv")):
        frame = pd.read_csv(prediction_path)
        errors = frame[frame["y_true"] != frame["y_pred"]].copy()
        errors.insert(0, "source_file", prediction_path.name)
        errors["true_class"] = errors["y_true"].map(dict(enumerate(class_names)))
        errors["pred_class"] = errors["y_pred"].map(dict(enumerate(class_names)))
        error_frames.append(errors)
    if error_frames:
        pd.concat(error_frames, ignore_index=True).to_csv(output / "error_cases.csv", index=False)
    _plot_confusion(final_group, class_names, output / "curves" / "F01_confusion_matrix.png")
    _write_report(
        output / "report.md", final_group, baseline_group, class_names, notebook_url,
        _evidence_markdown(drive_root),
    )
    _write_submission_readme(output / "README.md", notebook_url)
    checkpointing.atomic_json_dump({
        "created_from": str(drive_root),
        "final_predictions": [checkpointing.sha256_file(path) for path in _required_predictions(pred_dir)],
        "files": sum(1 for path in output.rglob("*") if path.is_file()),
    }, output / "submission_manifest.json")
    return output
