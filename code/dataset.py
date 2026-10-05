"""DeepWeeds data loading, split validation, transforms, and DataLoaders."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 9
EXPECTED_IMAGES = 17_509
REQUIRED_COLUMNS = {"Filename", "Label", "Species"}
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file split: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột: {sorted(missing)}")
    if df["Filename"].isna().any() or df["Label"].isna().any():
        raise ValueError(f"{path} có Filename/Label rỗng")
    labels = pd.to_numeric(df["Label"], errors="raise").astype(int)
    if not labels.between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"{path}: Label phải nằm trong [0, {NUM_CLASSES - 1}]")
    result = df.copy()
    result["Filename"] = result["Filename"].astype(str)
    result["Label"] = labels
    return result


def load_split(labels_dir: str | Path, fold: int = 0):
    """Load the author's unmodified train/val/test CSV files for one fold."""
    if not isinstance(fold, int) or not 0 <= fold <= 4:
        raise ValueError("fold phải là số nguyên từ 0 đến 4")
    root = Path(labels_dir)
    return tuple(_read_csv(root / f"{split}_subset{fold}.csv") for split in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Validate S1-S6 invariants and return report-ready counts."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names: dict[str, set[str]] = {}
    per_class: dict[str, dict[int, int]] = {}
    n: dict[str, int] = {}

    for split, df in frames.items():
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{split} thiếu cột: {sorted(missing)}")
        if df["Filename"].duplicated().any():
            dup = df.loc[df["Filename"].duplicated(), "Filename"].iloc[0]
            raise ValueError(f"{split} có Filename trùng: {dup}")
        labels = pd.to_numeric(df["Label"], errors="raise").astype(int)
        if not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{split}: Label phải nằm trong [0, {NUM_CLASSES - 1}]")
        names[split] = set(df["Filename"].astype(str))
        n[split] = len(df)
        counts = labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0).sort_index()
        per_class[split] = {int(k): int(v) for k, v in counts.items()}

    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    bad_overlap = {k: v[:10] for k, v in overlap.items() if v}
    if bad_overlap:
        raise ValueError(f"Các split bị giao nhau theo Filename: {bad_overlap}")

    union = names["train"] | names["val"] | names["test"]
    if len(union) != EXPECTED_IMAGES:
        raise ValueError(f"Hợp ba split có {len(union)} ảnh, phải là {EXPECTED_IMAGES}")

    ratios = {split: count / EXPECTED_IMAGES for split, count in n.items()}
    expected_ratios = {"train": 0.60, "val": 0.20, "test": 0.20}
    off_ratio = {s: ratios[s] for s in ratios if abs(ratios[s] - expected_ratios[s]) > 0.01}
    if off_ratio:
        raise ValueError(f"Tỉ lệ split lệch hơn 1 điểm phần trăm: {off_ratio}")

    image_root = Path(images_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_root}")
    missing_files = [name for name in sorted(union) if not (image_root / name).is_file()]
    if missing_files:
        preview = missing_files[:10]
        raise FileNotFoundError(f"Thiếu {len(missing_files)} ảnh trong {image_root}; ví dụ: {preview}")

    report = {
        "n": n,
        "ratios": ratios,
        "per_class": per_class,
        "overlap": {k: len(v) for k, v in overlap.items()},
        "union": len(union),
        "missing_files": 0,
    }
    print(pd.DataFrame(per_class).rename_axis("Label"))
    print(f"Số ảnh: {n}; giao: {report['overlap']}; hợp: {len(union)}")
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build deterministic evaluation or configurable training transforms."""
    if img_size <= 0:
        raise ValueError("img_size phải dương")
    aug = aug.lower()
    allowed = {"basic", "color", "trivial", "randaug"}
    if aug not in allowed:
        raise ValueError(f"aug phải thuộc {sorted(allowed)}, nhận được {aug!r}")

    normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        resize_size = int(round(img_size / 0.875))
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            *normalize,
        ])

    operations = [
        transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if aug == "color":
        operations.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.1))
    elif aug == "trivial":
        operations.append(transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR))
    elif aug == "randaug":
        operations.append(transforms.RandAugment(interpolation=InterpolationMode.BILINEAR))
    return transforms.Compose([*operations, *normalize])


class DeepWeedsDataset(Dataset):
    """Dataset returning ``(image_tensor, integer_label, filename)``."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame thiếu cột: {sorted(missing)}")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = self.images_dir / filename
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Không đọc được ảnh {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Create a reproducible DataLoader with optional inverse-frequency sampling."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size phải dương và num_workers không âm")
    if sampler not in {None, "balanced"}:
        raise ValueError("sampler chỉ nhận None hoặc 'balanced'")

    dataset = DeepWeedsDataset(df, images_dir, transform)
    sampler_obj = None
    if sampler == "balanced":
        if not train:
            raise ValueError("Balanced sampler chỉ được dùng cho train")
        counts = df["Label"].value_counts()
        sample_weights = df["Label"].map(lambda label: 1.0 / counts.loc[label]).to_numpy()
        sampler_obj = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True
        )

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train and sampler_obj is None),
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train and len(dataset) >= batch_size),
        worker_init_fn=_seed_worker,
        generator=generator,
        # Recreate workers each epoch so their seeds derive only from the saved generator state;
        # this makes Colab resume reproducible instead of depending on unsaved worker RNG state.
        persistent_workers=False,
    )
