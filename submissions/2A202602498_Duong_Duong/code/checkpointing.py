"""Crash-safe local writes and verified mirroring to persistent storage."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path

import torch


_PROFILE_STATE_NAMES = {"total_ops", "total_params"}


def clean_model_state_dict(state_dict):
    """Return a state dict without temporary buffers installed by profilers such as THOP."""
    cleaned = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key.rsplit(".", 1)[-1] not in _PROFILE_STATE_NAMES
    )
    if hasattr(state_dict, "_metadata"):
        cleaned._metadata = state_dict._metadata
    return cleaned


def load_model_state_dict(model, state_dict, strict: bool = True):
    """Load model weights while accepting checkpoints polluted by profiling helpers."""
    return model.load_state_dict(clean_model_state_dict(state_dict), strict=strict)


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(value, path: str | Path) -> Path:
    """Write JSON beside its destination, then replace the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(state, path: str | Path, verify: bool = True) -> Path:
    """Save a checkpoint atomically and optionally load it back before returning."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    torch.save(state, temporary)
    if verify:
        torch_load(temporary, map_location="cpu")
    os.replace(temporary, destination)
    return destination


def atomic_copy(source: str | Path, destination: str | Path,
                verify_hash: bool = True) -> str:
    """Copy through a temporary destination and return the destination SHA-256."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.uploading")
    shutil.copy2(source, temporary)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(temporary)
    if verify_hash and source_hash != destination_hash:
        temporary.unlink(missing_ok=True)
        raise IOError(f"Checksum không khớp khi copy {source} -> {destination}")
    os.replace(temporary, destination)
    return destination_hash


def mirror_files(mappings: list[tuple[str | Path, str | Path]], verify_hash: bool = True) -> dict:
    """Mirror existing files and return hashes keyed by persistent destination."""
    hashes = {}
    for source, destination in mappings:
        source_path = Path(source)
        if source_path.exists():
            hashes[str(destination)] = atomic_copy(source_path, destination, verify_hash=verify_hash)
    return hashes
