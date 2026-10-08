"""Reproducible inference latency measurement."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Measure a zero-argument callable and return latency percentiles in milliseconds."""
    if warmup < 0 or iters < 50:
        raise ValueError("warmup phải không âm và iters phải >= 50")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    timings = np.empty(iters, dtype=np.float64)
    for index in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        timings[index] = (time.perf_counter() - start) * 1000.0
    return {
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
        "p99": float(np.percentile(timings, 99)),
        "mean": float(timings.mean()),
        "n": int(iters),
    }


def _prepare(model, batch_size, img_size, dtype, device):
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng")
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype phải là fp32, amp hoặc fp16")
    if batch_size <= 0 or img_size <= 0:
        raise ValueError("batch_size và img_size phải dương")

    measured_model = copy.deepcopy(model).to(device_obj).eval()
    input_dtype = torch.float32
    if dtype == "fp16":
        if device_obj.type != "cuda":
            raise ValueError("fp16 benchmark chỉ hỗ trợ CUDA")
        measured_model.half()
        input_dtype = torch.float16
    batch = torch.randn(batch_size, 3, img_size, img_size, device=device_obj, dtype=input_dtype)
    return measured_model, batch, device_obj


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Measure model-only forward latency under a declared device/dtype condition."""
    measured_model, batch, device_obj = _prepare(model, batch_size, img_size, dtype, device)
    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None

    def forward():
        with torch.inference_mode():
            with torch.autocast(device_type=device_obj.type,
                                enabled=(dtype == "amp" and device_obj.type == "cuda")):
                measured_model(batch)

    stats = bench(forward, warmup=warmup, iters=iters, sync=sync)
    gpu = torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU"
    return {
        "gpu": gpu,
        "dtype": dtype,
        "batch": int(batch_size),
        "img_size": int(img_size),
        **stats,
        "images_per_s": float(batch_size / (stats["p50"] / 1000.0)),
        "torch": torch.__version__,
        "includes_preprocessing": False,
        "warmup": int(warmup),
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure K real forward passes and compare them with a measured single view."""
    if k_views < 1:
        raise ValueError("k_views phải >= 1")
    batch_size = int(kw.pop("batch_size", 1))
    img_size = int(kw.pop("img_size", 224))
    dtype = kw.pop("dtype", "fp32")
    device = kw.pop("device", "cuda")
    warmup = int(kw.pop("warmup", 10))
    iters = int(kw.pop("iters", 100))
    if kw:
        raise TypeError(f"Tham số không hỗ trợ: {sorted(kw)}")

    measured_model, batch, device_obj = _prepare(model, batch_size, img_size, dtype, device)
    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None

    def once():
        with torch.inference_mode():
            with torch.autocast(device_type=device_obj.type,
                                enabled=(dtype == "amp" and device_obj.type == "cuda")):
                measured_model(batch)

    single = bench(once, warmup=warmup, iters=iters, sync=sync)

    def k_forward():
        for _ in range(k_views):
            once()

    multi = bench(k_forward, warmup=warmup, iters=iters, sync=sync)
    gpu = torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU"
    return {
        "gpu": gpu,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "k_views": int(k_views),
        **multi,
        "single_p50": single["p50"],
        "expected_k_x_p50": float(k_views * single["p50"]),
        "relative_cost": float(multi["p50"] / single["p50"]),
        "images_per_s": float(batch_size / (multi["p50"] / 1000.0)),
        "torch": torch.__version__,
        "includes_preprocessing": False,
        "warmup": warmup,
    }
