"""Repeated compute-only benchmarking with explicit warm-up."""

from __future__ import annotations

import statistics
import time
from typing import Callable


def benchmark_callable(callable_: Callable, sample_count: int, warmup_runs: int = 1, repeats: int = 5, synchronize=None):
    if sample_count <= 0 or repeats <= 0 or warmup_runs < 0:
        raise ValueError("sample_count/repeats must be positive and warmup_runs non-negative")
    output = None
    for _ in range(warmup_runs):
        output = callable_()
    durations = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        output = callable_()
        if synchronize is not None:
            synchronize()
        durations.append(time.perf_counter() - started)
    mean = statistics.fmean(durations)
    return output, {
        "timing_scope": "model_compute_only; data loading and preprocessing excluded",
        "warmup_runs": int(warmup_runs),
        "timed_repeats": int(repeats),
        "inference_seconds_mean": mean,
        "inference_seconds_median": statistics.median(durations),
        "inference_seconds_std": statistics.pstdev(durations),
        "seconds_per_image": mean / sample_count,
        "throughput_images_per_second": sample_count / mean if mean else None,
    }

