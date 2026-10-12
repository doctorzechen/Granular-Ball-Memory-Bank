"""Construction protocol, synchronized timing, and repetition summaries."""
import time

import numpy as np
import torch
from baselines.medoid import MEDOID_BATCH, medoid_indices

SEED = 0
MIN_BALL_SIZE = 64
GB_THRESHOLD = 1.3
TIMING_WARMUP = 1
TIMING_REPEATS = 5
# Retained only as compatibility fields in archived configuration hashes.
STRESS_CATEGORY = "hazelnut"
STRESS_SCALES = (0.25, 0.50, 0.75, 1.00)

def summarize_repeats(durations) -> dict:
    """mean/median/std/min/max over timed repetitions (median is the
    headline statistic per protocol)."""
    durations = np.asarray(durations, dtype=np.float64)
    if durations.ndim != 1 or len(durations) == 0:
        raise ValueError("Durations must be a nonempty vector.")
    if not np.all(np.isfinite(durations)) or np.any(durations < 0):
        raise ValueError("Durations must be finite and nonnegative.")
    return {
        "n_repeats": int(len(durations)),
        "mean": float(np.mean(durations)),
        "median": float(np.median(durations)),
        "std": float(np.std(durations)),
        "min": float(np.min(durations)),
        "max": float(np.max(durations)),
    }


class ConstructionTimer:
    """CUDA-synchronized wall-clock timer with peak-memory capture."""

    def __init__(self, device: torch.device):
        self.device = device

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.started
        self.peak_gpu_allocated_bytes = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda" else 0)
        return False
