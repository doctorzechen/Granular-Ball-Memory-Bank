"""Exact global farthest-first control, extracted without algorithm changes.

Farthest-first traversal is an established baseline, not a proposed GB method.
The implementation comes from this project\'s frozen experimental code.
"""

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch


def global_farthest_first_order(
    memory_bank: np.ndarray,
    count: int,
    projection_dimension: int,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Scalable global max-min ordering without a Memory-Bank pairwise matrix."""
    memory_bank = np.asarray(memory_bank, dtype=np.float32)
    if count < 1 or count > len(memory_bank):
        raise ValueError("Invalid global farthest-first count.")
    projection_dimension = min(projection_dimension, memory_bank.shape[1])
    use_raw_space = projection_dimension == memory_bank.shape[1]
    generator = np.random.default_rng(seed)
    start_time = time.perf_counter()
    with torch.no_grad():
        data = torch.from_numpy(
            np.array(memory_bank, dtype=np.float32, order="C", copy=True)
        ).to(device)
        if use_raw_space:
            reduced = data
        else:
            projection = generator.normal(
                0.0, 1.0 / math.sqrt(projection_dimension),
                size=(memory_bank.shape[1], projection_dimension),
            ).astype(np.float32)
            projection_tensor = torch.from_numpy(projection).to(device)
            reduced = data @ projection_tensor
        norms = torch.sum(reduced * reduced, dim=1)
        center = torch.mean(reduced, dim=0)
        first_distances = norms + torch.sum(center * center) - 2.0 * (reduced @ center)
        current = int(torch.argmin(first_distances).item())
        order = np.empty(count, dtype=np.int64)
        chosen = torch.zeros(len(memory_bank), dtype=torch.bool, device=device)
        closest = torch.full((len(memory_bank),), torch.inf, device=device)
        for position in range(count):
            order[position] = current
            chosen[current] = True
            candidate = reduced[current]
            new_distance = norms + torch.sum(candidate * candidate) - 2.0 * (reduced @ candidate)
            closest = torch.minimum(closest, new_distance.clamp_min_(0.0))
            closest[chosen] = -1.0
            if position + 1 < count:
                current = int(torch.argmax(closest).item())
    report = {
        "seconds": time.perf_counter() - start_time,
        "selected_count": int(count),
        "projection_dimension": int(projection_dimension),
        "projection_seed": int(seed),
        "selection_space": (
            "exact Raw {}D Original MB".format(memory_bank.shape[1])
            if use_raw_space
            else "deterministic Gaussian projection of Original MB"
        ),
        "coverage_evaluation_space": "exact Raw 1024D",
        "full_memory_pairwise_matrix_created": False,
    }
    return order, report

