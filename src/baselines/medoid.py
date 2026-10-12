"""Per-ball nearest-center real-patch representative control.

This retains the frozen experimental function unchanged; it is an established
representative-selection control, not a new GB scoring rule.
"""

import numpy as np
import torch

MEDOID_BATCH = 4096


def medoid_indices(
    features: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    device: torch.device,
    batch_size: int = MEDOID_BATCH,
) -> np.ndarray:
    """Per-ball medoid: the real patch closest to the ball center.

    Deterministic (ties break toward the lowest row index). Balls are
    disjoint, so medoids are unique by construction; the contract is still
    asserted here.
    """
    features = np.asarray(features, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float32)
    if features.ndim != 2 or centers.ndim != 2:
        raise ValueError("Features and centers must be matrices.")
    if len(assignments) != len(features):
        raise ValueError("Assignment vector length mismatch.")
    if features.shape[1] != centers.shape[1]:
        raise ValueError("Feature and center dimensions differ.")
    number_of_balls = len(centers)
    if number_of_balls < 1:
        raise ValueError("At least one ball is required.")
    if len(features) and (
            assignments.min() < 0 or assignments.max() >= number_of_balls):
        raise ValueError("Assignments reference an invalid ball.")
    with torch.no_grad():
        feats = torch.from_numpy(features).to(device)
        cents = torch.from_numpy(centers).to(device)
        assign = torch.from_numpy(assignments).to(device)
        medoids = np.empty(number_of_balls, dtype=np.int64)
        for ball_id in range(number_of_balls):
            rows = torch.nonzero(assign == ball_id).reshape(-1)
            if len(rows) == 0:
                raise ValueError("Ball {} is empty.".format(ball_id))
            best_pos, best_dist = None, None
            for start in range(0, len(rows), batch_size):
                chunk = rows[start: start + batch_size]
                distances = torch.linalg.norm(
                    feats[chunk] - cents[ball_id], dim=1)
                value, position = torch.min(distances, dim=0)
                if best_dist is None or float(value) < float(best_dist):
                    best_dist = float(value)
                    best_pos = int(chunk[int(position)])
            medoids[ball_id] = best_pos
    if len(np.unique(medoids)) != number_of_balls:
        raise RuntimeError("Medoid indices are not unique (impossible for "
                           "disjoint balls).")
    return medoids

