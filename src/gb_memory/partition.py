"""Memory-safe granular-ball generation for offline feature analysis."""

import math
import time
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch


@dataclass
class GranularBallResult:
    """Hard granular-ball partition and its geometric summaries."""

    assignments: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    mean_radii: np.ndarray
    max_radii: np.ndarray
    representative_indices: np.ndarray
    point_center_distances: np.ndarray
    point_representative_distances: np.ndarray
    members: List[np.ndarray]
    build_seconds: float
    configuration: Dict[str, object]
    split_statistics: Dict[str, object]
    peak_gpu_allocated_bytes: Optional[int]
    peak_gpu_reserved_bytes: Optional[int]


def _rows(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Copy only selected feature rows into a contiguous float32 block."""
    return np.array(features[indices], dtype=np.float32, order="C", copy=True)


def _squared_distances(
    samples: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    """Return a bounded ``[batch, centers]`` squared Euclidean matrix."""
    sample_norms = torch.sum(samples * samples, dim=1, keepdim=True)
    center_norms = torch.sum(centers * centers, dim=1).unsqueeze(0)
    distances = sample_norms + center_norms
    distances.addmm_(samples, centers.T, beta=1.0, alpha=-2.0)
    return distances.clamp_min_(0.0)


def _kmeans_plus_plus(
    samples: torch.Tensor,
    number_of_centers: int,
    random_generator: np.random.Generator,
) -> torch.Tensor:
    """Initialize centers with seeded k-means++ without a sample-pair matrix."""
    number_of_samples = len(samples)
    first_index = int(random_generator.integers(number_of_samples))
    chosen = np.zeros(number_of_samples, dtype=bool)
    chosen[first_index] = True
    center_indices = [first_index]

    sample_norms = torch.sum(samples * samples, dim=1)
    first_center = samples[first_index]
    closest = sample_norms + torch.sum(first_center * first_center)
    closest.addmv_(samples, first_center, beta=1.0, alpha=-2.0)
    closest.clamp_min_(0.0)

    for _ in range(1, number_of_centers):
        probabilities = closest.detach().cpu().numpy().astype(np.float64)
        probabilities[chosen] = 0.0
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0 or not np.isfinite(probability_sum):
            next_index = int(np.flatnonzero(~chosen)[0])
        else:
            probabilities /= probability_sum
            next_index = int(
                random_generator.choice(number_of_samples, p=probabilities)
            )
        chosen[next_index] = True
        center_indices.append(next_index)

        center = samples[next_index]
        distances = sample_norms + torch.sum(center * center)
        distances.addmv_(samples, center, beta=1.0, alpha=-2.0)
        distances.clamp_min_(0.0)
        closest = torch.minimum(closest, distances)

    return samples[center_indices].clone()


def _fit_coarse_centers(
    features: np.ndarray,
    number_of_centers: int,
    fit_sample_size: int,
    iterations: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, int]:
    """Fit seeded coarse k-means centers on a bounded training-only sample."""
    random_generator = np.random.default_rng(seed)
    fit_sample_size = min(fit_sample_size, len(features))
    sample_indices = np.sort(
        random_generator.choice(len(features), fit_sample_size, replace=False)
    )
    sample_array = _rows(features, sample_indices)
    sample_tensor = torch.from_numpy(sample_array).to(device)
    number_of_centers = min(number_of_centers, fit_sample_size)

    with torch.no_grad():
        centers = _kmeans_plus_plus(
            sample_tensor, number_of_centers, random_generator
        )
        for _ in range(iterations):
            center_sums = torch.zeros_like(centers)
            center_counts = torch.zeros(
                number_of_centers, dtype=torch.float32, device=device
            )
            for start in range(0, fit_sample_size, batch_size):
                batch = sample_tensor[start : start + batch_size]
                labels = torch.argmin(
                    _squared_distances(batch, centers), dim=1
                )
                center_sums.index_add_(0, labels, batch)
                center_counts.index_add_(
                    0, labels, torch.ones_like(labels, dtype=torch.float32)
                )
            nonempty = center_counts > 0
            new_centers = centers.clone()
            new_centers[nonempty] = (
                center_sums[nonempty] / center_counts[nonempty].unsqueeze(1)
            )
            center_shift = torch.max(
                torch.linalg.vector_norm(new_centers - centers, dim=1)
            )
            centers = new_centers
            if float(center_shift) <= 1e-5:
                break

    return centers.cpu().numpy().astype(np.float32), int(fit_sample_size)


def _assign_to_centers(
    features: np.ndarray,
    centers: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Assign every row to its nearest center using bounded GPU/CPU batches."""
    assignments = np.empty(len(features), dtype=np.int32)
    center_tensor = torch.from_numpy(np.ascontiguousarray(centers)).to(device)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch_array = np.array(
                features[start:end], dtype=np.float32, order="C", copy=True
            )
            batch = torch.from_numpy(batch_array).to(device)
            assignments[start:end] = (
                torch.argmin(_squared_distances(batch, center_tensor), dim=1)
                .cpu()
                .numpy()
                .astype(np.int32)
            )
    return assignments


def _members_from_assignments(
    assignments: np.ndarray, number_of_centers: int
) -> List[np.ndarray]:
    """Convert flat assignments to nonempty, disjoint index arrays."""
    order = np.argsort(assignments, kind="stable")
    counts = np.bincount(assignments, minlength=number_of_centers)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return [
        np.ascontiguousarray(order[offsets[i] : offsets[i + 1]])
        for i in range(number_of_centers)
        if counts[i] > 0
    ]


def center_consistency_log(
    ball_features: np.ndarray, strong: bool = False
) -> float:
    """Evaluate the paper/code density-ratio criterion in the log domain."""
    if len(ball_features) == 0:
        raise ValueError("A granular ball must not be empty.")
    center = np.mean(ball_features, axis=0, dtype=np.float64)
    distances = np.linalg.norm(ball_features - center, axis=1)
    max_radius = float(np.max(distances))
    if max_radius <= 1e-12:
        return 0.0

    if strong:
        inner_count = int(np.count_nonzero(distances <= max_radius / 4.0))
        if inner_count == 0:
            return -math.inf
        radius_ratio_log = math.log(4.0)
    else:
        mean_radius = float(np.mean(distances))
        if mean_radius <= 1e-12:
            return 0.0
        inner_count = int(np.count_nonzero(distances <= mean_radius))
        if inner_count == 0:
            return -math.inf
        radius_ratio_log = math.log(max_radius / mean_radius)

    return (
        math.log(inner_count / len(ball_features))
        + ball_features.shape[1] * radius_ratio_log
    )


def _two_means_split(
    features: np.ndarray,
    indices: np.ndarray,
    iterations: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Deterministically split a ball with two data-driven centers."""
    ball = _rows(features, indices)
    center = np.mean(ball, axis=0, dtype=np.float64).astype(np.float32)
    first = int(np.argmax(np.linalg.norm(ball - center, axis=1)))
    second = int(np.argmax(np.linalg.norm(ball - ball[first], axis=1)))
    if first == second or np.array_equal(ball[first], ball[second]):
        return None

    centers = np.stack((ball[first], ball[second])).astype(np.float32)
    labels = None
    for _ in range(iterations):
        squared = (
            np.sum(ball * ball, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * ball @ centers.T
        )
        new_labels = np.argmin(squared, axis=1)
        if np.all(new_labels == 0) or np.all(new_labels == 1):
            return None
        if labels is not None and np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
        centers[0] = np.mean(ball[labels == 0], axis=0, dtype=np.float64)
        centers[1] = np.mean(ball[labels == 1], axis=0, dtype=np.float64)

    return (
        np.ascontiguousarray(indices[labels == 0]),
        np.ascontiguousarray(indices[labels == 1]),
    )


def _try_split(
    features: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    min_ball_size: int,
    two_means_iterations: int,
    strong: bool,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str, float]:
    """Apply the log-domain consistency decision and the size guard."""
    if len(indices) < 2 * min_ball_size:
        return None, "minimum_size", math.nan

    ball = _rows(features, indices)
    consistency_log = center_consistency_log(ball, strong=strong)
    threshold_log = math.log(threshold)
    should_split = consistency_log >= threshold_log or consistency_log < 0.0
    if not should_split:
        return None, "consistent", consistency_log

    children = _two_means_split(features, indices, two_means_iterations)
    if children is None:
        return None, "unsplittable", consistency_log
    if min(len(children[0]), len(children[1])) < min_ball_size:
        return None, "minimum_child_size", consistency_log
    return children, "accepted", consistency_log


def _split_initial_balls(
    features: np.ndarray,
    initial_members: Sequence[np.ndarray],
    threshold: float,
    min_ball_size: int,
    two_means_iterations: int,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Run the author's strong first pass followed by regular splitting."""
    regular_queue = []
    reason_counts: Dict[str, int] = {}
    accepted_strong = 0
    consistency_logs = []

    for indices in initial_members:
        children, reason, value = _try_split(
            features,
            indices,
            threshold,
            min_ball_size,
            two_means_iterations,
            strong=True,
        )
        reason_counts["strong_" + reason] = (
            reason_counts.get("strong_" + reason, 0) + 1
        )
        if np.isfinite(value):
            consistency_logs.append(value)
        if children is None:
            regular_queue.append(indices)
        else:
            accepted_strong += 1
            regular_queue.extend(children)

    final_members = []
    accepted_regular = 0
    while regular_queue:
        indices = regular_queue.pop()
        children, reason, value = _try_split(
            features,
            indices,
            threshold,
            min_ball_size,
            two_means_iterations,
            strong=False,
        )
        reason_counts["regular_" + reason] = (
            reason_counts.get("regular_" + reason, 0) + 1
        )
        if np.isfinite(value):
            consistency_logs.append(value)
        if children is None:
            final_members.append(indices)
        else:
            accepted_regular += 1
            regular_queue.extend(children)

    return final_members, {
        "accepted_strong_splits": accepted_strong,
        "accepted_regular_splits": accepted_regular,
        "reason_counts": reason_counts,
        "finite_log_consistency_min": (
            float(np.min(consistency_logs)) if consistency_logs else None
        ),
        "finite_log_consistency_median": (
            float(np.median(consistency_logs)) if consistency_logs else None
        ),
        "finite_log_consistency_max": (
            float(np.max(consistency_logs)) if consistency_logs else None
        ),
    }


def _summarize_geometry(
    features: np.ndarray, members: Sequence[np.ndarray]
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute centers, two radii, representatives, and pointwise errors."""
    dimension = features.shape[1]
    number_of_balls = len(members)
    centers = np.empty((number_of_balls, dimension), dtype=np.float32)
    sizes = np.empty(number_of_balls, dtype=np.int32)
    mean_radii = np.empty(number_of_balls, dtype=np.float32)
    max_radii = np.empty(number_of_balls, dtype=np.float32)
    representative_indices = np.empty(number_of_balls, dtype=np.int64)
    point_center_distances = np.empty(len(features), dtype=np.float32)
    point_representative_distances = np.empty(len(features), dtype=np.float32)

    for ball_id, indices in enumerate(members):
        ball = _rows(features, indices)
        center = np.mean(ball, axis=0, dtype=np.float64).astype(np.float32)
        center_distances = np.linalg.norm(ball - center, axis=1).astype(
            np.float32
        )
        representative_position = int(np.argmin(center_distances))
        representative = ball[representative_position]

        centers[ball_id] = center
        sizes[ball_id] = len(indices)
        mean_radii[ball_id] = float(np.mean(center_distances))
        max_radii[ball_id] = float(np.max(center_distances))
        representative_indices[ball_id] = indices[representative_position]
        point_center_distances[indices] = center_distances
        point_representative_distances[indices] = np.linalg.norm(
            ball - representative, axis=1
        )

    return (
        centers,
        sizes,
        mean_radii,
        max_radii,
        representative_indices,
        point_center_distances,
        point_representative_distances,
    )


def build_granular_balls(
    features: np.ndarray,
    min_ball_size: int = 64,
    center_consistency_threshold: float = 1.3,
    seed: int = 0,
    initial_ball_count: Optional[int] = None,
    coarse_fit_sample_size: int = 20000,
    coarse_kmeans_iterations: int = 8,
    two_means_iterations: int = 2,
    assignment_batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> GranularBallResult:
    """Build a hard GB partition using scalable center-consistency splitting."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Features must be a nonempty [N, D] matrix.")
    if min_ball_size < 1:
        raise ValueError("Minimum ball size must be positive.")
    if center_consistency_threshold <= 1.0:
        raise ValueError("Center-consistency threshold must be greater than one.")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if initial_ball_count is None:
        initial_ball_count = max(1, int(round(math.sqrt(len(features)))))
    initial_ball_count = min(initial_ball_count, len(features))

    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    coarse_centers, actual_fit_sample_size = _fit_coarse_centers(
        features,
        initial_ball_count,
        coarse_fit_sample_size,
        coarse_kmeans_iterations,
        seed,
        device,
        assignment_batch_size,
    )
    coarse_assignments = _assign_to_centers(
        features, coarse_centers, assignment_batch_size, device
    )
    initial_members = _members_from_assignments(
        coarse_assignments, len(coarse_centers)
    )
    members, split_statistics = _split_initial_balls(
        features,
        initial_members,
        center_consistency_threshold,
        min_ball_size,
        two_means_iterations,
    )
    members.sort(key=lambda indices: int(np.min(indices)))

    assignments = np.full(len(features), -1, dtype=np.int32)
    duplicate_assignments = 0
    for ball_id, indices in enumerate(members):
        duplicate_assignments += int(np.count_nonzero(assignments[indices] >= 0))
        assignments[indices] = ball_id
    missing_assignments = int(np.count_nonzero(assignments < 0))
    if missing_assignments or duplicate_assignments:
        raise RuntimeError(
            "Invalid GB coverage: missing={}, duplicates={}.".format(
                missing_assignments, duplicate_assignments
            )
        )

    (
        centers,
        sizes,
        mean_radii,
        max_radii,
        representative_indices,
        point_center_distances,
        point_representative_distances,
    ) = _summarize_geometry(features, members)

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache()

    split_statistics.update(
        {
            "initial_nonempty_ball_count": int(len(initial_members)),
            "final_ball_count": int(len(members)),
            "missing_assignments": missing_assignments,
            "duplicate_assignments": duplicate_assignments,
        }
    )
    configuration = {
        "seed": seed,
        "initial_ball_count_requested": int(initial_ball_count),
        "coarse_fit_sample_size": actual_fit_sample_size,
        "coarse_kmeans_iterations": coarse_kmeans_iterations,
        "two_means_iterations": two_means_iterations,
        "min_ball_size": min_ball_size,
        "center_consistency_threshold": center_consistency_threshold,
        "assignment_batch_size": assignment_batch_size,
        "device": str(device),
        "deterministic_given_seed": True,
        "full_n_by_n_matrix_created": False,
    }
    return GranularBallResult(
        assignments=assignments,
        centers=centers,
        sizes=sizes,
        mean_radii=mean_radii,
        max_radii=max_radii,
        representative_indices=representative_indices,
        point_center_distances=point_center_distances,
        point_representative_distances=point_representative_distances,
        members=members,
        build_seconds=time.perf_counter() - start_time,
        configuration=configuration,
        split_statistics=split_statistics,
        peak_gpu_allocated_bytes=peak_allocated,
        peak_gpu_reserved_bytes=peak_reserved,
    )
