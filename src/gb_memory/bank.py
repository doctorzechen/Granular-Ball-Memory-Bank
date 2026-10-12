"""Encoder-independent interface for the proposed mean-center memory bank.

Construction delegates to the unchanged frozen partition implementation.
After fitting, the inference object owns only its centers and configuration;
raw features, member indices, assignments and radii are not retained.
"""

import json
import math
from numbers import Integral, Real
from pathlib import Path
import tempfile

import numpy as np
import torch

from .partition import build_granular_balls


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name + " must be a positive integer.")
    return int(value)


def _finite_features(features):
    """Validate in bounded row blocks rather than allocating a full boolean bank."""
    array = np.asarray(features)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("Normal embeddings must be a nonempty [N, D] matrix.")
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise ValueError("Normal embeddings must contain real numeric values.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float32)
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start + 4096]).all():
            raise ValueError("Normal embeddings must be finite in float32.")
    return array


class GranularBallMemoryBank:
    """Fit a compact mean-center representation to normal feature vectors.

    ``fit(normal_embeddings)`` accepts an ``[N, D]`` NumPy-compatible real
    matrix, converts it to the paper's float32 representation, and returns
    this object. The encoder is intentionally outside this interface.

    This is the paper's existing construction, not a new neural encoder or
    anomaly scoring rule. Defaults exactly match ``build_granular_balls``.
    GPU floating-point reductions need not be bitwise identical across runs.
    """

    def __init__(
        self,
        min_ball_size=64,
        center_consistency_threshold=1.3,
        seed=0,
        initial_ball_count=None,
        coarse_fit_sample_size=20000,
        coarse_kmeans_iterations=8,
        two_means_iterations=2,
        assignment_batch_size=4096,
        device=None,
    ):
        if (isinstance(center_consistency_threshold, bool)
                or not isinstance(center_consistency_threshold, Real)
                or not math.isfinite(center_consistency_threshold)
                or center_consistency_threshold <= 1.0):
            raise ValueError("Center-consistency threshold must be finite and greater than one.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        self._parameters = {
            "min_ball_size": _positive_integer("min_ball_size", min_ball_size),
            "center_consistency_threshold": float(center_consistency_threshold),
            "seed": int(seed),
            "initial_ball_count": (
                None if initial_ball_count is None
                else _positive_integer("initial_ball_count", initial_ball_count)),
            "coarse_fit_sample_size": _positive_integer("coarse_fit_sample_size", coarse_fit_sample_size),
            "coarse_kmeans_iterations": _positive_integer("coarse_kmeans_iterations", coarse_kmeans_iterations),
            "two_means_iterations": _positive_integer("two_means_iterations", two_means_iterations),
            "assignment_batch_size": _positive_integer("assignment_batch_size", assignment_batch_size),
            "device": None if device is None else str(torch.device(device)),
        }
        self._centers = None
        self._configuration = None

    def _require_fitted(self):
        if self._centers is None:
            raise RuntimeError("The memory bank has not been fitted.")

    @property
    def parameters(self):
        """Construction settings; changes to the returned dictionary are isolated."""
        return dict(self._parameters)

    @property
    def centers(self):
        """Retained float32 ``[K, D]`` center vectors for downstream retrieval."""
        self._require_fitted()
        return self._centers

    @property
    def configuration(self):
        """Actual construction configuration, including the chosen device."""
        self._require_fitted()
        return dict(self._configuration)

    @property
    def feature_dimension(self):
        return int(self.centers.shape[1])

    @property
    def size(self):
        return int(len(self.centers))

    def fit(self, normal_embeddings):
        features = _finite_features(normal_embeddings)
        arguments = dict(self._parameters)
        if arguments["device"] is not None:
            arguments["device"] = torch.device(arguments["device"])
        result = build_granular_balls(features, **arguments)
        centers = np.array(result.centers, dtype=np.float32, order="C", copy=True)
        if centers.ndim != 2 or centers.shape[1] != features.shape[1] or not np.isfinite(centers).all():
            raise RuntimeError("Partition construction produced invalid centers.")
        self._centers = centers
        self._configuration = dict(result.configuration)
        # The local result, including assignments/member arrays, is released
        # on return. No reference to training embeddings is stored on self.
        return self

    def save(self, path):
        """Atomically save exactly centers plus a JSON configuration in NPZ."""
        self._require_fitted()
        if not np.isfinite(self._centers).all():
            raise ValueError("The retained centers contain non-finite values.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "format_version": 1,
            "parameters": self.parameters,
            "configuration": self.configuration,
            "embedding_dimension": self.feature_dimension,
            "center_count": self.size,
        }
        serialized = json.dumps(config, sort_keys=True, allow_nan=False)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp",
                                             delete=False) as temporary:
                temporary_path = Path(temporary.name)
                np.savez(temporary, centers=self._centers, config=np.asarray(serialized))
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path

    @classmethod
    def load(cls, path):
        """Load an inference bank without reading or recreating training members."""
        with np.load(Path(path), allow_pickle=False) as storage:
            if set(storage.files) != {"centers", "config"}:
                raise ValueError("A saved bank must contain only centers and config.")
            serialized = storage["config"]
            if serialized.ndim != 0 or serialized.dtype.kind != "U":
                raise ValueError("Saved configuration must be a scalar JSON string.")
            config = json.loads(str(serialized.item()))
            centers = np.array(storage["centers"], order="C", copy=True)
        if not isinstance(config, dict) or config.get("format_version") != 1:
            raise ValueError("Unsupported memory-bank format.")
        if (centers.ndim != 2 or min(centers.shape) < 1
                or centers.dtype != np.float32 or not np.isfinite(centers).all()):
            raise ValueError("Saved centers must be a finite nonempty float32 matrix.")
        if (config.get("embedding_dimension") != centers.shape[1]
                or config.get("center_count") != len(centers)):
            raise ValueError("Saved center dimensions disagree with configuration.")
        if not isinstance(config.get("parameters"), dict) or not isinstance(config.get("configuration"), dict):
            raise ValueError("Saved construction settings are missing.")
        bank = cls(**config["parameters"])
        bank._centers = centers
        bank._configuration = dict(config["configuration"])
        return bank
