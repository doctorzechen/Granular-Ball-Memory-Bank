"""Shared, exact PatchCore patch scoring and descriptive summaries.

The paper protocol uses 1024-D features, float32 squared-L2 FAISS search,
one nearest neighbour, and the same 8192-row chunks as the archived diagnostics.
This module does not build a memory bank or read labels/test masks.
"""

import torch  # Initialize the PyTorch DLL runtime before FAISS on Windows.
import faiss
import numpy as np


def summary(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p05": float(np.quantile(x, .05)),
            "p95": float(np.quantile(x, .95)),
            "p99": float(np.quantile(x, .99)),
            "p999": float(np.quantile(x, .999))}


def query(bank, features):
    index = faiss.IndexFlatL2(1024)
    index.add(np.ascontiguousarray(bank, dtype=np.float32))
    scores = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), 8192):
        end = min(start + 8192, len(features))
        distances, _ = index.search(
            np.ascontiguousarray(features[start:end], dtype=np.float32), 1)
        scores[start:end] = distances[:, 0]
    return scores
