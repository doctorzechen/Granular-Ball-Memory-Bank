"""Training-only, same-capacity coverage-compensated GB pilot for screw.

Rule fixed before test scoring: replace 25% of the most center-redundant GB
centers with diverse actual training patches having the largest residual to
the retained centers. Candidate patches come from the frozen same-K FF order.
"""

import argparse
import json
import sys
from pathlib import Path

import torch  # Initialize the PyTorch DLL runtime before FAISS on Windows.
import faiss
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.retrieval import query, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir = args.out_dir or args.project_root / "revision_evidence"
    faiss.omp_set_num_threads(8)
    root = args.project_root / "patchcore-inspection/results/MVTecAD_Results"
    train = root / "IM224_WR50_STAGE5B/screw/cache"
    comparison = root / "IM224_WR50_STAGE8C/screw"
    gb = np.load(comparison / "cache/gb_center_bank.npy")
    ff = np.load(comparison / "cache/ff_raw_bank.npy")
    ff_indices = np.load(comparison / "cache/ff_raw_order.npy")
    raw = np.load(train / "training_embeddings.npy", mmap_mode="r")
    test = np.load(train / "test_embeddings.npy", mmap_mode="r")
    labels = np.load(train / "labels.npy")
    masks = np.load(train / "masks_gt.npy", mmap_mode="r")
    k = len(gb)
    n_replace = int(round(.25 * k))
    assert gb.shape == ff.shape == (k, 1024) and k > 0
    assert np.array_equal(ff, np.asarray(raw[ff_indices]))
    assert len(np.unique(ff_indices)) == k

    # A low 2-NN distance flags a GB center with a nearby alternative center.
    index = faiss.IndexFlatL2(1024)
    index.add(np.ascontiguousarray(gb))
    neighbor_distance, neighbor_index = index.search(gb, 2)
    assert np.array_equal(neighbor_index[:, 0], np.arange(k))
    remove = np.argsort(neighbor_distance[:, 1], kind="stable")[:n_replace]
    keep = np.ones(k, dtype=bool)
    keep[remove] = False
    retained = np.ascontiguousarray(gb[keep])

    # FF candidates are diverse among raw training patches. Pick candidates
    # furthest from the retained GB centers; no test feature or label is read.
    index = faiss.IndexFlatL2(1024)
    index.add(retained)
    residual, _ = index.search(np.ascontiguousarray(ff), 1)
    chosen = np.argsort(-residual[:, 0], kind="stable")[:n_replace]
    exemplars = np.ascontiguousarray(ff[chosen])
    hybrid = np.ascontiguousarray(np.vstack([retained, exemplars]))
    assert hybrid.shape == (k, 1024) and np.isfinite(hybrid).all()

    # Test features and masks enter only after the complete bank is frozen.
    distances = query(hybrid, test).reshape(len(labels), 28, 28)
    image_score = distances.reshape(len(labels), -1).max(axis=1)
    occupancy = np.asarray(masks[:, 0], dtype=np.float32).reshape(
        len(labels), 28, 8, 28, 8).mean(axis=(2, 4))
    anomalous = labels == 1
    good = labels == 0
    core = occupancy >= .25
    hits = []
    for i in np.flatnonzero(anomalous):
        y, x = divmod(int(np.argmax(distances[i])), 28)
        hits.append(bool(core[i, y, x]))
    report = {
        "method": "GB center plus coverage-rescue actual patches",
        "design": "Drop 25% of GB centers with the smallest center-to-center 2-NN distances. From the frozen global FF order, add the same number of real training patches with the largest residual to retained centers. All selection uses training-normal features only.",
        "capacity": k,
        "retained_center_count": int(keep.sum()),
        "added_real_patch_count": n_replace,
        "image_auroc": float(roc_auc_score(labels, image_score)),
        "good_patch_distance": summary(distances[good]),
        "good_image_max": summary(image_score[good]),
        "anomaly_image_max": summary(image_score[anomalous]),
        "defect_core_patch_distance": summary(
            distances[core & anomalous[:, None, None]]),
        "max_patch_inside_defect_core_rate": float(np.mean(hits)),
        "selection_audit": {
            "removed_centers_unique": bool(len(np.unique(remove)) == n_replace),
            "added_raw_indices_unique": bool(
                len(np.unique(ff_indices[chosen])) == n_replace),
            "bank_rows": int(len(hybrid)),
            "min_removed_center_redundancy_distance": float(
                neighbor_distance[remove, 1].min()),
            "max_removed_center_redundancy_distance": float(
                neighbor_distance[remove, 1].max()),
        },
        "caution": "Exploratory single-category fixed-rule pilot, not a validated general improvement. The test labels were used only to evaluate the frozen bank.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "screw_hybrid_bank.npy", hybrid)
    np.save(args.out_dir / "screw_hybrid_image_scores.npy", image_score)
    np.save(args.out_dir / "screw_hybrid_removed_center_indices.npy", remove)
    np.save(args.out_dir / "screw_hybrid_added_raw_indices.npy", ff_indices[chosen])
    (args.out_dir / "screw_hybrid_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("K", k, "replaced", n_replace, "AUROC", report["image_auroc"],
          "normal max median", report["good_image_max"]["median"],
          "max-patch core hit", report["max_patch_inside_defect_core_rate"])


if __name__ == "__main__":
    main()
