"""Apply the fixed 25% coverage-rescue rule to all 15 cached categories.

This is an exploratory extension of the screw-motivated pilot, not an
independent held-out validation or a cross-framework test.
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
from experiments.retrieval import query


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)


def hybrid_bank(gb, ff):
    k = len(gb)
    replace = int(round(.25 * k))
    assert gb.shape == ff.shape and gb.shape[1] == 1024 and replace > 0
    index = faiss.IndexFlatL2(1024)
    index.add(np.ascontiguousarray(gb))
    center_distance, center_index = index.search(gb, 2)
    assert np.array_equal(center_index[:, 0], np.arange(k))
    remove = np.argsort(center_distance[:, 1], kind="stable")[:replace]
    keep = np.ones(k, dtype=bool)
    keep[remove] = False
    retained = np.ascontiguousarray(gb[keep])
    index = faiss.IndexFlatL2(1024)
    index.add(retained)
    residual, _ = index.search(np.ascontiguousarray(ff), 1)
    chosen = np.argsort(-residual[:, 0], kind="stable")[:replace]
    combined = np.ascontiguousarray(np.vstack([retained, ff[chosen]]))
    assert combined.shape == gb.shape
    return combined, replace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES,
                        default=list(CATEGORIES))
    args = parser.parse_args()
    args.out_dir = args.out_dir or args.project_root / "revision_evidence"
    faiss.omp_set_num_threads(8)
    root = args.project_root / "patchcore-inspection/results/MVTecAD_Results"
    results = []
    for category in args.categories:
        train = root / "IM224_WR50_STAGE5B" / category / "cache"
        comparison = root / "IM224_WR50_STAGE8C" / category
        labels = np.load(train / "labels.npy")
        test = np.load(train / "test_embeddings.npy", mmap_mode="r")
        gb = np.load(comparison / "cache/gb_center_bank.npy")
        ff = np.load(comparison / "cache/ff_raw_bank.npy")
        k = len(gb)
        bank, replace = hybrid_bank(gb, ff)
        hybrid_scores = query(bank, test).reshape(len(labels), -1).max(axis=1)
        center_scores = np.load(
            comparison / "runs/gb_center/image_scores.npy")
        ff_scores = np.load(comparison / "runs/ff_raw/image_scores.npy")
        metrics = {
            "category": category, "k": k, "replaced": replace,
            "n_test_images": int(len(labels)),
            "image_auroc_hybrid": float(roc_auc_score(labels, hybrid_scores)),
            "image_auroc_gb_center": float(roc_auc_score(labels, center_scores)),
            "image_auroc_ff_real": float(roc_auc_score(labels, ff_scores)),
            "normal_image_max_median_hybrid": float(
                np.median(hybrid_scores[labels == 0])),
            "normal_image_max_median_gb_center": float(
                np.median(center_scores[labels == 0])),
            "normal_image_max_median_ff_real": float(
                np.median(ff_scores[labels == 0])),
        }
        results.append(metrics)
        print(category, "K", k, "GB", round(metrics["image_auroc_gb_center"], 4),
              "hybrid", round(metrics["image_auroc_hybrid"], 4),
              "FF", round(metrics["image_auroc_ff_real"], 4), flush=True)
    mean = {
        method: float(np.mean([r["image_auroc_" + method] for r in results]))
        for method in ("gb_center", "hybrid", "ff_real")
    }
    report = {
        "categories": list(args.categories),
        "all_15_categories": set(args.categories) == set(CATEGORIES),
        "design": "Fixed, training-normal-only 25% center-replacement rule, applied unchanged to all 15 categories; exact same-K test query.",
        "caution": "The rule was motivated by observed screw test behavior, so these 15-category results are exploratory and not an independent held-out validation. This is still PatchCore, not cross-framework transfer.",
        "category_results": results,
        "mean_image_auroc": mean,
        "hybrid_better_than_center_count": int(sum(
            r["image_auroc_hybrid"] > r["image_auroc_gb_center"]
            for r in results)),
        "hybrid_better_than_ff_count": int(sum(
            r["image_auroc_hybrid"] > r["image_auroc_ff_real"]
            for r in results)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "hybrid_fixed_rule_15cat.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("mean AUROC", mean, flush=True)


if __name__ == "__main__":
    main()
