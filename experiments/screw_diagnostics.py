"""Reproducible screw feature audit: fixed queries, exact 1-NN, same K banks.

Test masks only group already-computed patch scores; they never tune a bank.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch  # Initialize the PyTorch DLL runtime before FAISS on Windows.
import faiss
import numpy as np
from sklearn.metrics import roc_auc_score


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.retrieval import query, summary


METHODS = ("gb_center", "gb_medoid", "ff_raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir = args.out_dir or args.project_root / "revision_evidence"
    faiss.omp_set_num_threads(8)
    root = args.project_root / "patchcore-inspection/results/MVTecAD_Results"
    cached = root / "IM224_WR50_STAGE5B/screw/cache"
    compared = root / "IM224_WR50_STAGE8C/screw"
    metadata = json.loads((cached / "test_cache.json").read_text(encoding="utf-8"))
    manifest = json.loads((compared / "cache/banks_manifest.json").read_text(encoding="utf-8"))
    labels = np.load(cached / "labels.npy")
    features = np.load(cached / "test_embeddings.npy", mmap_mode="r")
    masks = np.load(cached / "masks_gt.npy", mmap_mode="r")
    n = len(labels)
    assert n == 160 and features.shape == (n * 784, 1024)
    assert masks.shape == (n, 1, 224, 224)
    k_gb = int(manifest["k_gb"])
    assert k_gb > 0
    assert len(metadata["subtypes"]) == n
    assert np.array_equal(labels == 0, np.array(metadata["subtypes"]) == "good")
    mask_fraction = np.asarray(masks[:, 0], dtype=np.float32).reshape(
        n, 28, 8, 28, 8).mean(axis=(2, 4))
    core = mask_fraction >= .25
    background = mask_fraction == 0
    good = labels == 0
    anomalous = labels == 1
    assert np.all(core[anomalous].sum(axis=(1, 2)) > 0)

    scored, methods, rows = {}, {}, []
    for method in METHODS:
        bank = np.load(compared / "cache" / (method + "_bank.npy"))
        assert bank.shape == (k_gb, 1024) and np.isfinite(bank).all()
        distance = query(bank, features).reshape(n, 28, 28)
        scored[method] = distance
        image_score = distance.reshape(n, -1).max(axis=1)
        saved = np.load(compared / "runs" / method / "image_scores.npy")
        error = float(np.max(np.abs(image_score - saved)))
        assert np.allclose(image_score, saved, rtol=1e-5, atol=.05)
        defect = distance[core & anomalous[:, None, None]]
        normal = distance[background & anomalous[:, None, None]]
        pooled_auc = roc_auc_score(
            np.r_[np.zeros(len(normal)), np.ones(len(defect))],
            np.r_[normal, defect])
        hits, margins = [], []
        for i in range(n):
            y, x = divmod(int(np.argmax(distance[i])), 28)
            if anomalous[i]:
                hits.append(bool(core[i, y, x]))
                margins.append(float(np.median(distance[i][core[i]]) -
                                     np.median(distance[i][background[i]])))
            rows.append({
                "method": method, "image_index": i,
                "subtype": metadata["subtypes"][i],
                "is_anomaly": int(labels[i]),
                "image_max_squared_l2": float(image_score[i]),
                "max_patch_mask_fraction": float(mask_fraction[i, y, x]),
                "defect_core_median": (
                    float(np.median(distance[i][core[i]])) if core[i].any()
                    else None),
                "background_median": float(np.median(
                    distance[i][background[i]])),
            })
        subtypes = {}
        for subtype in sorted(set(metadata["subtypes"]) - {"good"}):
            selected = np.array(metadata["subtypes"]) == subtype
            subtypes[subtype] = {
                "n": int(selected.sum()),
                "image_auroc_vs_good": float(roc_auc_score(
                    np.r_[np.zeros(good.sum()), np.ones(selected.sum())],
                    np.r_[image_score[good], image_score[selected]])),
                "image_max": summary(image_score[selected]),
            }
        methods[method] = {
            "image_auroc": float(roc_auc_score(labels, image_score)),
            "max_abs_saved_score_difference": error,
            "good_image_max": summary(image_score[good]),
            "anomaly_image_max": summary(image_score[anomalous]),
            "good_patch": summary(distance[good]),
            "anomaly_background_patch": summary(normal),
            "anomaly_defect_core_patch": summary(defect),
            "descriptive_pooled_patch_auroc": float(pooled_auc),
            "within_image_defect_minus_background_median": summary(margins),
            "max_patch_inside_defect_core_rate": float(np.mean(hits)),
            "subtypes": subtypes,
        }
        print(method, "AUROC", methods[method]["image_auroc"],
              "saved-score error", error, flush=True)

    paired = {}
    for name, mask in (
        ("good", np.broadcast_to(good[:, None, None], core.shape)),
        ("anomaly_background", background & anomalous[:, None, None]),
        ("anomaly_defect_core", core & anomalous[:, None, None]),
    ):
        delta = (scored["gb_center"] - scored["ff_raw"])[mask]
        paired[name] = {
            "center_minus_ff_squared_l2": summary(delta),
            "fraction_center_larger": float(np.mean(delta > 0)),
        }
    report = {
        "design": "Same 160 screw test images and cached 1024-D features; three exact FAISS squared-L2 1-NN banks, each with K={}.".format(k_gb),
        "k_gb": k_gb,
        "mask_rule": "28x28 non-overlapping 8x8 cells. Defect core: >=25% mask occupancy. Background: zero occupancy. Intermediate cells excluded.",
        "caution": "Post-hoc, descriptive analysis. Patch scores within images are correlated. Pooled patch AUROC is not an independent-sample test. No causal mechanism is established by these statistics alone.",
        "n_images": n, "n_good": int(good.sum()),
        "n_anomalous": int(anomalous.sum()),
        "n_core_patches": int(core.sum()),
        "methods": methods, "paired_center_vs_ff": paired,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "screw_feature_audit_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out_dir / "screw_feature_image_rows_v1.csv", "w",
              newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
