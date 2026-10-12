"""Transfer GB storage to SPADE's DN2 image-level retrieval stage.

This is *not* full SPADE: it tests only the paper's image-retrieval stage,
not its conditioned multi-scale correspondence / segmentation stage.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import faiss
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms


def portable_basename(value):
    return Path(str(value).replace("\\", "/")).name


def extract(paths, model, transform, device, batch_size=12):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).to(device)
            rows.append(model(batch).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(rows))


def score(bank, test, neighbors=5):
    assert len(bank) >= neighbors
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(np.ascontiguousarray(bank, dtype=np.float32))
    distances, _ = index.search(np.ascontiguousarray(test), neighbors)
    return np.sqrt(np.maximum(distances, 0.0)).mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--category", default="screw")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--code-root", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir = args.out_dir or args.project_root / "revision_evidence"
    device = torch.device(args.device)
    faiss.omp_set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)

    repo = args.project_root / "patchcore-inspection"
    code_root = args.code_root or Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root / "third_party/patchcore/src"))
    sys.path.insert(0, str(code_root / "src"))
    from gb_memory.partition import build_granular_balls
    from baselines.farthest_first import global_farthest_first_order

    dataset = (args.dataset_root or repo / "datasets/mvtec") / args.category
    train_paths = sorted((dataset / "train/good").glob("*.png"))
    metadata_path = repo / (
        "results/MVTecAD_Results/IM224_WR50_STAGE5B/"
        + args.category + "/cache/test_cache.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_paths = [dataset / "test" / subtype / portable_basename(old)
                  for old, subtype in zip(metadata["paths"], metadata["subtypes"])]
    labels = np.load(metadata_path.parent / "labels.npy")
    assert len(test_paths) == len(labels)
    assert all(path.is_file() for path in train_paths + test_paths)
    assert np.array_equal(labels == 0,
                          np.array(metadata["subtypes"]) == "good")

    output = args.out_dir / "spade_image_stage" / args.category
    output.mkdir(parents=True, exist_ok=True)
    train_file, test_file = output / "train_global.npy", output / "test_global.npy"
    provenance_path = output / "feature_provenance.json"
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints/wide_resnet50_2-95faca4d.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Offline V1 checkpoint is missing: " + str(checkpoint))
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    provenance = {"weight_sha256": digest.hexdigest(),
        "weights": "IMAGENET1K_V1", "seed": 0,
        "preprocessing": "resize256_crop224_imagenet_normalization",
        "train_order": [path.name for path in train_paths],
        "test_order": [str(path.relative_to(dataset)).replace("\\", "/") for path in test_paths],
        "test_manifest_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    if train_file.is_file() and test_file.is_file():
        if not provenance_path.is_file() or json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError("SPADE feature provenance differs; use a fresh output directory.")
        train = np.load(train_file)
        test = np.load(test_file)
    else:
        model = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        started = time.perf_counter()
        train = extract(train_paths, model, transform, device)
        test = extract(test_paths, model, transform, device)
        np.save(train_file, train)
        np.save(test_file, test)
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print("extracted", args.category, len(train), len(test),
              "seconds", round(time.perf_counter() - started, 2), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert train.shape == (len(train_paths), 2048)
    assert test.shape == (len(test_paths), 2048)
    assert np.isfinite(train).all() and np.isfinite(test).all()

    # Same center-partition routine as the PatchCore GB bank, with a smaller
    # size guard appropriate to a few hundred image-level normal features.
    gb = build_granular_balls(
        train, min_ball_size=4, center_consistency_threshold=1.3,
        seed=0, device=device)
    k = len(gb.centers)
    assert 5 <= k < len(train)
    ff_order, _ = global_farthest_first_order(
        train, k, 2048, seed=0, device=device)
    ff_bank = np.ascontiguousarray(train[ff_order])
    scores = {
        "full_normal_gallery": score(train, test),
        "gb_centers": score(gb.centers, test),
        "same_k_ff_real_images": score(ff_bank, test),
    }
    np.savez(output / "image_scores.npz", **scores)
    np.save(output / "gb_image_centers.npy", gb.centers)
    np.save(output / "ff_image_indices.npy", ff_order)
    report = {
        "category": args.category,
        "scope": "SPADE paper Section 3.2 / DN2 image-level retrieval only; no pyramid correspondence or segmentation",
        "backbone": "ImageNet Wide-ResNet50-2, global average-pooled 2048-D feature",
        "preprocessing": "Resize 256, center crop 224, ImageNet normalization",
        "scorer": "mean of five exact Euclidean nearest-neighbor distances",
        "training_only_partition": "GB seed 0, consistency threshold 1.3, min_ball_size 4 (image-level adaptation)",
        "n_train_images": len(train), "n_test_images": len(test),
        "k_gb": k, "compression_ratio": float(len(train) / k),
        "ball_build_seconds": float(gb.build_seconds),
        "gb_split_statistics": gb.split_statistics,
        "image_auroc": {name: float(roc_auc_score(labels, value))
                        for name, value in scores.items()},
        "caution": "Only the image-retrieval stage of SPADE is tested. These numbers must not be called full-SPADE accuracy or transfer of its segmentation stage.",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.category, "K", k, "AUC", report["image_auroc"], flush=True)


if __name__ == "__main__":
    main()
