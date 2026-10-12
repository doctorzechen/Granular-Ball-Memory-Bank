"""Reproduce granular-ball memory-bank construction and controlled experiments.

Run --help without ML dependencies. Defaults deliberately keep frozen evidence
read-only. A fresh run writes only under --out-root.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "third_party/patchcore"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CONFIG = json.loads((ROOT / "configs/paper.json").read_text(encoding="utf-8"))
CATEGORIES = tuple(CONFIG["categories"])
RESULTS_REL = Path("patchcore-inspection/results/MVTecAD_Results")
STAGES = ("prepare", "banks", "detect", "capacity", "construction", "query",
          "screw", "hybrid", "spade", "report")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(assets_root=None):
    # This cache includes the exact V1 checkpoint; never silently choose V2.
    assets_root = ROOT if assets_root is None else Path(assets_root)
    os.environ["TORCH_HOME"] = str(assets_root / "weights")
    os.environ.setdefault("MPLBACKEND", "Agg")
    pythonpaths = [str(ROOT / "src"), str(REPO / "src"), str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpaths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpaths)


def runtime_info():
    names = ("torch", "torchvision", "faiss-cpu", "numpy", "scikit-learn",
             "scipy", "Pillow", "matplotlib", "psutil", "timm", "pytest")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def dataset_inventory(dataset, categories):
    rows = []
    for category in categories:
        directory = dataset / category
        train = sorted((directory / "train/good").glob("*.png"))
        test = sorted((directory / "test").glob("*/*.png"))
        masks = sorted((directory / "ground_truth").glob("*/*.png"))
        if not train or not test:
            raise FileNotFoundError("Missing training/test PNGs: " + str(directory))
        for image in test:
            if image.parent.name != "good":
                mask = directory / "ground_truth" / image.parent.name / (
                    image.stem + "_mask.png")
                if not mask.is_file():
                    raise FileNotFoundError("Missing ground-truth mask: " + str(mask))
        row = {"category": category, "train": len(train), "test": len(test),
               "masks": len(masks)}
        expected_path = ROOT / "reference/dataset_inventory.json"
        if expected_path.is_file():
            expected = {r["category"]: r for r in json.loads(
                expected_path.read_text(encoding="utf-8"))}[category]
            if row != expected:
                raise ValueError("Dataset split differs from paper: " + repr(row))
        rows.append(row)
    return rows


def guard_output(path, extra_inputs=()):
    path = Path(path).resolve()
    protected = tuple((ROOT / name).resolve() for name in (
        "frozen", "reference", "data", "weights", "third_party", "src",
        "configs", "scripts", "experiments", "tests", "verification", "docs", "legacy"))
    protected += tuple(Path(folder).resolve() for folder in extra_inputs)
    if path == ROOT.resolve() or path in ROOT.resolve().parents or any(folder == path or folder in path.parents
                                     for folder in protected):
        raise ValueError("Choose a run output folder, not a package/input/evidence folder.")


def doctor(args):
    guard_output(args.out_root, (args.dataset, args.frozen_root, args.assets_root / "weights"))
    if args.assets_root.resolve() != ROOT.resolve():
        guard_output(args.out_root, (args.assets_root,))
    info = runtime_info()
    missing = [key for key, value in info["packages"].items() if value is None]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing))
    import torch
    import faiss
    import patchcore.backbones
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    info["device"] = args.device
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose --device cpu for correctness checks.")
        info["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    checkpoint = args.assets_root / "weights/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Required offline ImageNet V1 weights missing: " + str(checkpoint))
    actual = sha256(checkpoint)
    expected = json.loads((ROOT / "reference/weight.json").read_text(encoding="utf-8"))
    if actual != expected["sha256"]:
        raise ValueError("ImageNet checkpoint checksum mismatch.")
    info["weight_sha256"] = actual
    require_dataset = args.mode == "fresh" or "spade" in args.stages
    info["dataset"] = dataset_inventory(args.dataset, args.categories) if require_dataset else []
    manifest_path = ROOT / "reference/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = 0
    for entry in manifest["files"]:
        if not require_dataset or entry["path"].split("/")[0] not in args.categories:
            continue
        image = args.dataset / entry["path"]
        if image.stat().st_size != entry["bytes"] or sha256(image) != entry["sha256"]:
            raise ValueError("Dataset PNG checksum differs from paper: " + str(image))
        verified += 1
    info["dataset_png_hashes_verified"] = verified
    info["dataset_manifest_sha256"] = sha256(manifest_path)
    info["faiss_version"] = faiss.__version__
    info["status"] = "pass"
    save_json(args.out_root / "doctor.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def validate_cache(root, category):
    import numpy as np
    directory = root / RESULTS_REL / "IM224_WR50_STAGE5B" / category / "cache"
    files = ("training_cache.json", "test_cache.json", "training_embeddings.npy",
             "original_memory_bank.npy", "selected_indices.npy", "gb_partition.npz",
             "test_embeddings.npy", "labels.npy", "masks_gt.npy")
    for filename in files:
        if not (directory / filename).is_file():
            raise FileNotFoundError("Missing frozen input: " + str(directory / filename))
    raw = np.load(directory / "training_embeddings.npy", mmap_mode="r")
    memory = np.load(directory / "original_memory_bank.npy", mmap_mode="r")
    selected = np.load(directory / "selected_indices.npy")
    test = np.load(directory / "test_embeddings.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy")
    masks = np.load(directory / "masks_gt.npy", mmap_mode="r")
    metadata = json.loads((directory / "test_cache.json").read_text(encoding="utf-8"))
    if raw.shape[1] != 1024 or test.shape != (len(labels) * 784, 1024):
        raise ValueError("Wrong cached embedding shape for " + category)
    if raw.dtype != np.float32 or test.dtype != np.float32:
        raise ValueError("Paper requires float32 features.")
    if masks.shape != (len(labels), 1, 224, 224):
        raise ValueError("Wrong mask dimensions.")
    if sum(metadata["batch_sizes"]) != len(labels):
        raise ValueError("Test metadata batch coverage mismatch.")
    if not np.array_equal(labels == 0, np.asarray(metadata["subtypes"]) == "good"):
        raise ValueError("Test image order/labels mismatch.")
    if len(np.unique(selected)) != len(selected) or len(selected) != len(memory):
        raise ValueError("Invalid coreset indices.")
    for start in range(0, len(selected), 2048):
        if not np.array_equal(raw[selected[start:start + 2048]], memory[start:start + 2048]):
            raise ValueError("Coreset rows do not match raw training features.")
    with np.load(directory / "gb_partition.npz") as partition:
        assignments = partition["assignments"]
        centers, sizes = partition["centers"], partition["sizes"]
        if len(assignments) != len(raw) or assignments.min() < 0:
            raise ValueError("Invalid granular-ball coverage.")
        if not np.array_equal(np.bincount(assignments, minlength=len(centers)), sizes):
            raise ValueError("Granular-ball member counts mismatch.")
        if centers.shape[1] != 1024 or int(sizes.sum()) != len(raw):
            raise ValueError("Invalid granular-ball dimensions.")
    return {"category": category, "train_patches": len(raw), "test_images": len(labels),
            "original_k": len(memory), "gb_k": len(centers), "status": "pass"}


def clone_inputs(source, destination, categories):
    """Copy immutable inputs; same-volume hard links avoid duplicating 17 GB."""
    if source.resolve() == destination.resolve():
        raise ValueError("Output directory cannot be the frozen-input directory.")
    for category in categories:
        validate_cache(source, category)
        src_category = source / RESULTS_REL / "IM224_WR50_STAGE5B" / category
        targets = list((src_category / "cache").iterdir())
        targets.extend((src_category / "runs/original").glob("*"))
        for src in targets:
            if not src.is_file():
                continue
            relative = src.relative_to(source)
            dst = destination / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # The initial baseline record is seeded from the archive, but
                # detect legitimately replaces it with regenerated timings.
                # Only cache inputs must stay identical across resumed runs.
                if src.parent.name == "original" and "runs" in relative.parts:
                    if dst.suffix == ".json":
                        baseline = json.loads(dst.read_text(encoding="utf-8"))
                        if baseline.get("status") != "complete" or "metrics" not in baseline:
                            raise RuntimeError("Invalid regenerated baseline: " + str(dst))
                    continue
                # Never accept stale inputs based only on shape or filename.
                if os.path.samefile(src, dst):
                    continue
                if src.stat().st_size != dst.stat().st_size or sha256(src) != sha256(dst):
                    raise RuntimeError("Output contains different inputs: " + str(dst))
                continue
            if src.parent.name == "cache" and src.suffix in (".npy", ".npz"):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
    provenance_path = destination / "frozen_input_provenance.json"
    previous = (json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance_path.is_file() else {})
    covered = set(previous.get("categories", [])) | set(categories)
    save_json(provenance_path,
              {"source": str(source.resolve()),
               "categories": [category for category in CATEGORIES if category in covered],
               "policy": "Binary inputs are read-only; hard links on same volume, copy otherwise."})


def save_scored(directory, scored, **extra):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "image_scores.npy", scored["scores"])
    public = {k: v for k, v in scored.items() if k not in ("scores", "maps")}
    public.update(status="complete", **extra)
    save_json(directory / "result.json", public)
    return public


def prepare_category(args, category):
    import numpy as np
    import torch
    import patchcore.utils
    from gb_memory.partition import build_granular_balls
    from adapters.patchcore import _build_original_model, _extract_test_cache, _score_cache, _TrainingCapture, _loaders
    from experiments.cache_io import _cache_test, _load_test_cache, canonical_config, config_hash
    device = torch.device(args.device)
    category_root = args.out_root / RESULTS_REL / "IM224_WR50_STAGE5B" / category
    cache = category_root / "cache"
    provenance = cache / "reproduction_config.json"
    config = canonical_config(category, 2, 8, device)
    config["dataset_file_manifest_sha256"] = sha256(ROOT / "reference/dataset_manifest.json")
    key = config_hash(config)
    if provenance.exists():
        old = json.loads(provenance.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError("Configuration changed; use a new --out-root.")
        validate_cache(args.out_root, category)
        return
    if (cache / "training_embeddings.npy").exists():
        raise RuntimeError("Cache exists without fresh-run provenance; select a new --out-root.")
    patchcore.utils.fix_seeds(0, with_torch=True, with_cuda=True)
    train, test, _, _ = _loaders(str(args.dataset), category, 2, 0)
    model = _build_original_model(device, 8)
    capture = _TrainingCapture()
    model.set_memory_bank_compressor(capture)
    started = time.perf_counter()
    model.fit(train)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started
    raw = np.asarray(capture.raw_features, dtype=np.float32)
    memory = np.asarray(model.anomaly_scorer.detection_features, dtype=np.float32)
    selected = np.asarray(model.featuresampler.last_selected_indices, dtype=np.int64)
    if not np.array_equal(raw[selected], memory):
        raise RuntimeError("Original coreset row mapping failed.")
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "training_embeddings.npy", raw)
    np.save(cache / "original_memory_bank.npy", memory)
    np.save(cache / "selected_indices.npy", selected)
    partition = build_granular_balls(raw, min_ball_size=64,
                                    center_consistency_threshold=1.3, seed=0, device=device)
    np.savez(cache / "gb_partition.npz", assignments=partition.assignments,
             centers=partition.centers, sizes=partition.sizes,
             mean_radii=partition.mean_radii, max_radii=partition.max_radii)
    save_json(cache / "training_cache.json", {
        "status": "complete", "config_hash": key, "fit_seconds": fit_seconds,
        "gb_build_seconds": partition.build_seconds,
        "training_embedding_shape": list(raw.shape),
        "original_memory_bank_shape": list(memory.shape),
        "selected_mapping_exact": True, "number_of_balls": len(partition.centers),
        "gb_configuration": partition.configuration,
        "gb_split_statistics": partition.split_statistics})
    test_cache = _extract_test_cache(model, test, device)
    _cache_test(cache, test_cache, key)
    del test_cache, raw, capture, partition
    gc.collect()
    test_cache = _load_test_cache(cache)
    scored = _score_cache(model, memory, test_cache, device, 8, keep_maps=False)
    save_scored(category_root / "runs/original", scored, method="original",
                memory_bank_size=len(memory), k=len(memory), config_hash=key)
    save_json(provenance, config)
    validate_cache(args.out_root, category)
    del model, scored, test_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


from experiments.capacity_study import capacity


def call_revision(script, args, *extras):
    command = [sys.executable, "-m", "experiments." + Path(script).stem,
               "--project-root", str(args.out_root), "--out-dir",
               str(args.out_root / "revision_evidence"), *map(str, extras)]
    subprocess.run(command, cwd=str(ROOT), env=os.environ.copy(), check=True)


def run(args):
    guard_output(args.out_root, (args.dataset, args.frozen_root, args.assets_root / "weights"))
    if args.assets_root.resolve() != ROOT.resolve():
        guard_output(args.out_root, (args.assets_root,))
    import torch
    import faiss
    import patchcore.utils
    device = torch.device(args.device)
    faiss.omp_set_num_threads(8)
    doctor(args)
    existing = args.out_root / "run_protocol.json"
    protocol = {"mode": args.mode, "device": args.device,
                "categories": list(args.categories), "paper_config": CONFIG}
    if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Run protocol differs; choose a new --out-root.")
    save_json(existing, protocol)
    save_json(args.out_root / "environment.json", runtime_info())
    stages = args.stages
    roots = args.out_root / RESULTS_REL
    five = roots / "IM224_WR50_STAGE5B"
    eight = roots / "IM224_WR50_STAGE8C"
    if args.mode == "frozen":
        clone_inputs(args.frozen_root, args.out_root, args.categories)
    if "prepare" in stages and args.mode == "fresh":
        for category in args.categories:
            print("PREPARE", category, flush=True)
            prepare_category(args, category)
    if "banks" in stages or "detect" in stages or "construction" in stages:
        from experiments.main_comparison import run_banks, run_detect, run_time, METHODS
        for category in args.categories:
            validate_cache(args.out_root, category)
            if "banks" in stages:
                print("BANKS", category, flush=True)
                run_banks(five, eight, category, device, 8)
            if "detect" in stages:
                print("DETECT", category, flush=True)
                from adapters.patchcore import _build_original_model, _score_cache
                from experiments.cache_io import _load_test_cache
                import numpy as np
                original_cache = five / category / "cache"
                original_bank = np.load(original_cache / "original_memory_bank.npy", mmap_mode="r")
                model = _build_original_model(device, 8)
                baseline = _score_cache(model, original_bank, _load_test_cache(original_cache),
                                        device, 8, keep_maps=False)
                save_scored(five / category / "runs/original", baseline, method="original",
                            k=len(original_bank), memory_bank_size=len(original_bank))
                del original_bank, model, baseline
                gc.collect()
                run_detect(five, eight, category, device, 8)
            if "construction" in stages:
                for method in METHODS:
                    print("CONSTRUCTION", category, method, flush=True)
                    for attempt in range(100):
                        result = run_time(five, eight, category, method, device, 8)
                        if result.get("status") == "complete":
                            break
                    else:
                        raise RuntimeError("Construction timing did not complete after 100 resumes.")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if "capacity" in stages:
        capacity(args)
    if "query" in stages:
        from experiments.query_benchmark import run_benchmark, aggregate_benchmarks
        output = roots / "IM224_WR50_QUERYBENCH_V2"
        for category in args.categories:
            print("QUERY", category, flush=True)
            benchmark = run_benchmark(five, eight, output, category, repeats=32,
                          warmup_rounds=10, warmup_batches=4,
                          faiss_workers=8, schedule_seed=20260922)
            if not benchmark.get("analysis", {}).get("validation", {}).get("pass", False):
                raise RuntimeError("Query timing quality gate failed for " + category +
                                   "; raw measurements are saved but no validated speed-up or completed query stage is claimed.")
        aggregate_benchmarks(output, five, eight, categories=args.categories)
    if "screw" in stages and "screw" in args.categories:
        call_revision("screw_diagnostics.py", args)
        call_revision("screw_hybrid_pilot.py", args)
    if "hybrid" in stages:
        call_revision("hybrid_exploration.py", args, "--categories", *args.categories)
    if "spade" in stages:
        for category in args.categories:
            call_revision("spade_retrieval.py", args, "--category", category,
                          "--dataset-root", args.dataset, "--device", args.device,
                          "--code-root", ROOT)
    if "report" in stages:
        from report_results import build_report
        build_report(args.out_root, categories=args.categories)
    completion_path = args.out_root / "execution_complete.json"
    previous = (json.loads(completion_path.read_text(encoding="utf-8"))
                if completion_path.is_file() else {})
    completed = set(stages)
    if (previous.get("status") == "complete"
            and previous.get("categories") == list(args.categories)
            and previous.get("mode") == args.mode):
        completed.update(previous.get("executed_stages", []))
    save_json(completion_path, {
        "status": "complete", "executed_stages": [s for s in STAGES if s in completed],
        "executed_stages_this_invocation": list(stages),
        "categories": list(args.categories), "mode": args.mode,
        "note": "Executed stages are cumulative successful invocations under this same protocol; omitted stages are not implied."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "run", "verify-reference", "report"))
    parser.add_argument("--assets-root", type=Path, default=ROOT,
                        help="Read-only asset folder containing data/, weights/ and optional frozen/.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    parser.add_argument("--mode", choices=("fresh", "frozen"), default="fresh")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--frozen-root", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--check-files", action="store_true",
                        help="SHA256 every packaged file; may take minutes for large inputs.")
    args = parser.parse_args()
    args.assets_root = args.assets_root.resolve()
    if args.dataset is None:
        args.dataset = args.assets_root / "data/mvtec"
    if args.frozen_root is None:
        args.frozen_root = args.assets_root / "frozen"
    if args.out_root is None:
        args.out_root = ROOT / "runs" / (args.mode + "_paper")
    args.categories = tuple(args.categories)
    if len(set(args.categories)) != len(args.categories):
        parser.error("Each category must appear only once.")
    args.dataset, args.out_root, args.frozen_root = (
        path.resolve() for path in (args.dataset, args.out_root, args.frozen_root))
    configure(args.assets_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "verify-reference":
        from verify_reference import verify
        result = verify(ROOT, ROOT / "verification/reference_check.json", args.check_files)
        print(json.dumps({"passed": result["passed"],
                          "checks_total": result.get("checks_total"),
                          "checks_failed": result.get("checks_failed"),
                          "scope": result.get("scope"),
                          "report": "verification/reference_check.json"},
                         ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "report":
        guard_output(args.out_root)
        if args.assets_root.resolve() != ROOT.resolve():
            guard_output(args.out_root, (args.assets_root,))
        from report_results import build_report
        print(build_report(args.out_root, categories=args.categories))
    elif args.command == "doctor":
        doctor(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
