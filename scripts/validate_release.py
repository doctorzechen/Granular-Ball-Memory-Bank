"""Check refactoring equivalence and sampled frozen-input scoring contracts.

This is not a full feature-extraction, construction, AUROC or timing rerun.
Source-only verification uses the standard library. Sample checks use CPU
FAISS; --device controls only the explicitly requested map post-processing.
The supplied full reproduction package is read-only input.
"""

import argparse
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
RESULTS_REL = Path("patchcore-inspection/results/MVTecAD_Results")
METHODS = ("original", "gb_center", "gb_medoid", "ff_raw")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _syntax(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"))


def _node(tree, name):
    return next(node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def _digest(node):
    # Comments, line numbers and CRLF/LF changes are not executable semantics.
    value = ast.dump(node, include_attributes=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _WithoutCheckpointGuards(ast.NodeTransformer):
    """Normalize only the new operational validation calls, not algorithm code."""
    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"_validate_generated_banks", "_validate_scored_checkpoint"}):
            return None
        return self.generic_visit(node)


def source_checks(assets_root, release_root=ROOT):
    """Compare explicit unchanged modules/functions with their frozen donors."""
    source = Path(assets_root) / "patchcore-inspection/src/patchcore"
    release_root = Path(release_root)
    checks = []
    old = _syntax(source / "granular_ball.py")
    new = _syntax(release_root / "src/gb_memory/partition.py")
    checks.append({"name": "partition_entire_module", "original_ast_sha256": _digest(old),
                   "release_ast_sha256": _digest(new), "passed": _digest(old) == _digest(new)})
    pairs = [
        ("intra_ball_redundancy_analysis.py", "src/baselines/farthest_first.py", ["global_farthest_first_order"]),
        ("stage8c_construction.py", "src/baselines/medoid.py", ["medoid_indices"]),
        ("stage5a_detection_validation.py", "src/adapters/patchcore.py", [
            "_synchronize", "_array_sha256", "_process_memory", "_build_original_model",
            "_extract_test_cache", "_index_vectors", "_score_cache"]),
        ("stage5b_multicategory_validation.py", "src/adapters/patchcore.py", ["_TrainingCapture", "_loaders"]),
        ("stage5b_multicategory_validation.py", "experiments/cache_io.py", [
            "canonical_config", "config_hash", "_save_json", "_cache_test", "_load_test_cache"]),
        ("stage8a_validation.py", "experiments/cache_io.py", ["_stage5b_category_root", "load_train_assets", "_load_raw"]),
        ("stage8c_construction.py", "experiments/construction_time.py", ["summarize_repeats", "ConstructionTimer"]),
        ("stage8c_validation.py", "experiments/main_comparison.py", [
            "stage8c_config", "_rss_bytes", "run_banks", "_construct_once", "_timing_path",
            "_timing_done", "run_time", "run_detect"]),
        ("common.py", "third_party/patchcore/src/patchcore/common.py", [
            "FaissNN", "_BaseMerger", "AverageMerger", "ConcatMerger", "Preprocessing",
            "MeanMapper", "Aggregator", "RescaleSegmentor", "NearestNeighbourScorer"]),
        ("metrics.py", "third_party/patchcore/src/patchcore/metrics.py", [
            "compute_imagewise_retrieval_metrics", "compute_pixelwise_retrieval_metrics"]),
    ]
    for donor, target, names in pairs:
        old_tree, new_tree = _syntax(source / donor), _syntax(release_root / target)
        for name in names:
            try:
                old_node, new_node = _node(old_tree, name), _node(new_tree, name)
            except StopIteration:
                checks.append({"name": donor + ":" + name, "passed": False,
                               "reason": "Function/class missing from donor or release"})
                continue
            full_release_digest = _digest(new_node)
            comparison_node = (_WithoutCheckpointGuards().visit(copy.deepcopy(new_node))
                               if target == "experiments/main_comparison.py" and name in {"run_banks", "run_detect"}
                               else new_node)
            original_digest, release_digest = _digest(old_node), _digest(comparison_node)
            checks.append({"name": donor + ":" + name, "release_file": target,
                           "original_ast_sha256": original_digest, "release_ast_sha256": release_digest,
                           "full_release_ast_sha256": full_release_digest,
                           "comparison": "AST with only added checkpoint-integrity calls excluded" if full_release_digest != release_digest else "Exact AST",
                           "passed": original_digest == release_digest})
    return checks


def _add(checks, name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))


def _sample_indices(labels, np):
    # Head/tail images protect ordering, while class boundaries ensure AUROC
    # and segmentation sanity checks see normal and anomalous cases.
    selected = set(range(min(2, len(labels))))
    selected.update(range(max(0, len(labels) - 2), len(labels)))
    for value in (0, 1):
        positions = np.flatnonzero(labels == value)
        if not len(positions):
            raise ValueError("Every sampled MVTec category requires both label classes.")
        selected.update((int(positions[0]), int(positions[-1])))
    return sorted(selected)


def _source_rows_equal(bank, raw, indices, np):
    if len(indices) != len(bank):
        return False
    for start in range(0, len(indices), 2048):
        end = start + 2048
        if not np.array_equal(bank[start:end], raw[indices[start:end]]):
            return False
    return True


def sampled_checks(assets_root, categories, device="cpu", rtol=1e-5, atol=1e-5):
    """Validate all bank rows and head/tail/class-boundary test-image scores."""
    # Imports happen only after the caller requests numerical checks.
    sys.path.insert(0, str(ROOT / "third_party/patchcore/src"))
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import torch
    import faiss
    import numpy as np
    from adapters.patchcore import _score_cache
    from experiments.retrieval import query
    from patchcore.common import RescaleSegmentor
    from patchcore.patchcore import PatchMaker

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for map checks but unavailable.")
    faiss.omp_set_num_threads(8)
    torch.set_num_threads(2)  # Contract/map validation, not a performance benchmark.
    assets_root = Path(assets_root).resolve()
    original_root = assets_root / "frozen" / RESULTS_REL / "IM224_WR50_STAGE5B"
    prior_root = assets_root / "runs/verified_15_accuracy" / RESULTS_REL
    compared_root = prior_root / "IM224_WR50_STAGE8C"
    checks, sampled, largest_score_difference, largest_map_difference = [], [], 0.0, 0.0
    for category in categories:
        print("VALIDATE SAMPLE", category, flush=True)
        cached = original_root / category / "cache"
        raw = np.load(cached / "training_embeddings.npy", mmap_mode="r")
        original = np.load(cached / "original_memory_bank.npy", mmap_mode="r")
        selected = np.load(cached / "selected_indices.npy")
        features = np.load(cached / "test_embeddings.npy", mmap_mode="r")
        labels = np.load(cached / "labels.npy")
        masks = np.load(cached / "masks_gt.npy", mmap_mode="r")
        metadata = _read_json(cached / "test_cache.json")
        _add(checks, category + ":test_shape", features.shape == (len(labels) * 784, 1024))
        _add(checks, category + ":mask_shape", masks.shape == (len(labels), 1, 224, 224))
        _add(checks, category + ":metadata_coverage", sum(metadata["batch_sizes"]) == len(labels))
        _add(checks, category + ":labels_order", np.array_equal(labels == 0, np.asarray(metadata["subtypes"]) == "good"))
        selected_valid = (selected.ndim == 1 and len(selected) == len(original)
                          and len(np.unique(selected)) == len(selected)
                          and selected.min() >= 0 and selected.max() < len(raw))
        _add(checks, category + ":baseline_indices", selected_valid)
        _add(checks, category + ":baseline_source_rows",
             selected_valid and _source_rows_equal(original, raw, selected, np))
        with np.load(cached / "gb_partition.npz", allow_pickle=False) as partition:
            center_reference = np.array(partition["centers"], copy=True)
        k = len(center_reference)
        indices = _sample_indices(labels, np)
        sampled_features = np.ascontiguousarray(np.concatenate([
            features[index * 784:(index + 1) * 784] for index in indices]), dtype=np.float32)
        sampled_masks = np.asarray(masks[indices], dtype=np.float32)
        sampled_labels = labels[indices]
        batches = []
        for start in range(0, len(indices), 2):
            batch_size = min(2, len(indices) - start)
            batches.append({"features": sampled_features[start * 784:(start + batch_size) * 784],
                            "batch_size": batch_size, "patch_shape": [28, 28]})
        cache = {"batches": batches, "labels": sampled_labels,
                 "masks_gt": list(sampled_masks), "report": {"feature_extraction_seconds": 0.0}}
        # _score_cache needs these scoring components, not a CNN backbone.
        model = SimpleNamespace(patch_maker=PatchMaker(3, stride=1),
                                anomaly_segmentor=RescaleSegmentor(selected_device, target_size=(224, 224)))
        method_rows = []
        for method in METHODS:
            if method == "original":
                bank = np.ascontiguousarray(original)
                previous_scores = prior_root / "IM224_WR50_STAGE5B" / category / "runs/original/image_scores.npy"
            else:
                bank_path = compared_root / category / "cache" / (method + "_bank.npy")
                bank = np.load(bank_path, mmap_mode="r")
                previous_scores = compared_root / category / "runs" / method / "image_scores.npy"
            _add(checks, category + ":" + method + ":float32", bank.dtype == np.float32)
            _add(checks, category + ":" + method + ":finite", np.isfinite(bank).all())
            _add(checks, category + ":" + method + ":dimension", bank.ndim == 2 and bank.shape[1] == 1024)
            if method != "original":
                _add(checks, category + ":" + method + ":same_k", len(bank) == k)
                if method == "gb_center":
                    _add(checks, category + ":center_frozen_partition", np.array_equal(bank, center_reference))
                else:
                    filename = "gb_medoid_indices.npy" if method == "gb_medoid" else "ff_raw_order.npy"
                    bank_indices = np.load(compared_root / category / "cache" / filename)
                    valid = (bank_indices.ndim == 1 and len(bank_indices) == k
                             and len(np.unique(bank_indices)) == k
                             and bank_indices.min() >= 0 and bank_indices.max() < len(raw))
                    _add(checks, category + ":" + method + ":indices", valid)
                    _add(checks, category + ":" + method + ":source_rows",
                         valid and _source_rows_equal(bank, raw, bank_indices, np))
            distances = query(bank, sampled_features).reshape(len(indices), 28, 28)
            image_scores = distances.reshape(len(indices), -1).max(axis=1)
            archived_scores = np.load(previous_scores, mmap_mode="r")
            _add(checks, category + ":" + method + ":saved_score_count", len(archived_scores) == len(labels))
            expected = archived_scores[indices]
            difference = float(np.max(np.abs(image_scores - expected)))
            largest_score_difference = max(largest_score_difference, difference)
            _add(checks, category + ":" + method + ":sampled_archived_scores",
                 np.allclose(image_scores, expected, rtol=rtol, atol=atol),
                 max_absolute_difference=difference, image_indices=indices)
            scored = _score_cache(model, bank, cache, selected_device, 8, keep_maps=True)
            adapter_difference = float(np.max(np.abs(scored["scores"] - image_scores)))
            _add(checks, category + ":" + method + ":adapter_scores",
                 np.allclose(scored["scores"], image_scores, rtol=rtol, atol=atol),
                 max_absolute_difference=adapter_difference)
            expected_maps = np.asarray(model.anomaly_segmentor.convert_to_segmentation(distances), dtype=np.float32)
            map_difference = float(np.max(np.abs(scored["maps"] - expected_maps)))
            largest_map_difference = max(largest_map_difference, map_difference)
            _add(checks, category + ":" + method + ":adapter_map_consistency",
                 np.allclose(scored["maps"], expected_maps, rtol=rtol, atol=atol),
                 max_absolute_difference=map_difference,
                 scope="direct sampled queries plus unchanged official RescaleSegmentor; archived maps are unavailable")
            _add(checks, category + ":" + method + ":adapter_faiss_vectors", scored["faiss_validation"]["pass"])
            method_rows.append({"method": method, "bank_rows": len(bank),
                                "sampled_image_count": len(indices), "sampled_image_indices": indices,
                                "max_saved_image_score_difference": difference,
                                "max_adapter_score_difference": adapter_difference,
                                "max_map_consistency_difference": map_difference,
                                "sampled_metrics_only": scored["metrics"]})
            del distances, expected_maps, scored, bank
        sampled.append({"category": category, "normal_images_in_sample": int((sampled_labels == 0).sum()),
                        "anomalous_images_in_sample": int((sampled_labels == 1).sum()),
                        "methods": method_rows})
        del raw, original, features, masks, cache, sampled_features, sampled_masks, center_reference, model
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
    return {"checks": checks, "categories": sampled, "faiss_workers": 8,
            "faiss_device": "CPU IndexFlatL2", "map_device": str(selected_device),
            "relative_tolerance": rtol, "absolute_tolerance": atol,
            "largest_saved_image_score_difference": largest_score_difference,
            "largest_internal_map_difference": largest_map_difference,
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__, "faiss": faiss.__version__}}


def validate(assets_root, out_path, categories=CATEGORIES, device="cpu", source_only=False):
    assets_root, out_path = Path(assets_root).resolve(), Path(out_path).resolve()
    if out_path == assets_root or assets_root in out_path.parents:
        raise ValueError("Validation report must not overwrite the full package's input files.")
    source = source_checks(assets_root)
    numerical = None if source_only else sampled_checks(assets_root, tuple(categories), device)
    checks = source + ([] if numerical is None else numerical["checks"])
    report = {"schema_version": 1, "passed": all(check["passed"] for check in checks),
              "scope": "AST preservation plus sampled frozen-input bank/query/score contracts; no full feature, construction, AUROC or timing rerun",
              "source_only": source_only, "assets_root": str(assets_root),
              "source_checks_count": len(source), "checks_count": len(checks),
              "failed_checks_count": sum(not check["passed"] for check in checks),
              "source_checks": source, "sampled_validation": numerical}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "verification/release_validation.json")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-only", action="store_true", help="Compare source ASTs without importing ML runtimes")
    args = parser.parse_args()
    if len(set(args.categories)) != len(args.categories):
        parser.error("Each category must appear only once.")
    try:
        report = validate(args.assets_root, args.out, args.categories, args.device, args.source_only)
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
        print("Release validation could not complete: " + str(error), file=sys.stderr)
        return 2
    print("Release validation: {} ({} checks, {} failures)".format(
        "PASS" if report["passed"] else "FAIL", report["checks_count"], report["failed_checks_count"]))
    print("Report: " + str(args.out.resolve()))
    for check in report["source_checks"] + ([] if report["sampled_validation"] is None else report["sampled_validation"]["checks"]):
        if not check["passed"]:
            print("FAIL: " + check["name"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
