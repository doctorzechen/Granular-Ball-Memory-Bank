"""Capacity controls: GB size guards and farthest-first over the original coreset."""
import gc
import json
from pathlib import Path
from experiments.result_io import save_json, save_scored, validate_existing_result

RESULTS_REL = Path("patchcore-inspection/results/MVTecAD_Results")

def _validate_capacity_order(order, count, pool_size):
    import numpy as np
    if (order.shape != (count,) or not np.issubdtype(order.dtype, np.integer)
            or len(np.unique(order)) != count or order.min() < 0 or order.max() >= pool_size):
        raise RuntimeError("Invalid capacity FF order; expected unique indices into the original coreset.")

def capacity(args):
    import numpy as np
    import torch
    from gb_memory.partition import build_granular_balls
    from baselines.farthest_first import global_farthest_first_order
    from adapters.patchcore import _build_original_model, _score_cache
    from experiments.cache_io import _load_test_cache
    device = torch.device(args.device)
    results_root = args.out_root / RESULTS_REL
    for category in args.categories:
        if category not in ("screw", "capsule", "cable", "pill"):
            continue
        cached = results_root / "IM224_WR50_STAGE5B" / category / "cache"
        raw = np.load(cached / "training_embeddings.npy", mmap_mode="r")
        memory = np.load(cached / "original_memory_bank.npy", mmap_mode="r")
        k_gb = json.loads((cached / "training_cache.json").read_text(encoding="utf-8"))["number_of_balls"]
        test = _load_test_cache(cached)
        model = _build_original_model(device, 8)
        for multiplier, child_size in ((2, 32), (4, 16)):
            if category != "screw" and multiplier == 4:
                continue
            method = "gb_center_x" + str(multiplier)
            target = results_root / "IM224_WR50_STAGE8C" / category / "runs" / method
            if (target / "result.json").is_file():
                validate_existing_result(target, len(test["labels"]), method, method + "_bank.npy")
                continue
            print("CAPACITY", category, method, "min_child", child_size, flush=True)
            partition = build_granular_balls(raw, min_ball_size=child_size,
                                             center_consistency_threshold=1.3,
                                             seed=0, device=device)
            bank = np.ascontiguousarray(partition.centers, dtype=np.float32)
            scored = _score_cache(model, bank, test, device, 8, keep_maps=False)
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / (method + "_bank.npy"), bank)
            save_scored(target, scored, method=method, k=len(bank),
                        memory_bank_size=len(bank), min_ball_size=child_size,
                        seed=0, gb_configuration=partition.configuration)
            del bank, partition, scored
        if category == "screw":
            # Fig.2(b) is FF over the ORIGINAL CORESET, not all raw patches.
            order_path = results_root / "IM224_WR50_STAGE8C/screw/cache/ff_coreset_capacity_order.npy"
            order_path.parent.mkdir(parents=True, exist_ok=True)
            count = min(4 * k_gb, len(memory))
            if order_path.exists():
                order = np.load(order_path)
            else:
                order, report = global_farthest_first_order(memory, count, 1024, 0, device)
                np.save(order_path, order)
                save_json(order_path.with_suffix(".json"), report)
            _validate_capacity_order(order, count, len(memory))
            for multiplier in (1, 2, 4):
                method = "ff_coreset_x" + str(multiplier)
                target = results_root / "IM224_WR50_STAGE8C/screw/runs" / method
                if (target / "result.json").is_file():
                    validate_existing_result(target, len(test["labels"]), method)
                    continue
                k = min(multiplier * k_gb, len(memory))
                print("CAPACITY", category, method, "K", k, flush=True)
                bank = np.ascontiguousarray(memory[order[:k]])
                scored = _score_cache(model, bank, test, device, 8, keep_maps=False)
                save_scored(target, scored, method=method, k=k, memory_bank_size=k,
                            source="original 10% coreset", seed=0)
        del raw, memory, model, test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


