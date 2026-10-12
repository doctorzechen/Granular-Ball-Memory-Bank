"""Small result persistence helpers shared by experiment runners."""
import json
from pathlib import Path

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_scored(directory, scored, **extra):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "image_scores.npy", scored["scores"])
    public = {k: v for k, v in scored.items() if k not in ("scores", "maps")}
    public.update(status="complete", **extra)
    save_json(directory / "result.json", public)
    return public


def validate_existing_result(directory, expected_images, method, bank_filename=None):
    """An interrupted/deleted capacity result must fail visibly on resume."""
    import numpy as np
    directory = Path(directory)
    record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if record.get("status") != "complete" or record.get("method") != method or "metrics" not in record:
        raise RuntimeError("Invalid result checkpoint: " + str(directory))
    path = directory / "image_scores.npy"
    if not path.is_file():
        raise RuntimeError("Incomplete result checkpoint: " + str(path))
    scores = np.load(path, allow_pickle=False)
    if scores.shape != (expected_images,) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid score checkpoint: " + str(path))
    if bank_filename is not None:
        path = directory / bank_filename
        if not path.is_file():
            raise RuntimeError("Incomplete capacity bank checkpoint: " + str(path))
        bank = np.load(path, mmap_mode="r", allow_pickle=False)
        if bank.shape != (int(record["k"]), 1024) or bank.dtype != np.float32 or not np.isfinite(bank).all():
            raise RuntimeError("Invalid capacity bank checkpoint: " + str(path))
    return record


