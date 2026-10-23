"""Configuration and paths, always resolved relative to the project root."""

import hashlib
import json
from pathlib import Path


def load_config(path="configs/baseline.json", root=None):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["root"] = str(Path(root).resolve() if root else path.parent.parent)
    if cfg["features"]["recipe"] not in {"baseline", "extended"}:
        raise ValueError("features.recipe must be baseline or extended")
    for key in ("partitions", "threads"):
        if cfg["processing"][key] < 1:
            raise ValueError(f"processing.{key} must be positive")
    if cfg["cv"]["folds"] < 2 or cfg["prediction"]["batch_size"] < 1:
        raise ValueError("Use at least two folds and a positive prediction batch size")
    return cfg


def resolve(cfg, key):
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else Path(cfg["root"]) / path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan(cfg):
    """Describe the workflow without opening any input files."""
    return {
        "experiment": cfg["experiment"],
        "recipe": cfg["features"]["recipe"],
        "paths": {key: str(resolve(cfg, key)) for key in cfg["paths"]},
        "steps": ["prepare train", "train", "report", "prepare test", "predict", "profile"],
        "folds": cfg["cv"]["folds"],
        "processing": cfg["processing"],
    }
