"""Shared helpers for the evaluation scripts: on-disk call cache, seeds, environment record."""

import hashlib
import importlib.metadata
import json
import platform
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".eval_cache"
DATA_DIR = ROOT / "eval" / "data"
RESULTS_DIR = ROOT / "eval" / "results"
BENCHMARK = DATA_DIR / "benchmark.json"
SEED = 0

PACKAGES = [
    "numpy",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "transformers",
    "torch",
    "arxiv",
]


def key_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cached(namespace: str, key, fn: Callable[[], object], cache_dir: Path | None = None):
    """Return fn()'s JSON-serializable result, computing it only the first time for this key.

    Every external call (arXiv, LLM) goes through here, so reruns are free and reproducible."""
    path = (cache_dir or CACHE_DIR) / namespace / f"{key_hash(key)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def environment(seeds: list[int] | None = None, **extra) -> dict:
    """What produced a result: versions, platform, date, seeds. Goes into every results file."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "seeds": seeds if seeds is not None else [SEED],
        **extra,
    }


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paper_text(p: dict) -> str:
    """The text that is embedded for a benchmark paper (same shape the app embeds)."""
    return f"{p['title']}. {p['abstract']}"
