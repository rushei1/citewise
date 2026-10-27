"""Compare clustering methods across embedding models on the benchmark.

Methods: KMeans and agglomerative (each with k = the true number of classes, and with k chosen
automatically by silhouette), HDBSCAN, and BERTopic when installed. Randomised methods run with
several seeds and are reported as mean +/- std (population std over seeds); deterministic ones run
once. Embeddings are L2-normalised, as in the app. Writes eval/results/clustering.json.
"""

import argparse
import csv
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.services.cluster import HDBSCANClusterer, _auto_k, _unit
from eval.common import (
    BENCHMARK,
    CACHE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    environment,
    key_hash,
    load_json,
    paper_text,
    save_json,
    set_seeds,
)
from eval.metrics import cluster_metrics, mean_std

log = logging.getLogger(__name__)

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "BAAI/bge-small-en-v1.5",
]
SEEDS = [0, 1, 2, 3, 4]
METRICS = [
    "ari",
    "nmi",
    "homogeneity",
    "completeness",
    "silhouette",
    "n_clusters",
    "noise_fraction",
]
HANDLABEL = DATA_DIR / "handlabel_template.csv"
MIN_LABELLED_FRACTION = 0.8

Encoder = Callable[[list[str]], np.ndarray]
# (name, needs_k, randomised)
METHODS = [
    ("kmeans", "k=true", True),
    ("kmeans", "auto", True),
    ("agglomerative", "k=true", False),
    ("agglomerative", "auto", False),
    ("hdbscan", "n/a", False),  # the app's own min_cluster_size rule
    ("hdbscan", "mcs=5", False),  # sensitivity: fixed min_cluster_size
    ("hdbscan", "mcs=10", False),
    ("bertopic", "n/a", True),
]


def hashing_encoder(texts: list[str]) -> np.ndarray:
    """Bag-of-words feature hashing. Only used by --quick, as a model-free stand-in."""
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(n_features=256, stop_words="english", alternate_sign=False)
    return vec.transform(texts).toarray()


def sbert_encoder(model_name: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: np.asarray(model.encode(texts, show_progress_bar=False))


def text_for(model_name: str, p: dict) -> str:
    # SPECTER was trained on "title [SEP] abstract"; the others get the app's "title. abstract"
    if "specter" in model_name:
        return f"{p['title']}[SEP]{p['abstract']}"
    return paper_text(p)


def embed(model_name: str, papers: list[dict], encoder: Encoder, cache_dir: Path) -> np.ndarray:
    texts = [text_for(model_name, p) for p in papers]
    path = cache_dir / "embeddings" / f"{key_hash([model_name, texts])}.npy"
    if path.exists():
        return np.load(path)
    x = np.asarray(encoder(texts), dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def fit_kmeans(x, k, seed):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(x)


def fit_agglomerative(x, k, seed=None):
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=k).fit_predict(x)


def bertopic_available() -> bool:
    try:
        import bertopic  # noqa: F401
        import umap  # noqa: F401
    except ImportError:
        return False
    return True


def fit_bertopic(x, docs, seed) -> list[int]:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    n = len(docs)
    model = BERTopic(
        embedding_model=None,  # embeddings are supplied, so every method sees the same vectors
        umap_model=UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed
        ),
        hdbscan_model=HDBSCAN(min_cluster_size=max(2, round(n * 0.1)), prediction_data=False),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=x)
    return [int(t) for t in topics]


def predict(method: str, k_mode: str, x: np.ndarray, k_true: int, seed: int, docs) -> list[int]:
    if method == "kmeans":
        k = k_true if k_mode == "k=true" else _auto_k(x, lambda v, kk: fit_kmeans(v, kk, seed))
        return [int(label) for label in fit_kmeans(x, k, seed)]
    if method == "agglomerative":
        k = k_true if k_mode == "k=true" else _auto_k(x, fit_agglomerative)
        return [int(label) for label in fit_agglomerative(x, k)]
    if method == "hdbscan":
        if k_mode.startswith("mcs="):
            from sklearn.cluster import HDBSCAN

            model = HDBSCAN(min_cluster_size=int(k_mode.removeprefix("mcs=")))
            return [int(label) for label in model.fit_predict(x)]
        return HDBSCANClusterer().cluster(x, None)
    if method == "bertopic":
        return fit_bertopic(x, docs, seed)
    raise ValueError(method)


def evaluate_config(method, k_mode, randomised, x, truth, docs, seeds) -> tuple[dict, list[int]]:
    """Run one (method, k_mode) on one embedding; returns the summary row and seed-0 labels."""
    k_true = len(set(truth))
    runs, first_labels = [], None
    for seed in seeds if randomised else seeds[:1]:
        t0 = time.perf_counter()
        labels = predict(method, k_mode, x, k_true, seed, docs)
        elapsed = time.perf_counter() - t0
        if first_labels is None:
            first_labels = labels
        runs.append({**cluster_metrics(truth, labels, x), "runtime_s": elapsed})
    row = {"method": method, "k_mode": k_mode, "n_runs": len(runs)}
    for m in [*METRICS, "runtime_s"]:
        mean, std = mean_std([r[m] for r in runs])
        row[m] = {"mean": mean, "std": std}
    return row, first_labels


def run_part(papers, truth, model_names, load_encoder, cache_dir, seeds, label):
    """All methods x all embedding models on one set of papers."""
    results, skipped, predictions = [], [], {}
    docs = [paper_text(p) for p in papers]
    for model_name in model_names:
        try:
            x = _unit(embed(model_name, papers, load_encoder(model_name), cache_dir))
        except Exception as e:  # noqa: BLE001 - a model that cannot load must not stop the run
            log.warning("%s: embedding model %s skipped: %s", label, model_name, e)
            skipped.append({"item": model_name, "reason": f"could not load/encode: {e}"})
            continue
        for method, k_mode, randomised in METHODS:
            if method == "bertopic" and not bertopic_available():
                reason = "bertopic is not installed (optional extra)"
                if not any(s["item"] == "bertopic" for s in skipped):
                    skipped.append({"item": "bertopic", "reason": reason})
                continue
            try:
                row, labels = evaluate_config(method, k_mode, randomised, x, truth, docs, seeds)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: %s/%s/%s failed: %s", label, model_name, method, k_mode, e)
                skipped.append(
                    {"item": f"{model_name}/{method}/{k_mode}", "reason": f"failed: {e}"}
                )
                continue
            results.append({"embedding": model_name, **row})
            predictions[f"{model_name}|{method}|{k_mode}"] = labels
            log.info(
                "%s %-22s %-13s %-6s ARI %.3f NMI %.3f",
                label,
                model_name,
                method,
                k_mode,
                row["ari"]["mean"],
                row["nmi"]["mean"],
            )
    return results, skipped, predictions


def load_handlabels(path: Path = HANDLABEL) -> dict[str, list[dict]] | str:
    """Topic -> labelled rows, or a 'not run' reason string."""
    if not path.exists():
        return "not run (no hand-label file)"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    topics: dict[str, list[dict]] = {}
    for r in rows:
        topics.setdefault(r["topic"], []).append(r)
    ready = {}
    for topic, items in topics.items():
        labelled = [r for r in items if r["cluster_label"].strip()]
        if (
            len(labelled) >= MIN_LABELLED_FRACTION * len(items)
            and len({r["cluster_label"].strip() for r in labelled}) >= 2
        ):
            ready[topic] = labelled
    if not ready:
        return "not run (awaiting hand labels)"
    return ready


def run(
    out: Path = RESULTS_DIR / "clustering.json",
    quick: bool = False,
    benchmark: Path = BENCHMARK,
    model_names: list[str] | None = None,
    load_encoder: Callable[[str], Encoder] | None = None,
    cache_dir: Path = CACHE_DIR,
    handlabel: Path = HANDLABEL,
    seeds: list[int] = SEEDS,
) -> dict:
    set_seeds()
    data = load_json(benchmark)
    papers = data["part_a"]
    if quick:  # smoke test: tiny subset, no models
        cats = data["meta"]["categories"][:3]
        papers = [p for c in cats for p in [q for q in papers if q["category"] == c][:10]]
        model_names = model_names or ["hashing-bow"]
        load_encoder = load_encoder or (lambda name: hashing_encoder)
        seeds = seeds[:2]
    model_names = model_names or EMBEDDING_MODELS
    load_encoder = load_encoder or sbert_encoder

    truth = [p["category"] for p in papers]
    results, skipped, preds = run_part(
        papers, truth, model_names, load_encoder, cache_dir, seeds, "A"
    )
    out_data = {
        "meta": environment(
            seeds=seeds,
            quick=quick,
            embedding_models=model_names,
            note="Quick mode: model-free hashing embeddings on a tiny subset. Smoke test only; "
            "these numbers say nothing about the real models."
            if quick
            else None,
        ),
        "part_a": {
            "description": "Proxy benchmark: ground truth = arXiv primary category.",
            "n_papers": len(papers),
            "n_classes": len(set(truth)),
            "results": results,
            "skipped": skipped,
        },
    }

    labels = load_handlabels(handlabel) if not quick else "not run (quick mode)"
    if isinstance(labels, str):
        out_data["part_b"] = {"status": labels}
    else:
        topics = {}
        for topic, rows in labels.items():
            b_papers = [
                {"title": r["title"], "abstract": r["abstract"], "paper_id": r["paper_id"]}
                for r in rows
            ]
            b_truth = [r["cluster_label"].strip() for r in rows]
            res, skip, _ = run_part(
                b_papers, b_truth, model_names, load_encoder, cache_dir, seeds, topic
            )
            topics[topic] = {
                "n_papers": len(rows),
                "n_classes": len(set(b_truth)),
                "results": res,
                "skipped": skip,
            }
        out_data["part_b"] = {"status": "run", "topics": topics}

    save_json(out, out_data)
    save_json(
        out.with_name(out.stem + "_predictions.json"),
        {"papers": [p["id"] for p in papers], "truth": truth, "predictions": preds},
    )
    log.info("wrote %s", out)
    return out_data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="tiny model-free smoke test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (RESULTS_DIR.with_name("results_quick") if args.quick else RESULTS_DIR)
    run(out=out / "clustering.json", quick=args.quick)


if __name__ == "__main__":
    main()
