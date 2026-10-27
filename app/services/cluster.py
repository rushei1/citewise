import threading
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

NOISE = -1  # HDBSCAN's label for papers that fit no cluster
MAX_AUTO_K = 8


class EmbedderLike(Protocol):
    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]: ...


class Embedder:
    """Sentence-transformers embedder; the model loads on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        return self._load().encode(texts)


def keyword_labels(docs: dict[str, list[str]], top_n: int = 3) -> dict[str, str]:
    """Short label per cluster from its top c-TF-IDF terms (no LLM). `docs` maps cluster id to
    the texts of its papers. The NOISE cluster gets a fixed label and takes no part in scoring."""
    labels = {"-1": "Misc / unclustered"} if str(NOISE) in docs else {}
    ids = [i for i in docs if i != str(NOISE)]
    if not ids:
        return labels
    from sklearn.feature_extraction.text import CountVectorizer

    try:
        vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        counts = vec.fit_transform([" ".join(docs[i]) for i in ids]).toarray().astype(float)
    except ValueError:  # nothing left after stop words
        return labels | {i: f"Cluster {int(i) + 1}" for i in ids}
    terms = vec.get_feature_names_out()
    # c-TF-IDF: term frequency inside the cluster, down-weighted by how common the term is overall
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + counts.sum() / len(ids) / counts.sum(axis=0))
    scores = tf * idf
    for row, cid in enumerate(ids):
        top = sorted(range(len(terms)), key=lambda j: (-scores[row, j], terms[j]))[:top_n]
        picked = [str(terms[j]) for j in top if scores[row, j] > 0]
        labels[cid] = ", ".join(picked) or f"Cluster {int(cid) + 1}"
    return labels


class Clusterer(Protocol):
    name: str

    def cluster(self, vectors: Sequence[Sequence[float]], k: int | None) -> list[int]:
        """One label per vector, in order. k=None means choose automatically; methods that
        find their own cluster count (HDBSCAN) ignore k. Never raises for tiny inputs."""
        ...


def _unit(vectors) -> np.ndarray:
    """L2-normalize so euclidean distance orders points like cosine similarity."""
    x = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def _auto_k(x: np.ndarray, fit: Callable[[np.ndarray, int], np.ndarray]) -> int:
    """k in 2..min(8, n-1) with the best silhouette score; the smaller k wins ties.
    Falls back to 1 when there are too few points to score a split."""
    best_score, best_k = -2.0, 1
    for k in range(2, min(MAX_AUTO_K, len(x) - 1) + 1):
        labels = fit(x, k)
        if len(set(labels)) < 2:
            continue
        from sklearn.metrics import silhouette_score

        score = float(silhouette_score(x, labels))
        if score > best_score:
            best_score, best_k = score, k
    return best_k


class _FixedK:
    """Shared k handling for methods that take a cluster count."""

    name: str

    def _fit(self, x: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError

    def cluster(self, vectors, k: int | None) -> list[int]:
        n = len(vectors)
        if n == 0:
            return []
        x = _unit(vectors)
        k = _auto_k(x, self._fit) if k is None else min(k, n)
        if k <= 1:
            return [0] * n
        return [int(label) for label in self._fit(x, k)]


class KMeansClusterer(_FixedK):
    name = "kmeans"

    def _fit(self, x, k):
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(x)


class AgglomerativeClusterer(_FixedK):
    name = "agglomerative"

    def _fit(self, x, k):
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k).fit_predict(x)  # ward linkage


class HDBSCANClusterer:
    """Density-based: finds its own number of clusters and marks outliers as NOISE."""

    name = "hdbscan"

    def cluster(self, vectors, k: int | None = None) -> list[int]:
        n = len(vectors)
        if n < 3:  # too few points to talk about density
            return [0] * n
        from sklearn.cluster import HDBSCAN

        # ponytail: min_cluster_size is a guess (10% of n, at least 2), tune with the Phase 4 eval
        labels = HDBSCAN(min_cluster_size=max(2, round(n * 0.1))).fit_predict(_unit(vectors))
        return [int(label) for label in labels]


CLUSTERERS: dict[str, Clusterer] = {
    c.name: c for c in (KMeansClusterer(), AgglomerativeClusterer(), HDBSCANClusterer())
}
