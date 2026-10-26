import threading
from collections.abc import Sequence
from typing import Protocol


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


class Clusterer:
    def __init__(self, embedder: EmbedderLike):
        self.embedder = embedder

    def cluster(self, texts: list[str], num_clusters: int) -> list[int]:
        """Return one cluster label per text, in order."""
        from sklearn.cluster import KMeans

        labels = (
            KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
            .fit(self.embedder.encode(texts))
            .labels_
        )
        return [int(label) for label in labels]
