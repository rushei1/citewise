import numpy as np

from app.schemas import PaperIn
from app.services.cluster import EmbedderLike


def rank_by_relevance(
    query: str, papers: list[PaperIn], embedder: EmbedderLike, k: int
) -> list[PaperIn]:
    """Top-k papers by cosine similarity between the query and title + abstract.

    Ties keep the incoming order, so output is deterministic for a deterministic embedder."""
    if not papers:
        return []
    vectors = np.asarray(
        embedder.encode([query] + [f"{p.title}. {p.abstract}" for p in papers]), dtype=float
    )
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0  # all-zero vector scores 0 instead of NaN
    unit = vectors / norms[:, None]
    scores = unit[1:] @ unit[0]
    order = sorted(range(len(papers)), key=lambda i: (-round(float(scores[i]), 12), i))
    return [papers[i] for i in order[:k]]
