"""Metric functions. Small and dependency-light so they can be checked against hand-computed
values in tests."""

import re
from collections import Counter

import numpy as np


def cluster_metrics(true: list, pred: list, vectors=None) -> dict:
    """External metrics against ground-truth classes, plus silhouette on the embeddings.

    A label of -1 (HDBSCAN noise) counts as one extra cluster for ARI/NMI/homogeneity/
    completeness (noise is a prediction like any other); `n_clusters` excludes it."""
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    n_pred = len({p for p in pred if p != -1})
    out = {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "homogeneity": float(homogeneity_score(true, pred)),
        "completeness": float(completeness_score(true, pred)),
        "n_clusters": n_pred,
        "noise_fraction": float(np.mean([p == -1 for p in pred])) if len(pred) else 0.0,
        "silhouette": None,
    }
    distinct = len(set(pred))
    if vectors is not None and 2 <= distinct <= len(pred) - 1:
        out["silhouette"] = float(silhouette_score(np.asarray(vectors), pred))
    return out


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))  # population std over seeds


def tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens. No stemming or stopword removal, so scores are not
    comparable with the `rouge-score` package's numbers (which stems)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _prf(overlap: int, n_ref: int, n_hyp: int) -> dict:
    p = overlap / n_hyp if n_hyp else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def rouge_n(reference: str, hypothesis: str, n: int) -> dict:
    def grams(toks):
        return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))

    ref, hyp = grams(tokens(reference)), grams(tokens(hypothesis))
    overlap = sum((ref & hyp).values())
    return _prf(overlap, sum(ref.values()), sum(hyp.values()))


def rouge_l(reference: str, hypothesis: str) -> dict:
    """Longest-common-subsequence ROUGE over tokens."""
    a, b = tokens(reference), tokens(hypothesis)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return _prf(prev[-1], len(a), len(b))


def rouge_all(reference: str, hypothesis: str) -> dict:
    return {
        "rouge1": rouge_n(reference, hypothesis, 1)["f1"],
        "rouge2": rouge_n(reference, hypothesis, 2)["f1"],
        "rougeL": rouge_l(reference, hypothesis)["f1"],
    }


def compression_ratio(abstract: str, summary: str) -> float:
    """Summary words per abstract word (lower = shorter)."""
    n = len(abstract.split())
    return len(summary.split()) / n if n else 0.0
