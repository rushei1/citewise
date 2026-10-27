"""Metric checks against values worked out by hand."""

import math

import pytest

from eval.metrics import (
    cluster_metrics,
    compression_ratio,
    mean_std,
    rouge_l,
    rouge_n,
    tokens,
)


def test_perfect_clustering_scores_one_regardless_of_label_names():
    m = cluster_metrics([0, 0, 1, 1], [5, 5, 9, 9])
    assert (m["ari"], m["nmi"], m["homogeneity"], m["completeness"]) == (1.0, 1.0, 1.0, 1.0)
    assert m["n_clusters"] == 2


def test_orthogonal_clustering_has_negative_ari_and_zero_nmi():
    # pairs: 2 of 6 agree on "same cluster"... ARI = -0.5 for this classic example
    m = cluster_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert m["ari"] == pytest.approx(-0.5)
    assert m["nmi"] == pytest.approx(0.0, abs=1e-12)


def test_one_big_cluster_is_complete_but_not_homogeneous():
    m = cluster_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert m["homogeneity"] == pytest.approx(0.0, abs=1e-12)
    assert m["completeness"] == pytest.approx(1.0)
    assert m["n_clusters"] == 1


def test_all_singletons_are_homogeneous_but_not_complete():
    m = cluster_metrics([0, 0, 1, 1], [0, 1, 2, 3])
    assert m["homogeneity"] == pytest.approx(1.0)
    # completeness = 1 - H(cluster | class) / H(cluster) = 1 - ln2 / ln4
    assert m["completeness"] == pytest.approx(1 - math.log(2) / math.log(4))


def test_noise_is_a_prediction_but_not_a_cluster():
    m = cluster_metrics([0, 0, 1, 1], [0, 0, -1, -1])
    assert m["n_clusters"] == 1
    assert m["noise_fraction"] == 0.5
    assert m["ari"] == pytest.approx(1.0)  # noise grouped together matches class 1 exactly


def test_silhouette_needs_two_clusters_and_vectors():
    pts = [[0, 0], [0, 1], [10, 10], [10, 11]]
    assert cluster_metrics([0, 0, 1, 1], [0, 0, 1, 1], pts)["silhouette"] > 0.9
    assert cluster_metrics([0, 0, 1, 1], [0, 0, 0, 0], pts)["silhouette"] is None
    assert cluster_metrics([0, 0, 1, 1], [0, 1, 2, 3], pts)["silhouette"] is None  # all singletons
    assert cluster_metrics([0, 0, 1, 1], [0, 0, 1, 1])["silhouette"] is None  # no vectors


def test_mean_std():
    assert mean_std([1.0, 3.0]) == (2.0, 1.0)
    assert mean_std([None, 2.0]) == (2.0, 0.0)
    assert mean_std([None]) == (None, None)
    assert mean_std([]) == (None, None)


def test_tokens():
    assert tokens("The U-Net, 2D!") == ["the", "u", "net", "2d"]


def test_rouge_1_hand_computed():
    # ref: the cat sat (3); hyp: the cat ran home (4); overlap = the, cat = 2
    r = rouge_n("the cat sat", "the cat ran home", 1)
    assert r["precision"] == pytest.approx(2 / 4)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_rouge_2_counts_bigrams_with_multiplicity():
    # ref bigrams: (a b)(b a)(a b) ; hyp bigrams: (a b)(b a) -> overlap min counts = 2
    r = rouge_n("a b a b", "a b a", 2)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["precision"] == pytest.approx(2 / 2)


def test_rouge_l_uses_longest_common_subsequence():
    # LCS of "a b c d" and "a x c d" is "a c d" (3)
    r = rouge_l("a b c d", "a x c d")
    assert r["precision"] == pytest.approx(3 / 4) and r["recall"] == pytest.approx(3 / 4)
    assert rouge_l("a b", "b a")["recall"] == pytest.approx(1 / 2)  # order matters


@pytest.mark.parametrize(("ref", "hyp"), [("", "a b"), ("a b", ""), ("", "")])
def test_rouge_empty_inputs_score_zero_without_dividing_by_zero(ref, hyp):
    assert rouge_n(ref, hyp, 1)["f1"] == 0.0
    assert rouge_l(ref, hyp)["f1"] == 0.0


def test_rouge_identical_text_is_one():
    assert rouge_n("same text here", "Same TEXT here!", 1)["f1"] == 1.0
    assert rouge_n("same text here", "same text here", 2)["f1"] == 1.0


def test_compression_ratio():
    assert compression_ratio("one two three four", "one two") == 0.5
    assert compression_ratio("", "x") == 0.0
