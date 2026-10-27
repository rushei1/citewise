import json

import numpy as np

from eval import failure_cases as fc
from eval.run_clustering import embed


def test_majority_class_breaks_ties_alphabetically():
    assert fc.majority_class(["b", "a", "a", "b", "b"], [0, 0, 1, 1, 1]) == {0: "a", 1: "b"}


def test_clustering_mistakes_are_ranked_by_distance_from_their_true_centroid():
    # class a: two near-origin points and one outlier that the clustering put with class b
    truth = ["a", "a", "a", "b", "b", "b"]
    labels = [0, 0, 1, 1, 1, 1]
    x = np.array([[0.0, 0], [0.1, 0], [5, 5], [5, 5.1], [5.1, 5], [5, 5]])
    res = fc.analyze_clustering(list("uvwxyz"), [f"T{c}" for c in "uvwxyz"], truth, labels, x)
    assert res["n_mistakes"] == 1
    assert res["farthest"][0]["id"] == "w" and res["farthest"][0]["assigned"] == "b"
    assert res["confused_pairs"] == [{"true": "a", "assigned": "b", "count": 1}]


def test_noise_counts_as_a_mistake_and_pairs_are_ordered_by_count():
    truth = ["a", "a", "b", "b", "b"]
    labels = [0, -1, 1, -1, -1]
    x = np.eye(5)
    res = fc.analyze_clustering(list("12345"), list("ABCDE"), truth, labels, x)
    assert res["n_mistakes"] == 3
    assert res["confused_pairs"][0] == {"true": "b", "assigned": "(noise)", "count": 2}


def row(i, kept, dropped, error=None):
    s = {
        f: [{"text": "t", "evidence": "e"}] * 0
        for f in ("problem", "method", "results", "limitations")
    }
    s["problem"] = [{"text": "t", "evidence": "e"}] * kept
    s["unverified_claims"] = dropped
    return {"id": f"p{i}", "error": error, "structured": None if error else s}


def test_lowest_grounding_orders_by_rate_then_dropped_and_lists_failures():
    titles = {f"p{i}": f"Title {i}" for i in range(5)}
    rows = [row(0, 3, 0), row(1, 1, 3), row(2, 1, 1), row(3, 0, 0), row(4, 0, 0, error="bad json")]
    res = fc.lowest_grounding(rows, titles, top=2)
    assert [w["id"] for w in res["worst"]] == ["p1", "p2"]  # 25% then 50%; p3 has no claims
    assert res["worst"][0]["rate"] == 0.25
    assert res["failed"] == [{"id": "p4", "title": "Title 4", "error": "bad json"}]


def test_best_config_and_empty_results():
    rows = [
        {"embedding": "a", "method": "m", "k_mode": "x", "ari": {"mean": 0.2}},
        {"embedding": "b", "method": "m", "k_mode": "x", "ari": {"mean": 0.7}},
    ]
    assert fc.best_config({"part_a": {"results": rows}})["embedding"] == "b"
    assert fc.best_config({"part_a": {"results": []}}) is None


def test_build_reports_not_run_when_inputs_are_missing(tmp_path):
    bench = tmp_path / "b.json"
    bench.write_text(
        json.dumps({"part_a": [{"id": "p", "title": "T", "abstract": "A", "category": "c"}]})
    )
    text = fc.build(tmp_path / "none", bench, tmp_path / "cache")
    assert "not run (no clustering results)" in text
    assert "not run (the LLM arm has not been run)" in text


def test_build_end_to_end_uses_cached_embeddings_without_loading_a_model(tmp_path):
    papers = [
        {"id": f"p{i}", "title": f"T{i}", "abstract": f"A{i}", "category": "ab"[i // 3]}
        for i in range(6)
    ]
    bench = tmp_path / "b.json"
    bench.write_text(json.dumps({"part_a": papers}))
    res = tmp_path / "r"
    res.mkdir()
    cfg = {"embedding": "m", "method": "kmeans", "k_mode": "k=true", "ari": {"mean": 0.5}}
    (res / "clustering.json").write_text(json.dumps({"part_a": {"results": [cfg]}}))
    (res / "clustering_predictions.json").write_text(
        json.dumps(
            {
                "papers": [p["id"] for p in papers],
                "truth": [p["category"] for p in papers],
                "predictions": {"m|kmeans|k=true": [0, 0, 1, 1, 1, 1]},
            }
        )
    )
    vecs = np.array([[1.0, 0], [1, 0.1], [1, 0.2], [0, 1], [0.1, 1], [0, 1.1]])
    embed("m", papers, lambda texts: vecs, tmp_path / "cache")  # prime the cache

    def no_model(name):
        raise AssertionError("model must not be loaded when embeddings are cached")

    text = fc.build(res, bench, tmp_path / "cache", encoder_for=no_model)
    assert "m / kmeans / k=true" in text and "1 of 6 papers" in text
    assert "`p2` (a -> b" in text
