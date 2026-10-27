import csv
import json

import numpy as np
import pytest

from eval import run_clustering as rc

VOCAB = {
    "cs.CL": "language grammar syntax sentence parsing token corpus linguistic",
    "cs.CV": "image pixel convolution camera segmentation detection vision object",
    "cs.DB": "database query index transaction relational schema storage join",
}


def make_benchmark(path, per=8):
    papers = []
    for cat, words in VOCAB.items():
        ws = words.split()
        for i in range(per):
            papers.append(
                {
                    "id": f"{cat}.{i}",
                    "title": f"{ws[i % 8]} {ws[(i + 1) % 8]} study",
                    "abstract": " ".join(ws[(i + j) % 8] for j in range(6)) + " " + words,
                    "category": cat,
                    "published": "2023-01-01",
                }
            )
    path.write_text(
        json.dumps({"meta": {"categories": list(VOCAB)}, "part_a": papers}), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def no_bertopic(monkeypatch):
    monkeypatch.setattr(rc, "bertopic_available", lambda: False)


def run(tmp_path, **kw):
    bench = tmp_path / "bench.json"
    make_benchmark(bench)
    return rc.run(
        out=tmp_path / "out" / "clustering.json",
        benchmark=bench,
        cache_dir=tmp_path / "cache",
        handlabel=tmp_path / "missing.csv",
        model_names=["hashing-bow"],
        load_encoder=lambda name: rc.hashing_encoder,
        seeds=[0, 1, 2],
        **kw,
    )


def test_full_run_recovers_obvious_clusters_and_reports_all_methods(tmp_path):
    out = run(tmp_path)
    rows = {(r["method"], r["k_mode"]): r for r in out["part_a"]["results"]}
    assert set(rows) == {
        ("kmeans", "k=true"),
        ("kmeans", "auto"),
        ("agglomerative", "k=true"),
        ("agglomerative", "auto"),
        ("hdbscan", "n/a"),
        ("hdbscan", "mcs=5"),
        ("hdbscan", "mcs=10"),
    }
    assert rows[("kmeans", "k=true")]["ari"]["mean"] > 0.9
    assert rows[("kmeans", "k=true")]["n_runs"] == 3  # randomised: one run per seed
    assert rows[("agglomerative", "k=true")]["n_runs"] == 1  # deterministic: once
    assert rows[("kmeans", "k=true")]["ari"]["std"] is not None
    assert out["part_a"]["n_classes"] == 3 and out["part_a"]["n_papers"] == 24
    assert out["meta"]["seeds"] == [0, 1, 2] and "packages" in out["meta"]


def test_missing_optional_and_unloadable_models_are_skipped_not_fatal(tmp_path):
    def loader(name):
        if name == "broken":
            raise OSError("no weights")
        return rc.hashing_encoder

    bench = tmp_path / "b.json"
    make_benchmark(bench)
    out = rc.run(
        out=tmp_path / "o.json",
        benchmark=bench,
        cache_dir=tmp_path / "c",
        handlabel=tmp_path / "none.csv",
        model_names=["broken", "ok"],
        load_encoder=loader,
        seeds=[0],
    )
    skipped = {s["item"]: s["reason"] for s in out["part_a"]["skipped"]}
    assert "no weights" in skipped["broken"]
    assert "not installed" in skipped["bertopic"]
    assert {r["embedding"] for r in out["part_a"]["results"]} == {"ok"}


def test_predictions_file_lets_failure_analysis_find_each_paper(tmp_path):
    run(tmp_path)
    preds = json.loads((tmp_path / "out" / "clustering_predictions.json").read_text())
    assert len(preds["papers"]) == len(preds["truth"]) == 24
    assert all(len(v) == 24 for v in preds["predictions"].values())
    assert "hashing-bow|kmeans|k=true" in preds["predictions"]


def test_part_b_not_run_without_labels(tmp_path):
    assert run(tmp_path)["part_b"]["status"] == "not run (no hand-label file)"
    assert rc.load_handlabels(tmp_path / "x.csv") == "not run (no hand-label file)"


def write_labels(path, labels_by_topic):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["topic", "paper_id", "title", "abstract", "cluster_label"]
        )
        w.writeheader()
        for topic, labels in labels_by_topic.items():
            for i, lab in enumerate(labels):
                words = VOCAB["cs.CL" if i % 2 else "cs.CV"]
                w.writerow(
                    {
                        "topic": topic,
                        "paper_id": f"{topic}{i}",
                        "title": words.split()[0],
                        "abstract": words,
                        "cluster_label": lab,
                    }
                )


def test_part_b_awaits_labels_when_template_is_blank_or_mostly_blank(tmp_path):
    p = tmp_path / "h.csv"
    write_labels(p, {"t": [""] * 10})
    assert rc.load_handlabels(p) == "not run (awaiting hand labels)"
    write_labels(p, {"t": ["a", "b"] + [""] * 8})  # only 20% labelled
    assert rc.load_handlabels(p) == "not run (awaiting hand labels)"


def test_part_b_runs_only_for_topics_that_are_labelled_enough(tmp_path):
    p = tmp_path / "h.csv"
    write_labels(p, {"done": ["a", "b"] * 6, "blank": [""] * 12})
    labelled = rc.load_handlabels(p)
    assert list(labelled) == ["done"]
    bench = tmp_path / "b.json"
    make_benchmark(bench)
    out = rc.run(
        out=tmp_path / "o.json",
        benchmark=bench,
        cache_dir=tmp_path / "c",
        handlabel=p,
        model_names=["h"],
        load_encoder=lambda name: rc.hashing_encoder,
        seeds=[0],
    )
    assert out["part_b"]["status"] == "run"
    assert out["part_b"]["topics"]["done"]["n_classes"] == 2
    assert out["part_b"]["topics"]["done"]["results"]


def test_quick_mode_is_model_free_and_labelled_as_a_smoke_test(tmp_path, monkeypatch):
    bench = tmp_path / "b.json"
    make_benchmark(bench)
    out = rc.run(out=tmp_path / "q.json", quick=True, benchmark=bench, cache_dir=tmp_path / "c")
    assert out["meta"]["quick"] and "Smoke test" in out["meta"]["note"]
    assert out["part_a"]["n_papers"] == 24  # 3 categories x 8 available
    assert out["part_b"]["status"] == "not run (quick mode)"


def test_quick_mode_works_without_benchmark_file(tmp_path):
    out = rc.run(
        out=tmp_path / "q.json",
        quick=True,
        benchmark=tmp_path / "missing.json",
        cache_dir=tmp_path / "c",
    )
    assert out["part_a"]["n_papers"] == 12
    assert out["part_a"]["n_classes"] == 3


def test_embeddings_are_cached_and_specter_gets_sep_format(tmp_path):
    calls = []

    def encoder(texts):
        calls.append(texts)
        return np.ones((len(texts), 4))

    papers = [{"title": "T", "abstract": "A"}]
    rc.embed("allenai-specter", papers, encoder, tmp_path)
    rc.embed("allenai-specter", papers, encoder, tmp_path)
    assert len(calls) == 1 and calls[0] == ["T[SEP]A"]
    assert rc.text_for("all-MiniLM-L6-v2", papers[0]) == "T. A"
