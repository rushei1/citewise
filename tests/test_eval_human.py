import csv
import json
import re

import pytest

from eval import make_human_eval as mh
from eval import score_human_eval as sh


def fixture(tmp_path, n=25, with_llm=True):
    bench = {
        "part_a": [{"id": f"p{i}", "abstract": f"abstract {i}", "title": "t"} for i in range(n)]
    }
    (tmp_path / "bench.json").write_text(json.dumps(bench))
    structured = lambda i: {  # noqa: E731
        "problem": [{"text": f"problem {i}", "evidence": "e"}],
        "method": [{"text": f"method {i}", "evidence": "e"}],
        "results": [],
        "limitations": [],
    }
    summaries = {"hf": [{"id": f"p{i}", "text": f"bart {i}", "error": None} for i in range(n)]}
    if with_llm:
        summaries["llm"] = [
            {
                "id": f"p{i}",
                "text": f"Problem: problem {i}",
                "structured": structured(i),
                "error": None,
            }
            for i in range(n)
        ]
        summaries["llm"][0]["error"] = "failed"  # excluded from sampling
    (tmp_path / "summaries.json").write_text(json.dumps(summaries))
    return tmp_path / "summaries.json", tmp_path / "bench.json"


def test_sheet_is_blinded_balanced_deterministic_and_consistent_with_the_key(tmp_path):
    summ, bench = fixture(tmp_path)
    key = mh.build(summ, bench, tmp_path / "h", n=20)
    rows = list(csv.DictReader((tmp_path / "h" / "ratings.csv").open(encoding="utf-8")))
    assert len(rows) == 20 and len({r["item_id"] for r in rows}) == 20
    assert "p0" not in {r["item_id"] for r in rows}  # the failed LLM paper is never sampled
    assert all(r[c] == "" for r in rows for c in mh.FIELDS[4:])
    assert set(rows[0]) == set(mh.FIELDS)
    for r in rows:  # the key really says which output is which
        a_is_hf = key[r["item_id"]]["A"] == "hf"
        assert r["output_A"].startswith("bart" if a_is_hf else "problem")
        assert r["output_B"].startswith("problem" if a_is_hf else "bart")
    sides = [k["A"] for k in key.values()]
    assert 0 < sides.count("hf") < 20  # not always the same system first
    again = mh.build(summ, bench, tmp_path / "h2", n=20)
    assert again == key


def test_llm_text_is_plain_prose_without_labels_or_evidence(tmp_path):
    summ, bench = fixture(tmp_path)
    mh.build(summ, bench, tmp_path / "h", n=20)
    text = (tmp_path / "h" / "ratings.csv").read_text(encoding="utf-8")
    assert "Problem:" not in text and re.search(r"problem (\d+) method \1", text)


def test_existing_ratings_are_never_overwritten_without_force(tmp_path):
    summ, bench = fixture(tmp_path)
    mh.build(summ, bench, tmp_path / "h", n=20)
    (tmp_path / "h" / "ratings.csv").write_text("my ratings")
    with pytest.raises(FileExistsError):
        mh.build(summ, bench, tmp_path / "h", n=20)
    assert (tmp_path / "h" / "ratings.csv").read_text() == "my ratings"
    mh.build(summ, bench, tmp_path / "h", n=20, force=True)


def test_needs_both_arms_and_enough_papers(tmp_path):
    summ, bench = fixture(tmp_path, with_llm=False)
    with pytest.raises(SystemExit, match="both arms"):
        mh.build(summ, bench, tmp_path / "h")
    summ, bench = fixture(tmp_path, n=10)
    with pytest.raises(SystemExit, match="need 20"):
        mh.build(summ, bench, tmp_path / "h")


# ---- scoring ----


def ratings(n, llm_better_by=1, hf_acc=3):
    key = {f"i{i}": {"A": "hf", "B": "llm"} if i % 2 else {"A": "llm", "B": "hf"} for i in range(n)}
    rows = []
    for i in range(n):
        a_is_hf = key[f"i{i}"]["A"] == "hf"
        hf, llm = hf_acc, hf_acc + llm_better_by
        rows.append(
            {
                "item_id": f"i{i}",
                "accuracy_A": hf if a_is_hf else llm,
                "usefulness_A": hf if a_is_hf else llm,
                "accuracy_B": llm if a_is_hf else hf,
                "usefulness_B": llm if a_is_hf else hf,
            }
        )
    return rows, key


def test_scores_are_mapped_through_the_key_and_aggregated():
    rows, key = ratings(10, llm_better_by=1)
    out = sh.score(rows, key)
    assert out["n_items"] == 10
    assert out["systems"]["hf"]["accuracy"] == {"mean": 3.0, "std": 0.0}
    assert out["systems"]["llm"]["usefulness"]["mean"] == 4.0
    assert out["llm_vs_hf"]["accuracy"]["llm_higher"] == 10
    assert out["llm_vs_hf"]["accuracy"]["mean_difference"] == 1.0
    assert out["llm_vs_hf"]["accuracy"]["wilcoxon_p"] < 0.05  # 10 pairs, all in one direction


def test_wilcoxon_is_skipped_when_too_few_differences():
    rows, key = ratings(5, llm_better_by=1)
    entry = sh.score(rows, key)["llm_vs_hf"]["accuracy"]
    assert entry["wilcoxon_p"] is None and "only 5 non-zero" in entry["wilcoxon_note"]
    tied, key = ratings(10, llm_better_by=0)
    e = sh.score(tied, key)["llm_vs_hf"]["usefulness"]
    assert e["tied"] == 10 and e["wilcoxon_p"] is None


@pytest.mark.parametrize("bad", ["", None, "x", "0", "6", "2.5"])
def test_refuses_missing_or_invalid_ratings(bad):
    rows, key = ratings(10)
    rows[3]["usefulness_B"] = bad
    with pytest.raises(sh.MissingRatings):
        sh.score(rows, key)


def test_refuses_unknown_items_and_empty_sheets():
    rows, key = ratings(10)
    with pytest.raises(sh.MissingRatings, match="not in key"):
        sh.score(rows, {k: v for k, v in key.items() if k != "i1"})
    with pytest.raises(sh.MissingRatings, match="no rows"):
        sh.score([], key)
