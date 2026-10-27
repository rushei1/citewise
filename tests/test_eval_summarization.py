import json
import re
from types import SimpleNamespace

import pytest

from app.schemas import PaperSummary
from eval import run_summarization as rs
from eval.common import cached


def make_papers(per_cat=4, cats=("cs.CL", "cs.CV", "cs.DB")):
    return [
        {
            "id": f"{c}.{i}",
            "title": f"Title {c} {i}",
            "abstract": f"First sentence about {c} number {i}. Second sentence adds detail here. "
            "Third sentence closes the abstract properly.",
            "category": c,
            "published": "2023-01-01",
        }
        for c in cats
        for i in range(per_cat)
    ]


class FakeLLM:
    """OpenAI-shaped client: quotes the first sentence as problem evidence, plus one invented
    quote, so the verifier has something to drop. Tracks usage like the real clients."""

    model = "fake-llm"

    def __init__(self, fail_on=()):
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        self.calls = 0
        self.fail_on = fail_on

    def complete(self, messages):
        self.calls += 1
        abstract = messages[1]["content"].split("Abstract:\n", 1)[1]
        if any(f in abstract for f in self.fail_on):
            return "not json"
        first = re.split(r"(?<=[.!?])\s", abstract)[0]
        self.usage.prompt_tokens += 100
        self.usage.completion_tokens += 20
        return json.dumps(
            {
                "problem": [{"text": "p", "evidence": first}],
                "method": [{"text": "m", "evidence": "a quote that is not in the abstract"}],
                "results": "not stated",
                "limitations": "not stated",
            }
        )


class FakeHF:
    id = "hf:fake"

    def summarize(self, paper):
        return PaperSummary(overview=paper.abstract.split(".")[0] + ".")


@pytest.fixture
def bench(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"meta": {}, "part_a": make_papers()}), encoding="utf-8")
    return path


def test_sample_is_round_robin_deterministic_and_bounded():
    papers = make_papers()
    s = rs.sample_papers(papers, 5)
    assert [p["category"] for p in s] == ["cs.CL", "cs.CV", "cs.DB", "cs.CL", "cs.CV"]
    assert s == rs.sample_papers(list(reversed(papers)), 5)  # input order does not matter
    assert len(rs.sample_papers(papers, 999)) == 12


def test_text_metrics_hand_checked():
    abstracts = {"a": "one two three four", "b": "x y"}
    rows = [
        {"id": "a", "text": "one two three four", "latency_s": 1.0, "error": None},
        {"id": "b", "text": "", "latency_s": 9.0, "error": "boom"},
    ]
    m = rs.text_metrics(rows, abstracts)
    assert (m["n"], m["n_failed"]) == (2, 1)
    assert m["rouge1"] == m["rouge2"] == m["rougeL"] == 1.0
    assert m["compression_ratio"] == 1.0 and m["latency_mean_s"] == 1.0  # failures excluded
    assert rs.text_metrics([rows[1]], abstracts) == {"n": 1, "n_failed": 1}


def test_grounding_metrics_count_dropped_before_verification():
    def row(problem, dropped, not_stated):
        s = {f: [] for f in ("problem", "method", "results", "limitations")}
        s["problem"] = [{"text": "t", "evidence": "e"}] * problem
        s |= {"not_stated": not_stated, "unverified_claims": dropped}
        return {"structured": s, "prompt_tokens": 10, "completion_tokens": 5}

    g = rs.grounding_metrics(
        [
            row(2, 1, ["limitations"]),
            row(1, 0, ["limitations", "results"]),
            {"structured": None, "prompt_tokens": 7, "completion_tokens": 0},
        ]
    )
    assert (g["claims_total"], g["claims_verified"], g["claims_dropped"]) == (4, 3, 1)
    assert g["verification_rate"] == 0.75 and g["dropped_rate"] == 0.25
    assert g["papers_with_any_dropped_claim"] == 1
    assert g["not_stated_rate"]["limitations"] == 1.0 and g["not_stated_rate"]["results"] == 0.5
    assert g["prompt_tokens"] == 27 and "not computed" in g["cost"]
    assert rs.grounding_metrics([])["verification_rate"] is None


def test_full_run_with_both_arms_and_nli(tmp_path, bench):
    client = FakeLLM()
    nli = lambda pairs: ["entailment"] * (len(pairs) - 1) + ["neutral"]  # noqa: E731
    out = rs.run(
        out=tmp_path / "o" / "summarization.json",
        limit=6,
        benchmark=bench,
        hf_summarizer=FakeHF(),
        llm_client=client,
        nli=nli,
        cache_dir=tmp_path / "c",
    )
    assert out["sample"]["n"] == 6 and client.calls == 6
    llm = out["arms"]["llm"]
    assert llm["status"] == "run" and llm["id"].startswith("llm:fake-llm:")
    assert (llm["claims_total"], llm["claims_verified"], llm["claims_dropped"]) == (12, 6, 6)
    assert llm["verification_rate"] == 0.5
    assert llm["prompt_tokens"] == 600 and llm["completion_tokens"] == 120
    assert out["arms"]["hf"]["n_failed"] == 0 and out["arms"]["hf"]["rouge1"] > 0
    assert out["nli"]["llm"]["unit"] == "claims" and out["nli"]["llm"]["n"] == 6
    assert out["nli"]["hf"]["unit"] == "sentences"
    saved = json.loads((tmp_path / "o" / "summaries.json").read_text())
    assert set(saved) == {"hf", "llm"} and len(saved["llm"]) == 6


def test_rerun_reads_the_cache_and_reports_identical_numbers(tmp_path, bench):
    kwargs = dict(
        limit=4,
        benchmark=bench,
        hf_summarizer=FakeHF(),
        nli="not run (test)",
        cache_dir=tmp_path / "c",
    )
    c1, c2 = FakeLLM(), FakeLLM()
    a = rs.run(out=tmp_path / "1.json", llm_client=c1, **kwargs)
    b = rs.run(out=tmp_path / "2.json", llm_client=c2, **kwargs)
    assert c1.calls == 4 and c2.calls == 0
    for k in ("verification_rate", "prompt_tokens", "completion_tokens", "latency_mean_s"):
        assert a["arms"]["llm"][k] == b["arms"]["llm"][k]


def test_failed_llm_papers_are_counted_and_not_cached(tmp_path, bench):
    client = FakeLLM(fail_on=("cs.CV number 0",))
    out = rs.run(
        out=tmp_path / "o.json",
        limit=3,
        benchmark=bench,
        hf_summarizer=FakeHF(),
        llm_client=client,
        nli="not run (test)",
        cache_dir=tmp_path / "c",
    )
    assert out["arms"]["llm"]["n_failed"] == 1
    again = FakeLLM()
    rs.run(
        out=tmp_path / "o2.json",
        limit=3,
        benchmark=bench,
        hf_summarizer=FakeHF(),
        llm_client=again,
        nli="not run (test)",
        cache_dir=tmp_path / "c",
    )
    assert again.calls == 1  # only the failed paper is retried


def test_no_credentials_means_llm_arm_not_run(tmp_path, bench):
    out = rs.run(
        out=tmp_path / "o.json",
        limit=3,
        benchmark=bench,
        hf_summarizer=FakeHF(),
        llm_client=None,
        nli="not run (NLI model unavailable: x)",
        cache_dir=tmp_path,
    )
    assert out["arms"]["llm"] == {"status": "not run (no credentials configured)"}
    assert out["nli"] == {"status": "not run (NLI model unavailable: x)"}
    assert "llm" not in json.loads((tmp_path / "summaries.json").read_text())


def test_quick_mode_uses_the_stub_and_says_so(tmp_path, bench):
    out = rs.run(
        out=tmp_path / "o.json", limit=3, quick=True, benchmark=bench, cache_dir=tmp_path / "c"
    )
    assert out["meta"]["quick"] and "Smoke test" in out["meta"]["note"]
    assert out["arms"]["hf"]["id"] == "stub:first-sentence"
    assert out["arms"]["llm"]["status"] == "not run (quick mode)"


def test_nli_helpers():
    assert rs.sentences("Short. This one has enough words. And so does this one.") == [
        "This one has enough words.",
        "And so does this one.",
    ]
    assert rs.nli_metrics([], {}, lambda p: [], "claims") == {
        "status": "not run (nothing to check)"
    }
    reason = rs.load_nli.__doc__
    assert "reason string" in reason


def test_arms_can_run_in_separate_processes_and_merge(tmp_path, bench):
    kw = dict(limit=6, benchmark=bench, cache_dir=tmp_path / "c")
    nli = lambda pairs: ["entailment"] * len(pairs)  # noqa: E731
    out = tmp_path / "s.json"
    first = rs.run(out=out, hf_summarizer=FakeHF(), arms=("hf",), **kw)
    assert first["arms"]["llm"] == {"status": "not run"} and first["nli"] == {"status": "not run"}
    rs.run(out=out, llm_client=FakeLLM(), arms=("llm",), **kw)
    merged = rs.run(out=out, nli=nli, arms=("nli",), **kw)

    together = rs.run(
        out=tmp_path / "t.json", hf_summarizer=FakeHF(), llm_client=FakeLLM(), nli=nli, **kw
    )
    for arm in ("hf", "llm"):
        for k in ("n", "n_failed", "rouge1", "rougeL", "id"):
            assert merged["arms"][arm][k] == together["arms"][arm][k]
    assert merged["nli"] == together["nli"]
    assert merged["arms"]["llm"]["verification_rate"] == 0.5
    assert set(merged["meta"]["arm_dates"]) == {"hf", "llm", "nli"}
    assert set(json.loads((tmp_path / "summaries.json").read_text())) == {"hf", "llm"}


def test_nli_arm_without_summaries_says_so_and_a_changed_sample_is_refused(tmp_path, bench):
    kw = dict(benchmark=bench, cache_dir=tmp_path / "c")
    out = tmp_path / "s.json"
    r = rs.run(out=out, limit=6, nli=lambda p: [], arms=("nli",), **kw)
    assert "run the hf arm first" in r["nli"]["status"]
    rs.run(out=out, limit=6, hf_summarizer=FakeHF(), arms=("hf",), **kw)
    with pytest.raises(ValueError, match="rerun all arms"):
        rs.run(out=out, limit=4, hf_summarizer=FakeHF(), arms=("hf",), **kw)


def test_cached_store_if_refuses_failures(tmp_path):
    calls = []
    for _ in range(2):
        cached(
            "n",
            "k",
            lambda: calls.append(1) or {"error": "x"},
            tmp_path,
            store_if=lambda v: v["error"] is None,
        )
    assert len(calls) == 2
