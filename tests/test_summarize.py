import json

import pytest

from app.main import app, get_summarizers
from app.schemas import PaperIn, PaperSummary
from app.services.llm import LLMError
from app.services.summarize import (
    LLMSummarizer,
    SummarizeError,
    evidence_in_abstract,
    parse_summary_json,
    summarize_all,
)
from tests.conftest import ScriptedLLM, make_papers

ABSTRACT = (
    "Transformers dominate NLP.  We propose a sparse attention method that reduces memory "
    "by 40% on long documents. Experiments on three benchmarks show improved accuracy."
)
PAPER = PaperIn(title="Sparse Attention", abstract=ABSTRACT, link="http://x/1")
OTHER = PaperIn(title="Other", abstract="Other abstract.", link="l2")


def claim(text, evidence):
    return {"text": text, "evidence": evidence}


def reply(**fields):
    base = {k: "not stated" for k in ("problem", "method", "results", "limitations")}
    return json.dumps(base | fields)


GOOD = reply(
    problem=[claim("Transformers dominate NLP.", "Transformers dominate NLP.")],
    method=[claim("Sparse attention.", "a sparse attention method that reduces memory")],
    results=[claim("Better accuracy.", "show improved accuracy")],
)


def test_valid_output_is_verified_and_not_stated_is_kept():
    s = LLMSummarizer(ScriptedLLM(GOOD)).summarize(PAPER)
    assert [c.text for c in s.problem] == ["Transformers dominate NLP."]
    assert s.limitations == []
    assert s.not_stated == ["limitations"]
    assert s.unverified_claims == 0
    assert s.is_structured
    assert "Method: Sparse attention." in s.text()


def test_evidence_check_normalizes_whitespace_and_case_only():
    assert evidence_in_abstract("TRANSFORMERS   dominate\nNLP.", ABSTRACT)  # double space in text
    assert not evidence_in_abstract("Transformers dominate vision.", ABSTRACT)
    assert not evidence_in_abstract("the", ABSTRACT)  # in the abstract, but too short to count
    assert not evidence_in_abstract("", ABSTRACT)


def test_fabricated_evidence_is_dropped_and_counted():
    bad = reply(
        problem=[claim("ok", "Transformers dominate NLP.")],
        method=[
            claim("real", "a sparse attention method that reduces memory"),
            claim("made up", "we also prove convergence in polynomial time"),
        ],
        results=[claim("made up", "accuracy improves by 12 points on GLUE")],
    )
    s = LLMSummarizer(ScriptedLLM(bad)).summarize(PAPER)
    assert len(s.method) == 1 and s.method[0].text == "real"
    assert s.results == [] and "results" not in s.not_stated  # dropped, not "not stated"
    assert s.unverified_claims == 2


def test_invalid_json_is_retried_with_the_error_fed_back():
    llm = ScriptedLLM("this is not json", GOOD)
    s = LLMSummarizer(llm).summarize(PAPER)
    assert len(llm.calls) == 2 and s.problem
    retry = llm.calls[1]
    assert retry[-2] == {"role": "assistant", "content": "this is not json"}
    assert "invalid" in retry[-1]["content"]


def test_schema_violation_is_retried_then_gives_up_after_two_retries():
    wrong = json.dumps({"problem": [], "method": "x"})
    llm = ScriptedLLM(wrong)
    with pytest.raises(SummarizeError, match="3 attempts"):
        LLMSummarizer(llm).summarize(PAPER)
    assert len(llm.calls) == 3  # first try + 2 retries


@pytest.mark.parametrize("value", ["Not stated.", "not stated", " NOT STATED "])
def test_not_stated_spellings_are_normalized(value):
    raw = parse_summary_json(reply(limitations=value))
    assert raw.limitations == "not stated"


def test_empty_list_must_say_not_stated():
    with pytest.raises(ValueError, match="not stated"):
        parse_summary_json(reply(limitations=[]))


def test_blank_claim_fields_are_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        parse_summary_json(reply(problem=[claim("x", "  ")]))


def test_json_inside_code_fence_is_accepted():
    assert parse_summary_json(f"```json\n{GOOD}\n```").problem


def test_llm_transport_error_becomes_summarize_error():
    class Down:
        model = "m"

        def complete(self, messages):
            raise LLMError("boom")

    with pytest.raises(SummarizeError, match="boom"):
        LLMSummarizer(Down()).summarize(PAPER)


def test_summarize_all_keeps_order_and_reports_failures():
    class Flaky:
        id = "flaky"
        workers = 3

        def summarize(self, paper):
            if paper.title.endswith("1"):
                raise SummarizeError("nope")
            return PaperSummary(overview=paper.title)

    summaries, errors = summarize_all(Flaky(), make_papers(3))
    assert [s.overview for s in summaries] == ["Paper 0", "Summary unavailable.", "Paper 2"]
    assert len(errors) == 1 and "Paper 1" in errors[0]
    assert summarize_all(Flaky(), []) == ([], [])


# ---- through the API ----


@pytest.fixture
def llm_client(client, summarizer):
    llm = ScriptedLLM(GOOD)
    app.dependency_overrides[get_summarizers] = lambda: {
        "hf": summarizer,
        "llm": LLMSummarizer(llm),
    }
    return client, llm


def test_review_with_llm_backend_returns_structured_claims(llm_client, fetcher):
    client, _ = llm_client
    fetcher.papers = [PAPER]
    r = client.get("/review", params={"topic": "t", "summarizer": "llm"})
    paper = r.json()["clusters"]["0"][0]
    assert paper["structured"]["problem"][0]["evidence"] == "Transformers dominate NLP."
    assert paper["structured"]["not_stated"] == ["limitations"]
    assert paper["summary"].startswith("Problem:")


def test_llm_and_hf_results_are_cached_separately(llm_client, fetcher):
    client, llm = llm_client
    fetcher.papers = [PAPER]
    params = {"topic": "t", "summarizer": "llm"}
    client.get("/review", params={"topic": "t"})
    assert client.get("/review", params=params).json()["cached"] is False
    assert client.get("/review", params=params).json()["cached"] is True
    assert len(llm.calls) == 1


def test_all_papers_failing_is_502(llm_client, fetcher):
    client, llm = llm_client
    fetcher.papers = [PAPER, OTHER]
    llm.replies = ["garbage"]  # every reply invalid -> both papers fail
    r = client.get("/review", params={"topic": "t", "summarizer": "llm"})
    assert r.status_code == 502 and "Summarization failed" in r.json()["detail"]


def test_partial_summary_failure_warns(llm_client, fetcher):
    client, llm = llm_client
    fetcher.papers = [PAPER, OTHER]
    # papers are summarized on parallel threads, so pick the reply by content, not call order
    llm.complete = lambda messages: GOOD if "Transformers" in messages[1]["content"] else "garbage"
    r = client.get("/review", params={"topic": "t", "summarizer": "llm", "max_results": 2}).json()
    assert len(r["warnings"]) == 1 and "Other" in r["warnings"][0]
