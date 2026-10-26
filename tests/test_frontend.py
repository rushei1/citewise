from pathlib import Path

import pytest

pytest.importorskip("streamlit")  # not in requirements-dev, so CI skips this file
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "frontend" / "app.py")

REVIEW = {
    "review_id": 3,
    "topic": "sparse attention",
    "created_at": "2025-01-02T03:04:00Z",
    "cached": True,
    "message": None,
    "warnings": ["semantic_scholar: HTTP 429"],
    "cluster_info": {
        "0": {"label": "sparse, attention", "theme_summary": "They share memory goals [1]."},
        "-1": {"label": "Misc / unclustered", "theme_summary": None},
    },
    "clusters": {
        "-1": [{"title": "Odd One", "summary": "plain summary", "link": "http://x/2"}],
        "0": [
            {
                "title": "Sparse Attention",
                "summary": "Problem: ...",
                "link": "http://x/1",
                "authors": ["Ann Lee"],
                "year": 2023,
                "structured": {
                    "problem": [{"text": "Memory is costly.", "evidence": "memory is costly"}],
                    "method": [],
                    "results": [],
                    "limitations": [],
                    "not_stated": ["method", "results", "limitations"],
                    "unverified_claims": 2,
                },
            }
        ],
    },
}


class FakeResponse:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code, self._body, self.content = status, body, content
        self.text = str(body)

    def json(self):
        return self._body


class FakeApi:
    def __init__(self, review_status=200):
        self.calls: list[tuple[str, dict]] = []
        self.review_status = review_status

    def __call__(self, url, params=None, timeout=None):
        path = url.removeprefix("http://127.0.0.1:8000")
        self.calls.append((path, params or {}))
        if path == "/reviews":
            return FakeResponse(body=[{"id": 3, "topic": "old", "created_at": "2025-01-01T00:00"}])
        if path == "/review":
            if self.review_status != 200:
                return FakeResponse(
                    self.review_status, {"detail": "Summarizer 'llm' is not configured."}
                )
            return FakeResponse(body=REVIEW)
        if path.startswith("/reviews/3/"):
            return FakeResponse(content=b"export-bytes")
        return FakeResponse(404, {"detail": "nope"})


@pytest.fixture
def api(monkeypatch):
    import streamlit as st

    st.cache_data.clear()  # fetch_export is cached in-process across tests
    fake = FakeApi()
    monkeypatch.setattr("requests.get", fake)
    return fake


def run_generate(topic="sparse attention") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value(topic)
    at.button[0].click().run()
    return at


def test_initial_page_loads_history_and_no_error(api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and not at.error
    assert ("/reviews", {}) in api.calls


def test_generate_sends_all_options_and_renders_grounded_review(api):
    at = run_generate()
    assert not at.exception
    sent = dict(next(p for path, p in api.calls if path == "/review"))
    assert sent == {
        "topic": "sparse attention",
        "num_clusters": 3,
        "max_results": 10,
        "cluster_method": "kmeans",
        "summarizer": "hf",
        "sources": "arxiv",
    }
    headings = [m.value for m in at.markdown]
    assert any("sparse, attention" in h for h in headings)
    assert any("Misc / unclustered" in h for h in headings)
    assert any("They share memory goals [1]." in h for h in headings)
    assert any("**Problem:** Memory is costly. ✅" in h for h in headings)
    assert any("**Method:** _not stated in the abstract_" in h for h in headings)
    assert any("2 claim(s) were dropped" in c.value for c in at.caption)
    assert any("memory is costly" in c.value for c in at.caption)  # evidence quote
    assert any("semantic_scholar: HTTP 429" in w.value for w in at.warning)
    assert [e.label for e in at.expander if e.label.startswith("📄")] == [
        "📄 [1] Sparse Attention",
        "📄 [1] Odd One",
    ]  # unclustered group is last, numbering restarts per cluster
    assert any("From cache" in i.value for i in at.info)


def test_unclustered_group_is_listed_after_numbered_clusters(api):
    at = run_generate()
    labels = [m.value for m in at.markdown if m.value.startswith("### ")]
    assert labels == ["### 🔗 sparse, attention", "### 🔗 Misc / unclustered"]


def test_all_export_formats_are_fetched(api):
    run_generate()
    exported = {path.rsplit("/", 1)[1] for path, _ in api.calls if path.startswith("/reviews/3/")}
    assert exported == {"pdf", "bibtex", "markdown", "latex"}


def test_options_are_forwarded(api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("t")
    at.multiselect[0].set_value(["arxiv", "semantic_scholar"])
    at.selectbox[1].set_value("llm")  # [0] is the history selector
    at.selectbox[2].set_value("hdbscan")
    at.checkbox[0].check()
    at.number_input[0].set_value(2020)
    at.number_input[2].set_value(10)
    at.button[0].click().run()
    sent = dict(next(p for path, p in api.calls if path == "/review"))
    assert sent["sources"] == "arxiv,semantic_scholar"
    assert sent["summarizer"] == "llm" and sent["cluster_method"] == "hdbscan"
    assert sent["num_clusters"] == "auto"
    assert sent["year_from"] == 2020 and sent["min_citations"] == 10
    assert "year_to" not in sent and "venue_contains" not in sent


def test_api_error_message_is_shown_and_nothing_rendered(monkeypatch):
    fake = FakeApi(review_status=503)
    monkeypatch.setattr("requests.get", fake)
    at = run_generate()
    assert any("not configured" in e.value for e in at.error)
    assert not at.success


def test_empty_topic_and_no_sources_are_rejected_client_side(api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()
    assert any("enter a topic" in w.value for w in at.warning)
    at.text_input[0].set_value("t")
    at.multiselect[0].set_value([])
    at.button[0].click().run()
    assert any("at least one source" in w.value for w in at.warning)
    assert not [c for c in api.calls if c[0] == "/review"]
