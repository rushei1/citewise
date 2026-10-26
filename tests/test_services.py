import sys
import time
import types
from datetime import UTC, datetime

import pytest

from app import main
from app.config import Settings, get_settings
from app.schemas import Paper, ReviewResponse, SearchFilters
from app.services import fetch
from app.services.cluster import Embedder
from app.services.export import build_pdf
from app.services.http import RateLimiter
from app.services.summarize import Summarizer


def test_arxiv_source_success(monkeypatch):
    monkeypatch.setattr(fetch, "_arxiv_search", lambda q, n: ["ok", q, n])
    assert fetch.ArxivSource(limiter=RateLimiter(0)).search("t", 3, SearchFilters()) == [
        "ok",
        "t",
        3,
    ]


def test_arxiv_source_wraps_errors(monkeypatch):
    def fail(q, n):
        raise RuntimeError("network down")

    monkeypatch.setattr(fetch, "_arxiv_search", fail)
    with pytest.raises(fetch.FetchError, match="network down"):
        fetch.ArxivSource(limiter=RateLimiter(0)).search("t", 3, SearchFilters())


def test_arxiv_source_timeout(monkeypatch):
    monkeypatch.setattr(fetch, "_arxiv_search", lambda q, n: time.sleep(1))
    with pytest.raises(fetch.FetchError, match="did not respond"):
        fetch.ArxivSource(timeout=0.05, limiter=RateLimiter(0)).search("t", 3, SearchFilters())


def test_arxiv_search_maps_results(monkeypatch):
    result = types.SimpleNamespace(
        title=" T ",
        summary=" S ",
        entry_id="http://arxiv.org/abs/2303.15105v2",
        authors=[types.SimpleNamespace(name="Ann Lee")],
        published=datetime(2023, 3, 27, tzinfo=UTC),
        journal_ref="CVPR 2023",
        doi="10.1/x",
    )

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return [result]

    monkeypatch.setattr(fetch.arxiv, "Client", FakeClient)
    [paper] = fetch._arxiv_search("topic", 1)
    assert (paper.title, paper.abstract, paper.link) == ("T", "S", result.entry_id)
    assert paper.arxiv_id == "2303.15105"
    assert (paper.authors, paper.year, paper.venue, paper.doi) == (
        ["Ann Lee"],
        2023,
        "CVPR 2023",
        "10.1/x",
    )
    assert paper.source == "arxiv"


def test_summarizer_loads_lazily_once(monkeypatch):
    loads = []

    def pipeline(task, model):
        loads.append(model)
        return lambda text, **kw: [{"summary_text": text[:10]}]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=pipeline))
    s = Summarizer("my-model", max_input_chars=20)
    assert loads == []  # nothing at construction
    assert s.summarize("Title", "x" * 100) == "Title. xxx"
    s.summarize("a", "b")
    assert loads == ["my-model"]


def test_embedder_loads_lazily_once(monkeypatch):
    loads = []

    class ST:
        def __init__(self, name):
            loads.append(name)

        def encode(self, texts):
            return [[0.0]] * len(texts)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=ST)
    )
    e = Embedder("emb")
    assert loads == []
    e.encode(["a"])
    e.encode(["b"])
    assert loads == ["emb"]


def test_default_factories_use_settings():
    main.get_summarizer.cache_clear()
    main.get_clusterer.cache_clear()
    assert main.get_summarizer().model_name == get_settings().summarizer_model
    assert main.get_clusterer().embedder.model_name == get_settings().embedding_model
    assert set(main.get_sources()) == {"arxiv", "semantic_scholar"}


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("ARXIV_MAX_RESULTS", "7")
    assert Settings().arxiv_max_results == 7


def test_pdf_empty_review():
    review = ReviewResponse(
        review_id=1, topic="x", created_at=datetime.now(UTC), clusters={}, message="nothing"
    )
    assert build_pdf(review).startswith(b"%PDF")


def test_pdf_many_pages():
    paper = Paper(title="T " * 30, summary="word " * 400, link="http://x")
    review = ReviewResponse(
        review_id=1,
        topic="x",
        created_at=datetime.now(UTC),
        clusters={"0": [paper] * 10, "1": [paper]},
    )
    assert build_pdf(review).startswith(b"%PDF")
