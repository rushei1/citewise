import sys

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import summarize
from app.services.fetch import FetchError
from tests.conftest import make_papers


def n_papers(body):
    return sum(len(v) for v in body["clusters"].values())


def test_review_happy_path(client):
    r = client.get("/review", params={"topic": "vision transformers"})
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] == "vision transformers"
    assert body["cached"] is False
    assert body["review_id"] >= 1
    assert n_papers(body) == 6
    assert len(body["clusters"]) == 3  # default_num_clusters
    paper = next(iter(body["clusters"].values()))[0]
    assert {"title", "summary", "link"} <= set(paper)  # v1 fields kept; new ones are additive


def test_review_params(client):
    body = client.get("/review", params={"topic": "t", "num_clusters": 2, "max_results": 4}).json()
    assert len(body["clusters"]) == 2
    assert n_papers(body) == 4


@pytest.mark.parametrize(
    "params", [{"num_clusters": 0}, {"num_clusters": 9}, {"max_results": 0}, {"max_results": 31}]
)
def test_review_param_validation(client, params):
    assert client.get("/review", params={"topic": "t", **params}).status_code == 422


def test_cache_hit(client, summarizer, fetcher):
    first = client.get("/review", params={"topic": "Deep  Learning"}).json()
    calls = (fetcher.calls, summarizer.calls)
    second = client.get("/review", params={"topic": " deep learning "}).json()
    assert second["cached"] is True
    assert second["review_id"] == first["review_id"]
    assert second["clusters"] == first["clusters"]
    assert (fetcher.calls, summarizer.calls) == calls  # no arXiv, no model


def test_different_params_miss_cache(client, fetcher):
    client.get("/review", params={"topic": "t", "num_clusters": 2})
    other = client.get("/review", params={"topic": "t", "num_clusters": 3}).json()
    assert other["cached"] is False
    assert fetcher.calls == 2


def test_arxiv_failure_is_502(client, fetcher):
    fetcher.error = FetchError("boom")
    r = client.get("/review", params={"topic": "t"})
    assert r.status_code == 502
    assert "boom" in r.json()["detail"]
    assert client.get("/reviews").json() == []  # failures are not stored


@pytest.mark.parametrize("topic", ["", "   "])
def test_empty_topic_is_422(client, topic):
    assert client.get("/review", params={"topic": topic}).status_code == 422


def test_no_papers_returns_message(client, fetcher):
    fetcher.papers = []
    r = client.get("/review", params={"topic": "zzzz"})
    assert r.status_code == 200
    assert r.json()["clusters"] == {}
    assert r.json()["message"]


def test_single_paper(client, fetcher):
    fetcher.papers = make_papers(1)
    assert list(client.get("/review", params={"topic": "t"}).json()["clusters"]) == ["0"]


def test_health_loads_no_models(monkeypatch):
    def boom(self):
        raise AssertionError("model loaded")

    monkeypatch.setattr(summarize.Summarizer, "_load", boom)
    r = TestClient(app).get("/health")
    assert r.json() == {"status": "ok"}
    assert not {"torch", "transformers", "sentence_transformers"} & set(sys.modules)


def test_history_list_get_delete(client):
    a = client.get("/review", params={"topic": "first"}).json()
    b = client.get("/review", params={"topic": "second"}).json()

    listing = client.get("/reviews").json()
    assert [x["topic"] for x in listing] == ["second", "first"]  # newest first

    got = client.get(f"/reviews/{a['review_id']}").json()
    assert got["topic"] == "first"
    assert got["clusters"] == a["clusters"]

    assert client.delete(f"/reviews/{b['review_id']}").status_code == 204
    assert client.get(f"/reviews/{b['review_id']}").status_code == 404
    assert client.delete(f"/reviews/{b['review_id']}").status_code == 404
    assert len(client.get("/reviews").json()) == 1


def test_pdf_endpoint(client):
    rid = client.get("/review", params={"topic": "pdf topic"}).json()["review_id"]
    r = client.get(f"/reviews/{rid}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert f"review_{rid}.pdf" in r.headers["content-disposition"]
    assert r.content.startswith(b"%PDF")
    assert client.get("/reviews/999/pdf").status_code == 404


def test_pdf_survives_markup_characters(client, fetcher):
    fetcher.papers = [
        make_papers(1)[0].model_copy(
            update={"title": 'A <b>& "quoted" x < y', "link": 'http://x/?a=1&b="2"'}
        )
    ]
    rid = client.get("/review", params={"topic": "<script>&"}).json()["review_id"]
    r = client.get(f"/reviews/{rid}/pdf")
    assert r.status_code == 200
    assert r.content.startswith(b"%PDF")


def test_path_traversal_topic_is_just_data(client, tmp_path):
    topic = "../../etc/passwd"
    rid = client.get("/review", params={"topic": topic}).json()["review_id"]
    assert client.get(f"/reviews/{rid}").json()["topic"] == topic
    assert [x["topic"] for x in client.get("/reviews").json()] == [topic]
    assert not (tmp_path.parent / "etc").exists()


def test_sql_injection_topic_is_just_data(client):
    topic = "x'; DROP TABLE reviews; --"
    client.get("/review", params={"topic": topic})
    assert [x["topic"] for x in client.get("/reviews").json()] == [topic]


# ---- Phase 2: sources, filters, warnings -----------------------------------------------------


@pytest.fixture
def two_sources(client, db, summarizer):
    from app.main import get_sources
    from tests.conftest import FakeFetcher

    arxiv, s2 = FakeFetcher("arxiv"), FakeFetcher("semantic_scholar")
    app.dependency_overrides[get_sources] = lambda: {"arxiv": arxiv, "semantic_scholar": s2}
    return arxiv, s2


def test_unknown_source_is_422(client):
    r = client.get("/review", params={"topic": "t", "sources": "arxiv,pubmed"})
    assert r.status_code == 422
    assert "pubmed" in r.json()["detail"]


def test_sources_param_selects_sources(two_sources, client):
    arxiv, s2 = two_sources
    client.get("/review", params={"topic": "t", "sources": "semantic_scholar"})
    assert (arxiv.calls, s2.calls) == (0, 1)
    client.get("/review", params={"topic": "u", "sources": " ArXiv , semantic_scholar "})
    assert (arxiv.calls, s2.calls) == (1, 2)


def test_partial_failure_gives_warning_and_is_not_cached(two_sources, client):
    arxiv, s2 = two_sources
    s2.error = FetchError("rate limited")
    params = {"topic": "t", "sources": "arxiv,semantic_scholar"}
    body = client.get("/review", params=params).json()
    assert body["warnings"] == ["semantic_scholar: rate limited"]
    assert n_papers(body) == 6
    # degraded result is stored as history but never served as a cache hit
    s2.error = None
    again = client.get("/review", params=params).json()
    assert again["cached"] is False
    assert again["warnings"] == []
    assert client.get("/review", params=params).json()["cached"] is True
    assert client.get(f"/reviews/{body['review_id']}").json()["warnings"] == body["warnings"]


def test_all_sources_failing_is_502(two_sources, client):
    arxiv, s2 = two_sources
    arxiv.error = s2.error = FetchError("down")
    r = client.get("/review", params={"topic": "t", "sources": "arxiv,semantic_scholar"})
    assert r.status_code == 502


def test_dedup_across_sources_via_api(two_sources, client):
    arxiv, s2 = two_sources
    arxiv.papers = make_papers(3)
    s2.papers = make_papers(3)  # identical titles
    body = client.get("/review", params={"topic": "t", "sources": "arxiv,semantic_scholar"}).json()
    assert n_papers(body) == 3


def test_filters_applied(client, fetcher):
    fetcher.papers = [
        p.model_copy(update={"year": 2010 + i * 5, "citation_count": i * 10})
        for i, p in enumerate(make_papers(4))
    ]
    body = client.get("/review", params={"topic": "t", "year_from": 2015}).json()
    assert n_papers(body) == 3
    body = client.get("/review", params={"topic": "t2", "min_citations": 20}).json()
    assert n_papers(body) == 2


def test_filters_are_part_of_cache_key(client, fetcher):
    client.get("/review", params={"topic": "t"})
    assert client.get("/review", params={"topic": "t", "year_from": 2000}).json()["cached"] is False
    assert client.get("/review", params={"topic": "t", "sources": "arxiv"}).json()["cached"] is True


@pytest.mark.parametrize(
    "params",
    [
        {"year_from": 2024, "year_to": 2020},
        {"year_from": 1500},
        {"year_to": 3000},
        {"min_citations": -1},
        {"venue_contains": "x" * 101},
    ],
)
def test_filter_validation(client, params):
    assert client.get("/review", params={"topic": "t", **params}).status_code == 422


def test_new_metadata_is_returned(client, fetcher):
    fetcher.papers = [
        make_papers(1)[0].model_copy(
            update={"authors": ["Ann Lee"], "year": 2021, "venue": "ICML", "doi": "10/x"}
        )
    ]
    paper = next(iter(client.get("/review", params={"topic": "t"}).json()["clusters"].values()))[0]
    assert (paper["authors"], paper["year"], paper["venue"], paper["doi"]) == (
        ["Ann Lee"],
        2021,
        "ICML",
        "10/x",
    )
