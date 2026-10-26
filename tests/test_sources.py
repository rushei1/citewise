import json
from pathlib import Path

import pytest

from app.schemas import SearchFilters
from app.services.fetch import FetchError, SemanticScholarSource
from app.services.http import HttpError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "s2_search_synthetic.json").read_text(encoding="utf-8")
)


class FakeHttp:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []

    def get_json(self, url, params=None, headers=None):
        self.calls.append((url, params, headers))
        if self.error:
            raise self.error
        return self.payload


def search(payload=FIXTURE, filters=None, **kw):
    http = FakeHttp(payload)
    papers = SemanticScholarSource(http, **kw).search("vit", 10, filters or SearchFilters())
    return papers, http


def test_maps_fixture_and_skips_papers_without_abstract():
    papers, _ = search()
    assert [p.title for p in papers] == [
        "Emerging Properties in Self-Supervised Vision Transformers",
        "Vision Transformers for Dense Prediction",
        "Untitled Venue Paper",
    ]
    dino = papers[0]
    assert dino.arxiv_id == "2104.14294"
    assert dino.doi == "10.1109/ICCV48922.2021.00951"
    assert dino.citation_count == 10729
    assert dino.year == 2021
    assert dino.authors == ["Mathilde Caron", "Hugo Touvron"]
    assert dino.source == "semantic_scholar"


def test_tolerates_missing_and_null_fields():
    _, dense, bare = search()[0]
    assert dense.venue is None  # "" becomes None
    assert dense.authors == ["Rene Ranftl"]  # nameless author dropped
    assert dense.link == "https://www.semanticscholar.org/paper/ccc333"  # url missing -> built
    assert (bare.year, bare.citation_count, bare.doi, bare.arxiv_id, bare.authors) == (
        None,
        None,
        None,
        None,
        [],
    )


def test_empty_and_missing_data():
    assert search({"total": 0})[0] == []
    assert search({"data": None})[0] == []


def test_request_params_and_api_key():
    _, http = search(api_key="secret")
    url, params, headers = http.calls[0]
    assert "semanticscholar.org" in url
    assert params["query"] == "vit"
    assert params["limit"] == 10
    assert "externalIds" in params["fields"]
    assert headers == {"x-api-key": "secret"}


def test_works_without_api_key():
    _, http = search()
    assert http.calls[0][2] is None


def test_limit_is_capped_at_api_maximum():
    http = FakeHttp({"data": []})
    SemanticScholarSource(http).search("q", 500, SearchFilters())
    assert http.calls[0][1]["limit"] == 100


def test_filter_pushdown():
    _, http = search(filters=SearchFilters(year_from=2020, year_to=2022, min_citations=5))
    params = http.calls[0][1]
    assert params["year"] == "2020-2022"
    assert params["minCitationCount"] == 5
    _, http = search(filters=SearchFilters(year_from=2020))
    assert http.calls[0][1]["year"] == "2020-"
    _, http = search(filters=SearchFilters(year_to=2022))
    assert http.calls[0][1]["year"] == "-2022"
    _, http = search()
    assert "year" not in http.calls[0][1]
    assert "minCitationCount" not in http.calls[0][1]


def test_http_failure_becomes_fetch_error():
    source = SemanticScholarSource(FakeHttp(error=HttpError("429")))
    with pytest.raises(FetchError, match="Semantic Scholar"):
        source.search("q", 5, SearchFilters())
