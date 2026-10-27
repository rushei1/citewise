import pytest

from app.schemas import PaperIn, SearchFilters
from app.services.fetch import FetchError
from app.services.rank import rank_by_relevance
from app.services.retrieve import apply_filters, retrieve
from tests.conftest import FakeEmbedder, FakeFetcher


def P(title, **kw):
    return PaperIn(title=title, abstract=kw.pop("abstract", title), link=f"http://x/{title}", **kw)


class KeywordEmbedder:
    """Vector = [count of 'cat', count of 'dog', 1]; the query 'cat' points at the cat axis."""

    def encode(self, texts):
        return [[t.lower().count("cat"), t.lower().count("dog"), 0.1] for t in texts]


# ---- ranking -----------------------------------------------------------------------------


def test_ranking_orders_by_cosine_similarity():
    papers = [P("dog dog"), P("cat dog"), P("cat cat"), P("neither")]
    ranked = rank_by_relevance("cat", papers, KeywordEmbedder(), 4)
    assert [p.title for p in ranked] == ["cat cat", "cat dog", "neither", "dog dog"]


def test_ranking_truncates_to_k():
    papers = [P("dog"), P("cat"), P("cat cat")]
    ranked = rank_by_relevance("cat", papers, KeywordEmbedder(), 2)
    assert [p.title for p in ranked] == ["cat", "cat cat"]


def test_ranking_ties_keep_input_order_and_are_deterministic():
    papers = [P("same one"), P("same two"), P("same three")]

    class Constant:
        def encode(self, texts):
            return [[1.0, 0.0]] * len(texts)

    first = rank_by_relevance("q", papers, Constant(), 3)
    assert [p.title for p in first] == ["same one", "same two", "same three"]
    assert first == rank_by_relevance("q", papers, Constant(), 3)


def test_ranking_deterministic_with_fake_embedder():
    papers = [P(f"paper {i}") for i in range(8)]
    a = rank_by_relevance("query", papers, FakeEmbedder(), 5)
    b = rank_by_relevance("query", list(papers), FakeEmbedder(), 5)
    assert a == b


def test_ranking_handles_empty_and_zero_vectors():
    assert rank_by_relevance("q", [], FakeEmbedder(), 3) == []

    class Zeros:
        def encode(self, texts):
            return [[0.0, 0.0]] * len(texts)

    assert len(rank_by_relevance("q", [P("a"), P("b")], Zeros(), 2)) == 2


# ---- filters -----------------------------------------------------------------------------

PAPERS = [
    P("old", year=2010, citation_count=500, venue="NeurIPS"),
    P("mid", year=2020, citation_count=50, venue="CVPR Workshops"),
    P("new", year=2024, citation_count=2, venue="arXiv preprint"),
    P("bare"),
]


def titles(f):
    return [p.title for p in apply_filters(PAPERS, f)]


def test_no_filters_keeps_everything():
    assert titles(SearchFilters()) == ["old", "mid", "new", "bare"]


def test_year_filters():
    assert titles(SearchFilters(year_from=2020)) == ["mid", "new"]
    assert titles(SearchFilters(year_to=2020)) == ["old", "mid"]
    assert titles(SearchFilters(year_from=2015, year_to=2022)) == ["mid"]


def test_min_citations():
    assert titles(SearchFilters(min_citations=50)) == ["old", "mid"]
    assert titles(SearchFilters(min_citations=0)) == ["old", "mid", "new", "bare"]
    assert "bare" not in titles(SearchFilters(min_citations=1))  # unknown count cannot qualify


def test_venue_contains_is_case_insensitive():
    assert titles(SearchFilters(venue_contains="cvpr")) == ["mid"]
    assert titles(SearchFilters(venue_contains="PRINT")) == ["new"]


def test_filters_combine():
    f = SearchFilters(year_from=2015, min_citations=10, venue_contains="cvpr")
    assert titles(f) == ["mid"]


# ---- orchestration -----------------------------------------------------------------------


def make_source(name, papers=None, error=None):
    src = FakeFetcher(name)
    src.papers = papers or []
    src.error = error
    return src


def run(sources, names, max_results=5, filters=None, query="cat"):
    return retrieve(
        query, sources, names, max_results, filters or SearchFilters(), KeywordEmbedder()
    )


def test_merges_dedups_and_ranks_across_sources():
    a = make_source("arxiv", [P("cat cat", arxiv_id="1.1"), P("dog", arxiv_id="2.2")])
    b = make_source("semantic_scholar", [P("cat cat renamed", arxiv_id="1.1"), P("cat", doi="d/3")])
    result = run({"arxiv": a, "semantic_scholar": b}, ["arxiv", "semantic_scholar"])
    assert [p.title for p in result.papers] == ["cat", "cat cat", "dog"]
    assert result.warnings == []


def test_partial_failure_returns_results_and_warning():
    ok = make_source("arxiv", [P("cat")])
    bad = make_source("semantic_scholar", error=FetchError("429 slow down"))
    result = run({"arxiv": ok, "semantic_scholar": bad}, ["arxiv", "semantic_scholar"])
    assert [p.title for p in result.papers] == ["cat"]
    assert result.warnings == ["semantic_scholar: 429 slow down"]


def test_all_sources_failing_raises():
    a = make_source("arxiv", error=FetchError("a down"))
    b = make_source("semantic_scholar", error=FetchError("b down"))
    with pytest.raises(FetchError, match="a down.*b down"):
        run({"arxiv": a, "semantic_scholar": b}, ["arxiv", "semantic_scholar"])


def test_single_failing_source_raises():
    with pytest.raises(FetchError):
        run({"arxiv": make_source("arxiv", error=FetchError("x"))}, ["arxiv"])


def test_requests_a_larger_candidate_pool_then_cuts_to_max_results():
    seen = []

    class Spy(FakeFetcher):
        def search(self, query, max_results, filters):
            seen.append(max_results)
            return [P(f"cat {i}") for i in range(max_results)]

    result = run({"arxiv": Spy("arxiv")}, ["arxiv"], max_results=4)
    assert seen == [12]
    assert len(result.papers) == 4


def test_candidate_pool_is_capped():
    seen = []

    class Spy(FakeFetcher):
        def search(self, query, max_results, filters):
            seen.append(max_results)
            return []

    run({"arxiv": Spy("arxiv")}, ["arxiv"], max_results=30)
    assert seen == [90]


def test_filters_applied_after_merge():
    src = make_source("arxiv", [P("cat", year=2010), P("cat cat", year=2022)])
    result = run({"arxiv": src}, ["arxiv"], filters=SearchFilters(year_from=2020))
    assert [p.title for p in result.papers] == ["cat cat"]


def test_zero_papers_is_not_an_error():
    result = run({"arxiv": make_source("arxiv", [])}, ["arxiv"])
    assert result.papers == [] and result.warnings == []
