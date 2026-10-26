from app.schemas import PaperIn
from app.services.dedup import deduplicate, norm_arxiv_id, norm_doi, norm_title


def P(title="Some Title", **kw):
    kw.setdefault("abstract", "abstract")
    kw.setdefault("link", "http://x")
    return PaperIn(title=title, **kw)


def test_no_duplicates_untouched_and_ordered():
    papers = [P("Alpha Beta"), P("Gamma Delta"), P("Epsilon Zeta")]
    assert deduplicate(papers) == papers


def test_same_paper_on_both_sources_merges_by_arxiv_id():
    arxiv = P(
        "Attention Is All You Need",
        arxiv_id="1706.03762",
        authors=["Ashish Vaswani"],
        year=2017,
        source="arxiv",
        link="http://arxiv.org/abs/1706.03762v7",
        abstract="short",
    )
    s2 = P(
        "Totally Different Title Text",  # id match wins regardless of title
        arxiv_id="1706.03762v5",
        doi="10.5555/3295222",
        citation_count=100000,
        venue="NeurIPS",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        source="semantic_scholar",
        abstract="a much longer abstract",
    )
    [merged] = deduplicate([arxiv, s2])
    assert merged.title == "Attention Is All You Need"  # first record wins ties
    assert merged.link == "http://arxiv.org/abs/1706.03762v7"
    assert merged.abstract == "a much longer abstract"
    assert merged.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert (merged.citation_count, merged.venue, merged.doi, merged.year) == (
        100000,
        "NeurIPS",
        "10.5555/3295222",
        2017,
    )
    assert merged.source == "arxiv+semantic_scholar"


def test_merges_by_doi_with_url_prefix_and_case():
    a = P("A", doi="10.1000/ABC")
    b = P("Z", doi="https://doi.org/10.1000/abc")
    assert len(deduplicate([a, b])) == 1


def test_merges_by_near_identical_title():
    a = P("Deep Residual Learning for Image Recognition", year=2016, authors=["Kaiming He"])
    b = P("Deep residual learning for image recognition.", year=2015, authors=["K. He"])
    assert len(deduplicate([a, b])) == 1


def test_title_match_without_metadata_still_merges():
    assert len(deduplicate([P("Graph Attention Networks"), P("graph attention networks")])) == 1


def test_similar_titles_but_different_papers_stay_separate():
    a = P("Attention Is All You Need", authors=["Ashish Vaswani"], year=2017)
    b = P("Attention Is Not All You Need", authors=["Yihe Dong"], year=2021)
    assert len(deduplicate([a, b])) == 2


def test_same_title_far_apart_years_stay_separate():
    a = P("Introduction to Reinforcement Learning", year=1998)
    b = P("Introduction to Reinforcement Learning", year=2021)
    assert len(deduplicate([a, b])) == 2


def test_same_title_disjoint_authors_stay_separate():
    a = P("A Survey of Methods", authors=["Alice Smith"], year=2020)
    b = P("A Survey of Methods", authors=["Bob Jones"], year=2020)
    assert len(deduplicate([a, b])) == 2


def test_different_ids_with_unrelated_titles_stay_separate():
    a = P("Alpha Paper", arxiv_id="1111.1111")
    b = P("Beta Paper", arxiv_id="2222.2222")
    assert len(deduplicate([a, b])) == 2


def test_three_way_merge_and_citation_none_handling():
    papers = [
        P("X", arxiv_id="1.1", citation_count=None),
        P("X", arxiv_id="1.1", citation_count=5),
        P("X", doi="10/z", arxiv_id="1.1", citation_count=3),
    ]
    [m] = deduplicate(papers)
    assert m.citation_count == 5
    assert m.doi == "10/z"


def test_empty_titles_do_not_all_merge():
    assert len(deduplicate([P(""), P("")])) == 2


def test_normalizers():
    assert norm_title("  Hello,  WORLD!! ") == "hello world"
    assert norm_doi("http://dx.doi.org/10.1/X") == "10.1/x"
    assert norm_doi("") is None and norm_doi(None) is None
    assert norm_arxiv_id("arXiv:2104.14294v3") == "2104.14294"
    assert norm_arxiv_id(None) is None
