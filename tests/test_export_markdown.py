from app.schemas import Claim, ClusterInfo, PaperSummary
from app.services.export import build_markdown
from tests.test_export import paper, review


def full_review():
    """Two clusters plus an unclustered group, one grounded paper, one plain-summary paper."""
    grounded = paper(
        title="Sparse *Attention* [v2]",
        authors=("Ann Lee", "Bo Chen", "Cy Diaz", "Di Evans"),
        structured=PaperSummary(
            problem=[Claim(text="Memory is the bottleneck.", evidence="memory is the bottleneck")],
            method=[Claim(text="Sparse attention.", evidence="a sparse attention method")],
            not_stated=["limitations"],
            unverified_claims=1,
        ),
    )
    plain = paper(title="Dense Baselines", authors=(), year=None).model_copy(
        update={"summary": "A <b>plain</b> BART summary.", "link": "http://x/a b(1)"}
    )
    return review(
        {"0": [grounded, plain], "-1": [paper(title="Odd One Out")]},
        info={
            "0": ClusterInfo(
                label="sparse, attention",
                theme_summary="Both study memory [1][2] and *speed* [1].",
                invalid_citations=[9],
            ),
            "-1": ClusterInfo(label="Misc / unclustered"),
        },
        topic="Sparse _attention_",
        warnings=["semantic_scholar: HTTP 429"],
    )


EXPECTED = """\
# Literature review: Sparse \\_attention\\_

Generated 2025-01-02 03:04 UTC.
Claims are grounded in each paper's abstract only; full texts were not read.

> **Warnings**
> - semantic\\_scholar: HTTP 429

## Cluster 1: sparse, attention

Both study memory [1][2] and \\*speed\\* [1].

_Removed citations that matched no paper in this cluster: [9]_

1. **Sparse \\*Attention\\* \\[v2\\]** (Ann Lee, Bo Chen, Cy Diaz et al., 2023) — [link](http://arxiv.org/abs/1)
   - **Problem:** Memory is the bottleneck.
     > “memory is the bottleneck” — verified in abstract
   - **Method:** Sparse attention.
     > “a sparse attention method” — verified in abstract
   - **Limitations:** _not stated in the abstract_
   - _1 claim(s) dropped: quote not found in abstract_
2. **Dense Baselines** — [link](http://x/a%20b%281%29)
   A \\<b\\>plain\\</b\\> BART summary.

## Misc / unclustered

1. **Odd One Out** (John Smith, 2023) — [link](http://arxiv.org/abs/1)
   s
"""


def test_markdown_snapshot():
    assert build_markdown(full_review()) == EXPECTED


def test_markdown_for_empty_review_shows_the_message():
    md = build_markdown(review({}, message="No papers found for this topic."))
    assert "No papers found for this topic." in md and "##" not in md


def test_markdown_without_structured_papers_has_no_grounding_note():
    assert "grounded" not in build_markdown(review({"0": [paper()]}))


def test_markdown_route(client):
    rid = client.get("/review", params={"topic": "t"}).json()["review_id"]
    r = client.get(f"/reviews/{rid}/markdown")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert f"review_{rid}.md" in r.headers["content-disposition"]
    assert r.text.startswith("# Literature review: t")
    assert client.get("/reviews/999/markdown").status_code == 404
