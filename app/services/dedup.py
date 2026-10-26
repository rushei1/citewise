import re
from difflib import SequenceMatcher

from app.schemas import PaperIn

# Title-only matches need near-identical text. Deliberately stricter than a token-set ratio, which
# scores 100 when one title's words are a subset of the other's ("... Is All You Need" vs
# "... Is Not All You Need").
TITLE_THRESHOLD = 0.95


def norm_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi.strip().lower()) or None


def norm_arxiv_id(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    return re.sub(r"v\d+$", "", arxiv_id.strip().lower().removeprefix("arxiv:")) or None


def _surnames(authors: list[str]) -> set[str]:
    return {a.split()[-1].lower() for a in authors if a.split()}


def _same_by_title(a: PaperIn, b: PaperIn) -> bool:
    ta, tb = norm_title(a.title), norm_title(b.title)
    if not ta or SequenceMatcher(None, ta, tb).ratio() < TITLE_THRESHOLD:
        return False
    # Same-ish title is not enough: reject when metadata clearly says they differ.
    if a.year and b.year and abs(a.year - b.year) > 1:
        return False
    sa, sb = _surnames(a.authors), _surnames(b.authors)
    return not (sa and sb and sa.isdisjoint(sb))


def _same_paper(a: PaperIn, b: PaperIn) -> bool:
    for ka, kb in (
        (norm_doi(a.doi), norm_doi(b.doi)),
        (norm_arxiv_id(a.arxiv_id), norm_arxiv_id(b.arxiv_id)),
    ):
        if ka and ka == kb:
            return True
    return _same_by_title(a, b)


def merge(a: PaperIn, b: PaperIn) -> PaperIn:
    """Combine two records of one paper, keeping the richest metadata. `a` wins ties."""
    sources = [s for s in dict.fromkeys(f"{a.source or ''}+{b.source or ''}".split("+")) if s]
    counts = [c for c in (a.citation_count, b.citation_count) if c is not None]
    return PaperIn(
        title=a.title,
        abstract=max((a.abstract, b.abstract), key=len),
        link=a.link,
        authors=max((a.authors, b.authors), key=len),
        year=a.year or b.year,
        venue=a.venue or b.venue,
        citation_count=max(counts) if counts else None,
        doi=a.doi or b.doi,
        arxiv_id=a.arxiv_id or b.arxiv_id,
        source="+".join(sources) or None,
    )


def deduplicate(papers: list[PaperIn]) -> list[PaperIn]:
    """Merge duplicates (same DOI, else same arXiv id, else near-identical title).

    Order of first appearance is kept.
    ponytail: O(n^2) pairwise scan; fine for the <=100 candidates per request."""
    merged: list[PaperIn] = []
    for paper in papers:
        for i, existing in enumerate(merged):
            if _same_paper(existing, paper):
                merged[i] = merge(existing, paper)
                break
        else:
            merged.append(paper)
    return merged
