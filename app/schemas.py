from datetime import datetime

from pydantic import BaseModel


class PaperIn(BaseModel):
    """A paper as fetched from a source, before summarization. Only the first three are required."""

    title: str
    abstract: str
    link: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: str | None = None


class SearchFilters(BaseModel):
    """Applied after sources are merged. None means no constraint."""

    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None
    venue_contains: str | None = None


CLAIM_FIELDS = ("problem", "method", "results", "limitations")


class Claim(BaseModel):
    text: str
    evidence: str  # verbatim quote from the abstract


class PaperSummary(BaseModel):
    """What a summarizer returns. HF fills only `overview`; the LLM backend fills the claim
    lists. Every kept claim has been checked against the abstract (see services/summarize.py)."""

    overview: str = ""
    problem: list[Claim] = []
    method: list[Claim] = []
    results: list[Claim] = []
    limitations: list[Claim] = []
    not_stated: list[str] = []  # fields the abstract says nothing about
    unverified_claims: int = 0  # claims dropped because their evidence was not in the abstract

    @property
    def is_structured(self) -> bool:
        return bool(
            self.problem
            or self.method
            or self.results
            or self.limitations
            or self.not_stated
            or self.unverified_claims
        )

    def text(self) -> str:
        """Flat text for clients and exports that only know the v1 `summary` string."""
        if self.overview:
            return self.overview
        parts = []
        for name in CLAIM_FIELDS:
            claims = getattr(self, name)
            if claims:
                parts.append(f"{name.capitalize()}: " + " ".join(c.text for c in claims))
        return "\n".join(parts) or "No summary available."


class Paper(BaseModel):
    title: str
    summary: str
    link: str
    structured: PaperSummary | None = None  # additive: only set for the claim-based backend
    # Additive metadata (v1.1 clients can ignore it)
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: str | None = None


class ClusterInfo(BaseModel):
    label: str  # short keyword label (c-TF-IDF), or "Misc / unclustered"
    # 3-5 sentences written only from the cluster's verified claims. [n] cites the n-th paper
    # (1-based) in clusters[<id>]. Only set with the llm backend.
    theme_summary: str | None = None
    invalid_citations: list[int] = []  # [n] the model wrote that named no paper; stripped


class ReviewResponse(BaseModel):
    review_id: int
    topic: str
    created_at: datetime
    cached: bool = False
    clusters: dict[str, list[Paper]]
    cluster_info: dict[str, ClusterInfo] = {}  # additive, keyed like clusters
    message: str | None = None
    warnings: list[str] = []


def sorted_cluster_ids(ids) -> list[str]:
    """Numeric order, with the "unclustered" group (-1) last."""
    return sorted(ids, key=lambda i: (int(i) < 0, int(i)))


class ReviewListItem(BaseModel):
    id: int
    topic: str
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
