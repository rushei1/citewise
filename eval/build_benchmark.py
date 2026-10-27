"""Build the evaluation data.

Part A (automatic, committed): a proxy clustering benchmark. 6 related arXiv categories x 30
papers, a fixed submission window, the earliest papers in that window whose *primary* category
matches. The label is the primary category. This is a PROXY: categories are an imperfect stand-in
for research topics (cs.CL and cs.IR overlap a lot, and a paper's category reflects the
author's choice, not its content).

Part B (needs a human): a CSV of papers retrieved by the app's own pipeline for 5 topics, with an
empty `cluster_label` column to fill in by hand. Never overwritten once it exists.

Only metadata is stored (id, title, abstract, category, date), not PDFs.
"""

import argparse
import csv
import logging
from collections.abc import Callable
from pathlib import Path

from eval.common import BENCHMARK, DATA_DIR, cached, environment, save_json, set_seeds

log = logging.getLogger(__name__)

CATEGORIES = ["cs.CL", "cs.CV", "cs.IR", "cs.RO", "cs.CR", "cs.DB"]
PER_CATEGORY = 30
WINDOW = ("202301010000", "202306302359")  # submittedDate range, inclusive
MIN_ABSTRACT_CHARS = 300  # drops withdrawn / stub records
FETCH_FACTOR = 4  # fetch this many per wanted paper: cross-lists and stubs are filtered out

TOPICS = [
    "vision transformers for medical imaging",
    "retrieval-augmented generation for question answering",
    "graph neural networks for molecular property prediction",
    "reinforcement learning for robot manipulation",
    "differential privacy in deep learning",
]
HANDLABEL_CSV = DATA_DIR / "handlabel_template.csv"
HANDLABEL_PER_TOPIC = 30

SearchFn = Callable[[str, int], list[dict]]  # (arXiv query, max_results) -> raw records


def arxiv_search(query: str, max_results: int) -> list[dict]:
    """Real arXiv query, earliest submissions first (a deterministic order)."""
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    return [
        {
            "id": r.entry_id.rsplit("/abs/", 1)[-1],
            "title": " ".join(r.title.split()),
            "abstract": " ".join(r.summary.split()),
            "category": r.primary_category,
            "published": r.published.date().isoformat(),
        }
        for r in client.results(search)
    ]


def fetch_category(category: str, per: int, search: SearchFn, cache_dir: Path | None = None):
    query = f"cat:{category} AND submittedDate:[{WINDOW[0]} TO {WINDOW[1]}]"
    n = per * FETCH_FACTOR
    raw = cached("arxiv", {"q": query, "n": n}, lambda: search(query, n), cache_dir)
    seen, papers = set(), []
    for r in raw:
        if (
            r["category"] != category  # cross-listed from another primary category
            or len(r["abstract"]) < MIN_ABSTRACT_CHARS
            or r["id"] in seen
        ):
            continue
        seen.add(r["id"])
        papers.append({**r, "id": r["id"].split("v")[0] if "v" in r["id"] else r["id"]})
    papers.sort(key=lambda p: (p["published"], p["id"]))  # deterministic
    return papers[:per]


def build_part_a(search: SearchFn = arxiv_search, per: int = PER_CATEGORY, cache_dir=None) -> dict:
    papers, short = [], {}
    for cat in CATEGORIES:
        got = fetch_category(cat, per, search, cache_dir)
        if len(got) < per:
            short[cat] = len(got)
        papers += got
    return {
        "meta": {
            "description": "Proxy clustering benchmark: label = arXiv primary category.",
            "categories": CATEGORIES,
            "per_category": per,
            "submitted_window": list(WINDOW),
            "order": "earliest submission first, within the window",
            # taking the earliest papers means the sample covers only the start of the window
            "effective_dates": [
                min(p["published"] for p in papers),
                max(p["published"] for p in papers),
            ]
            if papers
            else None,
            "shortfalls": short,  # categories that had fewer than `per` usable papers
            "n_papers": len(papers),
        },
        "part_a": papers,
    }


def build_handlabel_template(
    embedder,
    search: SearchFn,  # (topic, n) -> [{id, title, abstract}]
    path: Path = HANDLABEL_CSV,
    topics: list[str] = TOPICS,
    per_topic: int = HANDLABEL_PER_TOPIC,
    cache_dir: Path | None = None,
) -> str:
    """Retrieve papers per topic with the app's own dedup + ranking, write the CSV to label.

    Returns "exists" without touching anything if the file is already there."""
    from app.schemas import PaperIn
    from app.services.dedup import deduplicate
    from app.services.rank import rank_by_relevance

    if path.exists():
        return "exists"
    rows = []
    for topic in topics:
        # not date-windowed: relevance-ordered arXiv results, as the app would fetch them
        raw = cached(
            "arxiv_topic",
            {"q": topic, "n": per_topic * 3},
            lambda t=topic: search(t, per_topic * 3),
            cache_dir,
        )
        papers = deduplicate(
            [
                PaperIn(
                    title=r["title"],
                    abstract=r["abstract"],
                    link=f"arxiv:{r['id']}",
                    arxiv_id=r["id"],
                    source="arxiv",
                )
                for r in raw
            ]
        )
        for p in rank_by_relevance(topic, papers, embedder, per_topic):
            rows.append(
                {
                    "topic": topic,
                    "paper_id": p.arxiv_id,
                    "title": p.title,
                    "abstract": p.abstract,
                    "cluster_label": "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["topic", "paper_id", "title", "abstract", "cluster_label"]
        )
        w.writeheader()
        w.writerows(rows)
    return "written"


def arxiv_topic_search(topic: str, n: int) -> list[dict]:
    """Free-text arXiv search by relevance."""
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    s = arxiv.Search(query=topic, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
    return [
        {
            "id": r.entry_id.rsplit("/abs/", 1)[-1].split("v")[0],
            "title": " ".join(r.title.split()),
            "abstract": " ".join(r.summary.split()),
        }
        for r in client.results(s)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-handlabel", action="store_true", help="only build Part A")
    args = ap.parse_args()
    set_seeds()
    data = build_part_a()
    data["meta"]["environment"] = environment()
    save_json(BENCHMARK, data)
    log.info("wrote %s (%d papers)", BENCHMARK, data["meta"]["n_papers"])
    if data["meta"]["shortfalls"]:
        log.warning("categories with fewer papers than requested: %s", data["meta"]["shortfalls"])
    if not args.skip_handlabel:
        from app.services.cluster import Embedder

        status = build_handlabel_template(Embedder(), arxiv_topic_search)
        log.info("hand-label template: %s (%s)", HANDLABEL_CSV, status)


if __name__ == "__main__":
    main()
