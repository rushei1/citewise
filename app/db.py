import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import ReviewListItem, ReviewResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_normalized TEXT NOT NULL,
    topic_display TEXT NOT NULL,
    params_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_cache ON reviews (topic_normalized, params_json);
"""


def normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().split())


def cache_params(
    max_results: int,
    num_clusters: int | str | None,
    cluster_method: str,
    summarizer: str,
    embedder: str,
    sources: list[str],
    filters: dict,
) -> str:
    """Canonical JSON of everything besides the topic that affects a result."""
    return json.dumps(
        {
            "sources": sorted(sources),
            "filters": filters,
            "max_results": max_results,
            "num_clusters": num_clusters,
            "cluster_method": cluster_method,
            "summarizer_model": summarizer,
            "embedding_model": embedder,
        },
        sort_keys=True,
    )


class ReviewDB:
    """SQLite store for reviews. One short-lived connection per call, so it is safe to share
    across FastAPI's worker threads."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _to_response(row: sqlite3.Row, cached: bool) -> ReviewResponse:
        result = json.loads(row["result_json"])
        return ReviewResponse(
            review_id=row["id"],
            topic=row["topic_display"],
            created_at=row["created_at"],
            cached=cached,
            clusters=result["clusters"],
            cluster_info=result.get("cluster_info", {}),
            message=result.get("message"),
            warnings=result.get("warnings", []),
        )

    def find_cached(self, topic: str, params: str) -> ReviewResponse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reviews WHERE topic_normalized = ? AND params_json = ? "
                "ORDER BY id DESC LIMIT 1",
                (normalize_topic(topic), params),
            ).fetchone()
        if row is None or json.loads(row["result_json"]).get("warnings"):
            return None  # degraded (partial-source) results are history, not cache
        return self._to_response(row, cached=True)

    def save(self, topic: str, params: str, review: ReviewResponse) -> ReviewResponse:
        """Store a freshly generated review and return it with its real id."""
        created_at = datetime.now(UTC)
        result = {
            "clusters": review.model_dump()["clusters"],
            "cluster_info": review.model_dump()["cluster_info"],
            "message": review.message,
            "warnings": review.warnings,
        }
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reviews (topic_normalized, topic_display, params_json, result_json, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (normalize_topic(topic), topic, params, json.dumps(result), created_at.isoformat()),
            )
            review_id = cur.lastrowid
        return review.model_copy(update={"review_id": review_id, "created_at": created_at})

    def get(self, review_id: int) -> ReviewResponse | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        return self._to_response(row, cached=False) if row else None

    def list_reviews(self) -> list[ReviewListItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, topic_display, created_at FROM reviews ORDER BY id DESC"
            ).fetchall()
        return [
            ReviewListItem(id=r["id"], topic=r["topic_display"], created_at=r["created_at"])
            for r in rows
        ]

    def delete(self, review_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,)).rowcount > 0
