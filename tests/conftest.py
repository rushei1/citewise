import hashlib

import pytest
from fastapi.testclient import TestClient

from app.db import ReviewDB
from app.main import app, get_clusterer, get_db, get_sources, get_summarizers
from app.schemas import PaperIn, PaperSummary
from app.services.cluster import Clusterer

MARKERS = ("slow", "llm", "network")


def pytest_addoption(parser):
    for name in MARKERS:
        parser.addoption(
            f"--run-{name}", action="store_true", default=False, help=f"run @{name} tests"
        )


def pytest_collection_modifyitems(config, items):
    for item in items:
        for name in MARKERS:
            if name in item.keywords and not config.getoption(f"--run-{name}"):
                item.add_marker(pytest.mark.skip(reason=f"needs --run-{name}"))


class FakeSummarizer:
    id = "fake"

    def __init__(self):
        self.calls = 0

    def summarize(self, paper: PaperIn) -> PaperSummary:
        self.calls += 1
        return PaperSummary(overview=f"Summary of {paper.title}")


class FakeEmbedder:
    """Deterministic: the same text always maps to the same 8-dim vector."""

    def encode(self, texts):
        return [[b / 255 for b in hashlib.md5(t.encode()).digest()[:8]] for t in texts]


class ScriptedLLM:
    """Stands in for an LLMClient: returns the scripted replies in order (the last one repeats)
    and records every message list it was sent."""

    model = "fake-llm"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def make_papers(n: int, title: str = "Paper") -> list[PaperIn]:
    return [
        PaperIn(title=f"{title} {i}", abstract=f"Abstract {i}", link=f"http://arxiv.org/abs/{i}")
        for i in range(n)
    ]


class FakeFetcher:
    """Stands in for a PaperSource; set .papers or .error to script behaviour."""

    def __init__(self, name: str = "arxiv"):
        self.name = name
        self.papers = make_papers(6)
        self.error: Exception | None = None
        self.calls = 0

    def search(self, query, max_results, filters):
        self.calls += 1
        if self.error:
            raise self.error
        return self.papers[:max_results]


@pytest.fixture
def summarizer():
    return FakeSummarizer()


@pytest.fixture
def fetcher():
    return FakeFetcher()


@pytest.fixture
def db(tmp_path):
    return ReviewDB(tmp_path / "data" / "reviews.db")


@pytest.fixture
def client(summarizer, fetcher, db):
    app.dependency_overrides[get_summarizers] = lambda: {"hf": summarizer}
    app.dependency_overrides[get_clusterer] = lambda: Clusterer(FakeEmbedder())
    app.dependency_overrides[get_sources] = lambda: {fetcher.name: fetcher}
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()
