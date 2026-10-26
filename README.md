# 🧠 Lazy Researcher Toolkit

[![CI](https://github.com/rushei1/lazy-researcher-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rushei1/lazy-researcher-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Fetches arXiv papers for a topic, summarizes each with a transformer, clusters them by meaning, and exports a structured literature review as PDF. FastAPI backend, Streamlit frontend, SQLite history.

## 📸 Demo

![UI Screenshot](demo_screenshot.png)

## 🚀 Quickstart

### Docker

```bash
docker compose up --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

The first review downloads the models (roughly 2 GB for BART-large plus the embedding model) into the `hf_cache` volume; after that they are reused. Reviews persist in the `db_data` volume.

### Local (`make`)

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
make run-api     # terminal 1: http://127.0.0.1:8000
make run-ui      # terminal 2: http://localhost:8501
```

Other targets: `make lint`, `make format`, `make test`, `make docker-up`.

## 🔌 API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check. Does not load any model. |
| GET | `/review?topic=...` | Fetch, summarize and cluster papers. Optional `num_clusters` (1-8), `max_results` (1-30), `sources` (`arxiv`, `semantic_scholar`), `year_from`, `year_to`, `min_citations`, `venue_contains`. Repeated identical requests are served from the SQLite cache (`cached: true`). |
| GET | `/reviews` | List stored reviews, newest first. |
| GET | `/reviews/{id}` | One stored review. |
| GET | `/reviews/{id}/pdf` | The review as a PDF download. |
| DELETE | `/reviews/{id}` | Delete a review. |

Errors use `{"detail": "..."}`: `422` for an empty topic or invalid parameters, `404` for an unknown review, `502` when arXiv fails or times out. A topic with no matching papers returns `200` with empty `clusters` and a `message`. If one source fails but another works, the response is `200` with the partial results and a `warnings` list; such results are saved to history but never served from the cache.

The `clusters` field keeps the v1.0 shape: `{"0": [{"title", "summary", "link"}], ...}`.

## ⚙️ Configuration

Environment variables (or a `.env` file; see `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `ARXIV_MAX_RESULTS` | `10` | Papers fetched per review when `max_results` is not given |
| `ARXIV_TIMEOUT_SECONDS` | `30` | Give up on arXiv after this long (returns 502) |
| `DEFAULT_SOURCES` | `arxiv` | Comma list of sources used when the request gives none (`arxiv`, `semantic_scholar`) |
| `S2_API_KEY` | _(unset)_ | Optional Semantic Scholar key. Works without one, but the shared unauthenticated pool is heavily rate limited |
| `DEFAULT_NUM_CLUSTERS` | `3` | Clusters when `num_clusters` is not given; clamped to the number of papers |
| `SUMMARIZER_MODEL` | `facebook/bart-large-cnn` | Hugging Face summarization model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model used for clustering |
| `DB_PATH` | `data/reviews.db` | SQLite file for history and cache |
| `API_URL` | `http://127.0.0.1:8000` | Backend URL, read by the Streamlit frontend |

## 📁 Project structure

```
app/
  main.py            FastAPI app factory and routes
  config.py          pydantic-settings configuration
  schemas.py         request/response models
  db.py              SQLite history and cache (stdlib sqlite3)
  services/
    fetch.py         PaperSource protocol; arXiv and Semantic Scholar sources
    http.py          GET-JSON with retry/backoff/Retry-After and a rate limiter
    retrieve.py      query sources in parallel, dedup, filter, rank
    dedup.py         merge duplicates by DOI / arXiv id / title
    rank.py          cosine ranking of candidates against the query
    summarize.py     BART summarizer, lazy-loaded
    cluster.py       SBERT embedder + KMeans clusterer, lazy-loaded
    export.py        PDF export (reportlab platypus)
frontend/app.py      Streamlit UI
tests/               pytest suite (no network, no model downloads)
Dockerfile, Dockerfile.frontend, docker-compose.yml, Makefile
```

## ✅ Tests

```bash
make test        # or: pytest --cov=app
```

The suite uses fake summarizer/embedder/arXiv implementations injected through FastAPI dependency overrides, so it needs no network and no model weights. `pytest --run-slow` additionally loads the real models on three abstracts.

## ⚠️ Limitations (v1.1)

- The BART summarizer truncates title + abstract to 1024 characters, so long abstracts are cut off. Summaries are generic compressions, not structured or fact-checked against the source.
- Clustering is plain KMeans on sentence embeddings with a user-chosen cluster count; there is no automatic choice of k and no cluster labels.
- Semantic Scholar is optional and, without an API key, often rate limited (you then get a `warnings` entry and arXiv-only results). arXiv records have no citation counts or venues, so `min_citations` / `venue_contains` drop arXiv-only papers.
- Filters are applied after retrieval, so a strict filter can return fewer than `max_results` papers.
- Title-only deduplication is conservative (near-identical titles with compatible year and authors); some cross-source duplicates will survive.
- There is no evaluation yet, so nothing here claims a measured summary or clustering quality.
- PDF export uses the built-in Helvetica font: Latin text only, so Greek or CJK characters in titles will not render.
- A request that has to summarize papers on CPU can take minutes the first time; there is no background job queue.

## 🗺️ Roadmap

v2.0 (separate upgrade, not part of v1.1):

- LLM summarization with grounding: structured claims, each tied to a verbatim quote from the abstract and checked programmatically.
- Semantic Scholar as a second source, with deduplication.
- Pluggable clustering methods and an evaluation benchmark with reproducible results.

## 📄 License

MIT. See [LICENSE](LICENSE). Built by Rusheil Singh Baath.
