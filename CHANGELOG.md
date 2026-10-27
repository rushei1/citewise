# Changelog

Entries follow the commit history of this repository. Dates are omitted because nothing has been released from a tag yet.

## v2.0 (release candidate)

### Added
- Second paper source (Semantic Scholar) behind a `PaperSource` interface, with retry, backoff and `Retry-After` handling, per-source rate limits and partial-failure `warnings`.
- Cross-source deduplication (DOI, arXiv id, title similarity), semantic ranking of candidates against the query, and year / citation / venue / source filters.
- Pluggable summarizers: BART (`hf`) and an LLM backend (`llm`) for any OpenAI-compatible server or Anthropic. The LLM backend returns problem / method / results / limitations claims, each with a quote that is verified against the abstract; unverifiable claims are dropped and counted.
- Content-hash cache for LLM replies.
- Pluggable clustering (KMeans, agglomerative, HDBSCAN) with automatic k, keyword cluster labels, and cited theme summaries whose `[n]` citations are checked.
- BibTeX, Markdown and LaTeX exports; the PDF now includes labels, themes and structured claims.
- Streamlit UI for all of the above.
- Evaluation harness in `eval/`: proxy clustering benchmark, clustering and summarization experiments, grounding and optional NLI metrics, blinded human-evaluation kit, failure-case extraction, report generation, `make eval` and `make eval-quick`. Results from real runs are in `eval/results.md`.
- Architecture doc, research write-up draft, Hugging Face Spaces deployment files (not deployed), contributing guide and issue templates.

### Changed
- Papers are embedded from `title + abstract` instead of the generated summary.
- The review cache key includes the summarizer, cluster method and cluster count.

### Notes
- The LLM path has been evaluated with one local model (`mistral:latest`) on 40 papers; hand-labelled clustering and human ratings are not done.

## v1.1

### Added
- `app/` package with lazily loaded models, pydantic schemas and environment-based configuration.
- Error handling (`502` for source failures, `422` for bad input), `/health`, logging.
- SQLite history and caching with `GET /reviews`, `GET /reviews/{id}`, `DELETE /reviews/{id}`.
- Server-side PDF export with real text wrapping, page numbers and escaped text.
- Test suite with fakes (no network, no model downloads), Docker and compose, Makefile, GitHub Actions workflow, license, ruff and pre-commit configuration.

### Fixed
- KMeans no longer crashes when fewer papers than requested clusters come back.
- The Streamlit export button no longer disappears on rerun; history no longer goes through unsanitised file names on the client.
- The API URL is configurable (it was hardcoded and broke in Docker).
- Requirements are minimal pinned files instead of `pip freeze` dumps.

## v1.0

The original project: FastAPI + Streamlit app that fetches ten arXiv papers, summarizes each with BART, clusters them with sentence embeddings and KMeans (three clusters) and exports a PDF from the client.
