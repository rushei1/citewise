# Progress

Branch: `upgrade-v2` (renamed from `v1.1-infra`; no history was rewritten). Tag `v1.0` = original `main`.

## Done
- Phase 0: license, ruff/pytest/pre-commit config, env example, gitignore, this file.
- Phase 0: also v1.0 limitations section in README.
- Phase 1 (1.1-1.13): package split, schemas, config, clustering fix, errors/health, SQLite cache, server-side PDF, frontend fixes, pinned requirements, tests (35 pass, 99% coverage of `app/`), Docker/compose/Makefile, CI, README v1.1. Tagged `v1.1` locally.

## Docker verification (2026-10-26)
- `docker compose build` and `up` OK; backend healthy, `/docs` 200, UI 200, frontend reaches `http://backend:8000`.
- Container runs as uid 1000; data and HF cache volumes writable.
- Real `/review` (2 papers, real arXiv + BART + SBERT, cold start ~2.5 min incl. model download) succeeded; identical repeat was `cached: true`; PDF endpoint returned a valid `%PDF`; the review survived a backend restart.

## Phase 2 (done, 2026-10-27)
- 2.1 PaperSource + extended paper model; 2.2 Semantic Scholar; 2.3 tenacity retry/backoff, Retry-After, per-source rate limiter, partial-failure `warnings`; 2.4 dedup; 2.5 semantic ranking; 2.6 filters + `sources` param; 2.7 tests (104 pass, 99% of `app/`).
- Deviations: title dedup uses `difflib` ratio >= 0.95 plus year/author guards instead of rapidfuzz token-set >= 0.92, because token-set scores 100 for "X Is All You Need" vs "X Is Not All You Need". Papers missing a field a filter needs are dropped. Degraded (warnings) results are saved but not cache-served. Default source stays `arxiv` because unauthenticated Semantic Scholar is mostly rate limited.
- The S2 test fixture is SYNTHETIC (`tests/fixtures/s2_search_synthetic.json`, abstracts invented): live recording kept hitting HTTP 429. One real call earlier did succeed after 4 retries and confirmed the field mapping.
- Docker check: real `sources=arxiv,semantic_scholar&year_from=2019` request returned arXiv papers plus a `semantic_scholar: ... HTTP 429` warning, as designed. A real S2 success inside the app was not observed (needs `S2_API_KEY`).

## Not verified
- `slow` pytest (real models) was not run via pytest, but the same models ran for real inside Docker (see below).
- Pinned `requirements.txt` (torch etc.) was resolved with `pip install --dry-run` in a fresh 3.11 venv, not fully installed.
- GitHub Actions workflow has not run yet (nothing pushed).

## Next
- Phase 3 (LLM summarization with grounding).

## Decisions / deviations
- Work started from the narrower v1.1 brief, then the all-phases brief arrived mid-run. Phase 0 items
  were therefore added as follow-up commits instead of the order the new brief lists.
- Dev venv is Python 3.11 (`.venv/`, gitignored); the code uses 3.10+ syntax, so Python 3.9 (README v1) is no longer supported.
- `ARXIV_TIMEOUT_SECONDS` added to config: arxiv.py has no request timeout, so the call is bounded from outside with a thread.
- Empty-result reviews are stored in SQLite like any other (one code path); arXiv failures are never cached.
- PDF uses the built-in Helvetica: Latin text only. Greek/CJK in titles will not render.

## Blockers
- none
