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

## Not verified
- `slow` pytest (real models) was not run via pytest, but the same models ran for real inside Docker (see below).
- Pinned `requirements.txt` (torch etc.) was resolved with `pip install --dry-run` in a fresh 3.11 venv, not fully installed.
- GitHub Actions workflow has not run yet (nothing pushed).

## Next
- Phase 2 (better retrieval). Stopped at the v1.1 checkpoint as the brief requires.

## Decisions / deviations
- Work started from the narrower v1.1 brief, then the all-phases brief arrived mid-run. Phase 0 items
  were therefore added as follow-up commits instead of the order the new brief lists.
- Dev venv is Python 3.11 (`.venv/`, gitignored); the code uses 3.10+ syntax, so Python 3.9 (README v1) is no longer supported.
- `ARXIV_TIMEOUT_SECONDS` added to config: arxiv.py has no request timeout, so the call is bounded from outside with a thread.
- Empty-result reviews are stored in SQLite like any other (one code path); arXiv failures are never cached.
- PDF uses the built-in Helvetica: Latin text only. Greek/CJK in titles will not render.

## Blockers
- none
