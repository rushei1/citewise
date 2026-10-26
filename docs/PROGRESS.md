# Progress

Branch: `upgrade-v2` (renamed from `v1.1-infra`; no history was rewritten). Tag `v1.0` = original `main`.

## Done
- Phase 0: license, ruff/pytest/pre-commit config, env example, gitignore, this file.
- Phase 1.1-1.9: package split, schemas, config, clustering fix, errors/health, SQLite cache, server-side PDF, frontend fixes, pinned requirements.

## Decisions / deviations
- Work started from the narrower v1.1 brief, then the all-phases brief arrived mid-run. Phase 0 items
  were therefore added as follow-up commits instead of the order the new brief lists.
- Dev venv is Python 3.11 (`.venv/`, gitignored); the code uses 3.10+ syntax, so Python 3.9 (README v1) is no longer supported.
- `ARXIV_TIMEOUT_SECONDS` added to config: arxiv.py has no request timeout, so the call is bounded from outside with a thread.
- Empty-result reviews are stored in SQLite like any other (one code path); arXiv failures are never cached.
- PDF uses the built-in Helvetica: Latin text only. Greek/CJK in titles will not render.

## Blockers
- none
