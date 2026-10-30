# Contributing

Small, focused changes are easiest to review.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
pre-commit install   # optional: runs ruff on commit
```

## Before opening a pull request

```bash
make lint    # ruff check + ruff format --check
make test    # pytest with coverage of app/
```

Tests must not need the network, an API key or model downloads; use the fakes in `tests/conftest.py`. Tests that need those are marked `slow`, `llm` or `network` and are skipped unless run with `--run-slow`, `--run-llm` or `--run-network`.

## Conventions

- Conventional Commit messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `build:`, `ci:`, `chore:`), one logical change per commit.
- Keep API changes additive: fields added to responses must be optional so older clients keep working.
- Numbers in the README, report or `eval/results.md` must come from a real run of the scripts in `eval/`. If something was not run, say "not run" and why.
- No secrets in the repository; keys come from environment variables.

## Changing prompts

Bump `PROMPT_VERSION` in `app/services/summarize.py` whenever the summary prompt or schema changes. It is part of the LLM cache key and the review cache key, so stale results are not served.
