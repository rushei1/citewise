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

## Phase 3 (done, 2026-10-27; branch `upgrade-v2-phase3`, 241 tests pass, 99% of `app/`)
- 3.1 `Summarizer` protocol (`summarize(paper) -> PaperSummary`), `HFSummarizer`, `SUMMARIZER_BACKEND` + `summarizer` query param.
- 3.2 `LLMClient` protocol with OpenAI-compatible and Anthropic clients over the shared `Http` helper (now also `post_json`), token-usage logging, no default model name.
- 3.3 `LLMSummarizer`: JSON claims with quotes, pydantic validation, 2 retries with the error fed back, programmatic evidence check, `not stated` enforced by schema.
- 3.4 SQLite LLM cache keyed by sha256(model, prompt version, abstract); also used for theme replies.
- 3.5 kmeans / agglomerative / hdbscan behind a `Clusterer` protocol, `num_clusters=auto` (silhouette), embedding of `title + abstract`.
- 3.6 c-TF-IDF cluster labels; LLM theme summaries with checked `[n]` citations.
- 3.7 BibTeX, Markdown, LaTeX exports and a richer PDF (`/reviews/{id}/bibtex|markdown|latex|pdf`).
- 3.8 Streamlit UI for all of the above. 3.9 tests. 3.10 README, `docs/architecture.md` (Mermaid), compose passes the LLM variables through.
- Verified end to end in Docker with real arXiv and the real MiniLM embedder, and a local fake OpenAI-compatible server standing in for the LLM: invented evidence was dropped and counted, `[99]` was stripped from a theme, a repeat request was `cached: true`, a different `cluster_method` re-used all six cached summaries, and all four export routes returned valid files. Container, volumes and image were removed afterwards.
- Deviations: tests were committed together with each task instead of one trailing `test:` commit (plus one final commit for the gaps). Work is on `upgrade-v2-phase3`, branched from `upgrade-v2`, because the background job ran in a git worktree (a worktree cannot check out the branch the main checkout holds). HDBSCAN comes from scikit-learn (no extra package) and `min_cluster_size` is an untuned guess. No new dependencies were added.
- Decisions: evidence shorter than 12 characters never verifies (otherwise "the" would ground a claim); verification is literal (no quote or dash folding). Themes are written only for clusters of 2+ papers that have verified claims, and a theme without any valid citation is dropped with a warning. A failed paper summary is a warning with a placeholder; all papers failing is a 502. Degraded results (any warning) are stored but not cache-served, as before. The review cache key now includes the summarizer id and cluster method (old v1.1 cache rows simply miss). `DEFAULT_NUM_CLUSTERS` also accepts `auto`; hdbscan ignores the count and shares one cache entry across counts. The unclustered group has id `-1` and sorts last. Exports use Windows-1252 (PDF) / Latin-1 (LaTeX) text; other characters print as `?`.

## Phase 4 (done, 2026-10-27; branch `upgrade-v2-phase4`; results in `eval/results.md`)
- 4.1 benchmark builder + 180-paper proxy benchmark + hand-label template; 4.2 clustering comparison (4 embedding models x KMeans/agglomerative, k = true and silhouette auto-k, HDBSCAN incl. fixed `min_cluster_size` 5/10, BERTopic, 5 seeds for randomised methods); 4.3 summarization comparison; 4.4 blinded human-eval kit; 4.5 failure cases; 4.6 report + `make eval` / `make eval-quick`; 4.7 tests (eval code is covered by the normal suite).
- Actually run: clustering (all methods, all four embedding models, BERTopic installed in the dev venv only), BART arm, NLI, and the LLM arm with **local Ollama `mistral:latest`** (a 7B model, temperature 0), 40 papers, 1 failed. `eval/results/*.json`, `results.md` and `failure_cases.md` are generated from those runs.
- Headline numbers (see results.md for all): best clustering was agglomerative with k = true (ARI about 0.70 with MiniLM or mpnet) on the proxy benchmark. HDBSCAN scored ARI 0.00 with the app's own `min_cluster_size` rule (everything labelled noise) and at most 0.09 with fixed sizes 5 and 10, so it is a poor default here (the app default stays KMeans). Of 217 claims from the LLM arm, 192 (88.5%) had their quote verbatim in the abstract and 25 were dropped; every successful paper said `limitations` were "not stated". The NLI model marks only 21% of LLM claims and 8% of BART sentences as entailed (most are "neutral"), so it looks conservative and should not be read as a faithfulness score.
- Deviations / decisions: ROUGE is implemented in `eval/metrics.py` (lowercased alphanumeric tokens, no stemming), so it is not comparable with the `rouge-score` package. Cost is not computed (no price table); only tokens are reported. The benchmark sample covers 2023-01-01 to 2023-01-25 only, because the earliest 120 papers per category all fall in January (stored as `effective_dates`). The summarization arms run as separate processes (`--arm hf|llm|nli`) after two runs were killed for memory pressure; each merges into the result files. LLM latency (about 124 s per paper) is NOT representative: a 4 GB laptop GPU with the model partly on CPU, and some calls overlapped with other jobs. HF and LLM latencies are not comparable.
- Findings worth acting on: `mistral` produced one reply with an invalid JSON escape (`Invalid \escape`), which exhausted the schema retries; a more lenient JSON parse could recover such replies. This was not changed, so the results stay tied to the committed code.
- Not run: Part B (needs your labels), human evaluation (needs your ratings; the blinded `eval/human/ratings.csv` is generated and `key.json` is gitignored). The GitHub Actions run of `eval-quick` has not happened (nothing pushed to Actions yet).

## Not verified
- No live LLM was used anywhere: the LLM path is tested with a scripted fake client and the end-to-end check above used a fake server. The `llm`-marked test (`pytest --run-llm`, needs `LLM_MODEL` plus a key or base URL) has not been run. Real model behaviour (JSON compliance, how often quotes verify) is exactly what Phase 4 should measure.
- The generated `.tex` has not been compiled: no LaTeX installation here. It is checked structurally in tests, and a compile test runs automatically where `pdflatex` exists.
- `slow` pytest (real models) was not run via pytest, but the same models ran for real inside Docker (see below).
- Pinned `requirements.txt` (torch etc.) was resolved with `pip install --dry-run` in a fresh 3.11 venv, not fully installed.
- GitHub Actions workflow has not run yet (nothing pushed). Frontend tests need streamlit, which CI does not install, so they are skipped there.

## Next
- Phase 4 (evaluation). Merge Phase 3 first: from the main checkout on `upgrade-v2`, `git merge --ff-only upgrade-v2-phase3`.

## Decisions / deviations
- Work started from the narrower v1.1 brief, then the all-phases brief arrived mid-run. Phase 0 items
  were therefore added as follow-up commits instead of the order the new brief lists.
- Dev venv is Python 3.11 (`.venv/`, gitignored); the code uses 3.10+ syntax, so Python 3.9 (README v1) is no longer supported.
- `ARXIV_TIMEOUT_SECONDS` added to config: arxiv.py has no request timeout, so the call is bounded from outside with a thread.
- Empty-result reviews are stored in SQLite like any other (one code path); arXiv failures are never cached.
- PDF uses the built-in Helvetica: Latin text only. Greek/CJK in titles will not render.

## Blockers
- none
