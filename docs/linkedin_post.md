# Launch post (draft)

Edit freely. Anything in `[BRACKETS]` is for you to fill in from `eval/results.md` (the current values are listed below the draft so you can check them). Keep only claims you are comfortable defending.

---

I rebuilt a side project of mine, a tool that takes a research topic, pulls papers from arXiv and Semantic Scholar, groups them by theme and summarizes each one, so that it is tested and measured instead of just working on my machine.

Two things I learned from measuring it:

1. Asking a language model to back every claim with a quote from the abstract, and then checking the quote in code, is not a formality. With a local 7B model on [N_PAPERS] papers, [VERIFIED_PCT] of the [N_CLAIMS] claims had a quote that was really in the abstract. The rest would have looked just as fluent.
2. A density-based clusterer I had picked by intuition ([HDBSCAN]) scored [ARI] against a simple agglomerative baseline's [ARI_AGGLO] on my benchmark. I only found out because I wrote the evaluation.

What's in the repo: FastAPI + Streamlit app, Docker, CI, [N_TESTS] tests, a reproducible eval harness with cached API calls, and exports to PDF, Markdown, LaTeX and BibTeX. It also lists what it does badly: grounding is against abstracts only, the benchmark is a proxy built from arXiv categories, and I have not collected human ratings yet.

Code and write-up: [REPO_LINK]

---

## Numbers to check before posting (from the committed results)

| Placeholder | Source | Current value |
|---|---|---|
| `[N_PAPERS]` | summarization sample | 40 |
| `[N_CLAIMS]` | LLM claims produced | 217 |
| `[VERIFIED_PCT]` | claims with the quote found in the abstract | 192 of 217 (88.5%) |
| `[HDBSCAN]` ARI | HDBSCAN with the app's `min_cluster_size` rule, all four embeddings | 0.00 |
| `[ARI_AGGLO]` | agglomerative, true k, MiniLM / mpnet | 0.70 / 0.70 |
| `[N_TESTS]` | `pytest` | run `pytest -q` and use the number it prints |

The LLM numbers are for `mistral:latest` via Ollama, 40 papers, temperature 0: say "a local 7B model", not "LLMs". The benchmark labels are arXiv categories, a proxy for topics.
