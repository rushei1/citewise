---
title: Citewise
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Literature review from arXiv and Semantic Scholar
---

# Citewise (Space)

Searches arXiv and Semantic Scholar for a topic, clusters the papers and summarizes each one. Source code and full documentation: https://github.com/rushei1/citewise

## Notes for this Space

- **Deploying:** create a Docker Space, copy the repository into it, and rename this file to `README.md` and `Dockerfile.space` to `Dockerfile`. The YAML block above is what Spaces reads.
- **CPU only.** The default summarizer is `sshleifer/distilbart-cnn-12-6`, a smaller BART than the one used locally, so summaries are generated on CPU in reasonable time. The first review downloads the models and can take several minutes.
- **The LLM summarizer is off unless you configure it.** It needs a model and credentials supplied as Space secrets (Settings -> Variables and secrets): `LLM_MODEL` plus `LLM_API_KEY` (and optionally `LLM_BASE_URL`), or `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`. Anyone using the Space then spends the owner's key, so only add one if you accept that. Without it, choosing the "llm" summarizer shows an error.
- **Storage is ephemeral.** Reviews and the LLM cache live in the container and disappear when the Space restarts.
- **Semantic Scholar** works without a key but is heavily rate limited; set `S2_API_KEY` as a secret for reliable use.
