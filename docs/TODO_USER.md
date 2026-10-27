# Things only you can do

Everything here needs a person, an account or a decision. Each item says exactly what to do.

## 1. Hand-label the clustering benchmark (Part B)

Optional but worthwhile: the automatic benchmark uses arXiv categories as a proxy for topics, and your labels are the check on that.

1. Open `eval/data/handlabel_template.csv`. It has 150 rows (5 topics x 30 papers retrieved by the app's own pipeline).
2. For each topic, group its papers into the sub-themes you would use in a literature review. Write a short label of your choice in the `cluster_label` column, using the **same label text** for papers that belong together (e.g. `segmentation`, `classification`). Aim for 3-6 labels per topic. Do not edit any other column.
3. A topic is evaluated once at least 80% of its rows are labelled and there are two or more distinct labels. Topics you leave blank are reported as "not run".
4. Run `python -m eval.run_clustering` (embeddings are cached, so it is quick; BERTopic is the slow part), then `python -m eval.report`, then `python -m eval.write_report`, and commit the changed files in `eval/results/`, `eval/results.md` and `docs/report.md`.

## 2. Rate the summaries (blinded human evaluation)

1. `eval/human/ratings.csv` is already generated (20 papers, outputs A/B shuffled) and the private answer key is in `eval/human/key.json` (gitignored). **Do not open `key.json` before you finish rating.** To regenerate the sheet, run `python -m eval.make_human_eval --force` (it refuses to overwrite otherwise).
2. In `ratings.csv`, read each `abstract` and rate `output_A` and `output_B` from 1 (bad) to 5 (very good) in the columns `accuracy_A`, `usefulness_A`, `accuracy_B`, `usefulness_B`.
   - accuracy: does the output say only things the abstract supports?
   - usefulness: would it help you decide whether to read the paper?
   - `notes` is free text and optional.
3. Score it: `python -m eval.score_human_eval`. It refuses to run if any rating is missing or outside 1-5, and writes `eval/results/human_eval.json`.
4. Then `python -m eval.report` and `python -m eval.write_report`, and write your interpretation in the `TODO(rusheil)` block of section 5.3 in `docs/report_template.md` (regenerate again afterwards).

Caveats to keep in mind when you write about it: 20 items is small; the two outputs are shown as plain prose, so blinding is imperfect; and the evidence quotes and structured view are not part of what is rated. Note that `key.json` exists only on the machine that generated it; if you rate on another machine, copy it over.

## 3. Finish the report

`docs/report_template.md` has `TODO(rusheil)` blocks: your motivation, the interpretation of the ablations (the draft deliberately asserts no causes), a qualitative look at the listed failure cases, and the human-evaluation paragraph. Edit the template (not `docs/report.md`), then run `make report`; `make report-pdf` also makes a PDF if you install pandoc.

## 4. Record the demo GIF

1. Run the app (`docker compose up --build`, or `make run-api` and `make run-ui`).
2. Record 60-90 seconds with a screen recorder (e.g. ScreenToGif on Windows, Peek or Kooha on Linux): enter a topic, generate a review, open a cluster, expand a paper to show a claim with its evidence quote, then download one export. Keep the browser window small (about 1000 px wide) to keep the file under ~10 MB.
3. Save it as `docs/demo.gif`.
4. In `README.md`, under the `## 📸 Demo` heading, add the line `![Demo](docs/demo.gif)` (keep or replace the existing screenshot).
5. Commit: `docs: add demo gif`.

## 5. Deploy the hosted demo (Hugging Face Spaces)

Not deployed by me; the deployment files are `Dockerfile.space`, `start_space.sh` and `space_README.md`. The Docker build of this image was not completed in my environment (see `docs/PROGRESS.md`), so expect to read the build log on the first deploy.

1. On huggingface.co create a new Space, SDK **Docker**, hardware **CPU basic**.
2. Clone the Space repo, copy this repository's files into it, then rename `space_README.md` to `README.md` and `Dockerfile.space` to `Dockerfile` (the Space reads its configuration from the YAML block at the top of that README).
3. Push. The first build and the first review are slow (the models download on first use).
4. Optional LLM backend: in the Space's Settings -> Variables and secrets add `LLM_MODEL` plus `LLM_API_KEY` (and `LLM_BASE_URL` for a non-OpenAI server), or `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`. This spends your key on every visitor's request, so decide whether to accept that. Without these the demo offers the BART summarizer only.
5. Optional: `S2_API_KEY` (Semantic Scholar) as a secret for reliable results.
6. Put the Space URL in the README (e.g. a "Live demo" line under the badges).

## 6. GitHub settings

**Rename the repository to `citewise`** (Settings -> General -> Repository name). The project was renamed from "Lazy Researcher Toolkit" in the code and docs, but the repository URL, the CI badge in `README.md`, the link in `space_README.md` and your local `origin` remote still use `lazy-researcher-toolkit`. GitHub redirects the old URL, so nothing breaks in the meantime. After renaming: update those URLs, run `git remote set-url origin https://github.com/rushei1/citewise.git`, and retake `demo_screenshot.png` (it still shows the old title).

Repository **About** (gear icon next to it on the repo page):

- Description (suggestion): `Literature-review toolkit: arXiv + Semantic Scholar retrieval, clustering, and LLM summaries whose claims are checked against the abstract.`
- Topics: `llm`, `rag`, `arxiv`, `literature-review`, `fastapi`, `streamlit`, `nlp`, `semantic-scholar`
- Website: the Space URL once it exists.

## 7. Review, push and merge

The work is on stacked branches that are already pushed: `upgrade-v2-phase3`, `upgrade-v2-phase4`, `upgrade-v2-phase5`. The last one contains everything from the earlier phases. `upgrade-v2` itself (Phases 0-2) is only local.

```bash
git log --oneline v1.1..upgrade-v2-phase5     # read the history
git diff main --stat                          # see what changed overall
```

1. Open **one** pull request from `upgrade-v2-phase5` into `main` (the intermediate branches are redundant after that and can be deleted).
2. Check that the GitHub Actions run passes. It has never run yet: the workflow installs only `requirements-dev.txt`, runs ruff, pytest and the model-free `eval-quick` smoke test. Fix anything that differs from your local environment.
3. Merge with a **merge commit** (not squash, not rebase) so the commits that the tags point to stay in `main`'s history.
4. Push the tags, which exist locally only: `git push origin v1.0 v1.1 v2.0` (run it from the clone that has them, i.e. your original checkout; `git tag` lists them).
5. Optionally create GitHub releases from the tags using the matching section of `CHANGELOG.md`.

## 8. Publish the post and update your profile

1. Open `docs/linkedin_post.md`, fill the `[PLACEHOLDERS]` from `eval/results.md` (the table at the bottom lists the current values), and cut anything you would not defend in an interview.
2. Post it with the repository link (and the Space link, if deployed).
3. Profile / bio line (suggestion, edit to taste): `Built and evaluated a literature-review toolkit with claim-level quote verification (FastAPI, Streamlit, sentence-transformers, Docker, CI) - [repo link]`.

## 9. Optional: a stronger evaluation

The grounding result is for one local 7B model on 40 papers. If you have a hosted model key: `LLM_MODEL=<name> LLM_API_KEY=<key> python -m eval.run_summarization --arm llm` (then `--arm nli`, `python -m eval.report`, `python -m eval.write_report`). Results from different models are not merged: the file records only the model that ran last, so keep the previous results elsewhere if you want to compare.
