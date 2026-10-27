# Things only you can do

Work that needs a person. Each item says exactly what to do. (More items are added in Phase 5.)

## 1. Hand-label the clustering benchmark (Part B)

Optional but worthwhile: the automatic benchmark uses arXiv categories as a proxy for topics, and your labels are the check on that.

1. Open `eval/data/handlabel_template.csv`. It has 150 rows (5 topics x 30 papers retrieved by the app's own pipeline).
2. For each topic, group its papers into the sub-themes you would use in a literature review. Write a short label of your choice in the `cluster_label` column, using the **same label text** for papers that belong together (e.g. `segmentation`, `classification`). Aim for 3-6 labels per topic. Do not edit any other column.
3. A topic is evaluated once at least 80% of its rows are labelled and there are two or more distinct labels. Topics you leave blank are reported as "not run".
4. Run `make eval` again (the arXiv and LLM results are cached, so only the clustering re-runs quickly), or just `python -m eval.run_clustering`, then `python -m eval.report`.

## 2. Rate the summaries (blinded human evaluation)

1. Generate the sheet (only after `make eval` has produced `eval/results/summaries.json` with both arms):
   `python -m eval.make_human_eval`
   This writes `eval/human/ratings.csv` and a private `eval/human/key.json` (gitignored). **Do not open `key.json` before you finish rating.** The command refuses to overwrite an existing `ratings.csv`.
2. In `ratings.csv`, read each `abstract` and rate `output_A` and `output_B` from 1 (bad) to 5 (very good) in the columns `accuracy_A`, `usefulness_A`, `accuracy_B`, `usefulness_B`.
   - accuracy: does the output say only things the abstract supports?
   - usefulness: would it help you decide whether to read the paper?
   - `notes` is free text and is optional.
3. Score it: `python -m eval.score_human_eval`. It refuses to run if any rating is missing or outside 1-5, and writes `eval/results/human_eval.json`.
4. Re-run `python -m eval.report` to include the result in `eval/results.md`.

Caveats to keep in mind when you write about it: 20 items is small; the two outputs are shown as plain prose, so blinding is imperfect; and the evidence quotes and structured view are not part of what is rated.
