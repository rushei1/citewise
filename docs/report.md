# Grounded literature-review generation: a retrieval, clustering and summarization toolkit with an evaluation of its parts

Rusheil Singh Baath. Draft. Generated from `docs/report_template.md` by `python -m eval.write_report`; all numbers in the Results and Failure analysis sections are copied from `eval/results.md` and `eval/results/failure_cases.md`.

## 1. Motivation

A literature review starts with a pile of papers that has to be sorted into themes and read. Tools that summarize each abstract with a language model are easy to build and hard to trust: a fluent sentence says nothing about whether the abstract supports it. This project builds a small end-to-end pipeline (retrieve, deduplicate, cluster, summarize) and asks two measurable questions: which clustering configuration recovers known groups best, and how often does a language model's claim about an abstract come with a quote that is really in that abstract.

<!-- TODO(rusheil): add 2-3 sentences of personal motivation / the M.Tech angle in your own words. -->

## 2. System overview

Papers are fetched from arXiv and Semantic Scholar in parallel, merged by DOI, arXiv id and title similarity, filtered, and ranked by embedding similarity to the query. Each paper is then summarized by one of two backends and, independently, embedded (title plus abstract) and clustered. Clusters get keyword labels, and with the LLM backend a short theme summary written only from the verified claims of the cluster. The review is stored and exported as PDF, Markdown, LaTeX or BibTeX.

```mermaid
flowchart LR
    Q["topic + filters"] --> A["arXiv"]
    Q --> S["Semantic Scholar"]
    A --> D["Deduplicate"]
    S --> D
    D --> F["Filters"]
    F --> R["Rank by similarity"]
    R --> SUM["Summarize (BART or LLM)"]
    SUM --> V["Verify quotes against abstract"]
    R --> E["Embed + cluster"]
    E --> L["Labels (c-TF-IDF)"]
    V --> T["Theme summaries, checked citations"]
    E --> T
    L --> X["Store + export"]
    T --> X
    V --> X
```

## 3. Methods

**Clustering.** Embeddings are L2-normalised sentence-transformer vectors of `title. abstract`. Compared methods: KMeans and agglomerative (Ward) clustering, each with the true number of classes and with k chosen by silhouette in 2..min(8, n-1); HDBSCAN (the application's rule `min_cluster_size = max(2, 10% of n)` and fixed sizes 5 and 10); and BERTopic on the same embeddings. HDBSCAN noise is treated as one extra cluster when computing ARI, NMI, homogeneity and completeness.

**Grounded summarization.** The LLM is asked for JSON with four fields (problem, method, results, limitations). Each field is either a list of claims, each with a quote copied from the abstract, or the string "not stated". Replies are validated against a schema (two retries with the error fed back). A claim is kept only if its quote occurs in the abstract after whitespace and case normalization and is at least 12 characters long; other claims are dropped and counted. Grounding is against the abstract only.

**Baseline.** BART (`facebook/bart-large-cnn`) on the first 1024 characters of title plus abstract.

## 4. Experimental setup

**Clustering benchmark (proxy).** 180 arXiv papers: the earliest 30 submissions in each of six categories (cs.CL, cs.CV, cs.IR, cs.RO, cs.CR, cs.DB) within a fixed window, labelled with their primary category. The sample falls within 2023-01-01 to 2023-01-25. Four embedding models: all-MiniLM-L6-v2, all-mpnet-base-v2, allenai-specter, BAAI/bge-small-en-v1.5. Randomised methods use five seeds (0-4) and are reported as mean ± population standard deviation.

**Summarization sample.** 40 papers from the same benchmark (round-robin over categories). The LLM arm is a local `mistral:latest` served by Ollama at temperature 0, through the same code path as the application. ROUGE is computed against the abstract with a simple tokenizer and no stemming. Entailment is judged by `cross-encoder/nli-deberta-v3-small`.

**Environment.**

- Date: 2026-10-27 07:55 UTC
- Python 3.11.9 on Windows-10-10.0.26300-SP0
- Packages: numpy 2.0.2, scikit-learn 1.6.1, scipy 1.17.1, sentence-transformers 5.0.0, transformers 4.53.2, torch 2.7.1+cpu, arxiv 2.2.0
- Seeds: [0, 1, 2, 3, 4]

## 5. Results

### 5.1 Clustering

180 arXiv papers, 6 classes (primary category). The labels are a proxy for topics, so a low score can mean the categories overlap, not only that clustering failed. Randomised methods: mean ± std (population std) over seeds; deterministic methods ran once. `k=true` gives the method the real number of classes; `auto` chooses k by silhouette. HDBSCAN noise counts as an extra cluster for ARI/NMI but not in `Clusters`.

| Embedding | Method | k | ARI | NMI | Homog. | Compl. | Silhouette | Clusters | Noise | Time (s) | Runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | kmeans | k=true | 0.599 ± 0.055 | 0.670 ± 0.039 | 0.662 ± 0.041 | 0.677 ± 0.037 | 0.070 ± 0.004 | 6.0 | 0.0% | 0.19 | 5 |
| all-MiniLM-L6-v2 | kmeans | auto | 0.610 ± 0.068 | 0.682 ± 0.052 | 0.696 ± 0.064 | 0.669 ± 0.046 | 0.072 ± 0.002 | 6.8 ± 0.7 | 0.0% | 0.05 | 5 |
| all-MiniLM-L6-v2 | agglomerative | k=true | 0.702 | 0.751 | 0.744 | 0.758 | 0.065 | 6.0 | 0.0% | 0.12 | 1 |
| all-MiniLM-L6-v2 | agglomerative | auto | 0.613 | 0.714 | 0.755 | 0.677 | 0.069 | 8.0 | 0.0% | 0.04 | 1 |
| all-MiniLM-L6-v2 | hdbscan | n/a | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.03 | 1 |
| all-MiniLM-L6-v2 | hdbscan | mcs=5 | 0.051 | 0.127 | 0.094 | 0.197 | 0.020 | 2.0 | 51.1% | 0.03 | 1 |
| all-MiniLM-L6-v2 | hdbscan | mcs=10 | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.03 | 1 |
| all-MiniLM-L6-v2 | bertopic | n/a | 0.234 ± 0.162 | 0.387 ± 0.169 | 0.308 ± 0.162 | 0.591 ± 0.075 | 0.020 ± 0.020 | 2.6 ± 0.5 | 8.3% | 2.68 | 5 |
| all-mpnet-base-v2 | kmeans | k=true | 0.555 ± 0.096 | 0.656 ± 0.071 | 0.643 ± 0.074 | 0.670 ± 0.067 | 0.068 ± 0.011 | 6.0 | 0.0% | 0.01 | 5 |
| all-mpnet-base-v2 | kmeans | auto | 0.553 ± 0.142 | 0.664 ± 0.094 | 0.620 ± 0.148 | 0.742 ± 0.046 | 0.077 ± 0.003 | 5.0 ± 1.7 | 0.0% | 0.07 | 5 |
| all-mpnet-base-v2 | agglomerative | k=true | 0.697 | 0.754 | 0.748 | 0.760 | 0.072 | 6.0 | 0.0% | 0.01 | 1 |
| all-mpnet-base-v2 | agglomerative | auto | 0.605 | 0.719 | 0.759 | 0.682 | 0.077 | 8.0 | 0.0% | 0.08 | 1 |
| all-mpnet-base-v2 | hdbscan | n/a | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.05 | 1 |
| all-mpnet-base-v2 | hdbscan | mcs=5 | 0.087 | 0.220 | 0.172 | 0.305 | 0.026 | 3.0 | 47.2% | 0.05 | 1 |
| all-mpnet-base-v2 | hdbscan | mcs=10 | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.05 | 1 |
| all-mpnet-base-v2 | bertopic | n/a | 0.504 ± 0.011 | 0.689 ± 0.006 | 0.594 | 0.822 ± 0.017 | 0.040 ± 0.012 | 4.0 | 0.8% | 0.40 | 5 |
| allenai-specter | kmeans | k=true | 0.515 ± 0.050 | 0.627 ± 0.024 | 0.612 ± 0.034 | 0.642 ± 0.014 | 0.089 ± 0.007 | 6.0 | 0.0% | 0.01 | 5 |
| allenai-specter | kmeans | auto | 0.541 ± 0.042 | 0.650 ± 0.024 | 0.608 ± 0.064 | 0.706 ± 0.031 | 0.101 ± 0.005 | 5.2 ± 1.5 | 0.0% | 0.09 | 5 |
| allenai-specter | agglomerative | k=true | 0.562 | 0.662 | 0.649 | 0.675 | 0.090 | 6.0 | 0.0% | 0.01 | 1 |
| allenai-specter | agglomerative | auto | 0.526 | 0.636 | 0.594 | 0.685 | 0.094 | 5.0 | 0.0% | 0.09 | 1 |
| allenai-specter | hdbscan | n/a | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.06 | 1 |
| allenai-specter | hdbscan | mcs=5 | 0.066 | 0.337 | 0.286 | 0.411 | -0.045 | 6.0 | 65.6% | 0.06 | 1 |
| allenai-specter | hdbscan | mcs=10 | 0.024 | 0.142 | 0.093 | 0.297 | -0.020 | 2.0 | 83.3% | 0.05 | 1 |
| allenai-specter | bertopic | n/a | 0.531 ± 0.032 | 0.645 ± 0.023 | 0.608 ± 0.038 | 0.691 ± 0.035 | 0.081 ± 0.013 | 4.6 ± 0.5 | 5.2% | 0.35 | 5 |
| BAAI/bge-small-en-v1.5 | kmeans | k=true | 0.548 ± 0.109 | 0.636 ± 0.069 | 0.624 ± 0.076 | 0.648 ± 0.062 | 0.059 ± 0.010 | 6.0 | 0.0% | 0.01 | 5 |
| BAAI/bge-small-en-v1.5 | kmeans | auto | 0.252 ± 0.077 | 0.401 ± 0.062 | 0.299 ± 0.069 | 0.643 ± 0.062 | 0.075 ± 0.002 | 2.6 ± 0.5 | 0.0% | 0.06 | 5 |
| BAAI/bge-small-en-v1.5 | agglomerative | k=true | 0.674 | 0.717 | 0.711 | 0.724 | 0.059 | 6.0 | 0.0% | 0.01 | 1 |
| BAAI/bge-small-en-v1.5 | agglomerative | auto | 0.137 | 0.353 | 0.225 | 0.818 | 0.073 | 2.0 | 0.0% | 0.05 | 1 |
| BAAI/bge-small-en-v1.5 | hdbscan | n/a | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.03 | 1 |
| BAAI/bge-small-en-v1.5 | hdbscan | mcs=5 | 0.031 | 0.171 | 0.126 | 0.267 | -0.024 | 3.0 | 71.7% | 0.03 | 1 |
| BAAI/bge-small-en-v1.5 | hdbscan | mcs=10 | 0.000 | 0.000 | 0.000 | 1.000 | - | 0.0 | 100.0% | 0.03 | 1 |
| BAAI/bge-small-en-v1.5 | bertopic | n/a | 0.214 ± 0.207 | 0.377 ± 0.208 | 0.307 ± 0.218 | 0.602 ± 0.083 | 0.008 ± 0.032 | 2.8 ± 1.0 | 5.1% | 0.35 | 5 |

#### Part B (hand-labelled)

not run (awaiting hand labels)

### 5.2 Summarization and grounding

40 papers from the benchmark. ROUGE against the abstract is a weak proxy for summary quality. NLI uses a small general-purpose model and is itself imperfect.

| Arm | Papers | Failed | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency (s) |
|---|---|---|---|---|---|---|---|
| BART (hf) | 40 | 0 | 0.395 | 0.369 | 0.381 | 0.26 | 6.4 |
| LLM | 40 | 1 | 0.493 | 0.382 | 0.423 | 0.42 | 124.2 |

#### LLM grounding (before unverified claims are dropped)

| Metric | Value |
|---|---|
| Model | `mistral:latest` |
| Claims produced | 217 |
| Quote found in abstract (verified) | 192 (88.5%) |
| Quote not found (dropped) | 25 (11.5%) |
| Papers with at least one dropped claim | 22 of 39 |
| Papers with no valid JSON after retries | 1 |
| `problem` stated as 'not stated' | 0.0% |
| `method` stated as 'not stated' | 5.1% |
| `results` stated as 'not stated' | 20.5% |
| `limitations` stated as 'not stated' | 100.0% |
| Tokens (prompt / completion) | 24019 / 19871 |
| Cost | not computed (no price table) |

#### NLI faithfulness (optional)

Model `cross-encoder/nli-deberta-v3-small`; premise = abstract.

| Arm | Unit | N | Entailed | Neutral | Contradicted |
|---|---|---|---|---|---|
| hf | sentences | 97 | 8.2% | 88.7% | 3.1% |
| llm | claims | 192 | 21.4% | 77.1% | 1.6% |

### 5.3 Human evaluation

not run (awaiting ratings; see docs/TODO_USER.md)

<!-- TODO(rusheil): after collecting your ratings (docs/TODO_USER.md, item 2) and re-running `python -m eval.report` and `python -m eval.write_report`, write 2-3 sentences interpreting them. Do not draw conclusions before then. -->

## 6. Discussion of ablations

The comparisons below are limited to what the tables above show.

- *Embedding model.* With the true k, MiniLM and mpnet gave the highest agglomerative ARI, then bge-small, then SPECTER. On this proxy benchmark the scientific-text model was not better than the general ones.
- *Method and k.* Agglomerative with the true k beat KMeans with the true k for every embedding. Choosing k by silhouette cost accuracy for every embedding and collapsed to two clusters for bge-small.
- *Density-based methods.* HDBSCAN with the application's rule labelled every paper as noise; fixed sizes helped little. BERTopic varied strongly with the embedding and the seed.
- *Summarizer.* The LLM's ROUGE-1 against the abstract was higher than BART's, but BART's ROUGE-2 is close to its ROUGE-1, which suggests near-copying, so ROUGE alone does not rank the two.

<!-- TODO(rusheil): add your interpretation of why these differences occur. The data above does not establish causes (e.g. whether SPECTER underperforms because of the proxy labels or the model), so none is asserted here. -->

## 7. Failure analysis

**Clustering.**

Configuration: `all-MiniLM-L6-v2 / agglomerative / k=true` - 26 of 180 papers sit in a cluster whose majority category is not their own.

Most confused category pairs (true -> category of the cluster it landed in):

- cs.IR -> cs.CL: 10
- cs.IR -> cs.DB: 4
- cs.CV -> cs.CL: 2
- cs.DB -> cs.CL: 2
- cs.RO -> cs.CV: 2
- cs.CL -> cs.CR: 1
- cs.CL -> cs.DB: 1
- cs.CL -> cs.IR: 1
- cs.CV -> cs.RO: 1
- cs.IR -> cs.CV: 1

Misplaced papers farthest from their own category's centroid:

- `2301.00844` (cs.IR -> cs.DB, distance 1.10): Understanding the main failure scenarios of subsea blowout preventers systems: An approach through Latent Semantic Analysis
- `2301.02817` (cs.IR -> cs.DB, distance 1.09): Cost-optimal Seeding Strategy During a Botanical Pandemic in Domesticated Fields
- `2301.00379` (cs.RO -> cs.DB, distance 1.02): A review of Implementation and Challenges of Unmanned Aerial Vehicles for Spraying Applications and Crop Monitoring in Indonesia
- `2301.03656` (cs.DB -> cs.CL, distance 0.97): Towards Multifaceted Human-Centered AI
- `2301.03774` (cs.IR -> cs.DB, distance 0.94): How Data Scientists Review the Scholarly Literature
- `2301.04780` (cs.IR -> cs.CL, distance 0.90): Much Ado About Gender: Current Practices and Future Recommendations for Appropriate Gender-Aware Information Access
- `2301.00618` (cs.CV -> cs.RO, distance 0.89): An Event-based Algorithm for Simultaneous 6-DOF Camera Pose Tracking and Mapping
- `2301.07183` (cs.IR -> cs.CL, distance 0.85): Tracking Brand-Associated Polarity-Bearing Topics in User Reviews
- `2301.00866` (cs.RO -> cs.CV, distance 0.85): 3DSGrasp: 3D Shape-Completion for Robotic Grasp
- `2301.00503` (cs.CL -> cs.IR, distance 0.85): A Concept Knowledge Graph for User Next Intent Prediction at Alipay

**Grounding.**

Papers with the smallest share of claims whose quote was found in the abstract:

- `2301.00330`: 2 verified, 2 dropped (50% verified): Efficient On-device Training via Gradient Filtering
- `2301.00399`: 2 verified, 1 dropped (67% verified): Semantic Operator Prediction and Applications
- `2301.01227`: 2 verified, 1 dropped (67% verified): Semantic Units: Organizing knowledge graphs into semantically meaningful units of representation
- `2301.01228`: 2 verified, 1 dropped (67% verified): DMOps: Data Management Operation and Recipes
- `2301.00379`: 5 verified, 2 dropped (71% verified): A review of Implementation and Challenges of Unmanned Aerial Vehicles for Spraying Applications and Crop Monitoring in Indonesia
- `2301.00823`: 6 verified, 2 dropped (75% verified): Bringing data minimization to digital wallets at scale with general-purpose zero-knowledge proofs
- `2301.00320`: 3 verified, 1 dropped (75% verified): Relevance Classification of Flood-related Twitter Posts via Multiple Transformers
- `2301.00366`: 3 verified, 1 dropped (75% verified): SS-CPGAN: Self-Supervised Cut-and-Pasting Generative Adversarial Network for Object Segmentation
- `2301.00374`: 3 verified, 1 dropped (75% verified): Optimizing Readability Using Genetic Algorithms
- `2301.00521`: 3 verified, 1 dropped (75% verified): A Policy Optimization Method Towards Optimal-time Stability

Papers with no usable output (invalid JSON after retries, or LLM error):

- `2301.00397`: Inflected Forms Are Redundant in Question Generation Models (no valid JSON after 3 attempts: Invalid \escape: line 5 column 256 (char 443))

<!-- TODO(rusheil): read a few of the listed papers and add a short qualitative note on what kind of mistakes these are (for example, whether the misplaced cs.IR papers are about applications of retrieval to other domains). -->

## 8. Limitations

- The clustering benchmark uses arXiv categories as stand-ins for topics and covers 25 days of one month. The hand-labelled benchmark (Part B) has not been run.
- The grounding result is for one 7B local model on 40 papers, at temperature 0, with one prompt. It says nothing about other models or prompts.
- A verified quote shows that the claim has textual support in the abstract, not that the claim is a faithful reading of it. The NLI check was inconclusive. No human ratings have been collected.
- LLM latency in the evaluation was measured on a laptop with a 4 GB GPU and is not representative.
- Full texts are never read.

## 9. Future work

Collect the human ratings and the hand-labelled clusters; evaluate more models and prompts for grounding, including a more forgiving JSON parser (one of 40 replies failed on an invalid escape); tune HDBSCAN's `min_cluster_size` on a benchmark other than the one used to evaluate it; extend grounding to full text; and deploy the hosted demo.

## Not run in this evaluation

- Clustering Part B: not run (awaiting hand labels)
- Human evaluation: not run (awaiting ratings; see docs/TODO_USER.md)
