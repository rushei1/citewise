# Evaluation results

## Environment

- Date: 2026-10-27 07:55 UTC
- Python 3.11.9 on Windows-10-10.0.26300-SP0
- Packages: numpy 2.0.2, scikit-learn 1.6.1, scipy 1.17.1, sentence-transformers 5.0.0, transformers 4.53.2, torch 2.7.1+cpu, arxiv 2.2.0
- Seeds: [0, 1, 2, 3, 4]

## Clustering (Part A: proxy benchmark)

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

### Part B (hand-labelled)

not run (awaiting hand labels)

## Summarization

40 papers from the benchmark. ROUGE against the abstract is a weak proxy for summary quality. NLI uses a small general-purpose model and is itself imperfect.

| Arm | Papers | Failed | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency (s) |
|---|---|---|---|---|---|---|---|
| BART (hf) | 40 | 0 | 0.395 | 0.369 | 0.381 | 0.26 | 6.4 |
| LLM | 40 | 1 | 0.493 | 0.382 | 0.423 | 0.42 | 124.2 |

### LLM grounding (before unverified claims are dropped)

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

### NLI faithfulness (optional)

Model `cross-encoder/nli-deberta-v3-small`; premise = abstract.

| Arm | Unit | N | Entailed | Neutral | Contradicted |
|---|---|---|---|---|---|
| hf | sentences | 97 | 8.2% | 88.7% | 3.1% |
| llm | claims | 192 | 21.4% | 77.1% | 1.6% |

## Human evaluation

not run (awaiting ratings; see docs/TODO_USER.md)

## Failure analysis

See [results/failure_cases.md](results/failure_cases.md).

## Not run

- Clustering Part B: not run (awaiting hand labels)
- Human evaluation: not run (awaiting ratings; see docs/TODO_USER.md)
