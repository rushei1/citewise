"""List concrete failures so the write-up can discuss them instead of only averages.

Clustering: for the best configuration (highest mean ARI), the papers whose cluster does not match
their true category, ranked by distance from their true category's centroid, plus the most
confused category pairs. Summaries: the LLM-arm papers with the lowest share of claims whose quote
was found in the abstract, and the papers that failed outright.
Writes eval/results/failure_cases.md.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.services.cluster import _unit
from eval.common import BENCHMARK, CACHE_DIR, RESULTS_DIR, load_json
from eval.run_clustering import embed, sbert_encoder

TOP = 10


def best_config(clustering: dict) -> dict | None:
    rows = clustering["part_a"]["results"]
    return max(rows, key=lambda r: r["ari"]["mean"]) if rows else None


def majority_class(truth: list[str], labels: list[int]) -> dict[int, str]:
    """Each predicted cluster -> the true class most of its members have (ties: alphabetical)."""
    members: dict[int, Counter] = defaultdict(Counter)
    for t, label in zip(truth, labels, strict=True):
        members[label][t] += 1
    return {
        label: min(c, key=lambda cls: (-c[cls], cls))  # most common, then alphabetical
        for label, c in members.items()
    }


def analyze_clustering(ids, titles, truth, labels, vectors, top: int = TOP) -> dict:
    assigned = majority_class(truth, labels)
    x = np.asarray(vectors, dtype=float)
    centroids = {c: x[[i for i, t in enumerate(truth) if t == c]].mean(axis=0) for c in set(truth)}
    mistakes = []
    for i, (t, label) in enumerate(zip(truth, labels, strict=True)):
        # noise (-1) is never a correct assignment
        if label == -1 or assigned[label] != t:
            mistakes.append(
                {
                    "id": ids[i],
                    "title": titles[i],
                    "true": t,
                    "assigned": "(noise)" if label == -1 else assigned[label],
                    "distance_to_true_centroid": float(np.linalg.norm(x[i] - centroids[t])),
                }
            )
    mistakes.sort(key=lambda m: (-m["distance_to_true_centroid"], m["id"]))
    pairs = Counter((m["true"], m["assigned"]) for m in mistakes)
    return {
        "n_papers": len(truth),
        "n_mistakes": len(mistakes),
        "farthest": mistakes[:top],
        "confused_pairs": [
            {"true": a, "assigned": b, "count": n}
            for (a, b), n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        ][:top],
    }


def lowest_grounding(llm_rows: list[dict], titles: dict[str, str], top: int = TOP) -> dict:
    scored, failed = [], []
    for r in llm_rows:
        if r["error"] or not r["structured"]:
            failed.append({"id": r["id"], "title": titles[r["id"]], "error": r["error"]})
            continue
        s = r["structured"]
        kept = sum(len(s[f]) for f in ("problem", "method", "results", "limitations"))
        total = kept + s["unverified_claims"]
        if total:
            scored.append(
                {
                    "id": r["id"],
                    "title": titles[r["id"]],
                    "verified": kept,
                    "dropped": s["unverified_claims"],
                    "rate": kept / total,
                }
            )
    scored.sort(key=lambda d: (d["rate"], -d["dropped"], d["id"]))
    return {"worst": scored[:top], "failed": failed}


def render(clustering_part: dict | str, grounding_part: dict | str) -> str:
    out = ["# Failure cases", ""]
    out.append("## Clustering (best configuration by mean ARI)")
    if isinstance(clustering_part, str):
        out += ["", clustering_part, ""]
    else:
        c = clustering_part
        out += [
            "",
            f"Configuration: `{c['config']}` - {c['n_mistakes']} of {c['n_papers']} papers sit in "
            "a cluster whose majority category is not their own.",
            "",
            "Most confused category pairs (true -> category of the cluster it landed in):",
            "",
        ]
        out += [f"- {p['true']} -> {p['assigned']}: {p['count']}" for p in c["confused_pairs"]]
        out += ["", "Misplaced papers farthest from their own category's centroid:", ""]
        out += [
            f"- `{m['id']}` ({m['true']} -> {m['assigned']}, distance "
            f"{m['distance_to_true_centroid']:.2f}): {m['title']}"
            for m in c["farthest"]
        ]
        out.append("")
    out.append("## Lowest-grounding LLM summaries")
    if isinstance(grounding_part, str):
        out += ["", grounding_part, ""]
    else:
        out += [
            "",
            "Papers with the smallest share of claims whose quote was found in the abstract:",
            "",
        ]
        out += [
            f"- `{w['id']}`: {w['verified']} verified, {w['dropped']} dropped "
            f"({w['rate']:.0%} verified): {w['title']}"
            for w in grounding_part["worst"]
        ]
        if grounding_part["failed"]:
            out += [
                "",
                "Papers with no usable output (invalid JSON after retries, or LLM error):",
                "",
            ]
            out += [f"- `{f['id']}`: {f['title']} ({f['error']})" for f in grounding_part["failed"]]
        out.append("")
    return "\n".join(out)


def build(
    results_dir: Path = RESULTS_DIR,
    benchmark: Path = BENCHMARK,
    cache_dir: Path = CACHE_DIR,
    encoder_for=sbert_encoder,
) -> str:
    papers = {p["id"]: p for p in load_json(benchmark)["part_a"]}
    titles = {i: p["title"] for i, p in papers.items()}

    clustering_part: dict | str
    cl_path, pred_path = (
        results_dir / "clustering.json",
        results_dir / "clustering_predictions.json",
    )
    if not (cl_path.exists() and pred_path.exists()):
        clustering_part = "not run (no clustering results)"
    else:
        best = best_config(load_json(cl_path))
        preds = load_json(pred_path)
        if best is None:
            clustering_part = "not run (no configuration produced results)"
        else:
            name = f"{best['embedding']}|{best['method']}|{best['k_mode']}"
            ids = preds["papers"]
            ordered = [papers[i] for i in ids]
            x = _unit(
                embed(best["embedding"], ordered, _lazy(encoder_for, best["embedding"]), cache_dir)
            )
            clustering_part = analyze_clustering(
                ids, [titles[i] for i in ids], preds["truth"], preds["predictions"][name], x
            )
            clustering_part["config"] = name.replace("|", " / ")

    s_path = results_dir / "summaries.json"
    summaries = load_json(s_path) if s_path.exists() else {}
    grounding_part = (
        lowest_grounding(summaries["llm"], titles)
        if "llm" in summaries
        else "not run (the LLM arm has not been run)"
    )
    return render(clustering_part, grounding_part)


def _lazy(encoder_for, model_name):
    """Load the model only if the embeddings are not already cached."""
    state = {}

    def encode(texts):
        state.setdefault("e", encoder_for(model_name))
        return state["e"](texts)

    return encode


def main() -> None:
    text = build()
    (RESULTS_DIR / "failure_cases.md").write_text(text, encoding="utf-8")
    print(f"wrote {RESULTS_DIR / 'failure_cases.md'}")


if __name__ == "__main__":
    main()
