"""Turn eval/results/*.json into eval/results.md (and optionally a compact table in the README).

Nothing here is computed from anywhere but the result files, and anything missing or skipped is
listed under "Not run" instead of being left out silently.
"""

import argparse
import re
from pathlib import Path

from eval.common import RESULTS_DIR, ROOT, load_json

README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
OUT = ROOT / "eval" / "results.md"


def pm(stat: dict | None, digits: int = 3) -> str:
    """'mean' or 'mean ± std' (std shown only when it is not zero)."""
    if not stat or stat.get("mean") is None:
        return "-"
    mean, std = stat["mean"], stat.get("std") or 0.0
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std else "")


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def clustering_rows(results: list[dict]) -> list[list[str]]:
    return [
        [
            r["embedding"],
            r["method"],
            r["k_mode"],
            pm(r["ari"]),
            pm(r["nmi"]),
            pm(r["homogeneity"]),
            pm(r["completeness"]),
            pm(r["silhouette"]),
            pm(r["n_clusters"], 1),
            pct(r["noise_fraction"]["mean"]),
            f"{r['runtime_s']['mean']:.2f}",
            str(r["n_runs"]),
        ]
        for r in results
    ]


CL_HEADERS = [
    "Embedding",
    "Method",
    "k",
    "ARI",
    "NMI",
    "Homog.",
    "Compl.",
    "Silhouette",
    "Clusters",
    "Noise",
    "Time (s)",
    "Runs",
]


def best_per_method(results: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for r in results:
        key = (r["method"], r["k_mode"])
        if key not in best or r["ari"]["mean"] > best[key]["ari"]["mean"]:
            best[key] = r
    return sorted(best.values(), key=lambda r: -r["ari"]["mean"])


def env_lines(meta: dict) -> list[str]:
    pk = ", ".join(f"{k} {v}" for k, v in meta["packages"].items() if v)
    return [
        f"- Date: {meta['date']}",
        f"- Python {meta['python']} on {meta['platform']}",
        f"- Packages: {pk}",
        f"- Seeds: {meta['seeds']}",
    ]


def missing(name: str) -> str:
    return f"not run ({name} not found; run `make eval`)"


def build_report(results_dir: Path = RESULTS_DIR, quick: bool = False) -> tuple[str, str]:
    """Returns (full markdown, compact README table)."""
    not_run: list[str] = []
    out = ["# Evaluation results", ""]
    compact: list[str] = []

    cl_path = results_dir / "clustering.json"
    sm_path = results_dir / "summarization.json"
    cl = load_json(cl_path) if cl_path.exists() else None
    sm = load_json(sm_path) if sm_path.exists() else None

    if quick or any(d and d["meta"].get("quick") for d in (cl, sm)):
        out += [
            "> **SMOKE TEST ONLY.** Quick mode uses model-free stand-ins on a tiny subset. "
            "These numbers say nothing about the real models.",
            "",
        ]

    out += ["## Environment", ""]
    meta = (cl or sm or {}).get("meta")
    out += env_lines(meta) if meta else ["- not available (no result files)"]

    # ---- clustering ----
    out += ["", "## Clustering (Part A: proxy benchmark)", ""]
    if cl is None:
        not_run.append(f"Clustering: {missing('clustering.json')}")
        out.append("not run.")
    else:
        a = cl["part_a"]
        out += [
            f"{a['n_papers']} arXiv papers, {a['n_classes']} classes (primary category). The "
            "labels are a proxy for topics, so a low score can mean the categories overlap, not "
            "only that clustering failed. Randomised methods: mean ± std (population std) over "
            "seeds; deterministic methods ran once. `k=true` gives the method the real number of "
            "classes; `auto` chooses k by silhouette. HDBSCAN noise counts as an extra cluster "
            "for ARI/NMI but not in `Clusters`.",
            "",
            *table(CL_HEADERS, clustering_rows(a["results"])),
        ]
        for s in a["skipped"]:
            not_run.append(f"Clustering: {s['item']}: {s['reason']}")
        top = best_per_method(a["results"])
        compact += [
            f"| Clustering ({a['n_papers']} papers, proxy labels) | best `{r['method']}` "
            f"({r['k_mode']}), {r['embedding']} | ARI {pm(r['ari'], 2)}, NMI {pm(r['nmi'], 2)} |"
            for r in top[:3]
        ]
        out += ["", "### Part B (hand-labelled)", ""]
        b = cl["part_b"]
        if b["status"] != "run":
            out.append(b["status"])
            not_run.append(f"Clustering Part B: {b['status']}")
        else:
            for topic, t in b["topics"].items():
                out += [f"**{topic}** ({t['n_papers']} papers, {t['n_classes']} labels)", ""]
                out += table(CL_HEADERS, clustering_rows(t["results"])) + [""]

    # ---- summarization ----
    out += ["", "## Summarization", ""]
    if sm is None:
        not_run.append(f"Summarization: {missing('summarization.json')}")
        out.append("not run.")
    else:
        hf, llm = sm["arms"]["hf"], sm["arms"]["llm"]
        out += [
            f"{sm['sample']['n']} papers from the benchmark. {sm['meta']['caveat']}",
            "",
            *summarization_table(hf, llm),
        ]
        if hf.get("status", "run") == "run":
            compact.append(
                f"| Summarization ROUGE-1/2/L vs abstract ({sm['sample']['n']} papers, weak "
                f"proxy) | {hf['id']} | {num(hf.get('rouge1'))} / {num(hf.get('rouge2'))} / "
                f"{num(hf.get('rougeL'))} |"
            )
        else:
            not_run.append(f"Summarization BART arm: {hf['status']}")
        if llm.get("status") == "run":
            out += ["", "### LLM grounding (before unverified claims are dropped)", ""]
            out += table(
                ["Metric", "Value"],
                [
                    ["Model", f"`{sm['meta'].get('llm_model', llm['id'])}`"],
                    ["Claims produced", str(llm["claims_total"])],
                    [
                        "Quote found in abstract (verified)",
                        f"{llm['claims_verified']} ({pct(llm['verification_rate'])})",
                    ],
                    [
                        "Quote not found (dropped)",
                        f"{llm['claims_dropped']} ({pct(llm['dropped_rate'])})",
                    ],
                    [
                        "Papers with at least one dropped claim",
                        f"{llm['papers_with_any_dropped_claim']} of {llm['n'] - llm['n_failed']}",
                    ],
                    ["Papers with no valid JSON after retries", str(llm["n_failed"])],
                    *[
                        [f"`{f}` stated as 'not stated'", pct(v)]
                        for f, v in llm["not_stated_rate"].items()
                    ],
                    [
                        "Tokens (prompt / completion)",
                        f"{llm['prompt_tokens']} / {llm['completion_tokens']}",
                    ],
                    ["Cost", llm["cost"]],
                ],
            )
            compact.append(
                f"| LLM claims whose quote is in the abstract ({sm['sample']['n']} papers) | "
                f"{llm['id']} | {pct(llm['verification_rate'])} of {llm['claims_total']} claims |"
            )
        else:
            not_run.append(f"Summarization LLM arm: {llm['status']}")
            out += ["", f"LLM arm: {llm['status']}"]

        out += ["", "### NLI faithfulness (optional)", ""]
        nli = sm["nli"]
        if "status" in nli:
            out.append(nli["status"])
            not_run.append(f"NLI: {nli['status']}")
        else:
            rows = [
                [
                    arm,
                    d["unit"],
                    str(d["n"]),
                    pct(d["entailment_rate"]),
                    pct(d["neutral_rate"]),
                    pct(d["contradiction_rate"]),
                ]
                for arm, d in nli.items()
                if arm != "model" and d.get("status") == "run"
            ]
            out += [f"Model `{nli['model']}`; premise = abstract.", ""]
            out += table(["Arm", "Unit", "N", "Entailed", "Neutral", "Contradicted"], rows)

    # ---- human eval ----
    out += ["", "## Human evaluation", ""]
    he_path = results_dir / "human_eval.json"
    if he_path.exists():
        he = load_json(he_path)
        out += [f"{he['n_items']} items rated 1-5 (blinded A/B).", ""]
        out += table(
            ["System", "Accuracy", "Usefulness"],
            [[s, pm(v["accuracy"]), pm(v["usefulness"])] for s, v in he["systems"].items()],
        )
        for crit, d in he.get("llm_vs_hf", {}).items():
            p = f"p = {d['wilcoxon_p']:.3f}" if d["wilcoxon_p"] is not None else d["wilcoxon_note"]
            out.append(
                f"\n- {crit}: LLM higher on {d['llm_higher']}, lower on {d['llm_lower']}, "
                f"tied on {d['tied']} items; mean difference {d['mean_difference']:+.2f}; {p}"
            )
    else:
        msg = "not run (awaiting ratings; see docs/TODO_USER.md)"
        out.append(msg)
        not_run.append(f"Human evaluation: {msg}")

    # ---- failure cases ----
    out += ["", "## Failure analysis", ""]
    if (results_dir / "failure_cases.md").exists():
        out.append("See [results/failure_cases.md](results/failure_cases.md).")
    else:
        out.append("not run.")
        not_run.append("Failure analysis: failure_cases.md not found")

    out += ["", "## Not run", ""]
    out += [f"- {n}" for n in not_run] if not_run else ["Nothing was skipped."]
    compact_table = "\n".join(
        table(["Experiment", "Setting", "Result"], [])
        + compact
        + ["", "Full tables, seeds and environment: [eval/results.md](eval/results.md)."]
    )
    return "\n".join(out) + "\n", compact_table


def summarization_table(hf: dict, llm: dict) -> list[str]:
    rows = []
    for name, a in (("BART (hf)", hf), ("LLM", llm)):
        if a.get("status", "run") != "run":
            rows.append([name, a["status"], "", "", "", "", "", ""])
            continue
        rows.append(
            [
                name,
                str(a["n"]),
                str(a["n_failed"]),
                num(a.get("rouge1")),
                num(a.get("rouge2")),
                num(a.get("rougeL")),
                num(a.get("compression_ratio"), 2),
                num(a.get("latency_mean_s"), 1),
            ]
        )
    return table(
        ["Arm", "Papers", "Failed", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Compression", "Latency (s)"],
        rows,
    )


def inject(readme: str, compact_table: str) -> str:
    """Replace everything between the RESULTS markers; raises if they are missing."""
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise ValueError(f"README has no {START} ... {END} markers")
    return pattern.sub(lambda m: f"{START}\n{compact_table}\n{END}", readme)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="read results_quick/, write there")
    ap.add_argument("--readme", action="store_true", help="also inject the compact table")
    args = ap.parse_args()
    results_dir = RESULTS_DIR.with_name("results_quick") if args.quick else RESULTS_DIR
    full, compact = build_report(results_dir, quick=args.quick)
    out = results_dir.parent / ("results_quick.md" if args.quick else "results.md")
    out.write_text(full, encoding="utf-8")
    print(f"wrote {out}")
    if args.readme:
        README.write_text(inject(README.read_text(encoding="utf-8"), compact), encoding="utf-8")
        print(f"updated {README}")


if __name__ == "__main__":
    main()
