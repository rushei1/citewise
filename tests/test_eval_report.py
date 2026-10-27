import json

import pytest

from eval import report as rp


def stat(mean, std=0.0):
    return {"mean": mean, "std": std}


def cl_row(emb, method, k, ari, std=0.0, runs=5):
    return {
        "embedding": emb,
        "method": method,
        "k_mode": k,
        "n_runs": runs,
        "ari": stat(ari, std),
        "nmi": stat(0.6),
        "homogeneity": stat(0.7),
        "completeness": stat(0.5),
        "silhouette": stat(0.1),
        "n_clusters": stat(6.0),
        "noise_fraction": stat(0.0),
        "runtime_s": stat(0.25),
    }


META = {
    "date": "2026-10-27 10:00 UTC",
    "python": "3.11.9",
    "platform": "test-os",
    "packages": {"numpy": "2.0.2", "torch": None},
    "seeds": [0, 1],
}


def clustering(part_b=None, skipped=()):
    return {
        "meta": META,
        "part_a": {
            "n_papers": 180,
            "n_classes": 6,
            "results": [
                cl_row("m1", "kmeans", "k=true", 0.55, 0.012),
                cl_row("m1", "agglomerative", "k=true", 0.7, runs=1),
                cl_row("m2", "kmeans", "k=true", 0.6, 0.01),
            ],
            "skipped": list(skipped),
        },
        "part_b": part_b or {"status": "not run (awaiting hand labels)"},
    }


def summarization(llm=None, nli=None):
    arm = {
        "n": 40,
        "n_failed": 1,
        "rouge1": 0.4,
        "rouge2": 0.1,
        "rougeL": 0.3,
        "compression_ratio": 0.3,
        "latency_mean_s": 4.2,
    }
    return {
        "meta": {**META, "caveat": "ROUGE is a weak proxy.", "llm_model": "mistral"},
        "sample": {"n": 40, "ids": []},
        "arms": {
            "hf": {**arm, "id": "hf:bart"},
            "llm": llm or {"status": "not run (no credentials configured)"},
        },
        "nli": nli or {"status": "not run (NLI model unavailable: x)"},
    }


LLM_RUN = {
    "status": "run",
    "id": "llm:mistral:summ-v1",
    "n": 40,
    "n_failed": 3,
    "rouge1": 0.2,
    "rouge2": 0.05,
    "rougeL": 0.15,
    "compression_ratio": 0.5,
    "latency_mean_s": 30.0,
    "claims_total": 100,
    "claims_verified": 80,
    "claims_dropped": 20,
    "verification_rate": 0.8,
    "dropped_rate": 0.2,
    "papers_with_any_dropped_claim": 12,
    "not_stated_rate": {"problem": 0.0, "method": 0.1, "results": 0.2, "limitations": 0.9},
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "cost": "not computed (no price table)",
}


def write(tmp_path, **files):
    for name, data in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    return tmp_path


def test_formatters():
    assert rp.pm(stat(0.5556, 0.0123)) == "0.556 ± 0.012"
    assert rp.pm(stat(0.5)) == "0.500"
    assert rp.pm(None) == "-" and rp.pm(stat(None)) == "-"
    assert rp.pct(0.1234) == "12.3%" and rp.pct(None) == "-"


def test_full_report_has_every_section_and_real_numbers(tmp_path):
    res = write(
        tmp_path,
        clustering=clustering(skipped=[{"item": "bertopic", "reason": "not installed"}]),
        summarization=summarization(
            LLM_RUN,
            {
                "model": "nli-x",
                "hf": {
                    "status": "run",
                    "unit": "sentences",
                    "n": 50,
                    "entailment_rate": 0.6,
                    "neutral_rate": 0.3,
                    "contradiction_rate": 0.1,
                },
            },
        ),
    )
    full, compact = rp.build_report(res)
    for heading in (
        "## Environment",
        "## Clustering",
        "## Summarization",
        "## Human evaluation",
        "## Failure analysis",
        "## Not run",
    ):
        assert heading in full
    assert "| m1 | kmeans | k=true | 0.550 ± 0.012 |" in full
    assert "| m1 | agglomerative | k=true | 0.700 |" in full  # deterministic: no std
    assert "80 (80.0%)" in full and "20 (20.0%)" in full
    assert "`limitations` stated as 'not stated' | 90.0%" in full
    assert "60.0%" in full and "SMOKE TEST" not in full
    assert "- Clustering: bertopic: not installed" in full
    assert "- Clustering Part B: not run (awaiting hand labels)" in full
    assert "- Human evaluation: not run (awaiting ratings" in full
    assert "numpy 2.0.2" in full and "torch" not in full.split("Packages:")[1].split("\n")[0]
    assert "80.0% of 100 claims" in compact and "ARI 0.70" in compact


def test_missing_result_files_are_reported_not_fatal(tmp_path):
    full, compact = rp.build_report(tmp_path)
    assert "Clustering: not run (clustering.json not found" in full
    assert "Summarization: not run (summarization.json not found" in full
    assert "not available (no result files)" in full
    assert compact.startswith("| Experiment")


def test_llm_arm_not_run_and_nli_unavailable_are_listed(tmp_path):
    res = write(tmp_path, clustering=clustering(), summarization=summarization())
    full, _ = rp.build_report(res)
    assert "- Summarization LLM arm: not run (no credentials configured)" in full
    assert "- NLI: not run (NLI model unavailable: x)" in full
    assert "LLM claims whose quote" not in rp.build_report(res)[1]  # nothing invented


def test_part_b_and_human_eval_are_rendered_when_present(tmp_path):
    part_b = {
        "status": "run",
        "topics": {
            "topic one": {
                "n_papers": 30,
                "n_classes": 4,
                "results": [cl_row("m1", "kmeans", "k=true", 0.4)],
                "skipped": [],
            }
        },
    }
    he = {
        "n_items": 20,
        "systems": {
            "hf": {"accuracy": stat(3.0, 0.5), "usefulness": stat(3.1)},
            "llm": {"accuracy": stat(4.0), "usefulness": stat(3.9)},
        },
        "llm_vs_hf": {
            "accuracy": {
                "llm_higher": 12,
                "llm_lower": 3,
                "tied": 5,
                "mean_difference": 1.0,
                "wilcoxon_p": 0.012,
            },
            "usefulness": {
                "llm_higher": 8,
                "llm_lower": 6,
                "tied": 6,
                "mean_difference": 0.2,
                "wilcoxon_p": None,
                "wilcoxon_note": "not run: only 5 non-zero differences",
            },
        },
    }
    res = write(
        tmp_path, clustering=clustering(part_b), summarization=summarization(), human_eval=he
    )
    (tmp_path / "failure_cases.md").write_text("x")
    full, _ = rp.build_report(res)
    assert "**topic one** (30 papers, 4 labels)" in full
    assert "20 items rated" in full and "p = 0.012" in full
    assert "not run: only 5 non-zero differences" in full
    assert "results/failure_cases.md" in full
    assert "Human evaluation: not run" not in full


def test_quick_results_carry_a_smoke_test_banner(tmp_path):
    c = clustering()
    c["meta"] = {**META, "quick": True}
    full, _ = rp.build_report(write(tmp_path, clustering=c))
    assert "SMOKE TEST ONLY" in full


def test_inject_replaces_only_between_markers_and_is_idempotent():
    readme = f"before\n{rp.START}\nold table\n{rp.END}\nafter\n"
    once = rp.inject(readme, "NEW")
    assert once == f"before\n{rp.START}\nNEW\n{rp.END}\nafter\n"
    assert rp.inject(once, "NEW") == once
    with pytest.raises(ValueError, match="markers"):
        rp.inject("no markers here", "NEW")
