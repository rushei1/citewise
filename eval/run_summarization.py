"""Compare the BART summarizer (HFSummarizer) with the claim-based LLMSummarizer.

Metrics (all computed against the paper's own abstract; see the caveats printed in the output):
  * ROUGE-1/2/L F1 against the abstract. A WEAK PROXY: a good summary of an abstract is not
    supposed to copy it, and the LLM arm's claims are paraphrases plus quoted evidence.
  * Evidence verification rate (LLM arm): the share of the model's claims whose quote is found
    verbatim in the abstract, counted BEFORE the verifier drops the rest. This measures how
    honest the model's grounding is and is the headline metric.
  * Optional NLI faithfulness: does the abstract entail each claim / summary sentence, judged by
    a small cross-encoder. Skipped cleanly if the model cannot be loaded.
  * Compression ratio, latency per paper, token usage. Cost is NOT computed (no price table).
The LLM arm runs only if an LLM is configured (LLM_MODEL plus LLM_API_KEY or LLM_BASE_URL, or
LLM_PROVIDER=anthropic with ANTHROPIC_API_KEY); otherwise it is reported as not run.
"""

import argparse
import logging
import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.schemas import CLAIM_FIELDS, PaperIn, PaperSummary
from app.services.summarize import PROMPT_VERSION, SummarizeError
from eval.common import (
    BENCHMARK,
    CACHE_DIR,
    RESULTS_DIR,
    cached,
    environment,
    key_hash,
    load_json,
    save_json,
    set_seeds,
)
from eval.metrics import compression_ratio, rouge_all

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 40
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
CAVEAT = (
    "ROUGE against the abstract is a weak proxy for summary quality. NLI uses a small "
    "general-purpose model and is itself imperfect."
)


def sample_papers(papers: list[dict], limit: int) -> list[dict]:
    """A fixed sample: round-robin over categories (sorted by id inside each), so every
    category is represented and the choice does not depend on randomness."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(papers, key=lambda p: (p["category"], p["id"])):
        by_cat[p["category"]].append(p)
    out, queues = [], [by_cat[c] for c in sorted(by_cat)]
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


def as_paper_in(p: dict) -> PaperIn:
    return PaperIn(
        title=p["title"], abstract=p["abstract"], link=f"https://arxiv.org/abs/{p['id']}"
    )


class FirstSentence:
    """Quick-mode stand-in so the metric code runs without downloading BART."""

    id = "stub:first-sentence"

    def summarize(self, paper: PaperIn) -> PaperSummary:
        return PaperSummary(overview=re.split(r"(?<=[.!?])\s", paper.abstract.strip())[0])


def run_hf_arm(papers: list[dict], summarizer) -> list[dict]:
    rows = []
    for p in papers:
        t0 = time.perf_counter()
        try:
            s = summarizer.summarize(as_paper_in(p))
            err = None
        except Exception as e:  # noqa: BLE001 - one bad paper must not stop the arm
            s, err = None, str(e)
        rows.append(
            {
                "id": p["id"],
                "text": s.text() if s else "",
                "latency_s": time.perf_counter() - t0,
                "error": err,
            }
        )
    return rows


def run_llm_arm(papers: list[dict], summarizer, client, cache_dir: Path) -> list[dict]:
    """One uncached-or-cached call per paper. Cached results keep their original latency and
    token counts, so a rerun reports the same numbers. Failures are never cached."""
    rows = []
    for p in papers:

        def call(p=p):
            before = (client.usage.prompt_tokens, client.usage.completion_tokens)
            t0 = time.perf_counter()
            try:
                s = summarizer.summarize(as_paper_in(p))
                summary, err = s.model_dump(), None
            except SummarizeError as e:
                summary, err = None, str(e)
            return {
                "latency_s": time.perf_counter() - t0,
                "prompt_tokens": client.usage.prompt_tokens - before[0],
                "completion_tokens": client.usage.completion_tokens - before[1],
                "summary": summary,
                "error": err,
            }

        key = {
            "model": client.model,
            "prompt": PROMPT_VERSION,
            "id": p["id"],
            "a": key_hash(p["abstract"]),
        }
        r = cached("llm_summary", key, call, cache_dir, store_if=lambda v: v["error"] is None)
        s = PaperSummary(**r["summary"]) if r["summary"] else None
        rows.append(
            {
                "id": p["id"],
                "text": s.text() if s else "",
                "structured": r["summary"],
                "latency_s": r["latency_s"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "error": r["error"],
            }
        )
    return rows


def text_metrics(rows: list[dict], abstracts: dict[str, str]) -> dict:
    ok = [r for r in rows if not r["error"] and r["text"]]
    out = {"n": len(rows), "n_failed": len(rows) - len(ok)}
    if not ok:
        return out
    scores = [rouge_all(abstracts[r["id"]], r["text"]) for r in ok]
    for k in ("rouge1", "rouge2", "rougeL"):
        out[k] = float(np.mean([s[k] for s in scores]))
    out["compression_ratio"] = float(
        np.mean([compression_ratio(abstracts[r["id"]], r["text"]) for r in ok])
    )
    lat = [r["latency_s"] for r in ok]
    out["latency_mean_s"], out["latency_median_s"] = float(np.mean(lat)), float(np.median(lat))
    return out


def grounding_metrics(rows: list[dict]) -> dict:
    """Claim-level honesty of the LLM arm. kept + dropped = what the model produced."""
    kept = dropped = papers_with_drop = 0
    not_stated: dict[str, int] = {f: 0 for f in CLAIM_FIELDS}
    ok = [r for r in rows if r["structured"]]
    for r in ok:
        s = r["structured"]
        k = sum(len(s[f]) for f in CLAIM_FIELDS)
        kept += k
        dropped += s["unverified_claims"]
        papers_with_drop += bool(s["unverified_claims"])
        for f in s["not_stated"]:
            not_stated[f] += 1
    total = kept + dropped
    return {
        "claims_total": total,
        "claims_verified": kept,
        "claims_dropped": dropped,
        "verification_rate": kept / total if total else None,
        "dropped_rate": dropped / total if total else None,
        "papers_with_any_dropped_claim": papers_with_drop,
        "not_stated_rate": {f: n / len(ok) if ok else None for f, n in not_stated.items()},
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "cost": "not computed (no price table)",
        "latency_note": "wall time of the original call, including retries, even when this run "
        "read it from the cache",
    }


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", text.strip()) if len(s.split()) >= 3]


def load_nli() -> Callable[[list[tuple[str, str]]], list[str]] | str:
    """(premise, hypothesis) pairs -> labels ('entailment' / 'neutral' / 'contradiction'),
    or a reason string if the model is unavailable."""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(NLI_MODEL)
        id2label = {int(i): lab.lower() for i, lab in model.model.config.id2label.items()}
    except Exception as e:  # noqa: BLE001
        return f"not run (NLI model unavailable: {e})"

    def score(pairs):
        logits = np.asarray(model.predict(pairs, show_progress_bar=False))
        return [id2label[int(i)] for i in logits.argmax(axis=1)]

    return score


def nli_metrics(rows, abstracts, nli, kind: str) -> dict:
    """Share of claims (LLM arm) or summary sentences (HF arm) that the abstract entails."""
    pairs = []
    for r in rows:
        if r["error"]:
            continue
        if kind == "claims":
            s = r["structured"] or {}
            hyps = [c["text"] for f in CLAIM_FIELDS for c in s.get(f, [])]
        else:
            hyps = sentences(r["text"])
        pairs += [(abstracts[r["id"]], h) for h in hyps]
    if not pairs:
        return {"status": "not run (nothing to check)"}
    labels = nli(pairs)
    n = len(labels)
    return {
        "status": "run",
        "unit": kind,
        "n": n,
        "entailment_rate": labels.count("entailment") / n,
        "contradiction_rate": labels.count("contradiction") / n,
        "neutral_rate": labels.count("neutral") / n,
    }


ALL_ARMS = ("hf", "llm", "nli")


def run(
    out: Path = RESULTS_DIR / "summarization.json",
    limit: int = DEFAULT_LIMIT,
    quick: bool = False,
    benchmark: Path = BENCHMARK,
    hf_summarizer=None,
    llm_client="auto",
    nli="auto",
    cache_dir: Path = CACHE_DIR,
    arms: tuple[str, ...] = ALL_ARMS,
) -> dict:
    """Run the chosen arms and merge them into the existing result files.

    Arms can run as separate processes (`--arm hf`, then `--arm llm`, then `--arm nli`) to keep
    memory use down: each merges into summarization.json / summaries.json. NLI reads the saved
    summaries, so it must come after the arms it should check."""
    set_seeds()
    papers = sample_papers(load_json(benchmark)["part_a"], limit)
    abstracts = {p["id"]: p["abstract"] for p in papers}
    ids = [p["id"] for p in papers]
    sums_path = out.with_name("summaries.json")

    prev = load_json(out) if out.exists() else None
    if prev and prev["sample"]["ids"] != ids:
        if set(arms) != set(ALL_ARMS):
            raise ValueError("the sample differs from the saved results; rerun all arms")
        prev = None  # a full rerun with a different sample replaces everything
    summaries = load_json(sums_path) if prev and sums_path.exists() else {}
    result = prev or {
        "meta": environment(
            quick=quick,
            sample_size=len(papers),
            caveat=CAVEAT,
            decoding="HF: greedy (do_sample=False). LLM: temperature 0, so runs are near- but not "
            "guaranteed bit-identical; the on-disk cache is what makes reruns identical.",
        ),
        "sample": {"n": len(papers), "ids": ids},
        "arms": {"hf": {"status": "not run"}, "llm": {"status": "not run"}},
        "nli": {"status": "not run"},
    }
    meta = result["meta"]
    meta["quick"] = quick
    meta.setdefault("arm_dates", {})
    now = environment()["date"]
    if quick:
        meta["note"] = "Quick mode: a first-sentence stub replaces BART. Smoke test only."

    if "hf" in arms:
        if quick:
            hf_summarizer = FirstSentence()
        elif hf_summarizer is None:
            from app.services.summarize import HFSummarizer

            hf_summarizer = HFSummarizer()
        rows = run_hf_arm(papers, hf_summarizer)
        summaries["hf"] = rows
        result["arms"]["hf"] = {**text_metrics(rows, abstracts), "id": hf_summarizer.id}
        meta["hf_arm"] = hf_summarizer.id
        meta["arm_dates"]["hf"] = now

    if "llm" in arms:
        if llm_client == "auto":
            from app.config import Settings
            from app.services.llm import make_llm_client

            llm_client = None if quick else make_llm_client(Settings())
        if llm_client is None:
            result["arms"]["llm"] = {
                "status": "not run (quick mode)" if quick else "not run (no credentials configured)"
            }
            summaries.pop("llm", None)
        else:
            from app.services.summarize import LLMSummarizer

            summarizer = LLMSummarizer(llm_client, workers=1)
            rows = run_llm_arm(papers, summarizer, llm_client, cache_dir)
            summaries["llm"] = rows
            result["arms"]["llm"] = {
                "status": "run",
                "id": summarizer.id,
                **text_metrics(rows, abstracts),
                **grounding_metrics(rows),
            }
            meta["llm_model"] = llm_client.model
            meta["arm_dates"]["llm"] = now

    if "nli" in arms:
        if quick:
            nli = "not run (quick mode)"
        elif nli == "auto":
            nli = load_nli()
        if isinstance(nli, str):
            result["nli"] = {"status": nli}
        elif "hf" not in summaries:
            result["nli"] = {"status": "not run (no summaries to check; run the hf arm first)"}
        else:
            result["nli"] = {"hf": nli_metrics(summaries["hf"], abstracts, nli, "sentences")}
            if "llm" in summaries:
                result["nli"]["llm"] = nli_metrics(summaries["llm"], abstracts, nli, "claims")
            result["nli"]["model"] = NLI_MODEL
            meta["arm_dates"]["nli"] = now

    save_json(out, result)
    save_json(sums_path, summaries)
    log.info("wrote %s (arms: %s)", out, ", ".join(arms))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--quick", action="store_true", help="tiny stub-based smoke test")
    ap.add_argument(
        "--arm",
        action="append",
        choices=ALL_ARMS,
        help="run only this arm (repeatable); results merge into the existing files. "
        "Default: all, in one process",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (RESULTS_DIR.with_name("results_quick") if args.quick else RESULTS_DIR)
    limit = min(args.limit, 6) if args.quick else args.limit
    arms = tuple(args.arm) if args.arm else ALL_ARMS
    run(out=out / "summarization.json", limit=limit, quick=args.quick, arms=arms)


if __name__ == "__main__":
    main()
