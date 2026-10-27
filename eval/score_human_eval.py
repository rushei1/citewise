"""Aggregate the human ratings by system.

Refuses to run if any rating is missing or outside 1-5. Reports, per system, the mean (and std) of
accuracy and usefulness; the share of items where the LLM arm was rated higher, lower or equal
(per criterion); and a paired Wilcoxon signed-rank test on the per-item differences when there are
enough non-zero differences for it to mean anything. With ~20 items, treat all of this as
indicative, not conclusive.
"""

import csv
import json
from pathlib import Path

import numpy as np

from eval.common import RESULTS_DIR, save_json
from eval.make_human_eval import HUMAN_DIR

CRITERIA = ("accuracy", "usefulness")
MIN_NONZERO_FOR_TEST = 6  # fewer non-zero pairs than this and Wilcoxon cannot reach p < 0.05


class MissingRatings(SystemExit):
    pass


def _rating(value: str, where: str) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise MissingRatings(f"missing or non-integer rating at {where}: {value!r}") from None
    if not 1 <= v <= 5:
        raise MissingRatings(f"rating out of range 1-5 at {where}: {v}")
    return v


def score(rows: list[dict], key: dict) -> dict:
    per_item = []
    for r in rows:
        item = r["item_id"]
        if item not in key:
            raise MissingRatings(f"item {item} is not in key.json")
        scores = {}
        for side in ("A", "B"):
            system = key[item][side]
            scores[system] = {c: _rating(r[f"{c}_{side}"], f"{item}/{c}_{side}") for c in CRITERIA}
        per_item.append(scores)
    if not per_item:
        raise MissingRatings("no rows in ratings.csv")

    systems = sorted(per_item[0])
    out: dict = {"n_items": len(per_item), "systems": {}, "llm_vs_hf": {}}
    for s in systems:
        out["systems"][s] = {
            c: {
                "mean": float(np.mean([it[s][c] for it in per_item])),
                "std": float(np.std([it[s][c] for it in per_item])),
            }
            for c in CRITERIA
        }
    if set(systems) == {"hf", "llm"}:
        from scipy.stats import wilcoxon

        for c in CRITERIA:
            diff = np.array([it["llm"][c] - it["hf"][c] for it in per_item])
            n_nonzero = int(np.count_nonzero(diff))
            entry = {
                "llm_higher": int((diff > 0).sum()),
                "llm_lower": int((diff < 0).sum()),
                "tied": int((diff == 0).sum()),
                "mean_difference": float(diff.mean()),
            }
            if n_nonzero >= MIN_NONZERO_FOR_TEST:
                res = wilcoxon(diff[diff != 0])
                entry["wilcoxon_p"] = float(res.pvalue)
            else:
                entry["wilcoxon_p"] = None
                entry["wilcoxon_note"] = f"not run: only {n_nonzero} non-zero differences"
            out["llm_vs_hf"][c] = entry
    return out


def main() -> None:
    with (HUMAN_DIR / "ratings.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = json.loads((HUMAN_DIR / "key.json").read_text(encoding="utf-8"))
    result = score(rows, key)
    save_json(Path(RESULTS_DIR / "human_eval.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
