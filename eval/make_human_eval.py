"""Build the blinded human-evaluation sheet (needs ratings from a person).

Samples papers where both arms produced a summary, shuffles which system is A and which is B per
item, and writes
  eval/human/ratings.csv  - abstract + two anonymous outputs + empty rating columns to fill (1-5)
  eval/human/key.json     - which system is A/B per item (gitignored; don't open before rating)

Blinding is imperfect: both outputs are shown as plain prose (the LLM arm's claims joined as
sentences, without field labels or evidence quotes), but a rater may still guess the system from
style. The structured view and the evidence quotes, which are the product's main feature, are NOT
part of this comparison; it only rates the text content.
"""

import argparse
import csv
import random
from pathlib import Path

from eval.common import BENCHMARK, RESULTS_DIR, ROOT, load_json, save_json

HUMAN_DIR = ROOT / "eval" / "human"
FIELDS = [
    "item_id",
    "abstract",
    "output_A",
    "output_B",
    "accuracy_A",
    "usefulness_A",
    "accuracy_B",
    "usefulness_B",
    "notes",
]
N_ITEMS = 20


def prose(row: dict, arm: str) -> str:
    """The text a rater sees. For the LLM arm: verified claim texts only, as running prose."""
    if arm == "llm" and row.get("structured"):
        s = row["structured"]
        claims = [
            c["text"].strip() for f in ("problem", "method", "results", "limitations") for c in s[f]
        ]
        return " ".join(claims)
    return row["text"].strip()


def build(
    summaries_path: Path,
    benchmark: Path,
    out_dir: Path,
    n: int = N_ITEMS,
    seed: int = 0,
    force: bool = False,
) -> dict:
    ratings, key_path = out_dir / "ratings.csv", out_dir / "key.json"
    if ratings.exists() and not force:
        raise FileExistsError(f"{ratings} already exists; refusing to overwrite your ratings")
    summaries = load_json(summaries_path)
    if "llm" not in summaries:
        raise SystemExit("needs both arms: the LLM arm has not been run (no summaries.json 'llm')")
    abstracts = {p["id"]: p["abstract"] for p in load_json(benchmark)["part_a"]}
    by_arm = {
        a: {r["id"]: r for r in rows if not r["error"] and prose(r, a)}
        for a, rows in summaries.items()
    }
    ids = sorted(set(by_arm["hf"]) & set(by_arm["llm"]))
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} papers have output from both arms; need {n}")
    rng = random.Random(seed)
    chosen = rng.sample(ids, n)
    rows, key = [], {}
    for item_id in chosen:
        order = ["hf", "llm"]
        rng.shuffle(order)
        key[item_id] = {"A": order[0], "B": order[1]}
        rows.append(
            {
                "item_id": item_id,
                "abstract": abstracts[item_id],
                "output_A": prose(by_arm[order[0]][item_id], order[0]),
                "output_B": prose(by_arm[order[1]][item_id], order[1]),
                **{f: "" for f in FIELDS[4:]},
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with ratings.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    save_json(key_path, key)
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=N_ITEMS)
    ap.add_argument("--force", action="store_true", help="overwrite an existing ratings.csv")
    args = ap.parse_args()
    build(RESULTS_DIR / "summaries.json", BENCHMARK, HUMAN_DIR, n=args.n, force=args.force)
    print(
        f"wrote {HUMAN_DIR / 'ratings.csv'} - fill the rating columns (1-5), then run "
        "python -m eval.score_human_eval"
    )


if __name__ == "__main__":
    main()
