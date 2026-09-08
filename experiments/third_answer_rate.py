"""Coherent-third-answer rate: the second registered arm of EXP2.

The registration commits to this alongside the IIA baseline:

    we record the coherent-third-answer rate ... for each random subspace to test
    whether the decomposable structure is direction-specific

and fixes the definition before the data: a third answer counts as coherent if it is a
token from the story's entity vocabulary distinct from both A and B. The identified
subspace's rate must exceed the random subspaces' rate to support directional specificity.

This reads the per-observation records rather than re-running anything. It requires the
`clean_ans` and `story_vocab` fields, which only observations written after that logging was
added carry; rows without them are counted as unclassifiable and reported, never guessed at.

    uv run --python .venv/bin/python experiments/third_answer_rate.py --obs-dir results/exp2_third_answer
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def classify(row):
    """A: clean answer. B: intervention target. C: coherent third. D: off-vocabulary."""
    pred = (row.get("pred") or "").lower().strip()
    a = (row.get("clean_ans") or "").lower().strip()
    b = (row.get("target") or "").lower().strip()
    vocab = row.get("story_vocab")
    if not a or vocab is None:
        return None
    if pred == a:
        return "A"
    if pred == b:
        return "B"
    return "C" if pred in set(vocab) else "D"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs-dir", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.obs_dir / "third_answer_rate.json")

    # mask_index is absent on the identified-subspace pass and present on random draws,
    # which is what separates the two arms of the comparison.
    tally = defaultdict(lambda: defaultdict(int))
    for f in sorted(args.obs_dir.glob("*.observations.jsonl")):
        name = f.name.replace(".observations.jsonl", "")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            arm = "random" if row.get("mask_index") is not None else "identified"
            cls = classify(row)
            tally[(name, arm)]["n"] += 1
            tally[(name, arm)]["unclassifiable" if cls is None else cls] += 1

    result = {"observations_dir": str(args.obs_dir), "subspaces": {}}
    for (name, arm), t in sorted(tally.items()):
        scored = t["n"] - t["unclassifiable"]
        entry = {
            "n_observations": t["n"],
            "n_unclassifiable": t["unclassifiable"],
            "n_scored": scored,
            "counts": {k: t[k] for k in ("A", "B", "C", "D")},
            "coherent_third_rate": (t["C"] / scored) if scored else None,
        }
        result["subspaces"].setdefault(name, {})[arm] = entry

    for name, arms in result["subspaces"].items():
        i, r = arms.get("identified"), arms.get("random")
        if i and r and i["coherent_third_rate"] is not None and r["coherent_third_rate"] is not None:
            arms["identified_minus_random"] = i["coherent_third_rate"] - r["coherent_third_rate"]

    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}\n")
    print(f"{'subspace':<28}{'arm':<12}{'scored':>8}{'A':>6}{'B':>6}{'C':>6}{'D':>6}{'third rate':>12}")
    for name, arms in result["subspaces"].items():
        for arm in ("identified", "random"):
            e = arms.get(arm)
            if not e:
                continue
            c = e["counts"]
            rate = e["coherent_third_rate"]
            print(f"{name:<28}{arm:<12}{e['n_scored']:>8}{c['A']:>6}{c['B']:>6}{c['C']:>6}{c['D']:>6}"
                  f"{(f'{rate:.4f}' if rate is not None else 'n/a'):>12}")
        if e and e["n_unclassifiable"]:
            print(f"{'':<28}{'':<12}({e['n_unclassifiable']} rows lack clean_ans/story_vocab and are not scored)")


if __name__ == "__main__":
    main()
