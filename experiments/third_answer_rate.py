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
    """A clean answer, B intervention target, C coherent third, E donor-only, D neither.

    C is membership in the *recipient* story's vocabulary. A prediction found only in the
    donor story is transferred content rather than a third answer built from the recipient's
    own entities, and counting it as coherent would let content leakage register as evidence
    for directional specificity. A token in both vocabularies is C, because it is available
    to the recipient.
    """
    pred = (row.get("pred") or "").lower().strip()
    a = (row.get("clean_ans") or "").lower().strip()
    b = (row.get("target") or "").lower().strip()
    clean_vocab = row.get("clean_vocab")
    cf_vocab = row.get("cf_vocab")
    if not pred or not a or not b or clean_vocab is None:
        return None
    if pred == a:
        return "A"
    if pred == b:
        return "B"
    if pred in set(clean_vocab):
        return "C"
    return "E" if pred in set(cf_vocab or ()) else "D"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs-dir", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.obs_dir / "third_answer_rate.json")

    # mask_index is absent on the identified-subspace pass and present on random draws,
    # which is what separates the two arms of the comparison.
    tally = defaultdict(lambda: defaultdict(int))
    per_mask = defaultdict(dict)          # subspace -> mask_index -> (n_scored, n_coherent)
    for f in sorted(args.obs_dir.glob("*.observations.jsonl")):
        name = f.name.replace(".observations.jsonl", "")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            arm = row.get("arm") or ("random" if row.get("mask_index") is not None else "identified")
            cls = classify(row)
            tally[(name, arm)]["n"] += 1
            tally[(name, arm)]["unclassifiable" if cls is None else cls] += 1
            if arm == "random" and cls is not None:
                mi = row.get("mask_index")
                n, c = per_mask[name].get(mi, (0, 0))
                per_mask[name][mi] = (n + 1, c + int(cls == "C"))

    result = {"observations_dir": str(args.obs_dir), "subspaces": {}}
    for (name, arm), t in sorted(tally.items()):
        scored = t["n"] - t["unclassifiable"]
        entry = {
            "n_observations": t["n"],
            "n_unclassifiable": t["unclassifiable"],
            "n_scored": scored,
            "counts": {k: t[k] for k in ("A", "B", "C", "E", "D")},
            "coherent_third_rate": (t["C"] / scored) if scored else None,
        }
        result["subspaces"].setdefault(name, {})[arm] = entry

    # The rank test runs over per-mask rates, not over pooled observations. Pooling
    # M x 80 random outcomes treats interventions sharing a mask and a story pair as
    # independent, which they are not, and shrinks the interval by roughly sqrt(80).
    for name, arms in result["subspaces"].items():
        rates = per_mask.get(name, {})
        ident = arms.get("identified")
        if not rates or not ident or ident["coherent_third_rate"] is None:
            continue
        p_id = ident["coherent_third_rate"]
        per = sorted((c / n) for n, c in rates.values() if n)
        # Ties count against the identified subspace.
        k = sum(1 for x in per if x >= p_id)
        arms["mask_level_test"] = {
            "n_masks": len(per),
            "identified_rate": p_id,
            "random_rates": per,
            "random_mean": sum(per) / len(per),
            "random_median": per[len(per) // 2],
            "random_min": per[0],
            "random_max": per[-1],
            "n_masks_ge_identified": k,
            "p_rank_test": (k + 1) / (len(per) + 1),
            "p_floor": 1 / (len(per) + 1),
            "identified_minus_random_mean": p_id - sum(per) / len(per),
            "note": ("Rank over per-mask rates with ties against the identified subspace. "
                     "p_floor is the smallest value this many masks can return, so a "
                     "registered threshold below it is unreachable by design."),
        }

    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}\n")
    print(f"{'subspace':<28}{'arm':<12}{'scored':>8}{'A':>6}{'B':>6}{'C':>6}{'E':>6}{'D':>6}{'third rate':>12}")
    for name, arms in result["subspaces"].items():
        for arm in ("identified", "random"):
            e = arms.get(arm)
            if not e:
                continue
            c = e["counts"]
            rate = e["coherent_third_rate"]
            print(f"{name:<28}{arm:<12}{e['n_scored']:>8}{c['A']:>6}{c['B']:>6}{c['C']:>6}{c['E']:>6}{c['D']:>6}"
                  f"{(f'{rate:.4f}' if rate is not None else 'n/a'):>12}")
        if e and e["n_unclassifiable"]:
            print(f"{'':<28}{'':<12}({e['n_unclassifiable']} rows lack clean_ans/story_vocab and are not scored)")


if __name__ == "__main__":
    main()
