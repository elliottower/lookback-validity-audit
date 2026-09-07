"""EXP20 offline analysis: what does the model output in the both-condition?

Registered in AMENDMENTS.md Amendment 13, corrected by Amendment 14 (exact conditional
binomial tests, Holm-adjusted over six comparisons) and Amendment 16b (IIA scored against
the intervention target, by normalized decoded string).

The collection script stores raw readouts and assigns no category. This assigns them, so
the classification can be revised without re-running anything on a GPU.

Classification is ordered and mutually exclusive; the first match wins:

  1 exact clean answer
  2 exact intervention target
  3 other canonical drink appearing in either source prompt
  4 canonical drink appearing in neither
  5 not a complete canonical drink

Confirmatory endpoint collapses these to three macro-categories:
  p_clean = 1,  p_cf = 2,  p_off = 3 + 4 + 5
and a conclusion requires one to exceed BOTH others under exact conditional binomial
tests, Holm-adjusted across the six comparisons (three pairwise, two layers).
"""

import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OBS = REPO / "results" / "exp19_exp20" / "observations.jsonl"
DEFAULT_PAIRS = REPO / "results" / "exp19_exp20_dryrun"
OUT = REPO / "results" / "exp19_exp20" / "exp20_analysis.json"

MACRO = {1: "p_clean", 2: "p_cf", 3: "p_off", 4: "p_off", 5: "p_off"}


def binom_two_sided(k, n):
    """Exact two-sided binomial test against p = 1/2, by summing tail probabilities."""
    if n == 0:
        return 1.0
    obs = comb(n, k)
    total = sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= obs)
    return min(1.0, total / (2 ** n))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adjusted, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def classify(row, drinks, prompts):
    pred = row["pred_normalized"]
    clean = (row.get("clean_ans") or "").lower().strip()
    target = (row.get("cf_ans") or "").lower().strip()
    if pred == clean:
        return 1
    if pred == target:
        return 2
    if pred in drinks:
        text = prompts.get(row["pair_index"], "")
        return 3 if pred in text else 4
    return 5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", type=Path, default=DEFAULT_OBS)
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.observations) if l.strip()]
    exp20 = [r for r in rows if r["exp"] == "EXP20" and not r.get("undefined")]

    pair_files = sorted(args.pairs_dir.glob("filtered_pairs_*.json"))
    prompts, drinks = {}, set()
    if pair_files:
        data = json.loads(pair_files[-1].read_text())
        for i, p in enumerate(data["pairs"]):
            prompts[i] = (p["clean_prompt"] + " " + p["counterfactual_prompt"]).lower()
            drinks.update(s.lower().strip() for s in p["clean_states"])
            drinks.update(s.lower().strip() for s in p["counterfactual_states"])

    layers = sorted({r["layer"] for r in exp20})
    out = {
        "observations": str(args.observations),
        "n_exp20_defined": len(exp20),
        "layers": layers,
        "canonical_drinks_seen": sorted(drinks),
        "conditions": {},
        "confirmatory": {},
    }

    # Descriptive: every condition, every layer.
    for layer in layers:
        for cond in sorted({r["condition"] for r in exp20}):
            sel = [r for r in exp20 if r["layer"] == layer and r["condition"] == cond]
            if not sel:
                continue
            cats = Counter(classify(r, drinks, prompts) for r in sel)
            n = len(sel)
            out["conditions"][f"L{layer}/{cond}"] = {
                "n": n,
                "categories": {str(k): cats.get(k, 0) for k in range(1, 6)},
                "macro": {m: sum(v for k, v in cats.items() if MACRO[k] == m)
                          for m in ("p_clean", "p_cf", "p_off")},
                "iia_vs_intervention_target": cats.get(2, 0) / n,
                "returns_clean_answer": cats.get(1, 0) / n,
                "top_token": Counter(r["pred_repr"] for r in sel).most_common(3),
            }

    # Confirmatory: the both-condition only, three pairwise contrasts per layer.
    tests = []
    for layer in layers:
        sel = [r for r in exp20 if r["layer"] == layer and r["condition"] == "both"]
        cats = Counter(classify(r, drinks, prompts) for r in sel)
        macro = {m: sum(v for k, v in cats.items() if MACRO[k] == m)
                 for m in ("p_clean", "p_cf", "p_off")}
        for a, b in (("p_clean", "p_cf"), ("p_clean", "p_off"), ("p_cf", "p_off")):
            n = macro[a] + macro[b]
            tests.append({"layer": layer, "contrast": f"{a} vs {b}",
                          "k": macro[a], "n": n, "p_raw": binom_two_sided(macro[a], n)})
        out["confirmatory"][f"L{layer}"] = {"macro_counts": macro, "n": len(sel)}

    for t, adj in zip(tests, holm([t["p_raw"] for t in tests])):
        t["p_holm"] = adj
    out["confirmatory"]["tests"] = tests
    out["confirmatory"]["family"] = "3 pairwise macro-contrasts x 2 layers, Holm-adjusted"

    # Amendment 14a: the margin is reported, never thresholded. Argmax alone cannot
    # separate a near-cancellation from a decisive restoration.
    for layer in layers:
        sel = [r for r in exp20 if r["layer"] == layer and r["condition"] == "both"]
        by_pair = {r["pair_index"]: r for r in sel}
        clean_rows = {r["pair_index"]: r for r in exp20
                      if r["layer"] == layer and r["condition"] == "clean"}
        margins = sorted(r["clean_answer_logit"] - r["cf_answer_logit"] for r in sel)
        n = len(margins)
        same_token = sum(1 for i, r in by_pair.items()
                         if r["pred_id"] == clean_rows[i]["pred_id"])
        same_top = sum(1 for i, r in by_pair.items()
                       if r["top_ids"] == clean_rows[i]["top_ids"])
        out["confirmatory"][f"L{layer}"]["both_condition_margin"] = {
            "definition": "clean-answer logit minus intervention-target logit",
            "n": n,
            "min": margins[0], "median": margins[n // 2], "max": margins[-1],
            "q05": margins[int(0.05 * n)], "q95": margins[int(0.95 * n)],
            "proportion_above_zero": sum(1 for m in margins if m > 0) / n,
        }
        out["confirmatory"][f"L{layer}"]["both_vs_unpatched"] = {
            "same_predicted_token": same_token, "n": n,
            "same_top10_ordering": same_top,
            "note": ("identical argmax but different logit ordering means the combined "
                     "patch is not a no-op: it perturbs the distribution and lands back "
                     "on the clean answer"),
        }

    def verdict(layer):
        m = out["confirmatory"][f"L{layer}"]["macro_counts"]
        rel = [t for t in tests if t["layer"] == layer]
        best = max(m, key=m.get)
        beats = [t for t in rel if best in t["contrast"]]
        if all(t["p_holm"] < 0.05 for t in beats):
            return {"p_clean": "cancellation", "p_cf": "counterfactual transfer",
                    "p_off": "off-target interference"}[best]
        return "unresolved"

    out["confirmatory"]["verdict"] = {f"L{l}": verdict(l) for l in layers}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}\n")

    for layer in layers:
        c = out["confirmatory"][f"L{layer}"]
        print(f"L{layer}  n={c['n']}  {c['macro_counts']}  -> {out['confirmatory']['verdict'][f'L{layer}']}")
    print()
    print(f"{'condition':<24}{'n':>5}{'clean':>8}{'target':>8}{'off':>7}   top token")
    for k, v in out["conditions"].items():
        m = v["macro"]
        print(f"{k:<24}{v['n']:>5}{m['p_clean']:>8}{m['p_cf']:>8}{m['p_off']:>7}   {v['top_token'][0]}")


if __name__ == "__main__":
    main()
