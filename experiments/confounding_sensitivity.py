"""
Confounding sensitivity analysis (I8): E-value computation.

Quantifies how strong an unmeasured confounder would need to be to
explain away each observed IIA effect. Uses the VanderWeele &
Ding (2017) E-value formula:

    RR = IIA / baseline_IIA
    E  = RR + sqrt(RR * (RR - 1))

where baseline_IIA = 0.5 for binary-outcome IIA (chance level).

An E-value of 3.0 means a confounder would need to triple both the
exposure-confounder and confounder-outcome associations to nullify
the finding. Higher is more robust.

No NDIF required -- pure statistics on existing result files.

Usage:
    uv run python experiments/confounding_sensitivity.py
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "confounding_sensitivity"

BASELINE_IIA = 0.5


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def compute_evalue(iia, baseline=BASELINE_IIA):
    """Compute E-value from observed IIA and baseline.

    Returns (evalue, rr) tuple. Returns (1.0, rr) if IIA <= baseline
    (no effect to explain away).
    """
    if iia <= baseline or baseline <= 0:
        return 1.0, iia / baseline if baseline > 0 else 0.0
    rr = iia / baseline
    evalue = rr + math.sqrt(rr * (rr - 1))
    return evalue, rr


def compute_evalue_ci(ci_lo, baseline=BASELINE_IIA):
    """E-value for the lower confidence bound (more conservative)."""
    if ci_lo <= baseline or baseline <= 0:
        return 1.0, ci_lo / baseline if baseline > 0 else 0.0
    rr = ci_lo / baseline
    return rr + math.sqrt(rr * (rr - 1)), rr


def load_all_iia_results(results_root):
    """Scan results directories for completed IIA experiments.

    Returns list of dicts: {source, condition, subspace, iia, n_pairs}
    """
    records = []

    # Positive control replication
    pc_path = results_root / "positive_control_replication.json"
    if pc_path.exists():
        d = json.load(open(pc_path))
        if not d.get("dry_run", True):
            for sub_name, sub_data in d.get("subspaces", {}).items():
                records.append({
                    "source": "positive_control",
                    "condition": "replication",
                    "subspace": sub_name,
                    "iia": sub_data["our_iia"],
                    "n_pairs": sub_data.get("n_pairs", 80),
                })

    # Experiments with conditions/{condition}/subspace_iias structure
    cond_experiments = {
        "false_belief": "false_belief",
        "observability": "observability",
        "distractor_insertion": "distractor_insertion",
    }
    for dirname, source_label in cond_experiments.items():
        summary = results_root / dirname / "summary.json"
        if not summary.exists():
            continue
        d = json.load(open(summary))
        if d.get("dry_run", True):
            continue
        for cond_name, cond_data in d.get("conditions", {}).items():
            if cond_data.get("dry_run", True):
                continue
            n = cond_data.get("n_pairs", cond_data.get("n_filtered", 0))
            if n == 0:
                continue
            for sub_name, iia in cond_data.get("subspace_iias", {}).items():
                records.append({
                    "source": source_label,
                    "condition": cond_name,
                    "subspace": sub_name,
                    "iia": iia,
                    "n_pairs": n,
                })

    # Question framing (framings/{framing}/subspace_iias)
    qf_path = results_root / "question_framing" / "summary.json"
    if qf_path.exists():
        d = json.load(open(qf_path))
        if not d.get("dry_run", True):
            for framing_name, framing_data in d.get("framings", {}).items():
                if framing_data.get("dry_run", True):
                    continue
                n = framing_data.get("n_pairs", 0)
                if n == 0:
                    continue
                for sub_name, iia in framing_data.get("subspace_iias", {}).items():
                    records.append({
                        "source": "question_framing",
                        "condition": framing_name,
                        "subspace": sub_name,
                        "iia": iia,
                        "n_pairs": n,
                    })

    # Cross-task (tasks/{task}/subspace_iias — skip nulls/skipped)
    ct_path = results_root / "cross_task" / "summary.json"
    if ct_path.exists():
        d = json.load(open(ct_path))
        if not d.get("synthetic", True):
            for task_name, task_data in d.get("tasks", {}).items():
                n = task_data.get("n_pairs", 0)
                if n == 0:
                    continue
                for sub_name, sub_info in task_data.get("subspace_iias", {}).items():
                    if isinstance(sub_info, dict):
                        if sub_info.get("skipped"):
                            continue
                        iia = sub_info.get("iia")
                    else:
                        iia = sub_info
                    if iia is None:
                        continue
                    records.append({
                        "source": "cross_task",
                        "condition": task_name,
                        "subspace": sub_name,
                        "iia": iia,
                        "n_pairs": n,
                    })

    # Adversarial heuristic (template_0.json)
    ah_path = results_root / "adversarial_heuristic" / "template_0.json"
    if ah_path.exists():
        d = json.load(open(ah_path))
        if not d.get("dry_run", True):
            n = d.get("n_filtered", d.get("n_pairs", 0))
            if n > 0:
                for sub_name, iia in d.get("subspace_iias", {}).items():
                    records.append({
                        "source": "adversarial_heuristic",
                        "condition": "template_0",
                        "subspace": sub_name,
                        "iia": iia,
                        "n_pairs": n,
                    })

    return records


def load_ci_data(results_root):
    """Load Wilson CI lower bounds from statistical analysis if available."""
    sa_path = results_root / "statistical_analysis_results.json"
    if not sa_path.exists():
        return {}
    d = json.load(open(sa_path))
    ci_map = {}
    for entry in d.get("results", []):
        key = f"L{entry['layer']}_{entry['metric']}"
        ci_map[key] = entry.get("ci_lo", entry.get("iia", 0.5))
    return ci_map


def main():
    parser = argparse.ArgumentParser(
        description="I8 confounding sensitivity: E-value analysis")
    parser.add_argument("--baseline", type=float, default=BASELINE_IIA,
                        help="Baseline IIA (chance level, default 0.5)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{ts()}] I8 Confounding sensitivity analysis")
    print(f"[{ts()}] Baseline IIA: {args.baseline}")
    print(f"[{ts()}] E-value formula: RR + sqrt(RR * (RR - 1))")
    print(f"[{ts()}] Reference: VanderWeele & Ding (2017)")

    results_root = REPO_ROOT / "results"
    records = load_all_iia_results(results_root)
    print(f"[{ts()}] Loaded {len(records)} IIA measurements from results/")

    if not records:
        print(f"[{ts()}] ERROR: No completed IIA results found.")
        return

    evalue_records = []
    for rec in records:
        ev, rr = compute_evalue(rec["iia"], args.baseline)
        evalue_records.append({
            **rec,
            "baseline_iia": args.baseline,
            "risk_ratio": round(rr, 4),
            "evalue": round(ev, 4),
        })

    evalue_records.sort(key=lambda x: x["evalue"])

    print(f"\n{'='*80}")
    print(f"{'Source':<25} {'Condition':<25} {'Subspace':<22} {'IIA':>5} {'RR':>6} {'E-value':>8}")
    print(f"{'='*80}")
    for r in evalue_records:
        print(f"{r['source']:<25} {r['condition']:<25} {r['subspace']:<22} "
              f"{r['iia']:5.3f} {r['risk_ratio']:6.2f} {r['evalue']:8.2f}")

    positive_effects = [r for r in evalue_records if r["iia"] > args.baseline]
    null_effects = [r for r in evalue_records if r["iia"] <= args.baseline]

    if positive_effects:
        min_ev = min(positive_effects, key=lambda x: x["evalue"])
        max_ev = max(positive_effects, key=lambda x: x["evalue"])
        median_ev = sorted(positive_effects, key=lambda x: x["evalue"])[len(positive_effects) // 2]
    else:
        min_ev = max_ev = median_ev = None

    # Per-subspace summary
    subspace_evalues = {}
    for r in positive_effects:
        sub = r["subspace"]
        if sub not in subspace_evalues:
            subspace_evalues[sub] = []
        subspace_evalues[sub].append(r["evalue"])

    subspace_summary = {}
    for sub, evs in subspace_evalues.items():
        subspace_summary[sub] = {
            "n_positive_effects": len(evs),
            "min_evalue": round(min(evs), 4),
            "max_evalue": round(max(evs), 4),
            "median_evalue": round(sorted(evs)[len(evs) // 2], 4),
        }

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total measurements: {len(evalue_records)}")
    print(f"Positive effects (IIA > {args.baseline}): {len(positive_effects)}")
    print(f"Null/negative effects: {len(null_effects)}")
    if min_ev:
        print(f"\nWeakest positive effect:")
        print(f"  {min_ev['source']}/{min_ev['condition']} {min_ev['subspace']}: "
              f"IIA={min_ev['iia']:.3f}, E={min_ev['evalue']:.2f}")
        print(f"\nStrongest positive effect:")
        print(f"  {max_ev['source']}/{max_ev['condition']} {max_ev['subspace']}: "
              f"IIA={max_ev['iia']:.3f}, E={max_ev['evalue']:.2f}")
    print(f"\nPer-subspace:")
    for sub, ss in sorted(subspace_summary.items()):
        print(f"  {sub}: min E={ss['min_evalue']:.2f}, "
              f"median E={ss['median_evalue']:.2f}, "
              f"max E={ss['max_evalue']:.2f} "
              f"(n={ss['n_positive_effects']})")

    interpretation_threshold = 2.0
    robust = [r for r in positive_effects if r["evalue"] >= interpretation_threshold]
    fragile = [r for r in positive_effects if r["evalue"] < interpretation_threshold]
    print(f"\nRobust (E >= {interpretation_threshold}): {len(robust)}/{len(positive_effects)}")
    print(f"Fragile (E < {interpretation_threshold}): {len(fragile)}/{len(positive_effects)}")
    if fragile:
        print("  Fragile effects:")
        for r in fragile:
            print(f"    {r['source']}/{r['condition']} {r['subspace']}: "
                  f"IIA={r['iia']:.3f}, E={r['evalue']:.2f}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "criterion": "I8",
        "criterion_name": "Confounding sensitivity",
        "method": "E-value (VanderWeele & Ding, 2017)",
        "formula": "E = RR + sqrt(RR * (RR - 1)); RR = IIA / baseline_IIA",
        "baseline_iia": args.baseline,
        "n_measurements": len(evalue_records),
        "n_positive_effects": len(positive_effects),
        "n_null_effects": len(null_effects),
        "weakest_positive": {
            "source": min_ev["source"],
            "condition": min_ev["condition"],
            "subspace": min_ev["subspace"],
            "iia": min_ev["iia"],
            "evalue": min_ev["evalue"],
        } if min_ev else None,
        "strongest_positive": {
            "source": max_ev["source"],
            "condition": max_ev["condition"],
            "subspace": max_ev["subspace"],
            "iia": max_ev["iia"],
            "evalue": max_ev["evalue"],
        } if max_ev else None,
        "n_robust": len(robust),
        "n_fragile": len(fragile),
        "robustness_threshold": interpretation_threshold,
        "subspace_summary": subspace_summary,
        "all_evalues": evalue_records,
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Saved {summary_path}")

    per_record_path = RESULTS_DIR / "all_evalues.json"
    with open(per_record_path, "w") as f:
        json.dump(evalue_records, f, indent=2)
    print(f"[{ts()}] Saved {per_record_path}")

    print(f"\n[{ts()}] Done.")


if __name__ == "__main__":
    main()
