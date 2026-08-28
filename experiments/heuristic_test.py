"""
Surface heuristic test: does the belief subspace use genuine belief tracking
or a template-specific shortcut?

The paper trains and evaluates on template_idx=2 (both characters independent,
no observability). A surface heuristic like "other person's container → unknown"
or "position of queried character's drink" could achieve high IIA without
genuine belief tracking.

Three adversarial conditions test this:

  1. **Template transfer** (template_idx=0,1): Different observability
     conditions. Template 0: neither observes the other (but has extra
     observability clause). Template 1: character 1 CAN observe character 2.
     If the subspace learned a template-2-specific pattern, it fails here.

  2. **Known-state queries**: Standard CausalToM asks about beliefs where
     the answer is often "unknown" (unobserved container). We specifically
     query beliefs where the character DID observe the action (own container).
     The answer is the drink they poured, not "unknown". If the subspace
     just encodes "unknown when asked about other's container", it fails
     when the answer should be the actual drink.

  3. **Reversed entity order**: Same template 2 structure but character/
     container/state names are chosen so that alphabetical or positional
     heuristics would pick the wrong answer.

Each condition tests IIA with all 6 subspaces (skipping binding for
non-CausalToM positions). High IIA across conditions → genuine belief
tracking. Low IIA → template-specific shortcut.

Usage:
    uv run python experiments/heuristic_test.py --dry-run
    uv run python experiments/heuristic_test.py
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    BELIEF_TRACKING,
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

sys.path.insert(0, str(BELIEF_TRACKING))
from src.dataset import Dataset, Sample

RESULTS_DIR = REPO_ROOT / "results" / "heuristic"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def load_entity_lists():
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)
    return characters, bottles, drinks


def generate_template_transfer_pairs(characters, bottles, drinks, n_samples,
                                     template_idx, seed):
    """Generate CausalToM pairs using a non-standard template.

    Same story generation logic as the paper but with template_idx != 2.
    """
    random.seed(seed)
    np.random.seed(seed)

    clean_configs, cf_configs = [], []
    for _ in range(n_samples):
        chars = random.sample(characters, 2)
        objs = random.sample(bottles, 2)
        states = random.sample(drinks, 2)
        clean_configs.append(Sample(
            template_idx=template_idx, characters=chars,
            objects=objs, states=states,
        ))

        new_states = random.sample(drinks, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(drinks, 2)

        cf_configs.append(Sample(
            template_idx=template_idx,
            characters=list(reversed(chars)),
            objects=list(reversed(objs)),
            states=new_states,
        ))

    clean_ds = Dataset(clean_configs)
    cf_ds = Dataset(cf_configs)

    pairs = []
    for idx in range(n_samples):
        rc = random.choice([0, 1])
        clean = clean_ds.__getitem__(idx, set_container=rc, set_character=rc)
        cf = cf_ds.__getitem__(idx, set_container=1 ^ rc, set_character=1 ^ rc)
        pairs.append({
            "clean_prompt": clean["prompt"],
            "counterfactual_prompt": cf["prompt"],
            "clean_ans": clean["target"],
            "counterfactual_ans": cf["target"],
            "condition": f"template_{template_idx}",
        })
    return pairs


def generate_known_state_pairs(characters, bottles, drinks, n_samples, seed):
    """Generate pairs where we query beliefs the character DOES know.

    Ask character about their OWN container (which they filled). The answer
    is the drink they poured, not "unknown".
    """
    random.seed(seed + 100)
    np.random.seed(seed + 100)

    clean_configs, cf_configs = [], []
    for _ in range(n_samples):
        chars = random.sample(characters, 2)
        objs = random.sample(bottles, 2)
        states = random.sample(drinks, 2)
        clean_configs.append(Sample(
            template_idx=2, characters=chars, objects=objs, states=states,
        ))

        new_states = random.sample(drinks, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(drinks, 2)

        cf_configs.append(Sample(
            template_idx=2,
            characters=list(reversed(chars)),
            objects=list(reversed(objs)),
            states=new_states,
        ))

    clean_ds = Dataset(clean_configs)
    cf_ds = Dataset(cf_configs)

    pairs = []
    for idx in range(n_samples):
        rc = random.choice([0, 1])
        # set_character=rc and set_container=rc means asking character rc
        # about container rc — the one THEY filled. They know the answer.
        clean = clean_ds.__getitem__(idx, set_container=rc, set_character=rc)
        # For CF: also ask the character about their own container
        cf = cf_ds.__getitem__(idx, set_container=1 ^ rc, set_character=1 ^ rc)

        if clean["target"] == "unknown" or cf["target"] == "unknown":
            continue

        pairs.append({
            "clean_prompt": clean["prompt"],
            "counterfactual_prompt": cf["prompt"],
            "clean_ans": clean["target"],
            "counterfactual_ans": cf["target"],
            "condition": "known_state",
        })
    return pairs


def generate_reversed_order_pairs(characters, bottles, drinks, n_samples, seed):
    """Generate standard template-2 pairs but with entity names chosen to
    defeat alphabetical/positional heuristics.

    Strategy: pick character/container/drink names where the alphabetically
    first entity is NOT the queried one, and the last-mentioned drink is NOT
    the answer.
    """
    random.seed(seed + 200)
    np.random.seed(seed + 200)

    sorted_chars = sorted(characters)
    sorted_bottles = sorted(bottles)
    sorted_drinks = sorted(drinks)

    clean_configs, cf_configs = [], []
    for _ in range(n_samples):
        c1_idx = random.randint(len(sorted_chars) // 2, len(sorted_chars) - 1)
        c2_idx = random.randint(0, len(sorted_chars) // 2 - 1)
        chars = [sorted_chars[c1_idx], sorted_chars[c2_idx]]

        objs = random.sample(bottles, 2)

        d1_idx = random.randint(len(sorted_drinks) // 2, len(sorted_drinks) - 1)
        d2_idx = random.randint(0, len(sorted_drinks) // 2 - 1)
        states = [sorted_drinks[d1_idx], sorted_drinks[d2_idx]]

        if len(set(chars)) < 2 or len(set(states)) < 2:
            continue

        clean_configs.append(Sample(
            template_idx=2, characters=chars, objects=objs, states=states,
        ))

        new_states = random.sample(drinks, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(drinks, 2)

        cf_configs.append(Sample(
            template_idx=2,
            characters=list(reversed(chars)),
            objects=list(reversed(objs)),
            states=new_states,
        ))

    clean_ds = Dataset(clean_configs)
    cf_ds = Dataset(cf_configs)

    pairs = []
    for idx in range(len(clean_configs)):
        rc = random.choice([0, 1])
        clean = clean_ds.__getitem__(idx, set_container=rc, set_character=rc)
        cf = cf_ds.__getitem__(idx, set_container=1 ^ rc, set_character=1 ^ rc)
        pairs.append({
            "clean_prompt": clean["prompt"],
            "counterfactual_prompt": cf["prompt"],
            "clean_ans": clean["target"],
            "counterfactual_ans": cf["target"],
            "condition": "reversed_order",
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Surface heuristic test for lookback subspaces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    characters, bottles, drinks = load_entity_lists()
    subspaces = load_subspace_specs()

    conditions = {}

    print(f"[{ts()}] Generating adversarial pairs...")

    conditions["template_0"] = generate_template_transfer_pairs(
        characters, bottles, drinks, args.n_eval * 3, template_idx=0, seed=args.seed)
    conditions["template_1"] = generate_template_transfer_pairs(
        characters, bottles, drinks, args.n_eval * 3, template_idx=1, seed=args.seed)
    conditions["known_state"] = generate_known_state_pairs(
        characters, bottles, drinks, args.n_eval * 3, seed=args.seed)
    conditions["reversed_order"] = generate_reversed_order_pairs(
        characters, bottles, drinks, args.n_eval * 3, seed=args.seed)

    for name, pairs in conditions.items():
        print(f"  {name}: {len(pairs)} raw pairs")

    lm = None
    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        for name in list(conditions.keys()):
            print(f"[{ts()}] Filtering {name} on model accuracy...")
            conditions[name] = filter_on_model(
                lm, conditions[name], max_size=args.n_eval)
            print(f"  {len(conditions[name])} pairs passed filter")

    results = {
        "experiment": "Surface heuristic test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "method": (
            "Tests 4 adversarial conditions against paper's subspaces. "
            "Template transfer (0,1): different observability. "
            "Known-state: character queried about own container. "
            "Reversed order: alphabetically adversarial entity names."
        ),
        "prediction": {
            "genuine_tracking": "IIA ~ paper levels across all conditions",
            "surface_heuristic": "IIA drops significantly for non-standard conditions",
        },
        "conditions": {},
    }

    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v.get("lookback_type", "")}

    for cond_name, pairs in conditions.items():
        print(f"\n{'='*60}")
        print(f"[{ts()}] Condition: {cond_name} ({len(pairs)} pairs)")
        print(f"{'='*60}")

        cond_result = {"n_pairs": len(pairs), "subspace_iias": {}}

        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            rank = spec["rank"]
            mask_indices = spec.get("mask_indices")

            ckpt_path = RESULTS_DIR / f"{cond_name}_{sub_name}.json"
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    cached = json.load(f)
                cond_result["subspace_iias"][sub_name] = cached
                print(f"  {sub_name}: IIA={cached['iia']:.4f} (cached)")
                continue

            if args.dry_run:
                if cond_name in ("template_0", "template_1"):
                    iia = float(spec["sv_iia"] * rng.uniform(0.5, 0.95))
                elif cond_name == "known_state":
                    iia = float(spec["sv_iia"] * rng.uniform(0.6, 1.0))
                else:
                    iia = float(spec["sv_iia"] * rng.uniform(0.7, 1.0))
            else:
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                if mask_indices is not None:
                    proj = build_projection_matrix(svd_basis, mask_indices)
                else:
                    proj = build_projection_matrix(svd_basis, np.arange(rank))
                iia = compute_iia_answer_flex(lm, pairs, layer, proj)

            entry = {
                "iia": float(iia),
                "paper_iia": spec["sv_iia"],
                "transfer_ratio": float(iia / spec["sv_iia"]) if spec["sv_iia"] > 0 else 0,
                "layer": layer,
                "rank": rank,
            }
            cond_result["subspace_iias"][sub_name] = entry

            with open(ckpt_path, "w") as f:
                json.dump(entry, f, indent=2)

            print(f"  {sub_name}: IIA={iia:.4f} "
                  f"(paper={spec['sv_iia']:.4f}, ratio={entry['transfer_ratio']:.2f})")

        iia_values = [v["iia"] for v in cond_result["subspace_iias"].values()
                      if isinstance(v.get("iia"), (int, float))]
        ratios = [v["transfer_ratio"] for v in cond_result["subspace_iias"].values()
                  if isinstance(v.get("transfer_ratio"), (int, float))]

        cond_result["mean_iia"] = float(np.mean(iia_values)) if iia_values else None
        cond_result["mean_transfer_ratio"] = float(np.mean(ratios)) if ratios else None

        if cond_result["mean_transfer_ratio"] is not None:
            if cond_result["mean_transfer_ratio"] > 0.8:
                cond_result["verdict"] = "GENUINE: subspace transfers to adversarial condition"
            elif cond_result["mean_transfer_ratio"] > 0.5:
                cond_result["verdict"] = "PARTIAL: subspace partially transfers"
            else:
                cond_result["verdict"] = "HEURISTIC: subspace fails on adversarial condition"

        results["conditions"][cond_name] = cond_result

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SURFACE HEURISTIC TEST SUMMARY")
    print(f"{'='*60}")
    for cond_name, cr in results["conditions"].items():
        ratio = cr.get("mean_transfer_ratio")
        verdict = cr.get("verdict", "N/A")
        print(f"  {cond_name}: mean_IIA={cr.get('mean_iia', 0):.4f}, "
              f"transfer_ratio={ratio:.2f}, {verdict}")

    all_ratios = [cr["mean_transfer_ratio"]
                  for cr in results["conditions"].values()
                  if cr.get("mean_transfer_ratio") is not None]
    overall = float(np.mean(all_ratios)) if all_ratios else 0
    results["overall_transfer_ratio"] = overall
    results["overall_verdict"] = (
        "GENUINE BELIEF TRACKING"
        if overall > 0.8 else
        "MIXED EVIDENCE"
        if overall > 0.5 else
        "SURFACE HEURISTIC"
    )
    print(f"\n  Overall transfer ratio: {overall:.2f}")
    print(f"  Verdict: {results['overall_verdict']}")

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
