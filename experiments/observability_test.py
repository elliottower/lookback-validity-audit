"""
Observability dissociation: does the subspace encode who can observe
what, or only self-action bindings?

Template 1 creates an asymmetry: char1 CAN observe char2's actions, but
char2 CANNOT observe char1's. Two conditions:

A. Observed-other-container: ask char1 about the container char2 filled
   (which char1 observed). If the subspace tracks observability-mediated
   belief, IIA should be high.

B. Self-container control: ask char1 about their own container (standard
   case). Should match the positive control baseline.

The paper's subspaces were trained exclusively on self-container queries
(set_character=rc, set_container=rc). Condition A tests whether they
generalize to knowledge acquired through observation rather than action.

Usage:
    uv run python experiments/observability_test.py --dry-run
    uv run python experiments/observability_test.py
"""

import argparse
import json
import random
import sys
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

RESULTS_DIR = REPO_ROOT / "results" / "observability"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def generate_template1_pairs(n_samples, seed, set_character, set_container):
    """Generate clean/CF pairs using template 1 with specified query indices.

    Both clean and CF use independent characters/objects/states but the
    same template structure and query indices, so the observability
    asymmetry is identical across the pair.
    """
    sys.path.insert(0, str(BELIEF_TRACKING))
    from src.dataset import Dataset, Sample

    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    random.seed(seed)
    np.random.seed(seed)

    pairs = []
    for _ in range(n_samples):
        chars = random.sample(characters, 2)
        objs = random.sample(bottles, 2)
        states = random.sample(drinks, 2)

        clean_sample = Sample(
            template_idx=1,
            characters=chars,
            objects=objs,
            states=states,
        )
        clean_dataset = Dataset([clean_sample])
        clean = clean_dataset.__getitem__(
            0, set_character=set_character, set_container=set_container)

        cf_chars = random.sample(characters, 2)
        while set(cf_chars) == set(chars):
            cf_chars = random.sample(characters, 2)
        cf_objs = random.sample(bottles, 2)
        cf_states = random.sample(drinks, 2)
        while cf_states[0] in states or cf_states[1] in states:
            cf_states = random.sample(drinks, 2)

        cf_sample = Sample(
            template_idx=1,
            characters=cf_chars,
            objects=cf_objs,
            states=cf_states,
        )
        cf_dataset = Dataset([cf_sample])
        cf = cf_dataset.__getitem__(
            0, set_character=set_character, set_container=set_container)

        pairs.append({
            "clean_prompt": clean["prompt"],
            "counterfactual_prompt": cf["prompt"],
            "clean_ans": clean["target"],
            "counterfactual_ans": cf["target"],
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Observability dissociation test for lookback subspaces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Observability dissociation test")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    lm = None
    if not args.dry_run:
        lm = setup_nnsight()

    conditions = {
        "observed_other_container": {
            "set_character": 0,
            "set_container": 1,
            "description": (
                "Char1 asked about container2 (filled by char2, "
                "which char1 observed)"
            ),
        },
        "self_container_control": {
            "set_character": 0,
            "set_container": 0,
            "description": (
                "Char1 asked about container1 (filled by char1 themselves)"
            ),
        },
    }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "conditions": {},
    }

    for cond_name, cond_spec in conditions.items():
        print(f"\n{'='*60}")
        print(f"[{ts()}] Condition: {cond_name}")
        print(f"  {cond_spec['description']}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{cond_name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) != args.dry_run:
                print(f"  Stale checkpoint (dry_run mismatch), recomputing")
            else:
                summary["conditions"][cond_name] = cached
                print(f"  Resumed from checkpoint")
                for sub_name, iia in cached.get("subspace_iias", {}).items():
                    print(f"    {sub_name}: IIA={iia:.4f}")
                continue

        pairs_cache = RESULTS_DIR / f"filtered_pairs_{cond_name}.json"
        if pairs_cache.exists() and not args.dry_run:
            with open(pairs_cache) as f:
                saved = json.load(f)
            pairs = saved["pairs"]
            behavioral_accuracy = saved["behavioral_accuracy"]
            print(f"[{ts()}] Loaded {len(pairs)} cached pairs "
                  f"(accuracy={behavioral_accuracy:.2f})")
        else:
            print(f"[{ts()}] Generating template-1 pairs...")
            pairs_raw = generate_template1_pairs(
                args.n_eval * 3, args.seed,
                cond_spec["set_character"], cond_spec["set_container"])

            if not args.dry_run:
                print(f"[{ts()}] Filtering on model accuracy...")
                all_passing = filter_on_model(lm, pairs_raw, max_size=len(pairs_raw))
                behavioral_accuracy = len(all_passing) / len(pairs_raw) if pairs_raw else 0
                pairs = all_passing[:args.n_eval]
                print(f"  {len(all_passing)}/{len(pairs_raw)} passed filter "
                      f"(accuracy={behavioral_accuracy:.2f}), using {len(pairs)}")
                with open(pairs_cache, "w") as f:
                    json.dump({"pairs": pairs, "behavioral_accuracy": behavioral_accuracy}, f, indent=2)
            else:
                pairs = pairs_raw[:args.n_eval]
                behavioral_accuracy = len(pairs) / len(pairs_raw) if pairs_raw else 0

        sub_iias = {}
        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                if cond_name == "self_container_control":
                    iia = spec["sv_iia"] + rng.normal(0, 0.02)
                else:
                    iia = rng.uniform(0.05, 0.35)
                iia = float(np.clip(iia, 0, 1))
            else:
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                proj = build_projection_matrix(svd_basis, mask_indices)
                iia = compute_iia_answer_flex(lm, pairs, layer, proj)

            sub_iias[sub_name] = float(iia)
            print(f"  {sub_name}: IIA={iia:.4f}")

        result = {
            "dry_run": args.dry_run,
            "condition": cond_name,
            "description": cond_spec["description"],
            "template_idx": 1,
            "set_character": cond_spec["set_character"],
            "set_container": cond_spec["set_container"],
            "n_generated": len(pairs_raw),
            "n_filtered": len(pairs),
            "behavioral_accuracy": float(behavioral_accuracy),
            "subspace_iias": sub_iias,
        }
        summary["conditions"][cond_name] = result

        with open(ckpt_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{ts()}] Saved {ckpt_path}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("OBSERVABILITY DISSOCIATION SUMMARY")
    print(f"{'='*60}")

    for cond_name, cres in summary["conditions"].items():
        print(f"\n{cond_name}:")
        print(f"  Behavioral accuracy: {cres.get('behavioral_accuracy', '?'):.2f}")
        print(f"  N pairs: {cres.get('n_filtered', '?')}")
        for sub_name, iia in cres.get("subspace_iias", {}).items():
            print(f"  {sub_name}: IIA={iia:.4f}")

    obs = summary["conditions"].get("observed_other_container", {})
    ctrl = summary["conditions"].get("self_container_control", {})
    if obs.get("subspace_iias") and ctrl.get("subspace_iias"):
        print(f"\nDissociation (self - observed):")
        for sub_name in ctrl["subspace_iias"]:
            if sub_name in obs["subspace_iias"]:
                diff = ctrl["subspace_iias"][sub_name] - obs["subspace_iias"][sub_name]
                print(f"  {sub_name}: {diff:+.4f}")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
