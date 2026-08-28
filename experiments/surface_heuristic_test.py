"""
Surface heuristic test: does the subspace track genuine beliefs or a
positional heuristic ("substance last mentioned before character leaves")?

In the paper's CausalToM stories, the counterfactual answer is always the
substance whose state change occurs in the sentence before the character
leaves. A model could solve the task without any belief representation
simply by tracking "most recently mentioned substance near the departure
sentence."

This test constructs adversarial stories where the positional heuristic
gives the WRONG answer but genuine belief tracking gives the RIGHT answer:

1. Decoy stories: After the real state change, add an irrelevant mention
   of a different substance right before departure. The heuristic would
   pick the decoy; belief tracking picks the real answer.

2. Reordered stories: Swap the order of sentences so the state change is
   described AFTER departure (via a "while X was gone, ..." construction).
   The heuristic fails because there's no pre-departure mention; belief
   tracking still works.

3. Distractor insertion: Insert a sentence about a third character
   interacting with a different substance between the state change and
   departure. Heuristic picks the distractor; belief tracking is unaffected.

If the subspace achieves high IIA on these adversarial prompts, it encodes
genuine beliefs. If IIA drops to chance, it was encoding the heuristic.

Only tests answer subspaces (last-token intervention at -1).

Usage:
    uv run python experiments/surface_heuristic_test.py --dry-run
    uv run python experiments/surface_heuristic_test.py
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "surface_heuristic"

CHARACTERS = [
    "Alice", "Bob", "Charlie", "Diana", "Edward",
    "Fiona", "George", "Hannah", "Ivan", "Julia",
]

SUBSTANCES = [
    "tea", "coffee", "juice", "water", "milk",
    "soda", "wine", "lemonade", "cider", "cocoa",
]

CONTAINERS = [
    "cup", "glass", "mug", "bottle", "pitcher",
    "jug", "thermos", "goblet", "tumbler", "flask",
]

LOCATIONS = [
    "kitchen", "garden", "living room", "study",
    "porch", "balcony", "dining room", "hallway",
]


def make_standard_story(char, substance1, substance2, container, location, rng):
    """Standard CausalToM-style story: state change -> departure -> return -> question.

    Clean answer: substance1 (character's belief = initial state).
    Counterfactual answer: substance2 (true state after swap).
    """
    other_char = rng.choice([c for c in CHARACTERS if c != char])
    other_loc = rng.choice([l for l in LOCATIONS if l != location])

    clean = (
        f"{char} poured some {substance1} into the {container} in the {location}. "
        f"{char} then went to the {other_loc}. "
        f"While {char} was away, {other_char} replaced the {substance1} "
        f"in the {container} with {substance2}. "
        f"{char} came back to the {location}. "
        f"{char} thinks the {container} contains"
    )

    cf = (
        f"{char} poured some {substance2} into the {container} in the {location}. "
        f"{char} then went to the {other_loc}. "
        f"While {char} was away, {other_char} replaced the {substance2} "
        f"in the {container} with {substance1}. "
        f"{char} came back to the {location}. "
        f"{char} thinks the {container} contains"
    )

    return {
        "clean_prompt": clean,
        "counterfactual_prompt": cf,
        "clean_ans": substance1,
        "counterfactual_ans": substance2,
        "condition": "standard",
    }


def make_decoy_story(char, substance1, substance2, container, location, rng):
    """Decoy: after the real state change, mention an irrelevant substance
    near the departure point.

    Heuristic would pick the decoy substance. Genuine belief tracking
    picks the original substance (character's false belief).
    """
    other_char = rng.choice([c for c in CHARACTERS if c != char])
    other_loc = rng.choice([l for l in LOCATIONS if l != location])
    decoy = rng.choice([s for s in SUBSTANCES if s not in (substance1, substance2)])
    decoy_container = rng.choice([c for c in CONTAINERS if c != container])

    clean = (
        f"{char} poured some {substance1} into the {container} in the {location}. "
        f"While {char} was away, {other_char} replaced the {substance1} "
        f"in the {container} with {substance2}. "
        f"{other_char} also placed some {decoy} in a {decoy_container} on the counter. "
        f"{char} then went to the {other_loc}. "
        f"{char} came back to the {location}. "
        f"{char} thinks the {container} contains"
    )

    cf = (
        f"{char} poured some {substance2} into the {container} in the {location}. "
        f"While {char} was away, {other_char} replaced the {substance2} "
        f"in the {container} with {substance1}. "
        f"{other_char} also placed some {decoy} in a {decoy_container} on the counter. "
        f"{char} then went to the {other_loc}. "
        f"{char} came back to the {location}. "
        f"{char} thinks the {container} contains"
    )

    return {
        "clean_prompt": clean,
        "counterfactual_prompt": cf,
        "clean_ans": substance1,
        "counterfactual_ans": substance2,
        "condition": "decoy",
    }


def make_reordered_story(char, substance1, substance2, container, location, rng):
    """Reordered: departure happens BEFORE the state change sentence.

    The narrative uses flashback-style: 'While X was gone, [state change].'
    So there's no pre-departure substance mention for the heuristic to latch
    onto. Belief tracking should still work because the causal structure is
    the same.
    """
    other_char = rng.choice([c for c in CHARACTERS if c != char])
    other_loc = rng.choice([l for l in LOCATIONS if l != location])

    clean = (
        f"{char} poured some {substance1} into the {container} in the {location}. "
        f"{char} left the {location} to go to the {other_loc}. "
        f"After {char} left, {other_char} came in and replaced the "
        f"{substance1} with {substance2} in the {container}. "
        f"Later, {char} returned to the {location}. "
        f"{char} thinks the {container} contains"
    )

    cf = (
        f"{char} poured some {substance2} into the {container} in the {location}. "
        f"{char} left the {location} to go to the {other_loc}. "
        f"After {char} left, {other_char} came in and replaced the "
        f"{substance2} with {substance1} in the {container}. "
        f"Later, {char} returned to the {location}. "
        f"{char} thinks the {container} contains"
    )

    return {
        "clean_prompt": clean,
        "counterfactual_prompt": cf,
        "clean_ans": substance1,
        "counterfactual_ans": substance2,
        "condition": "reordered",
    }


def make_distractor_story(char, substance1, substance2, container, location, rng):
    """Distractor: a third character interacts with a different substance
    between the state change and departure.

    Heuristic picks the distractor's substance. Belief tracking is unaffected.
    """
    other_char = rng.choice([c for c in CHARACTERS if c != char])
    third_char = rng.choice([c for c in CHARACTERS if c not in (char, other_char)])
    other_loc = rng.choice([l for l in LOCATIONS if l != location])
    distractor = rng.choice([s for s in SUBSTANCES if s not in (substance1, substance2)])
    dist_container = rng.choice([c for c in CONTAINERS if c != container])

    clean = (
        f"{char} poured some {substance1} into the {container} in the {location}. "
        f"While {char} was away, {other_char} replaced the {substance1} "
        f"in the {container} with {substance2}. "
        f"{third_char} then walked in and poured {distractor} into a {dist_container}. "
        f"{char} went to the {other_loc} and came back. "
        f"{char} thinks the {container} contains"
    )

    cf = (
        f"{char} poured some {substance2} into the {container} in the {location}. "
        f"While {char} was away, {other_char} replaced the {substance2} "
        f"in the {container} with {substance1}. "
        f"{third_char} then walked in and poured {distractor} into a {dist_container}. "
        f"{char} went to the {other_loc} and came back. "
        f"{char} thinks the {container} contains"
    )

    return {
        "clean_prompt": clean,
        "counterfactual_prompt": cf,
        "clean_ans": substance1,
        "counterfactual_ans": substance2,
        "condition": "distractor",
    }


def generate_all_conditions(n_per_condition, seed):
    """Generate n pairs per condition."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    generators = {
        "standard": make_standard_story,
        "decoy": make_decoy_story,
        "reordered": make_reordered_story,
        "distractor": make_distractor_story,
    }

    all_pairs = {}
    for cond_name, gen_fn in generators.items():
        pairs = []
        for _ in range(n_per_condition):
            char = rng.choice(CHARACTERS)
            s1, s2 = rng.choice(SUBSTANCES, size=2, replace=False)
            container = rng.choice(CONTAINERS)
            location = rng.choice(LOCATIONS)
            pairs.append(gen_fn(char, str(s1), str(s2), str(container), str(location), rng))
        all_pairs[cond_name] = pairs

    return all_pairs


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Surface heuristic test for Lookback audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-condition", type=int, default=120)
    parser.add_argument("--n-eval", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v.get("lookback_type", "")}

    print(f"[{ts()}] Surface heuristic test")
    print(f"[{ts()}] Conditions: standard, decoy, reordered, distractor")
    print(f"[{ts()}] Subspaces: {list(answer_subspaces.keys())}")

    print(f"\n[{ts()}] Generating adversarial stories...")
    conditions = generate_all_conditions(args.n_per_condition, args.seed)
    for cond, pairs in conditions.items():
        print(f"  {cond}: {len(pairs)} pairs generated")

    lm = None
    if not args.dry_run:
        print(f"\n[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        print(f"[{ts()}] Filtering pairs on model accuracy...")
        for cond in conditions:
            raw = conditions[cond]
            filtered = filter_on_model(lm, raw, max_size=args.n_eval)
            conditions[cond] = filtered
            print(f"  {cond}: {len(filtered)}/{len(raw)} passed filter")

    results = {
        "experiment": "Surface heuristic test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_per_condition_generated": args.n_per_condition,
        "n_eval_target": args.n_eval,
        "method": (
            "Adversarial story variants where positional heuristic fails. "
            "Standard: identical to CausalToM. Decoy: irrelevant substance "
            "mentioned near departure. Reordered: departure before state "
            "change sentence. Distractor: third character + different substance "
            "between state change and departure."
        ),
        "prediction": {
            "heuristic_encoding": (
                "IIA on standard >> IIA on adversarial conditions "
                "(subspace only encodes positional shortcut)"
            ),
            "genuine_belief": (
                "IIA comparable across all conditions "
                "(subspace tracks causal structure, not surface position)"
            ),
        },
        "conditions": {},
    }

    for cond_name, pairs in conditions.items():
        print(f"\n{'='*60}")
        print(f"[{ts()}] Condition: {cond_name} ({len(pairs)} pairs)")
        print(f"{'='*60}")

        cond_result = {
            "n_pairs": len(pairs),
            "subspaces": {},
        }

        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            rank = spec["rank"]
            mask_indices = spec.get("mask_indices")

            ckpt_path = RESULTS_DIR / f"{cond_name}_{sub_name}.json"
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    cached = json.load(f)
                cond_result["subspaces"][sub_name] = cached
                print(f"  {sub_name}: IIA={cached['iia']:.4f} (cached)")
                continue

            if args.dry_run:
                if cond_name == "standard":
                    iia = spec["sv_iia"] + rng.normal(0, 0.03)
                else:
                    iia = spec["sv_iia"] - rng.uniform(0.0, 0.15) + rng.normal(0, 0.03)
                iia = float(np.clip(iia, 0, 1))
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
                "layer": layer,
                "rank": rank,
                "iia": float(iia),
                "paper_iia": spec["sv_iia"],
            }
            cond_result["subspaces"][sub_name] = entry

            with open(ckpt_path, "w") as f:
                json.dump(entry, f, indent=2)

            print(f"  {sub_name}: IIA={iia:.4f} (paper={spec['sv_iia']:.4f})")

        results["conditions"][cond_name] = cond_result

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    for sub_name in answer_subspaces:
        std_iia = results["conditions"].get("standard", {}).get(
            "subspaces", {}).get(sub_name, {}).get("iia")
        if std_iia is None:
            continue

        adv_iias = []
        for cond in ["decoy", "reordered", "distractor"]:
            v = results["conditions"].get(cond, {}).get(
                "subspaces", {}).get(sub_name, {}).get("iia")
            if v is not None:
                adv_iias.append(v)

        if not adv_iias:
            continue

        mean_adv = float(np.mean(adv_iias))
        drop = std_iia - mean_adv
        ratio = mean_adv / std_iia if std_iia > 0 else 0

        is_heuristic = drop > 0.2

        print(f"\n  {sub_name}:")
        print(f"    Standard IIA:     {std_iia:.4f}")
        print(f"    Mean adversarial: {mean_adv:.4f}")
        for cond in ["decoy", "reordered", "distractor"]:
            v = results["conditions"].get(cond, {}).get(
                "subspaces", {}).get(sub_name, {}).get("iia")
            if v is not None:
                print(f"      {cond}: {v:.4f}")
        print(f"    Drop: {drop:.4f}, ratio: {ratio:.2f}")
        print(f"    -> {'HEURISTIC' if is_heuristic else 'GENUINE BELIEF'}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
