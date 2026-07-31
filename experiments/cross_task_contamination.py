"""
Cross-task contamination test: are the belief-tracking subspaces generic
answer pointers or belief-specific circuits?

Applies the CausalToM-derived subspace projections to completely unrelated
tasks (factual recall, arithmetic, color/property association). If IIA is
high on these non-belief tasks, the subspace encodes "the answer to any
question" — a generic answer mechanism, not a belief tracker.

Three task types as counterfactual pairs:
  1. Factual recall: capitals, elements, animal categories
  2. Simple arithmetic: addition/subtraction with single-token answers
  3. Color/property association: common object-property pairings

Binding subspaces (L34-36) are skipped — custom prompts lack CausalToM
state token positions. Only answer subspaces (L38, L52, L53) are tested
at the last token position.

Prediction:
  IIA ~ chance (0.0-0.2): subspace is belief-specific (SUPPORTS paper)
  IIA > 0.5: subspace is a generic answer mechanism (WEAKENS paper)

Usage:
    uv run python experiments/cross_task_contamination.py --dry-run
    uv run python experiments/cross_task_contamination.py --output results/cross_task/
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Factual recall pairs ──
# Each entry: (clean_prompt, clean_ans, cf_prompt, cf_ans)
# Answers MUST be single tokens for filter_on_model to work.

CAPITAL_FACTS = [
    ("The capital of France is", "Paris", "The capital of Japan is", "Tokyo"),
    ("The capital of Germany is", "Berlin", "The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid", "The capital of Egypt is", "Cairo"),
    ("The capital of Canada is", "Ottawa", "The capital of Mexico is", "Mexico"),
    ("The capital of Australia is", "Canberra", "The capital of Brazil is", "Bras"),
    ("The capital of Russia is", "Moscow", "The capital of China is", "Beijing"),
    ("The capital of India is", "Delhi", "The capital of Turkey is", "Ankara"),
    ("The capital of Poland is", "Warsaw", "The capital of Sweden is", "Stockholm"),
    ("The capital of Norway is", "Oslo", "The capital of Denmark is", "Copenhagen"),
    ("The capital of Greece is", "Athens", "The capital of Portugal is", "Lisbon"),
    ("The capital of Austria is", "Vienna", "The capital of Ireland is", "Dublin"),
    ("The capital of Finland is", "Helsinki", "The capital of Belgium is", "Brussels"),
    ("The capital of Argentina is", "Buenos", "The capital of Chile is", "Santiago"),
    ("The capital of Thailand is", "Bangkok", "The capital of Vietnam is", "Han"),
]

ELEMENT_FACTS = [
    ("The chemical symbol for gold is", "Au", "The chemical symbol for silver is", "Ag"),
    ("The chemical symbol for iron is", "Fe", "The chemical symbol for copper is", "Cu"),
    ("The chemical symbol for sodium is", "Na", "The chemical symbol for potassium is", "K"),
    ("The chemical symbol for hydrogen is", "H", "The chemical symbol for oxygen is", "O"),
    ("The chemical symbol for carbon is", "C", "The chemical symbol for nitrogen is", "N"),
    ("The chemical symbol for lead is", "Pb", "The chemical symbol for tin is", "Sn"),
    ("The chemical symbol for mercury is", "Hg", "The chemical symbol for zinc is", "Zn"),
    ("The chemical symbol for calcium is", "Ca", "The chemical symbol for magnesium is", "Mg"),
]

ANIMAL_FACTS = [
    ("A group of lions is called a", "pride", "A group of wolves is called a", "pack"),
    ("A baby dog is called a", "puppy", "A baby cat is called a", "kitten"),
    ("The largest land animal is the", "elephant", "The largest marine animal is the", "whale"),
    ("A female horse is called a", "mare", "A male horse is called a", "stall"),
    ("Bees produce", "honey", "Cows produce", "milk"),
    ("Silk is produced by", "silk", "Wool is produced by", "sheep"),
    ("The fastest land animal is the", "che", "The tallest animal is the", "g"),
]

MISC_FACTS = [
    ("Water freezes at", "32", "Water boils at", "212"),
    ("The number of continents is", "seven", "The number of oceans is", "five"),
    ("The Earth orbits the", "Sun", "The Moon orbits the", "Earth"),
    ("There are 60 seconds in a", "minute", "There are 60 minutes in an", "hour"),
    ("The opposite of hot is", "cold", "The opposite of light is", "dark"),
    ("The opposite of up is", "down", "The opposite of left is", "right"),
    ("The square root of 4 is", "2", "The square root of 9 is", "3"),
    ("The square root of 16 is", "4", "The square root of 25 is", "5"),
    ("The square root of 36 is", "6", "The square root of 49 is", "7"),
    ("The square root of 64 is", "8", "The square root of 81 is", "9"),
    ("A triangle has", "three", "A square has", "four"),
]


def generate_factual_pairs():
    """Generate counterfactual pairs from factual knowledge templates."""
    all_facts = CAPITAL_FACTS + ELEMENT_FACTS + ANIMAL_FACTS + MISC_FACTS
    pairs = []
    for clean_p, clean_a, cf_p, cf_a in all_facts:
        pairs.append({
            "clean_prompt": clean_p,
            "counterfactual_prompt": cf_p,
            "clean_ans": clean_a,
            "counterfactual_ans": cf_a,
            "task_type": "factual_recall",
        })
        pairs.append({
            "clean_prompt": cf_p,
            "counterfactual_prompt": clean_p,
            "clean_ans": cf_a,
            "counterfactual_ans": clean_a,
            "task_type": "factual_recall",
        })
    return pairs


# ── Arithmetic pairs ──

def generate_arithmetic_pairs(n, rng):
    """Generate addition/subtraction pairs with single-token answers (0-99)."""
    pairs = []
    for _ in range(n):
        a1, b1 = int(rng.integers(10, 50)), int(rng.integers(10, 50))
        a2, b2 = int(rng.integers(10, 50)), int(rng.integers(10, 50))
        s1, s2 = a1 + b1, a2 + b2
        if s1 == s2:
            b2 += 1
            s2 = a2 + b2

        pairs.append({
            "clean_prompt": f"What is {a1} + {b1}? The answer is",
            "counterfactual_prompt": f"What is {a2} + {b2}? The answer is",
            "clean_ans": str(s1),
            "counterfactual_ans": str(s2),
            "task_type": "arithmetic",
        })
    return pairs


# ── Color/property pairs ──

COLOR_PROPS = [
    ("Grass is typically the color", "green", "The sky is typically the color", "blue"),
    ("Snow is typically the color", "white", "Coal is typically the color", "black"),
    ("A ripe banana is the color", "yellow", "A ripe tomato is the color", "red"),
    ("The sun appears to be the color", "yellow", "A clear sky is the color", "blue"),
    ("Chocolate is typically the color", "brown", "A lemon is the color", "yellow"),
    ("An orange fruit is the color", "orange", "A blueberry is the color", "blue"),
    ("Milk is the color", "white", "Coffee is the color", "brown"),
    ("A firetruck is typically the color", "red", "A school bus is typically the color", "yellow"),
    ("Gold is the color", "gold", "Silver is the color", "silver"),
    ("A stop sign is the color", "red", "A go traffic light is the color", "green"),
    ("Wood is typically the color", "brown", "Paper is typically the color", "white"),
    ("Rust is the color", "red", "Moss is the color", "green"),
]

PROPERTY_PAIRS = [
    ("Ice is", "cold", "Fire is", "hot"),
    ("A rock is", "hard", "A pillow is", "soft"),
    ("A feather is", "light", "A boulder is", "heavy"),
    ("Sugar tastes", "sweet", "A lemon tastes", "sour"),
    ("Honey is", "sweet", "Vinegar is", "sour"),
    ("The desert is", "dry", "The ocean is", "wet"),
    ("A snail is", "slow", "A cheetah is", "fast"),
    ("A whisper is", "quiet", "Thunder is", "loud"),
    ("Summer is", "warm", "Winter is", "cold"),
    ("Glass is", "transparent", "A wall is", "op"),
    ("The sun is", "bright", "A cave is", "dark"),
    ("A highway is", "wide", "An alley is", "narrow"),
    ("A razor is", "sharp", "A spoon is", "dull"),
    ("Sandpaper is", "rough", "Silk is", "smooth"),
    ("Steel is", "strong", "Paper is", "weak"),
    ("Day is", "bright", "Night is", "dark"),
]


def generate_color_property_pairs():
    """Generate color/property association pairs."""
    all_props = COLOR_PROPS + PROPERTY_PAIRS
    pairs = []
    for clean_p, clean_a, cf_p, cf_a in all_props:
        pairs.append({
            "clean_prompt": clean_p,
            "counterfactual_prompt": cf_p,
            "clean_ans": clean_a,
            "counterfactual_ans": cf_a,
            "task_type": "color_property",
        })
        pairs.append({
            "clean_prompt": cf_p,
            "counterfactual_prompt": clean_p,
            "clean_ans": cf_a,
            "counterfactual_ans": clean_a,
            "task_type": "color_property",
        })
    return pairs


TASK_GENERATORS = {
    "factual_recall": ("Factual recall (capitals, elements, animals, misc)", generate_factual_pairs),
    "arithmetic": ("Simple arithmetic (addition, single-token answers)", None),
    "color_property": ("Color/property associations", generate_color_property_pairs),
}


def main():
    parser = argparse.ArgumentParser(
        description="Cross-task contamination test for Lookback audit")
    parser.add_argument("--n-arithmetic", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/cross_task/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    subspaces = load_subspace_specs()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lm = None
    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

    task_pairs = {
        "factual_recall": generate_factual_pairs(),
        "arithmetic": generate_arithmetic_pairs(args.n_arithmetic, rng),
        "color_property": generate_color_property_pairs(),
    }

    summary = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "method": "transfer_test",
        "method_note": (
            "Applies CausalToM-derived subspace projections to non-belief tasks. "
            "Binding subspaces skipped (custom prompts lack CausalToM state positions). "
            "High IIA on non-belief tasks would indicate the subspace is a generic "
            "answer pointer, not a belief-tracking mechanism."
        ),
        "prediction": {
            "supports_paper": "IIA ~ chance (0.0-0.2) on all non-belief tasks",
            "weakens_paper": "IIA > 0.5 on any non-belief task",
        },
        "tasks": {},
    }

    for task_name, pairs in task_pairs.items():
        desc = TASK_GENERATORS[task_name][0] if task_name in TASK_GENERATORS else task_name

        print(f"\n{'='*60}")
        print(f"[{ts()}] Task: {task_name}")
        print(f"  {desc}")
        print(f"  N raw pairs: {len(pairs)}")
        print(f"{'='*60}")

        if not args.dry_run:
            print(f"[{ts()}] Filtering pairs on model accuracy...")
            pairs = filter_on_model(lm, pairs, max_size=80)
            print(f"[{ts()}] {len(pairs)} pairs passed filter")

        task_result = {
            "description": desc,
            "n_pairs": len(pairs),
            "subspace_iias": {},
        }

        for sub_name, sub_spec in tqdm(subspaces.items(), desc=f"{task_name} subspaces"):
            layer = sub_spec["layer"]
            rank = sub_spec["rank"]
            lookback_type = sub_spec["lookback_type"]

            if "binding" in lookback_type:
                task_result["subspace_iias"][sub_name] = {
                    "iia": None,
                    "skipped": True,
                    "skip_reason": "binding subspace requires CausalToM state token positions",
                    "layer": layer,
                    "rank": rank,
                }
                continue

            if args.dry_run:
                iia = float(rng.random() * 0.15)
            else:
                vec_type = "last_token"
                svd_basis = load_svd_basis(layer, vec_type)
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    task_result["subspace_iias"][sub_name] = {
                        "iia": None,
                        "skipped": True,
                        "skip_reason": "SVD basis not found",
                        "layer": layer,
                        "rank": rank,
                    }
                    continue

                mask_indices = sub_spec.get("mask_indices")
                if mask_indices is not None:
                    proj = build_projection_matrix(svd_basis, mask_indices)
                else:
                    proj = build_projection_matrix(svd_basis, np.arange(rank))

                iia = compute_iia_answer_flex(lm, pairs, layer, proj)

            task_result["subspace_iias"][sub_name] = {
                "iia": float(iia),
                "skipped": False,
                "original_iia": sub_spec["sv_iia"],
                "layer": layer,
                "rank": rank,
                "lookback_type": lookback_type,
            }

            print(f"  {sub_name}: IIA={iia:.4f} (belief IIA={sub_spec['sv_iia']:.4f})")

        summary["tasks"][task_name] = task_result

        task_path = output_dir / f"{task_name}.json"
        with open(task_path, "w") as f:
            json.dump({"timestamp": summary["timestamp"], **task_result}, f, indent=2)

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    any_high = False
    for task_name, task_result in summary["tasks"].items():
        answer_iias = [
            v["iia"] for v in task_result["subspace_iias"].values()
            if not v.get("skipped") and v["iia"] is not None
        ]
        if not answer_iias:
            continue
        mean_iia = float(np.mean(answer_iias))
        max_iia = float(np.max(answer_iias))

        if max_iia > 0.5:
            any_high = True
            print(f"  {task_name}: mean={mean_iia:.4f}, max={max_iia:.4f} — HIGH (WEAKENS paper)")
        elif max_iia > 0.2:
            print(f"  {task_name}: mean={mean_iia:.4f}, max={max_iia:.4f} — MODERATE (inconclusive)")
        else:
            print(f"  {task_name}: mean={mean_iia:.4f}, max={max_iia:.4f} — LOW (supports paper)")

    summary["verdict"] = (
        "Generic answer pointer: subspace transfers to non-belief tasks"
        if any_high else
        "Belief-specific: subspace does NOT transfer to non-belief tasks"
    )
    print(f"\nVerdict: {summary['verdict']}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Results saved to {summary_path}")


if __name__ == "__main__":
    main()
