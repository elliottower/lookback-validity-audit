"""
Task-specificity controls (C4): discriminant validity via factorial design.

Tests whether the lookback pattern appears on non-belief-tracking tasks
with matched template structure. Four conditions in a 2x2 design:

  1. Echo/copy (no binding, no belief): pure surface retrieval.
     STRONG control — if lookback appears here, it is generic retrieval.
  2. Entity tracking (binding, no belief): requires state tracking but
     no mental states. WEAK control — authors' own prior work predicts
     the pattern may appear.
  3. True-belief matched (binding + belief, but belief = reality): the
     character SAW the move, so belief matches truth. Tests whether
     the lookback is sensitive to belief-reality divergence.
  4. False-belief (binding + belief): matches CausalToM — the character
     did NOT see the move, so belief diverges from reality.

Stories are programmatically generated as counterfactual pairs (120 per
condition). Each pair has a clean prompt and a counterfactual prompt with
a different answer, matching the interchange intervention protocol.

Limitation: binding_lookback subspaces (layers 34-36) are skipped for
custom stories because the CausalToM state token positions [155, 156,
167, 168] do not apply. Only answer_lookback subspaces (last-token
intervention at layers 38, 52, 53) are tested.

Usage:
    uv run python experiments/non_tom_control.py --dry-run --output results/DRYRUN_task_specificity/
    uv run python experiments/non_tom_control.py --output results/task_specificity/
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
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CHARACTERS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
              "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Paul"]
OBJECTS = ["book", "key", "ball", "cup", "hat", "ring", "phone", "wallet",
           "pen", "watch", "coin", "letter", "toy", "scarf", "bag", "box"]
LOCATIONS = ["shelf", "table", "drawer", "basket", "closet", "desk",
             "counter", "cabinet", "chair", "bench", "floor", "windowsill"]
CONTAINERS = ["box", "bag", "jar", "bottle", "basket", "bucket", "pouch", "case"]
SUBSTANCES = ["water", "sand", "rice", "flour", "salt", "sugar", "beans", "marbles"]


def generate_echo_copy_pairs(n, rng):
    """Pure retrieval, no binding (strong control).

    Clean: "X said the obj is in loc1. What did X say?" -> loc1
    Counterfactual: "X said the obj is in loc2. What did X say?" -> loc2
    IIA test: swapping the subspace component should make the model
    answer loc2 instead of loc1 (if the subspace encodes the stated location).
    """
    pairs = []
    for _ in range(n):
        c1 = rng.choice(CHARACTERS)
        obj = rng.choice(OBJECTS)
        loc1, loc2 = rng.choice(LOCATIONS, size=2, replace=False)

        pairs.append({
            "clean_prompt": (
                f"{c1} said: 'the {obj} is in the {loc1}.' "
                f"What did {c1} say the {obj} is in?"
            ),
            "counterfactual_prompt": (
                f"{c1} said: 'the {obj} is in the {loc2}.' "
                f"What did {c1} say the {obj} is in?"
            ),
            "clean_ans": loc1,
            "counterfactual_ans": loc2,
            "condition": "echo_copy",
            "has_binding": False,
            "has_belief": False,
        })
    return pairs


def generate_entity_tracking_pairs(n, rng):
    """Binding without belief attribution (weak control).

    Clean: obj moved to loc2. "Where is the obj?" -> loc2
    Counterfactual: obj moved to loc3. "Where is the obj?" -> loc3
    """
    pairs = []
    for _ in range(n):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        obj = rng.choice(OBJECTS)
        loc1, loc2, loc3 = rng.choice(LOCATIONS, size=3, replace=False)

        pairs.append({
            "clean_prompt": (
                f"{c1} put the {obj} in the {loc1}. "
                f"{c2} moved the {obj} to the {loc2}. "
                f"Where is the {obj}?"
            ),
            "counterfactual_prompt": (
                f"{c1} put the {obj} in the {loc1}. "
                f"{c2} moved the {obj} to the {loc3}. "
                f"Where is the {obj}?"
            ),
            "clean_ans": loc2,
            "counterfactual_ans": loc3,
            "condition": "entity_tracking",
            "has_binding": True,
            "has_belief": False,
        })
    return pairs


def generate_true_belief_pairs(n, rng):
    """Binding + belief, but character SAW the move (belief = reality).

    Clean: c1 watches c2 refill cont2 with s1. "What does c1 believe?" -> s1
    Counterfactual: c1 watches c2 refill cont2 with s3. "What does c1 believe?" -> s3
    """
    pairs = []
    for _ in range(n):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        cont1, cont2 = rng.choice(CONTAINERS, size=2, replace=False)
        s1, s2, s3 = rng.choice(SUBSTANCES, size=3, replace=False)
        setting = rng.choice(["kitchen", "workshop", "laboratory", "office"])

        pairs.append({
            "clean_prompt": (
                f"{c1} and {c2} are working in a {setting}. "
                f"{c1} grabs an opaque {cont1} and fills it with {s1}. "
                f"{c2} grabs another opaque {cont2} and fills it with {s2}. "
                f"{c1} watches as {c2} empties {cont2} and refills it with {s1}. "
                f"What does {c1} believe {cont2} contains?"
            ),
            "counterfactual_prompt": (
                f"{c1} and {c2} are working in a {setting}. "
                f"{c1} grabs an opaque {cont1} and fills it with {s1}. "
                f"{c2} grabs another opaque {cont2} and fills it with {s2}. "
                f"{c1} watches as {c2} empties {cont2} and refills it with {s3}. "
                f"What does {c1} believe {cont2} contains?"
            ),
            "clean_ans": s1,
            "counterfactual_ans": s3,
            "condition": "true_belief",
            "has_binding": True,
            "has_belief": True,
            "belief_matches_reality": True,
        })
    return pairs


def generate_false_belief_pairs(n, rng):
    """False-belief: character did NOT see the change (positive control).

    Clean: c1 absent when c2 refills cont2 with s1. c1 believes s2. -> s2
    Counterfactual: c1 absent when c2 refills cont2 with s3. c1 believes s2. -> s2
    But the counterfactual changes the *original* fill so c1's belief changes:
    Clean: c2 originally fills with s2, c1 believes s2
    CF: c2 originally fills with s3, c1 believes s3
    """
    pairs = []
    for _ in range(n):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        cont1, cont2 = rng.choice(CONTAINERS, size=2, replace=False)
        s1, s2, s3 = rng.choice(SUBSTANCES, size=3, replace=False)
        setting = rng.choice(["kitchen", "workshop", "laboratory", "office"])

        pairs.append({
            "clean_prompt": (
                f"{c1} and {c2} are working in a {setting}. "
                f"{c1} grabs an opaque {cont1} and fills it with {s1}. "
                f"{c2} grabs another opaque {cont2} and fills it with {s2}. "
                f"{c1} leaves the {setting}. "
                f"{c2} empties {cont2} and refills it with {s1}. "
                f"What does {c1} believe {cont2} contains?"
            ),
            "counterfactual_prompt": (
                f"{c1} and {c2} are working in a {setting}. "
                f"{c1} grabs an opaque {cont1} and fills it with {s1}. "
                f"{c2} grabs another opaque {cont2} and fills it with {s3}. "
                f"{c1} leaves the {setting}. "
                f"{c2} empties {cont2} and refills it with {s1}. "
                f"What does {c1} believe {cont2} contains?"
            ),
            "clean_ans": s2,
            "counterfactual_ans": s3,
            "condition": "false_belief",
            "has_binding": True,
            "has_belief": True,
            "belief_matches_reality": False,
        })
    return pairs


def compute_effect_size(tom_iias, control_iias):
    """Cohen's d between ToM and control IIA distributions."""
    n1, n2 = len(tom_iias), len(control_iias)
    s1, s2 = np.std(tom_iias, ddof=1), np.std(control_iias, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(tom_iias) - np.mean(control_iias)) / pooled_std


CONDITION_GENERATORS = {
    "echo_copy": ("Pure retrieval, no binding (strong control)", generate_echo_copy_pairs),
    "entity_tracking": ("Binding without belief (weak control)", generate_entity_tracking_pairs),
    "true_belief": ("Binding + belief, character observed (belief = reality)", generate_true_belief_pairs),
    "false_belief": ("Binding + belief, character absent (belief != reality, positive control)", generate_false_belief_pairs),
}


def main():
    parser = argparse.ArgumentParser(description="Task-specificity controls for Lookback audit")
    parser.add_argument("--n-stories", type=int, default=120,
                        help="Stories per control condition (>= 100 for power)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/task_specificity/")
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

    conditions = {}
    for cond_name, (desc, generator) in CONDITION_GENERATORS.items():
        conditions[cond_name] = {
            "description": desc,
            "pairs": generator(args.n_stories, rng),
        }

    summary = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_stories_per_condition": args.n_stories,
        "design": "2x2 factorial: binding (yes/no) x belief (yes/no)",
        "conditions": list(CONDITION_GENERATORS.keys()),
        "limitation": (
            "Binding_lookback subspaces (L34-36) skipped: custom stories have "
            "different token lengths than CausalToM, so the fixed state positions "
            "[155, 156, 167, 168] do not apply. Only answer_lookback subspaces "
            "(last-token intervention at L38, L52, L53) are tested."
        ),
        "hypothesis": {
            "echo_copy": "IIA ~ chance (lookback absent)",
            "entity_tracking": "IIA may be elevated (binding without belief)",
            "true_belief": "IIA elevated but lower than false_belief if mechanism is belief-specific",
            "false_belief": "IIA matches CausalToM (positive control)",
        },
        "results_by_condition": {},
    }

    for cond_name, cond_data in conditions.items():
        desc = cond_data["description"]
        pairs = cond_data["pairs"]

        print(f"\n{'='*60}")
        print(f"[{ts()}] Condition: {cond_name}")
        print(f"  {desc}")
        print(f"  N pairs: {len(pairs)}")
        print(f"{'='*60}")

        cond_results = {"description": desc, "n_pairs": len(pairs), "subspace_iias": {}}

        for sub_name, sub_spec in tqdm(subspaces.items(), desc=f"{cond_name} subspaces"):
            layer = sub_spec["layer"]
            rank = sub_spec["rank"]
            lookback_type = sub_spec["lookback_type"]

            if "binding" in lookback_type:
                print(f"  SKIP {sub_name}: binding subspace, custom stories lack CausalToM positions")
                cond_results["subspace_iias"][sub_name] = {
                    "iia_mean": None,
                    "skipped": True,
                    "skip_reason": "binding subspace requires CausalToM state token positions [155, 156, 167, 168]",
                    "layer": layer,
                    "rank": rank,
                    "lookback_type": lookback_type,
                    "concept": sub_spec["concept"],
                    "original_sv_iia": sub_spec["sv_iia"],
                }
                continue

            if args.dry_run:
                if cond_name == "echo_copy":
                    per_story = rng.random(len(pairs)) * 0.3
                elif cond_name == "entity_tracking":
                    per_story = rng.random(len(pairs)) * 0.5 + 0.2
                elif cond_name == "true_belief":
                    per_story = rng.random(len(pairs)) * 0.4 + 0.3
                else:
                    per_story = rng.random(len(pairs)) * 0.3 + 0.5
                mean_iia = float(np.mean(per_story))
            else:
                vec_type = "last_token"
                svd_basis = load_svd_basis(layer, vec_type)
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: SVD basis not found (run extract_svd.py first)")
                    cond_results["subspace_iias"][sub_name] = {
                        "iia_mean": None,
                        "skipped": True,
                        "skip_reason": "SVD basis not found",
                        "layer": layer,
                        "rank": rank,
                        "lookback_type": lookback_type,
                        "concept": sub_spec["concept"],
                        "original_sv_iia": sub_spec["sv_iia"],
                    }
                    continue

                selected = np.arange(rank)
                projection = build_projection_matrix(svd_basis, selected)
                mean_iia = compute_iia_answer_flex(lm, pairs, layer, projection)

            cond_results["subspace_iias"][sub_name] = {
                "iia_mean": mean_iia,
                "skipped": False,
                "layer": layer,
                "rank": rank,
                "lookback_type": lookback_type,
                "concept": sub_spec["concept"],
                "original_sv_iia": sub_spec["sv_iia"],
            }

            print(f"  {sub_name}: IIA={mean_iia:.4f} (original={sub_spec['sv_iia']:.4f})")

        summary["results_by_condition"][cond_name] = cond_results

    print(f"\n{'='*60}")
    print("Effect sizes (delta IIA) vs false_belief (answer subspaces only):")
    print(f"{'='*60}")

    effect_sizes = {}
    fb_results = summary["results_by_condition"]["false_belief"]["subspace_iias"]
    for cond_name in ["echo_copy", "entity_tracking", "true_belief"]:
        cond_iias = summary["results_by_condition"][cond_name]["subspace_iias"]
        effect_sizes[cond_name] = {}
        for sub_name in subspaces:
            fb_entry = fb_results[sub_name]
            ctrl_entry = cond_iias[sub_name]
            if fb_entry.get("skipped") or ctrl_entry.get("skipped"):
                continue
            fb_iia = fb_entry["iia_mean"]
            ctrl_iia = ctrl_entry["iia_mean"]
            d = fb_iia - ctrl_iia
            effect_sizes[cond_name][sub_name] = d
            print(f"  {cond_name} vs false_belief @ {sub_name}: delta={d:.4f}")

    summary["effect_sizes_vs_false_belief"] = effect_sizes

    for cond_name, cond_results in summary["results_by_condition"].items():
        cond_path = output_dir / f"{cond_name}.json"
        with open(cond_path, "w") as f:
            json.dump({"timestamp": summary["timestamp"], **cond_results}, f, indent=2)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
