"""
Non-ToM retrieval control (C4): discriminant validity test.

Tests whether the lookback pattern appears on non-belief-tracking tasks
with matched template structure. If the pattern appears at comparable
strength, it is generic attention-mediated retrieval.

Two control conditions:
  1. Entity tracking (no mental states): "X put obj in loc1. Y moved obj to loc2.
     Where is obj?" — requires binding but no belief attribution. If lookback
     appears here, that is consistent with the authors' own entity-tracking
     prior work, so this is a *weak* control (included for completeness).
  2. Echo/copy (no binding): "X said: the obj is in loc1. What did X say the
     obj is in?" — surface form matches but requires no state binding across
     characters. This is the *strong* control. If lookback appears here, the
     pattern is generic retrieval.

Stories are programmatically generated (100+ per condition) to ensure
statistical power.

Usage:
    python non_tom_control.py --output results/non_tom_control.json
    python non_tom_control.py --dry-run --output results/DRYRUN_non_tom_control.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CHARACTERS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
              "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Paul"]
OBJECTS = ["book", "key", "ball", "cup", "hat", "ring", "phone", "wallet",
           "pen", "watch", "coin", "letter", "toy", "scarf", "bag", "box"]
LOCATIONS = ["shelf", "table", "drawer", "basket", "closet", "desk",
             "counter", "cabinet", "chair", "bench", "floor", "windowsill"]


def generate_entity_tracking_stories(n: int, rng: np.random.Generator) -> list:
    """Generate entity-tracking stories with matched structure but no mental states.

    Template: "{char1} put the {obj} in the {loc1}. {char2} moved the {obj}
    to the {loc2}. Where is the {obj}?"
    Answer: loc2

    This requires binding (track which character moved what where) but no
    belief attribution. It is a WEAK control — the Lookback paper's own
    entity-tracking work predicts the pattern may appear here.
    """
    stories = []
    for _ in range(n):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        obj = rng.choice(OBJECTS)
        loc1, loc2 = rng.choice(LOCATIONS, size=2, replace=False)
        stories.append({
            "text": (
                f"{c1} put the {obj} in the {loc1}. "
                f"{c2} moved the {obj} to the {loc2}. "
                f"Where is the {obj}?"
            ),
            "answer": loc2,
            "char1": c1,
            "char2": c2,
            "obj": obj,
            "loc1": loc1,
            "loc2": loc2,
        })
    return stories


def generate_echo_copy_stories(n: int, rng: np.random.Generator) -> list:
    """Generate echo/copy stories that require no state binding.

    Template: "{char1} said: 'the {obj} is in the {loc1}.'
    What did {char1} say the {obj} is in?"
    Answer: loc1

    Surface form includes characters, objects, and locations (matching
    CausalToM token types), but the task is pure retrieval — no tracking
    of who knows what or where objects actually are.
    """
    stories = []
    for _ in range(n):
        c1 = rng.choice(CHARACTERS)
        obj = rng.choice(OBJECTS)
        loc1 = rng.choice(LOCATIONS)
        stories.append({
            "text": (
                f"{c1} said: 'the {obj} is in the {loc1}.' "
                f"What did {c1} say the {obj} is in?"
            ),
            "answer": loc1,
            "char1": c1,
            "obj": obj,
            "loc1": loc1,
        })
    return stories


# Per-layer subspace specs from released results.
# Ranks are placeholders — extract from https://github.com/Nix07/belief_tracking/results/
LOOKBACK_SUBSPACES = {
    "binding_L35": {"layer": 35, "rank": 7},
    "binding_L36": {"layer": 36, "rank": 5},
    "binding_L38": {"layer": 38, "rank": 3},
    "answer_L52": {"layer": 52, "rank": 18},
    "answer_L53": {"layer": 53, "rank": 10},
    "answer_L54": {"layer": 54, "rank": 8},
}


def main():
    parser = argparse.ArgumentParser(description="Non-ToM retrieval control for Lookback audit")
    parser.add_argument("--n-stories", type=int, default=120,
                        help="Stories per control condition (>= 100 for power)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/non_tom_control.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    control_tasks = {
        "entity_tracking": {
            "description": "Binding without belief attribution (weak control)",
            "stories": generate_entity_tracking_stories(args.n_stories, rng),
        },
        "echo_copy": {
            "description": "Pure retrieval, no binding (strong control)",
            "stories": generate_echo_copy_stories(args.n_stories, rng),
        },
    }

    results = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_stories_per_condition": args.n_stories,
        "hypothesis": "IIA on non-ToM tasks < 0.5 * IIA on ToM tasks",
        "test": "Permutation test comparing ToM vs non-ToM IIA distributions",
        "control_tasks": {},
    }

    for task_name, task_spec in control_tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Description: {task_spec['description']}")
        print(f"N stories: {len(task_spec['stories'])}")
        print(f"{'='*60}")

        task_results = {
            "description": task_spec["description"],
            "n_stories": len(task_spec["stories"]),
            "subspace_iias": {},
        }

        for sub_name, sub_spec in tqdm(LOOKBACK_SUBSPACES.items(), desc="Subspaces"):
            print(f"  Testing {sub_name} (rank={sub_spec['rank']}, layer={sub_spec['layer']})")

            if args.dry_run:
                per_story_iias = rng.random(len(task_spec["stories"])) * 0.4
                mean_iia = float(np.mean(per_story_iias))
            else:
                raise NotImplementedError(
                    "Full IIA computation requires model loading and NNsight hooks. "
                    "Run interchange intervention on each story using the identified "
                    "subspace at the specified layers."
                )

            task_results["subspace_iias"][sub_name] = {
                "iia": mean_iia,
                "layer": sub_spec["layer"],
                "rank": sub_spec["rank"],
            }

            print(f"    IIA: {mean_iia:.4f}")

        results["control_tasks"][task_name] = task_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
