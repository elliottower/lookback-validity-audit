"""
Shuffled-label subspace control: the matched-capacity trained baseline.

The random subspace baseline tests "are k optimized singular vectors better
than k random ones?" — the answer is trivially yes. This script tests the
informative question: "does a subspace selected on *real belief labels*
beat one selected on *shuffled belief labels*?"

If the real-label subspace beats shuffled-label subspaces, the mask selection
captures belief-specific structure. If shuffled-label subspaces achieve
comparable IIA, the procedure finds generic high-variance directions.

The Lookback paper's actual method (from https://github.com/Nix07/belief_tracking):
    1. Collect residual stream activations at target token positions
    2. SVD → 500 singular vectors (capped)
    3. Learn binary mask over singular vectors via Adam (lr=0.1) + L1 (λ=0.1)
    4. Round mask to {0,1}

Protocol for this control:
    1. Run SVD + mask selection on original CausalToM labels → real_iia
    2. For each of K permutations, shuffle story-pair assignments,
       re-run SVD + mask with identical hyperparameters → shuffled_iias[k]
    3. p-value = (rank + 1) / (K + 1)

Cost estimate (K=100, each mask fit = 1 epoch of Adam on 80 stories):
    Much cheaper than full DCM optimization — ~1 min per fit on A100, so ~2 GPU-hours total.

Usage:
    python shuffled_label_control.py --n-shuffles 100 --output results/shuffled_label.json
    python shuffled_label_control.py --n-shuffles 5 --dry-run --output results/DRYRUN_shuffled_label.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def shuffle_story_pairs(stories, rng):
    """Shuffle which story's belief state serves as source for interchange.

    Preserves the marginal distribution of belief states but breaks
    the association between story content and belief labels.
    """
    shuffled = list(stories)
    rng.shuffle(shuffled)
    return shuffled


def run_svd_mask_selection(model, stories, layer, n_components, l1_weight, lr, device, label="real"):
    """Run the Lookback paper's subspace identification procedure.

    Steps:
    1. Collect residual stream at layer for all stories at target token positions
    2. SVD of activation matrix → top n_components singular vectors
    3. Learn binary mask via Adam + L1 to maximize IIA
    4. Round mask, return selected basis and achieved IIA

    Source: https://github.com/Nix07/belief_tracking
    Key files: src/patching_utils.py, run_single_layer_patching_exps.py
    """
    raise NotImplementedError(
        "Requires running the Lookback paper's code. "
        "Clone https://github.com/Nix07/belief_tracking and call their "
        "SVD + mask selection pipeline. The SVD bases may need to be "
        "requested from the authors (svd/ directory not committed)."
    )


def main():
    parser = argparse.ArgumentParser(description="Shuffled-label subspace control for Lookback audit")
    parser.add_argument("--n-shuffles", type=int, default=100,
                        help="Number of shuffled-label fits (100 gives p-floor=0.0099)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/shuffled_label.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Per-layer specs matching the Lookback paper.
    # Ranks are placeholders — extract from their released results.
    subspaces_per_layer = {
        "binding_L35": {"layer": 35, "l1_weight": 0.1},
        "binding_L36": {"layer": 36, "l1_weight": 0.1},
        "binding_L38": {"layer": 38, "l1_weight": 0.1},
        "answer_L52": {"layer": 52, "l1_weight": 0.1},
        "answer_L53": {"layer": 53, "l1_weight": 0.1},
        "answer_L54": {"layer": 54, "l1_weight": 0.1},
    }
    n_svd_components = 500
    lr = 0.1

    results = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_shuffles": args.n_shuffles,
        "method": "SVD + L1-regularized binary mask (matching Lookback paper)",
        "n_svd_components": n_svd_components,
        "lr": lr,
        "subspaces": {},
    }

    for name, spec in subspaces_per_layer.items():
        print(f"\n{'='*60}")
        print(f"Shuffled-label control for {name} (layer={spec['layer']})")
        print(f"{'='*60}")

        if args.dry_run:
            real_iia = 0.75 + rng.random() * 0.2
        else:
            raise NotImplementedError("Run SVD + mask on real labels first.")

        shuffled_iias = []
        for k in tqdm(range(args.n_shuffles), desc=f"Shuffled fits for {name}"):
            if args.dry_run:
                shuffled_iia = 0.3 + rng.random() * 0.4
            else:
                raise NotImplementedError("Run SVD + mask on shuffled labels.")

            shuffled_iias.append(shuffled_iia)

        shuffled_arr = np.array(shuffled_iias)
        rank = int(np.sum(shuffled_arr >= real_iia))
        p_value = (rank + 1) / (args.n_shuffles + 1)

        results["subspaces"][name] = {
            "layer": spec["layer"],
            "l1_weight": spec["l1_weight"],
            "real_iia": float(real_iia),
            "shuffled_iia_mean": float(np.mean(shuffled_arr)),
            "shuffled_iia_std": float(np.std(shuffled_arr, ddof=1)),
            "shuffled_iia_max": float(np.max(shuffled_arr)),
            "shuffled_iia_min": float(np.min(shuffled_arr)),
            "shuffled_iia_p95": float(np.quantile(shuffled_arr, 0.95)),
            "rank": rank,
            "p_value": float(p_value),
            "p_value_formula": "(rank + 1) / (n + 1)",
            "significant_at_001": p_value < 0.01,
            "interpretation": (
                "Subspace captures belief-specific structure"
                if p_value < 0.01
                else "Subspace finds generic high-variance directions, not belief structure"
            ),
        }

        print(f"Real IIA:         {real_iia:.4f}")
        print(f"Shuffled mean:    {float(np.mean(shuffled_arr)):.4f} +/- {float(np.std(shuffled_arr, ddof=1)):.4f}")
        print(f"Shuffled max:     {float(np.max(shuffled_arr)):.4f}")
        print(f"Rank:             {rank}/{args.n_shuffles}")
        print(f"p-value:          {p_value:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
