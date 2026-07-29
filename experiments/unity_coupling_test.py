"""
Unity coupling test (I6): do the binding and answer sub-mechanisms interact
non-additively, or are they independent processes sharing a label?

Protocol:
  (a) Baseline: no ablation, measure task IIA via interchange intervention.
  (b) Ablate binding alone (mean-ablate binding subspace at layers 33-38),
      measure task IIA.
  (c) Ablate answer alone, measure task IIA.
  (d) Ablate both, measure task IIA.

Uses mean-ablation (replace subspace component with its mean across the
dataset) rather than zero-ablation, which pushes activations off the
data manifold. All conditions use interchange intervention for measurement
(matching the other experiments' instrument).

The interaction term = joint_loss - (binding_loss + answer_loss).
Bootstrap CI on the interaction term replaces the bare threshold.

Usage:
    python unity_coupling_test.py --output results/unity_coupling.json
    python unity_coupling_test.py --dry-run --output results/DRYRUN_unity_coupling.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# For the unity test, we test coupling between binding and answer mechanisms.
# Each mechanism spans multiple layers; we ablate all layers in the range.
# Binding uses binding_lookback/address_and_payload; answer uses answer_lookback/pointer.
# Source: https://github.com/Nix07/belief_tracking @ 0579347e
# See reference/extracted_results_llama70b.json for full extraction.
LOOKBACK_SUBSPACES = {
    "binding": {
        "lookback_type": "binding_lookback",
        "concept": "address_and_payload",
        "layers": {34: 3, 35: 7, 36: 8},
    },
    "answer": {
        "lookback_type": "answer_lookback",
        "concept": "pointer",
        "layers": {38: 3, 52: 18, 53: 19},
    },
}


def mean_ablate_subspace(activation, basis, mean_projection):
    """Replace the subspace component of activation with its dataset mean.

    Args:
        activation: (..., d_model)
        basis: (d_sub, d_model), orthonormal rows
        mean_projection: (d_model,), mean of basis @ x across the dataset

    Returns:
        activation with subspace component replaced by mean
    """
    current_projection = activation @ basis.T @ basis  # (..., d_model)
    return activation - current_projection + mean_projection


def compute_mean_projection(model, stories, basis, layer_range, device):
    """Compute the mean projection of the residual stream onto the subspace across all stories."""
    raise NotImplementedError(
        "Requires running model on all stories, extracting residual stream at the "
        "specified layers, projecting onto basis, and averaging."
    )


def measure_task_iia(model, stories, ablation_spec, device):
    """Measure task-level IIA under a given ablation configuration.

    Uses interchange intervention (not ablation) as the measurement instrument.
    ablation_spec maps subspace names to (basis, mean_projection) pairs that
    define which subspaces are mean-ablated during the forward pass.
    Empty dict = no ablation (baseline).

    Returns per-story IIA values for bootstrapping.
    """
    raise NotImplementedError(
        "Requires NNsight hooks. For each story pair: "
        "1. Apply mean-ablation to specified subspaces "
        "2. Run interchange intervention on the remaining mechanism "
        "3. Check if model output matches counterfactual target"
    )


def bootstrap_interaction_ci(per_story_baseline, per_story_binding, per_story_answer,
                              per_story_both, n_bootstrap=10000, ci_level=0.95, rng=None):
    """Bootstrap confidence interval on the interaction term.

    interaction = joint_loss - (binding_loss + answer_loss)
    where loss_X = baseline_iia - X_iia

    Returns (point_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(per_story_baseline)
    interaction_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_baseline = np.mean(per_story_baseline[idx])
        b_binding = np.mean(per_story_binding[idx])
        b_answer = np.mean(per_story_answer[idx])
        b_both = np.mean(per_story_both[idx])

        binding_loss = b_baseline - b_binding
        answer_loss = b_baseline - b_answer
        joint_loss = b_baseline - b_both
        interaction = joint_loss - (binding_loss + answer_loss)
        interaction_samples.append(interaction)

    interaction_samples = np.array(interaction_samples)
    alpha = (1 - ci_level) / 2
    ci_lower = float(np.quantile(interaction_samples, alpha))
    ci_upper = float(np.quantile(interaction_samples, 1 - alpha))

    baseline_mean = np.mean(per_story_baseline)
    point = (baseline_mean - np.mean(per_story_both)) - (
        (baseline_mean - np.mean(per_story_binding)) + (baseline_mean - np.mean(per_story_answer))
    )

    return float(point), ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description="Unity coupling test for Lookback audit")
    parser.add_argument("--output", type=str, default="results/unity_coupling.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    n_stories = 80  # CausalToM size

    results = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "ablation_method": "mean-ablation (not zero-ablation)",
        "measurement_instrument": "interchange intervention (matching other experiments)",
        "hypothesis": "Super-additive interaction supports unified mechanism; "
                      "CI including zero supports independent processes",
        "conditions": {},
    }

    conditions = {
        "baseline": {"ablate": [], "description": "No ablation"},
        "binding_only": {"ablate": ["binding"], "description": "Mean-ablate binding subspace only"},
        "answer_only": {"ablate": ["answer"], "description": "Mean-ablate answer subspace only"},
        "both": {"ablate": ["binding", "answer"], "description": "Mean-ablate both subspaces"},
    }

    per_story_iias = {}

    for cond_name, cond_spec in conditions.items():
        print(f"\nCondition: {cond_name} -- {cond_spec['description']}")

        if args.dry_run:
            if cond_name == "baseline":
                stories_iia = rng.beta(8, 2, size=n_stories)
            elif cond_name == "binding_only":
                stories_iia = rng.beta(5, 3, size=n_stories)
            elif cond_name == "answer_only":
                stories_iia = rng.beta(4, 3, size=n_stories)
            else:
                stories_iia = rng.beta(2, 4, size=n_stories)
        else:
            raise NotImplementedError("Full computation requires model loading.")

        per_story_iias[cond_name] = stories_iia
        mean_iia = float(np.mean(stories_iia))

        results["conditions"][cond_name] = {
            "description": cond_spec["description"],
            "ablated_subspaces": cond_spec["ablate"],
            "task_iia_mean": mean_iia,
            "task_iia_std": float(np.std(stories_iia, ddof=1)),
        }

        print(f"  Task IIA: {mean_iia:.4f} +/- {float(np.std(stories_iia, ddof=1)):.4f}")

    interaction_point, ci_lower, ci_upper = bootstrap_interaction_ci(
        per_story_iias["baseline"],
        per_story_iias["binding_only"],
        per_story_iias["answer_only"],
        per_story_iias["both"],
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )

    baseline_mean = float(np.mean(per_story_iias["baseline"]))
    binding_loss = baseline_mean - float(np.mean(per_story_iias["binding_only"]))
    answer_loss = baseline_mean - float(np.mean(per_story_iias["answer_only"]))
    joint_loss = baseline_mean - float(np.mean(per_story_iias["both"]))

    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    results["analysis"] = {
        "baseline_iia": baseline_mean,
        "binding_loss": binding_loss,
        "answer_loss": answer_loss,
        "joint_loss": joint_loss,
        "sum_individual_losses": binding_loss + answer_loss,
        "interaction_term": interaction_point,
        "interaction_95ci_lower": ci_lower,
        "interaction_95ci_upper": ci_upper,
        "ci_excludes_zero": ci_excludes_zero,
        "interpretation": (
            "Super-additive: supports unified staged mechanism (CI entirely above zero)"
            if ci_lower > 0
            else (
                "Sub-additive: partial redundancy (CI entirely below zero)"
                if ci_upper < 0
                else "Consistent with additive independence (CI includes zero)"
            )
        ),
    }

    print(f"\n{'='*60}")
    print(f"Baseline IIA:       {baseline_mean:.4f}")
    print(f"Binding loss:       {binding_loss:.4f}")
    print(f"Answer loss:        {answer_loss:.4f}")
    print(f"Joint loss:         {joint_loss:.4f}")
    print(f"Sum individual:     {binding_loss + answer_loss:.4f}")
    print(f"Interaction term:   {interaction_point:.4f}")
    print(f"95% CI:             [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"CI excludes zero:   {ci_excludes_zero}")
    print(f"Verdict:            {results['analysis']['interpretation']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
