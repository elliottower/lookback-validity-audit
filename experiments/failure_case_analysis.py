"""
Failure case analysis (I1): test whether the lookback pattern is absent
when the model answers incorrectly.

The Lookback paper excludes incorrect answers by design (selection on
the dependent variable). If the lookback is present and intact on failure
cases, it is a correlate of processing, not a cause of correct answers.

SCOPING NOTE: This experiment's viability depends on the model's error rate
on CausalToM. At 90% accuracy on 80 stories, expect ~8 failures — likely
too few for a powered test. Options:
  (a) Generate a larger CausalToM pool (500+ stories) to harvest ~50 failures.
  (b) Use adversarial perturbations to increase failure rate.
  (c) Report the experiment as underpowered if N_fail < 20.
Option (a) is preferred: it preserves the original distribution.

Usage:
    python failure_case_analysis.py --output results/failure_cases.json
    python failure_case_analysis.py --dry-run --output results/DRYRUN_failure_cases.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Per-layer subspace specs from released results.
LOOKBACK_SUBSPACES = {
    "binding_L35": {"layer": 35, "rank": 7},
    "binding_L36": {"layer": 36, "rank": 5},
    "binding_L38": {"layer": 38, "rank": 3},
    "answer_L52": {"layer": 52, "rank": 18},
    "answer_L53": {"layer": 53, "rank": 10},
    "answer_L54": {"layer": 54, "rank": 8},
}

MIN_FAILURES_FOR_POWER = 20


def identify_failure_cases(model, stories, device):
    """Run the model on all CausalToM stories and partition into correct/incorrect.

    Returns:
        correct_indices: list of story indices where model answers correctly
        incorrect_indices: list of story indices where model answers incorrectly
    """
    raise NotImplementedError(
        "Requires loading CausalToM dataset and running Llama-3-70B-Instruct. "
        "The model's answer is the argmax of the next-token logits at the final position. "
        "Use a LARGER pool than the original 80 stories to ensure enough failures."
    )


def permutation_test_iia_difference(correct_iias, incorrect_iias, n_permutations=10000, rng=None):
    """Two-sample permutation test for IIA difference between correct and incorrect cases.

    H0: IIA distributions are identical.
    H1: IIA on correct cases > IIA on incorrect cases.

    Returns p-value (one-sided), using (count + 1)/(n_perm + 1) to avoid exact zero.
    """
    if rng is None:
        rng = np.random.default_rng()

    observed_diff = np.mean(correct_iias) - np.mean(incorrect_iias)
    pooled = np.concatenate([correct_iias, incorrect_iias])
    n_correct = len(correct_iias)

    count_ge = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_diff = np.mean(pooled[:n_correct]) - np.mean(pooled[n_correct:])
        if perm_diff >= observed_diff:
            count_ge += 1

    return (count_ge + 1) / (n_permutations + 1)


def main():
    parser = argparse.ArgumentParser(description="Failure case analysis for Lookback audit")
    parser.add_argument("--output", type=str, default="results/failure_cases.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.dry_run:
        n_total = 500
        n_correct = 450
        n_incorrect = 50
    else:
        raise NotImplementedError(
            "Must first run model on CausalToM pool to determine correct/incorrect split. "
            "Generate 500+ stories to ensure enough failures for a powered test."
        )

    results = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "hypothesis": "Lookback IIA is significantly lower on incorrect-answer cases",
        "test": "Permutation test, p < 0.01, one-sided",
        "n_stories_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "powered": n_incorrect >= MIN_FAILURES_FOR_POWER,
        "min_failures_for_power": MIN_FAILURES_FOR_POWER,
        "subspace_results": {},
    }

    if n_incorrect < MIN_FAILURES_FOR_POWER:
        print(f"WARNING: Only {n_incorrect} failure cases. Need >= {MIN_FAILURES_FOR_POWER} for power.")
        print("Consider generating a larger CausalToM pool.")

    for sub_name, sub_spec in tqdm(LOOKBACK_SUBSPACES.items(), desc="Subspaces"):
        print(f"\nTesting {sub_name} (rank={sub_spec['rank']}, layer={sub_spec['layer']})")

        if args.dry_run:
            correct_iias = rng.beta(7, 3, size=n_correct)
            incorrect_iias = rng.beta(3, 5, size=n_incorrect)
        else:
            raise NotImplementedError("Full IIA computation requires model loading.")

        p_value = permutation_test_iia_difference(
            correct_iias, incorrect_iias,
            n_permutations=args.n_permutations, rng=rng,
        )

        correct_mean = float(np.mean(correct_iias))
        incorrect_mean = float(np.mean(incorrect_iias))

        results["subspace_results"][sub_name] = {
            "correct_iia_mean": correct_mean,
            "correct_iia_std": float(np.std(correct_iias, ddof=1)),
            "incorrect_iia_mean": incorrect_mean,
            "incorrect_iia_std": float(np.std(incorrect_iias, ddof=1)),
            "difference": correct_mean - incorrect_mean,
            "p_value": float(p_value),
            "p_value_formula": "(count + 1) / (n_perm + 1)",
            "significant_at_001": p_value < 0.01,
            "interpretation": (
                "Lookback absent on failures (supports causal role)"
                if p_value < 0.01
                else "Lookback present on failures (correlate, not cause)"
            ),
        }

        print(f"  Correct IIA:   {correct_mean:.4f} +/- {float(np.std(correct_iias, ddof=1)):.4f}")
        print(f"  Incorrect IIA: {incorrect_mean:.4f} +/- {float(np.std(incorrect_iias, ddof=1)):.4f}")
        print(f"  p-value:       {p_value:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
