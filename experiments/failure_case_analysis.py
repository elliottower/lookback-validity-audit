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

Uses NDIF for remote Llama 3.1-70B inference (no local GPU needed).

Usage:
    uv run python experiments/failure_case_analysis.py --dry-run
    uv run python experiments/failure_case_analysis.py --output results/failure_cases.json
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
    compute_iia_answer_per_sample,
    compute_iia_binding_per_sample,
    generate_counterfactual_pairs,
    get_model_prediction,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_FAILURES_FOR_POWER = 20


def partition_pairs_by_accuracy(lm, pairs, pair_type):
    """Run model on all pairs, partition into correct/incorrect on the clean prompt.

    Args:
        lm: nnsight LanguageModel
        pairs: list of counterfactual pair dicts
        pair_type: "answer" or "binding" (for logging)

    Returns:
        correct_pairs: list of pairs where model gets clean_prompt correct
        incorrect_pairs: list of pairs where model gets clean_prompt wrong
    """
    correct_pairs = []
    incorrect_pairs = []

    for sample in tqdm(pairs, desc=f"Partitioning {pair_type} pairs"):
        clean_prompt = sample["clean_prompt"]
        clean_target = sample["clean_ans"].lower().strip()

        pred_tok, _ = get_model_prediction(lm, clean_prompt)
        if pred_tok is None:
            continue

        if pred_tok == clean_target:
            correct_pairs.append(sample)
        else:
            incorrect_pairs.append(sample)

    return correct_pairs, incorrect_pairs


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
    parser.add_argument("--n-pairs", type=int, default=500,
                        help="Number of counterfactual pairs to generate (oversample for failures)")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    subspaces = load_subspace_specs()

    if args.dry_run:
        n_total = 500
        n_correct = 450
        n_incorrect = 50
        answer_correct_pairs = None
        answer_incorrect_pairs = None
        binding_correct_pairs = None
        binding_incorrect_pairs = None
    else:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        print(f"[{ts()}] Generating {args.n_pairs} counterfactual pairs...")
        answer_pairs, binding_pairs = generate_counterfactual_pairs(
            n_samples=args.n_pairs, seed=args.seed,
        )
        print(f"[{ts()}] Generated {len(answer_pairs)} answer pairs, {len(binding_pairs)} binding pairs")

        print(f"[{ts()}] Partitioning answer pairs by model accuracy...")
        answer_correct_pairs, answer_incorrect_pairs = partition_pairs_by_accuracy(
            lm, answer_pairs, "answer"
        )
        print(f"[{ts()}] Answer: {len(answer_correct_pairs)} correct, {len(answer_incorrect_pairs)} incorrect")

        print(f"[{ts()}] Partitioning binding pairs by model accuracy...")
        binding_correct_pairs, binding_incorrect_pairs = partition_pairs_by_accuracy(
            lm, binding_pairs, "binding"
        )
        print(f"[{ts()}] Binding: {len(binding_correct_pairs)} correct, {len(binding_incorrect_pairs)} incorrect")

        n_total = len(answer_pairs) + len(binding_pairs)
        n_correct = len(answer_correct_pairs) + len(binding_correct_pairs)
        n_incorrect = len(answer_incorrect_pairs) + len(binding_incorrect_pairs)

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

    if not args.dry_run:
        results["partition_details"] = {
            "answer_correct": len(answer_correct_pairs),
            "answer_incorrect": len(answer_incorrect_pairs),
            "binding_correct": len(binding_correct_pairs),
            "binding_incorrect": len(binding_incorrect_pairs),
        }

    if n_incorrect < MIN_FAILURES_FOR_POWER:
        print(f"WARNING: Only {n_incorrect} failure cases. Need >= {MIN_FAILURES_FOR_POWER} for power.")
        print("Consider generating a larger CausalToM pool.")

    for sub_name, sub_spec in tqdm(subspaces.items(), desc="Subspaces"):
        layer = sub_spec["layer"]
        rank = sub_spec["rank"]
        lookback = sub_spec["lookback_type"]

        print(f"\nTesting {sub_name} (rank={rank}, layer={layer})")

        if args.dry_run:
            correct_iias = rng.beta(7, 3, size=450)
            incorrect_iias = rng.beta(3, 5, size=50)
        else:
            is_answer = "answer" in lookback
            vec_type = "last_token" if is_answer else "state_tokens"
            compute_fn = compute_iia_answer_per_sample if is_answer else compute_iia_binding_per_sample
            correct_pairs = answer_correct_pairs if is_answer else binding_correct_pairs
            incorrect_pairs = answer_incorrect_pairs if is_answer else binding_incorrect_pairs

            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: SVD basis not found for layer {layer} ({vec_type})")
                continue
            proj = build_projection_matrix(svd_basis, np.arange(rank))

            print(f"  Computing IIA on {len(correct_pairs)} correct pairs...")
            correct_iias = np.array(compute_fn(lm, correct_pairs, layer, proj))

            if len(incorrect_pairs) > 0:
                print(f"  Computing IIA on {len(incorrect_pairs)} incorrect pairs...")
                incorrect_iias = np.array(compute_fn(lm, incorrect_pairs, layer, proj))
            else:
                print("  WARNING: No incorrect pairs — cannot run permutation test")
                incorrect_iias = np.array([])

        if len(incorrect_iias) == 0:
            results["subspace_results"][sub_name] = {
                "correct_iia_mean": float(np.mean(correct_iias)),
                "correct_iia_std": float(np.std(correct_iias, ddof=1)) if len(correct_iias) > 1 else 0.0,
                "incorrect_iia_mean": None,
                "incorrect_iia_std": None,
                "difference": None,
                "p_value": None,
                "significant_at_001": None,
                "interpretation": "No incorrect cases — test not run",
                "n_correct": len(correct_iias),
                "n_incorrect": 0,
            }
            print(f"  Correct IIA: {float(np.mean(correct_iias)):.4f}")
            print("  No incorrect cases — skipping permutation test")
            continue

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
            "incorrect_iia_std": float(np.std(incorrect_iias, ddof=1)) if len(incorrect_iias) > 1 else 0.0,
            "difference": correct_mean - incorrect_mean,
            "p_value": float(p_value),
            "p_value_formula": "(count + 1) / (n_perm + 1)",
            "significant_at_001": p_value < 0.01,
            "n_correct": len(correct_iias),
            "n_incorrect": len(incorrect_iias),
            "interpretation": (
                "Lookback absent on failures (supports causal role)"
                if p_value < 0.01
                else "Lookback present on failures (correlate, not cause)"
            ),
        }

        print(f"  Correct IIA:   {correct_mean:.4f} +/- {float(np.std(correct_iias, ddof=1)):.4f}")
        print(f"  Incorrect IIA: {incorrect_mean:.4f} +/- {float(np.std(incorrect_iias, ddof=1)) if len(incorrect_iias) > 1 else 0.0:.4f}")
        print(f"  p-value:       {p_value:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
