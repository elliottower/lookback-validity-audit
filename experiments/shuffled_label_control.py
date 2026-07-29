"""
Shuffled-label subspace control: matched-capacity baselines.

Tests two related questions:
  A. "Shuffled evaluation pairs" — does the identified subspace's IIA depend
     on the correct pairing of stories, or does it achieve high IIA even when
     counterfactual assignments are scrambled? Same projection matrix, shuffled
     clean/counterfactual assignments.
  B. "Random mask" — does a random selection of r-of-500 SVD components
     achieve comparable IIA to the paper's optimized mask? (Floor control.)

If the real subspace beats both controls, the mask captures belief-specific
structure that depends on correct label assignment.

NOTE: The full shuffled-label control (re-training the binary mask on shuffled
labels via Adam + L1) requires ~millions of NDIF traces per permutation and is
infeasible on NDIF alone. That version needs GPU compute (Modal/RunPod) and is
not implemented here. See COMPUTE_ESTIMATE.md for details.

The Lookback paper's actual method (from https://github.com/Nix07/belief_tracking):
    1. Collect residual stream activations at target token positions
    2. SVD -> 500 singular vectors (capped)
    3. Learn binary mask over singular vectors via Adam (lr=0.1) + L1 (lambda=0.1)
    4. Round mask to {0,1}

Design: 2 primary combos x 100 permutations + 4 exploratory x 20 permutations.
Primary combos: binding_addr_payload_L34 (highest binding IIA) and
answer_pointer_L52 (canonical answer layer). Exploratory: the other 4 combos.

Uses NDIF for remote Llama 3.1-70B inference (no local GPU needed).

Usage:
    uv run python experiments/shuffled_label_control.py --dry-run
    uv run python experiments/shuffled_label_control.py --n-shuffles-primary 100
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer,
    compute_iia_binding,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

N_SVD_COMPONENTS = 500
PRIMARY_COMBOS = {"binding_addr_payload_L34", "answer_pointer_L52"}


def shuffle_pair_assignments(pairs, rng):
    """Shuffle counterfactual assignments while preserving story prompts.

    For each pair, randomly decide whether to swap the clean/counterfactual
    assignment. This breaks the association between story content and belief
    labels while preserving the marginal distribution.

    Returns a new list of pair dicts with swapped assignments.
    """
    shuffled = []
    cf_prompts = [p["counterfactual_prompt"] for p in pairs]
    cf_answers = [p["counterfactual_ans"] for p in pairs]

    perm = rng.permutation(len(pairs))

    for i, sample in enumerate(pairs):
        j = perm[i]
        shuffled.append({
            "clean_prompt": sample["clean_prompt"],
            "clean_ans": sample["clean_ans"],
            "counterfactual_prompt": cf_prompts[j],
            "counterfactual_ans": cf_answers[j],
        })
    return shuffled


def sample_random_svd_mask(n_components, rank, rng):
    """Sample a random selection of rank-many SVD component indices."""
    return np.sort(rng.choice(n_components, size=rank, replace=False))


def main():
    parser = argparse.ArgumentParser(
        description="Shuffled-label subspace control for Lookback audit"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval-samples", type=int, default=80)
    parser.add_argument("--n-shuffles-primary", type=int, default=100)
    parser.add_argument("--n-shuffles-exploratory", type=int, default=20)
    parser.add_argument("--n-random-masks", type=int, default=200)
    parser.add_argument(
        "--output", type=str,
        default=str(REPO_ROOT / "results" / "shuffled_label"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)

    print(f"[{ts()}] Setting up nnsight + NDIF...")
    lm = setup_nnsight()

    subspaces = load_subspace_specs()
    print(f"[{ts()}] Loaded {len(subspaces)} subspace specs")

    # Generate and filter evaluation data
    print(f"[{ts()}] Generating counterfactual pairs...")
    answer_pairs, binding_pairs = generate_counterfactual_pairs(
        n_samples=args.n_eval_samples * 3, seed=args.seed
    )

    if not args.dry_run:
        print(f"[{ts()}] Filtering answer pairs on model accuracy...")
        answer_pairs = filter_on_model(lm, answer_pairs, max_size=args.n_eval_samples)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

        print(f"[{ts()}] Filtering binding pairs on model accuracy...")
        binding_pairs = filter_on_model(lm, binding_pairs, max_size=args.n_eval_samples)
        print(f"[{ts()}] {len(binding_pairs)} binding pairs passed filter")

    # Load SVD bases
    svd_dir = REPO_ROOT / "results" / "svd" / "CausalToM"
    if not svd_dir.exists() and not args.dry_run:
        print(f"[{ts()}] ERROR: SVD bases not found at {svd_dir}")
        print("Run scripts/extract_svd.py first.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval_samples": args.n_eval_samples,
        "design": {
            "primary_combos": sorted(PRIMARY_COMBOS),
            "n_shuffles_primary": args.n_shuffles_primary,
            "n_shuffles_exploratory": args.n_shuffles_exploratory,
            "n_random_masks": args.n_random_masks,
            "p_floor_primary": 1.0 / (args.n_shuffles_primary + 1),
            "p_floor_exploratory": 1.0 / (args.n_shuffles_exploratory + 1),
        },
        "method_note": (
            "Two NDIF-feasible controls: (A) shuffled evaluation pairs "
            "(same projection, scrambled clean/cf assignments) and "
            "(B) random SVD masks (random r-of-500 components). "
            "The full shuffled-label control (re-training binary mask on "
            "shuffled labels) requires GPU compute and is not implemented here."
        ),
        "subspaces": {},
    }

    for name, spec in subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        lookback = spec["lookback_type"]
        is_primary = name in PRIMARY_COMBOS
        n_shuffles = args.n_shuffles_primary if is_primary else args.n_shuffles_exploratory
        tier = "primary" if is_primary else "exploratory"

        pairs = answer_pairs if "answer" in lookback else binding_pairs
        compute_fn = compute_iia_answer if "answer" in lookback else compute_iia_binding
        vec_type = "last_token" if "answer" in lookback else "state_tokens"

        print(f"\n{'='*60}")
        print(f"[{ts()}] [{tier.upper()}] {name}")
        print(f"  layer={layer}, rank={rank}, lookback={lookback}")
        print(f"  n_shuffles={n_shuffles}, n_random_masks={args.n_random_masks}")
        print(f"{'='*60}")

        # Load SVD basis and build identified projection
        if not args.dry_run:
            svd_basis = load_svd_basis(layer, vec_type, svd_dir)
            if svd_basis is None:
                print(f"  SKIP: SVD basis not found for {vec_type}/L{layer}")
                continue
            identified_indices = np.arange(rank)
            identified_proj = build_projection_matrix(svd_basis, identified_indices)
            print(f"  SVD basis loaded: {svd_basis.shape}")

        # ── Real IIA ──
        if args.dry_run:
            real_iia = spec["sv_iia"]
        else:
            print(f"[{ts()}] Computing real IIA...")
            real_iia = compute_fn(lm, pairs, layer, identified_proj)
        print(f"  Real IIA: {real_iia:.4f}")

        # ── Control A: Shuffled evaluation pairs ──
        shuffled_pair_iias = []
        for k in tqdm(range(n_shuffles), desc=f"Shuffled pairs ({name})"):
            if args.dry_run:
                shuffled_pair_iias.append(0.3 + rng.random() * 0.4)
            else:
                shuffled_pairs = shuffle_pair_assignments(pairs, rng)
                iia = compute_fn(lm, shuffled_pairs, layer, identified_proj)
                shuffled_pair_iias.append(iia)

        sp_arr = np.array(shuffled_pair_iias)
        sp_rank = int(np.sum(sp_arr >= real_iia))
        sp_pval = (sp_rank + 1) / (n_shuffles + 1)

        # ── Control B: Random SVD masks ──
        random_mask_iias = []
        for k in tqdm(range(args.n_random_masks), desc=f"Random masks ({name})"):
            if args.dry_run:
                random_mask_iias.append(rng.random() * 0.5)
            else:
                selected = sample_random_svd_mask(N_SVD_COMPONENTS, rank, rng)
                rand_proj = build_projection_matrix(svd_basis, selected)
                iia = compute_fn(lm, pairs, layer, rand_proj)
                random_mask_iias.append(iia)

        rm_arr = np.array(random_mask_iias)
        rm_rank = int(np.sum(rm_arr >= real_iia))
        rm_pval = (rm_rank + 1) / (args.n_random_masks + 1)

        result = {
            "tier": tier,
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback,
            "concept": spec["concept"],
            "real_iia": float(real_iia),
            "control_a_shuffled_pairs": {
                "description": "Same projection, scrambled clean/cf pair assignments",
                "n_shuffles": n_shuffles,
                "mean": float(np.mean(sp_arr)),
                "std": float(np.std(sp_arr, ddof=1)) if len(sp_arr) > 1 else 0.0,
                "max": float(np.max(sp_arr)),
                "min": float(np.min(sp_arr)),
                "p95": float(np.quantile(sp_arr, 0.95)),
                "rank": sp_rank,
                "p_value": float(sp_pval),
                "p_floor": 1.0 / (n_shuffles + 1),
                "significant_at_005": sp_pval < 0.05,
                "significant_at_001": sp_pval < 0.01 if is_primary else None,
            },
            "control_b_random_mask": {
                "description": "Random r-of-500 SVD components, real pair assignments",
                "n_random": args.n_random_masks,
                "mean": float(np.mean(rm_arr)),
                "std": float(np.std(rm_arr, ddof=1)) if len(rm_arr) > 1 else 0.0,
                "max": float(np.max(rm_arr)),
                "min": float(np.min(rm_arr)),
                "p95": float(np.quantile(rm_arr, 0.95)),
                "p99": float(np.quantile(rm_arr, 0.99)),
                "rank": rm_rank,
                "p_value": float(rm_pval),
                "p_floor": 1.0 / (args.n_random_masks + 1),
                "significant_at_001": rm_pval < 0.01,
            },
        }

        summary["subspaces"][name] = result

        # Save per-combo detail
        per_combo_path = output_dir / f"{name}.json"
        with open(per_combo_path, "w") as f:
            json.dump({
                "timestamp": summary["timestamp"],
                **result,
                "shuffled_pair_iias": shuffled_pair_iias,
                "random_mask_iias": random_mask_iias,
            }, f, indent=2)

        print(f"  Real IIA:         {real_iia:.4f}")
        print(f"  Shuffled pairs:   mean={np.mean(sp_arr):.4f}, p={sp_pval:.6f}")
        print(f"  Random masks:     mean={np.mean(rm_arr):.4f}, p={rm_pval:.6f}")
        print(f"  Saved: {per_combo_path}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
