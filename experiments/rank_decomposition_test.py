"""
Rank decomposition test: does every SVD direction in the paper's mask
contribute to IIA, or does one direction do all the work?

For each answer subspace (L38, L52, L53):
  1. Full mask IIA (baseline — should match positive control)
  2. Each individual direction's IIA (rank-1 projections)
  3. Leave-one-out: full mask minus each direction

If binding_L34 (rank 3, mask [2,3,5]) gets IIA=0.96 and direction 2
alone gets 0.95, the other 2 are noise and the "rank-3 subspace" is
really rank-1.

Binding subspaces (L34-36) are skipped because the CausalToM state
positions do not apply to the answer-style (last-token) protocol.

Usage:
    uv run python experiments/rank_decomposition_test.py --dry-run
    uv run python experiments/rank_decomposition_test.py
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "rank_decomposition"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Rank decomposition test for lookback subspaces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Rank decomposition test")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    lm = None
    pairs = None

    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight...")
        lm = setup_nnsight()

        pairs_path = RESULTS_DIR / "filtered_answer_pairs.json"
        if pairs_path.exists():
            with open(pairs_path) as f:
                pairs = json.load(f)
            print(f"[{ts()}] Loaded cached pairs: {len(pairs)}")
        else:
            print(f"[{ts()}] Generating pairs...")
            answer_raw, _ = generate_counterfactual_pairs(
                n_samples=args.n_eval * 3, seed=args.seed)
            print(f"[{ts()}] Filtering on model accuracy...")
            pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
            with open(pairs_path, "w") as f:
                json.dump(pairs, f, indent=2)
            print(f"[{ts()}] {len(pairs)} pairs passed filter")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "subspaces": {},
    }

    for sub_name, spec in answer_subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        mask_indices = spec["mask_indices"]

        print(f"\n{'='*60}")
        print(f"[{ts()}] {sub_name}: layer={layer}, rank={rank}, mask={mask_indices}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            summary["subspaces"][sub_name] = cached
            print(f"[{ts()}] Resumed from checkpoint")
            continue

        svd_basis = load_svd_basis(layer, "last_token")
        if svd_basis is None and not args.dry_run:
            print(f"  SKIP: no SVD basis for L{layer}")
            continue

        result = {
            "layer": layer,
            "rank": rank,
            "mask_indices": mask_indices,
            "paper_iia": spec["sv_iia"],
        }

        # 1. Full mask IIA
        print(f"[{ts()}] Computing full mask IIA...")
        if args.dry_run:
            result["full_mask_iia"] = spec["sv_iia"] + rng.normal(0, 0.02)
        else:
            proj = build_projection_matrix(svd_basis, mask_indices)
            result["full_mask_iia"] = float(
                compute_iia_answer(lm, pairs, layer, proj))
        print(f"  Full mask: {result['full_mask_iia']:.4f}")

        # 2. Individual direction IIAs
        print(f"[{ts()}] Computing individual direction IIAs ({len(mask_indices)} directions)...")
        individual = {}
        for idx in tqdm(mask_indices, desc="Individual directions"):
            if args.dry_run:
                individual[str(idx)] = float(rng.uniform(0.1, 0.6))
            else:
                proj_i = build_projection_matrix(svd_basis, [idx])
                individual[str(idx)] = float(
                    compute_iia_answer(lm, pairs, layer, proj_i))
            print(f"  Direction {idx}: IIA={individual[str(idx)]:.4f}")
        result["individual_iias"] = individual

        # 3. Leave-one-out IIAs
        print(f"[{ts()}] Computing leave-one-out IIAs ({len(mask_indices)} combos)...")
        loo = {}
        for drop_idx in tqdm(mask_indices, desc="Leave-one-out"):
            remaining = [i for i in mask_indices if i != drop_idx]
            if args.dry_run:
                loo[str(drop_idx)] = float(
                    result["full_mask_iia"] - rng.uniform(0.0, 0.1))
            else:
                proj_loo = build_projection_matrix(svd_basis, remaining)
                loo[str(drop_idx)] = float(
                    compute_iia_answer(lm, pairs, layer, proj_loo))
            drop_effect = result["full_mask_iia"] - loo[str(drop_idx)]
            print(f"  Drop {drop_idx}: IIA={loo[str(drop_idx)]:.4f} "
                  f"(effect={drop_effect:+.4f})")
        result["loo_iias"] = loo

        # Interpretation
        best_single = max(individual.values())
        best_single_idx = max(individual, key=individual.get)
        full = result["full_mask_iia"]
        single_ratio = best_single / full if full > 0 else 0

        result["interpretation"] = {
            "best_single_direction": int(best_single_idx),
            "best_single_iia": best_single,
            "full_mask_iia": full,
            "single_to_full_ratio": float(single_ratio),
            "effectively_rank_1": single_ratio > 0.9,
        }

        summary["subspaces"][sub_name] = result

        with open(ckpt_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{ts()}] Saved {ckpt_path}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for sub_name, res in summary["subspaces"].items():
        interp = res.get("interpretation", {})
        print(f"{sub_name}:")
        print(f"  Full mask IIA:        {res.get('full_mask_iia', '?'):.4f}")
        print(f"  Best single dir:      {interp.get('best_single_direction', '?')} "
              f"(IIA={interp.get('best_single_iia', 0):.4f})")
        print(f"  Single/full ratio:    {interp.get('single_to_full_ratio', 0):.2f}")
        print(f"  Effectively rank-1:   {interp.get('effectively_rank_1', '?')}")
        print()

    print(f"[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
