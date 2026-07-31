"""
Layer spread test: is subspace IIA specific to the paper's chosen layers
or does it spread to adjacent layers?

If IIA is equally high at layers 33, 37, 39, 51, 54, the layer-specificity
claims dissolve — the representation is distributed and the paper just
cherry-picked peaks.

For each paper subspace at layer L, tests L-2, L-1, L, L+1, L+2.
At the original layer: uses the paper's binary mask (selected SVD indices).
At adjacent layers: uses top-k SVD directions (same rank) from that layer's
SVD basis, or computes a quick SVD from collected activations if no basis
exists on disk.

Only tests answer_lookback subspaces (last-token intervention at -1).
Binding subspaces require CausalToM-specific position handling.

Usage:
    uv run python experiments/layer_spread_test.py --dry-run
    uv run python experiments/layer_spread_test.py
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
    collect_activations_last_token,
    compute_iia_answer,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "layer_spread"
SVD_DIR = REPO_ROOT / "results" / "svd" / "CausalToM"

ANSWER_SUBSPACES = ["answer_pointer_L38", "answer_pointer_L52", "answer_pointer_L53"]

SPREAD_OFFSETS = [-2, -1, 0, 1, 2]


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def compute_onthefly_svd(lm, pairs, layer, rank):
    """Collect last-token activations and compute top-k SVD directions.

    80 x 8192 matrix SVD is instant on CPU. Returns (rank, d_model) basis.
    """
    acts = []
    for sample in tqdm(pairs, desc=f"Collecting acts L{layer}"):
        act = collect_activations_last_token(lm, sample["clean_prompt"], layer)
        if act is not None:
            acts.append(act)

    if len(acts) < rank:
        return None

    X = torch.stack(acts)  # (n_samples, d_model)
    X = X - X.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    return Vt[:rank]  # (rank, d_model)


def main():
    parser = argparse.ArgumentParser(description="Layer spread test for Lookback audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subspaces = load_subspace_specs()

    lm = None
    answer_pairs = None

    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_raw, _ = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed
        )

        print(f"[{ts()}] Filtering answer pairs...")
        answer_pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "spread_offsets": SPREAD_OFFSETS,
        "method": (
            "At original layer: paper's binary mask. "
            "At adjacent layers: top-k SVD from existing basis or on-the-fly SVD. "
            "Answer subspaces only (last-token intervention)."
        ),
        "subspaces": {},
    }

    for sub_name in ANSWER_SUBSPACES:
        spec = subspaces[sub_name]
        center_layer = spec["layer"]
        rank = spec["rank"]
        paper_iia = spec["sv_iia"]
        mask_indices = spec.get("mask_indices")

        print(f"\n{'='*60}")
        print(f"[{ts()}] {sub_name} (center=L{center_layer}, rank={rank})")
        print(f"{'='*60}")

        layer_results = {}

        for offset in SPREAD_OFFSETS:
            test_layer = center_layer + offset
            if test_layer < 0 or test_layer > 79:
                continue

            is_center = offset == 0
            label = f"L{test_layer}" + (" [PAPER]" if is_center else "")

            if args.dry_run:
                if is_center:
                    iia = paper_iia + rng.normal(0, 0.02)
                else:
                    decay = 0.15 * abs(offset)
                    iia = max(0.1, paper_iia - decay + rng.normal(0, 0.05))
                iia = float(np.clip(iia, 0, 1))
                basis_source = "paper_mask" if is_center else "simulated"
            else:
                if is_center and mask_indices is not None:
                    svd_basis = load_svd_basis(test_layer, "last_token", SVD_DIR)
                    if svd_basis is None:
                        print(f"  SKIP {label}: no SVD basis")
                        continue
                    proj = build_projection_matrix(svd_basis, mask_indices)
                    basis_source = "paper_mask"
                else:
                    svd_basis = load_svd_basis(test_layer, "last_token", SVD_DIR)
                    if svd_basis is not None:
                        proj = build_projection_matrix(svd_basis, np.arange(rank))
                        basis_source = "existing_svd_topk"
                    else:
                        print(f"  [~] L{test_layer}: no stored SVD, computing on-the-fly...")
                        basis = compute_onthefly_svd(lm, answer_pairs, test_layer, rank)
                        if basis is None:
                            print(f"  SKIP {label}: not enough activations")
                            continue
                        proj = (basis.T @ basis)
                        basis_source = "onthefly_svd"

                iia = compute_iia_answer(lm, answer_pairs, test_layer, proj)

            layer_results[str(test_layer)] = {
                "layer": test_layer,
                "offset": offset,
                "is_paper_layer": is_center,
                "iia": float(iia),
                "basis_source": basis_source,
                "rank": rank,
            }

            print(f"  {label}: IIA={iia:.4f} (basis={basis_source})")

        center_iia = layer_results.get(str(center_layer), {}).get("iia", 0)
        adjacent_iias = [
            v["iia"] for k, v in layer_results.items()
            if not v["is_paper_layer"]
        ]
        mean_adjacent = float(np.mean(adjacent_iias)) if adjacent_iias else None
        spread_ratio = (mean_adjacent / center_iia) if center_iia > 0 and mean_adjacent is not None else None

        results["subspaces"][sub_name] = {
            "center_layer": center_layer,
            "rank": rank,
            "paper_iia": paper_iia,
            "layer_results": layer_results,
            "center_iia": center_iia,
            "mean_adjacent_iia": mean_adjacent,
            "spread_ratio": spread_ratio,
            "interpretation": (
                "SPREAD (ratio > 0.8): IIA not layer-specific"
                if spread_ratio is not None and spread_ratio > 0.8
                else "SPECIFIC (ratio <= 0.8): IIA peaks at paper's layer"
                if spread_ratio is not None
                else "INSUFFICIENT DATA"
            ),
        }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("LAYER SPREAD SUMMARY")
    print(f"{'='*60}")
    for sub_name, sub_res in results["subspaces"].items():
        print(f"\n{sub_name}:")
        print(f"  Center L{sub_res['center_layer']}: IIA={sub_res['center_iia']:.4f}")
        print(f"  Mean adjacent: {sub_res['mean_adjacent_iia']:.4f}" if sub_res['mean_adjacent_iia'] else "  Mean adjacent: N/A")
        print(f"  Spread ratio: {sub_res['spread_ratio']:.3f}" if sub_res['spread_ratio'] else "  Spread ratio: N/A")
        print(f"  -> {sub_res['interpretation']}")
        for layer_key, lr in sorted(sub_res["layer_results"].items(), key=lambda x: int(x[0])):
            marker = " *" if lr["is_paper_layer"] else ""
            print(f"    L{lr['layer']}: {lr['iia']:.4f} ({lr['basis_source']}){marker}")

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
