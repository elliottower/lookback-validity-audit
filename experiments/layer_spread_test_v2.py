"""
Layer spread test v2: is subspace IIA specific to the paper's chosen layers
or does it spread to adjacent layers?

v2 fixes from v1:
  - Random subspace control at each layer (same rank, random orthogonal
    directions from the stored SVD basis). If random rank-k also scores
    high, the finding is "any k-d subspace works" — stronger than spread.
  - More samples (200 default vs 80) for better statistics.
  - Per-subspace checkpointing so crashes don't lose progress.
  - Only uses stored SVD bases (no on-the-fly SVD from tiny samples).
    Layers without stored SVD are skipped with a note.

For each paper subspace at layer L, tests L-2, L-1, L, L+1, L+2.
At the original layer: paper's binary mask.
At adjacent layers: top-k SVD + random-k SVD control.

Usage:
    uv run python experiments/layer_spread_test_v2.py --dry-run
    uv run python experiments/layer_spread_test_v2.py
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

RESULTS_DIR = REPO_ROOT / "results" / "layer_spread_v2"
SVD_DIR = REPO_ROOT / "results" / "svd" / "CausalToM"

ANSWER_SUBSPACES = ["answer_pointer_L38", "answer_pointer_L52", "answer_pointer_L53"]

SPREAD_OFFSETS = [-2, -1, 0, 1, 2]
N_RANDOM_TRIALS = 10


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def random_subspace_iia(lm, pairs, layer, svd_basis, rank, n_trials, rng):
    """Compute IIA for n_trials random k-dimensional subspaces.

    Selects rank random indices from the full SVD basis (500 directions),
    builds projection, computes IIA. Returns list of IIA values.
    """
    n_total = svd_basis.shape[0]
    iias = []
    for trial in range(n_trials):
        random_indices = rng.choice(n_total, size=rank, replace=False)
        random_indices.sort()
        proj = build_projection_matrix(svd_basis, random_indices)
        iia = float(compute_iia_answer(lm, pairs, layer, proj))
        iias.append(iia)
        print(f"    random trial {trial+1}/{n_trials}: IIA={iia:.4f} (indices={random_indices[:5].tolist()}...)")
    return iias


def main():
    parser = argparse.ArgumentParser(description="Layer spread test v2 with random controls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-random-trials", type=int, default=N_RANDOM_TRIALS)
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

        pairs_path = RESULTS_DIR / "filtered_answer_pairs.json"
        if pairs_path.exists():
            with open(pairs_path) as f:
                answer_pairs = json.load(f)
            print(f"[{ts()}] Loaded cached pairs: {len(answer_pairs)}")
        else:
            print(f"[{ts()}] Generating counterfactual pairs...")
            answer_raw, _ = generate_counterfactual_pairs(
                n_samples=args.n_eval * 3, seed=args.seed
            )
            print(f"[{ts()}] Filtering answer pairs on model accuracy...")
            answer_pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
            with open(pairs_path, "w") as f:
                json.dump(answer_pairs, f, indent=2)
            print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter, cached")

    for sub_name in ANSWER_SUBSPACES:
        spec = subspaces[sub_name]
        center_layer = spec["layer"]
        rank = spec["rank"]
        paper_iia = spec["sv_iia"]
        mask_indices = spec.get("mask_indices")

        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            print(f"\n[{ts()}] {sub_name}: checkpoint exists, skipping")
            continue

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

            svd_basis = load_svd_basis(test_layer, "last_token", SVD_DIR)
            if svd_basis is None:
                print(f"  SKIP {label}: no stored SVD basis (not computing on-the-fly)")
                continue

            if args.dry_run:
                if is_center:
                    topk_iia = paper_iia + rng.normal(0, 0.02)
                else:
                    decay = 0.15 * abs(offset)
                    topk_iia = max(0.1, paper_iia - decay + rng.normal(0, 0.05))
                topk_iia = float(np.clip(topk_iia, 0, 1))
                basis_source = "paper_mask" if is_center else "topk_svd"
                random_iias = [float(np.clip(rng.uniform(0.3, 0.7), 0, 1))
                               for _ in range(args.n_random_trials)]
            else:
                if is_center and mask_indices is not None:
                    proj = build_projection_matrix(svd_basis, mask_indices)
                    basis_source = "paper_mask"
                else:
                    proj = build_projection_matrix(svd_basis, np.arange(rank))
                    basis_source = "topk_svd"

                print(f"  [{ts()}] {label}: computing top-k IIA (rank={rank})...")
                topk_iia = float(compute_iia_answer(lm, answer_pairs, test_layer, proj))

                print(f"  [{ts()}] {label}: computing {args.n_random_trials} random controls (rank={rank})...")
                random_iias = random_subspace_iia(
                    lm, answer_pairs, test_layer, svd_basis, rank,
                    args.n_random_trials, rng)

            random_mean = float(np.mean(random_iias))
            random_std = float(np.std(random_iias))

            layer_results[str(test_layer)] = {
                "layer": test_layer,
                "offset": offset,
                "is_paper_layer": is_center,
                "topk_iia": float(topk_iia),
                "random_iia_mean": random_mean,
                "random_iia_std": random_std,
                "random_iia_trials": random_iias,
                "topk_vs_random_gap": float(topk_iia - random_mean),
                "basis_source": basis_source,
                "rank": rank,
            }

            print(f"  {label}: topk={topk_iia:.4f}, random={random_mean:.4f}±{random_std:.4f}, "
                  f"gap={topk_iia - random_mean:+.4f}")

        center_res = layer_results.get(str(center_layer), {})
        center_iia = center_res.get("topk_iia", 0)
        adjacent_topk = [v["topk_iia"] for v in layer_results.values()
                         if not v["is_paper_layer"]]
        adjacent_random = [v["random_iia_mean"] for v in layer_results.values()
                           if not v["is_paper_layer"]]

        mean_adj_topk = float(np.mean(adjacent_topk)) if adjacent_topk else None
        mean_adj_random = float(np.mean(adjacent_random)) if adjacent_random else None
        spread_ratio = (mean_adj_topk / center_iia) if center_iia > 0 and mean_adj_topk is not None else None

        if spread_ratio is None:
            interpretation = "INSUFFICIENT DATA"
        elif mean_adj_random is not None and mean_adj_random > 0.8:
            interpretation = (
                f"ANY-SUBSPACE (random={mean_adj_random:.3f}): "
                f"rank-{rank} is generous enough that random directions also work. "
                f"IIA reflects subspace dimensionality, not specific content."
            )
        elif spread_ratio > 0.8:
            interpretation = (
                f"SPREAD (ratio={spread_ratio:.3f}): "
                f"top-k IIA is high at adjacent layers and random is low "
                f"(random={mean_adj_random:.3f}). Signal follows dominant "
                f"variance directions but is not layer-specific."
            )
        else:
            interpretation = (
                f"SPECIFIC (ratio={spread_ratio:.3f}): "
                f"IIA peaks at paper's layer. Adjacent layers are lower."
            )

        sub_result = {
            "center_layer": center_layer,
            "rank": rank,
            "paper_iia": paper_iia,
            "layer_results": layer_results,
            "center_iia": center_iia,
            "mean_adjacent_topk": mean_adj_topk,
            "mean_adjacent_random": mean_adj_random,
            "spread_ratio": spread_ratio,
            "interpretation": interpretation,
        }

        with open(ckpt_path, "w") as f:
            json.dump(sub_result, f, indent=2)
        print(f"[{ts()}] Saved checkpoint: {ckpt_path}")

    # Write combined summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "n_random_trials": args.n_random_trials,
        "spread_offsets": SPREAD_OFFSETS,
        "method": (
            "At original layer: paper's binary mask + random control. "
            "At adjacent layers: top-k SVD + random-k SVD from stored 500-direction basis. "
            "No on-the-fly SVD — layers without stored basis are skipped. "
            "Answer subspaces only (last-token intervention)."
        ),
        "subspaces": {},
    }

    for sub_name in ANSWER_SUBSPACES:
        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                summary["subspaces"][sub_name] = json.load(f)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("LAYER SPREAD v2 SUMMARY")
    print(f"{'='*60}")
    for sub_name, sub_res in summary["subspaces"].items():
        print(f"\n{sub_name}:")
        print(f"  Center L{sub_res['center_layer']}: IIA={sub_res['center_iia']:.4f}")
        if sub_res['mean_adjacent_topk'] is not None:
            print(f"  Mean adjacent topk:   {sub_res['mean_adjacent_topk']:.4f}")
        if sub_res['mean_adjacent_random'] is not None:
            print(f"  Mean adjacent random: {sub_res['mean_adjacent_random']:.4f}")
        if sub_res['spread_ratio'] is not None:
            print(f"  Spread ratio:         {sub_res['spread_ratio']:.3f}")
        print(f"  -> {sub_res['interpretation']}")
        for layer_key, lr in sorted(sub_res["layer_results"].items(), key=lambda x: int(x[0])):
            marker = " *" if lr["is_paper_layer"] else ""
            print(f"    L{lr['layer']}: topk={lr['topk_iia']:.4f}  "
                  f"random={lr['random_iia_mean']:.4f}±{lr['random_iia_std']:.4f}  "
                  f"gap={lr['topk_vs_random_gap']:+.4f}  "
                  f"({lr['basis_source']}){marker}")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
