"""
Rival mechanism exclusion (I5): is this THE mechanism or A mechanism?

Tests whether the identified subspaces are uniquely privileged or whether
alternative subspaces of the same rank at the same layers achieve comparable
IIA. Two conditions per subspace:

A. Random SVD subspaces: randomly select r indices from the full SVD basis
   (where r = rank of the identified subspace). Repeat 50 times. If many
   random subspaces achieve high IIA, the identified subspace is not special.

B. Orthogonal complement: use all SVD indices EXCEPT the identified ones.
   If IIA is still high on the complement, the mechanism has rival encodings
   in the residual stream at that layer.

Pre-registered decision rule (fix N, M, delta before running):
  N = 50 random draws, delta = 0.10, M = 0.
  UNIQUE: if 0 of 50 random subspaces achieve IIA within 0.10 of the
    identified subspace, the mechanism is unique at this layer.
  NON-UNIQUE: if >= 1 random subspace achieves IIA within 0.10, the
    identified subspace is one of multiple encodings.
  RIVAL: if orthogonal complement IIA > 0.50, a competing mechanism
    exists in the residual directions.

Interpretation:
  - UNIQUE + no RIVAL: strong support for the paper's subspace being THE mechanism.
  - UNIQUE + RIVAL: subspace is locally optimal but alternatives exist.
  - NON-UNIQUE: the layer encodes the information diffusely.

Usage:
    uv run python experiments/rival_mechanism_test.py --dry-run
    uv run python experiments/rival_mechanism_test.py
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "rival_mechanism"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Rival mechanism exclusion test (I5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--n-random", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Rival mechanism exclusion test (I5)")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}, "
          f"N random: {args.n_random}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    lm = None
    pairs = None

    if not args.dry_run:
        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_raw, _ = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed)
        lm = setup_nnsight()

        pairs_cache = RESULTS_DIR / "filtered_pairs.json"
        if pairs_cache.exists():
            with open(pairs_cache) as f:
                saved = json.load(f)
            pairs = saved["pairs"]
            print(f"[{ts()}] Loaded {len(pairs)} cached filtered pairs")
        else:
            print(f"[{ts()}] Filtering on model accuracy...")
            pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
            print(f"[{ts()}] {len(pairs)} pairs passed filter")
            with open(pairs_cache, "w") as f:
                json.dump({"pairs": pairs}, f, indent=2)

    DELTA = 0.10

    summary = {
        "experiment": "Rival mechanism exclusion (I5)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval": args.n_eval,
        "n_random_draws": args.n_random,
        "decision_rule": {
            "N": args.n_random,
            "delta": DELTA,
            "M_threshold": 0,
            "rule": (
                f"UNIQUE if 0 of {args.n_random} random subspaces achieve "
                f"IIA within {DELTA} of identified. NON-UNIQUE otherwise. "
                f"RIVAL if complement IIA > 0.50."
            ),
        },
        "method": (
            "For each subspace: (1) compute IIA with identified directions, "
            "(2) compute IIA with n_random random subspaces of same rank "
            "drawn from the same SVD basis, (3) compute IIA with the "
            "orthogonal complement (all SVD directions except identified). "
            "Decision rule: unique if no random draw within delta of identified."
        ),
        "subspaces": {},
    }

    for sub_name, spec in answer_subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        mask_indices = spec.get("mask_indices")

        print(f"\n{'='*60}")
        print(f"[{ts()}] {sub_name}: L{layer}, rank={rank}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) == args.dry_run:
                summary["subspaces"][sub_name] = cached
                print(f"  Resumed from checkpoint")
                print(f"  Identified IIA: {cached['identified_iia']:.4f}")
                print(f"  Random mean: {cached['random_iia_mean']:.4f}")
                print(f"  Complement IIA: {cached['complement_iia']:.4f}")
                print(f"  Percentile: {cached['percentile_rank']:.1f}")
                continue

        svd_basis = load_svd_basis(layer, "last_token")
        if svd_basis is None and not args.dry_run:
            print(f"  SKIP: no SVD basis for L{layer}")
            continue

        n_total = svd_basis.shape[0] if svd_basis is not None else 500

        if args.dry_run:
            identified_iia = spec["sv_iia"]
            random_iias = [float(rng.uniform(0.0, 0.3)) for _ in range(args.n_random)]
            complement_iia = float(rng.uniform(0.0, 0.2))
        else:
            # 1. Identified subspace IIA
            print(f"  [{ts()}] Computing identified subspace IIA...")
            proj_id = build_projection_matrix(svd_basis, mask_indices)
            identified_iia = compute_iia_answer_flex(lm, pairs, layer, proj_id)
            print(f"  Identified IIA: {identified_iia:.4f}")

            # 2. Random subspaces of same rank
            random_iias = []
            all_indices = list(range(n_total))
            id_set = set(mask_indices)

            for i in tqdm(range(args.n_random),
                          desc=f"  Random subspaces (rank={rank})"):
                rand_indices = rng.choice(all_indices, size=rank, replace=False)
                proj_rand = build_projection_matrix(svd_basis, rand_indices)
                iia = compute_iia_answer_flex(lm, pairs, layer, proj_rand)
                random_iias.append(float(iia))

                if (i + 1) % 10 == 0:
                    print(f"    [{ts()}] {i+1}/{args.n_random} done, "
                          f"mean so far: {np.mean(random_iias):.4f}")

            # 3. Orthogonal complement
            print(f"  [{ts()}] Computing complement IIA...")
            complement_indices = [i for i in all_indices if i not in id_set]
            if len(complement_indices) > 0:
                proj_comp = build_projection_matrix(svd_basis, complement_indices)
                complement_iia = compute_iia_answer_flex(
                    lm, pairs, layer, proj_comp)
            else:
                complement_iia = 0.0

        percentile = float(
            100.0 * np.mean(np.array(random_iias) < identified_iia))
        n_within_delta = int(
            np.sum(np.abs(np.array(random_iias) - identified_iia) <= DELTA))

        entry = {
            "dry_run": args.dry_run,
            "layer": layer,
            "rank": rank,
            "n_svd_total": n_total,
            "mask_indices": list(mask_indices) if mask_indices is not None else None,
            "identified_iia": float(identified_iia),
            "random_iia_mean": float(np.mean(random_iias)),
            "random_iia_std": float(np.std(random_iias)),
            "random_iia_max": float(np.max(random_iias)),
            "random_iia_min": float(np.min(random_iias)),
            "random_iias": random_iias,
            "complement_iia": float(complement_iia),
            "complement_rank": n_total - rank,
            "percentile_rank": percentile,
            "n_within_delta": n_within_delta,
            "delta": DELTA,
            "is_unique": n_within_delta == 0,
            "has_rival": float(complement_iia) > 0.5,
        }
        summary["subspaces"][sub_name] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"  Identified IIA:   {identified_iia:.4f}")
        print(f"  Random mean/std:  {np.mean(random_iias):.4f} +/- "
              f"{np.std(random_iias):.4f}")
        print(f"  Random max:       {np.max(random_iias):.4f}")
        print(f"  Complement IIA:   {complement_iia:.4f}")
        print(f"  Percentile rank:  {percentile:.1f}%")
        print(f"  Within delta={DELTA}: {n_within_delta}/{args.n_random}")
        print(f"  UNIQUE:           {n_within_delta == 0}")
        print(f"  Has rival (>0.5): {complement_iia > 0.5}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("RIVAL MECHANISM SUMMARY")
    print(f"{'='*60}")
    for name, r in summary["subspaces"].items():
        tag = "UNIQUE" if r["is_unique"] else "NON-UNIQUE"
        rival = " + HAS RIVAL" if r["has_rival"] else ""
        print(f"  {name}: identified={r['identified_iia']:.3f}, "
              f"random_mean={r['random_iia_mean']:.3f}, "
              f"within_delta={r['n_within_delta']}, "
              f"complement={r['complement_iia']:.3f} "
              f"({tag}{rival})")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
