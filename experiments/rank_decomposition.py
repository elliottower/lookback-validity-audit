"""
Rank-1 decomposition: does each SVD direction contribute, or is the
subspace effectively rank-1?

For each subspace, tests each SVD direction individually and measures IIA.
If a single direction captures most of the IIA, the stated rank is inflated
and the subspace is effectively one-dimensional.

Protocol:
  1. For each subspace (rank r), test directions 0..r-1 individually
  2. Measure IIA with rank-1 projection from each direction
  3. Compare individual IIA to full-rank IIA
  4. Also test cumulative: top-1, top-2, ..., top-r

Predictions:
  All directions contribute roughly equally: genuine high-rank (SUPPORTS paper)
  Direction 0 captures >80% of full IIA: effectively rank-1 (WEAKENS paper)

Usage:
    uv run python experiments/rank_decomposition.py --dry-run
    uv run python experiments/rank_decomposition.py --output results/rank_decomp/
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
    build_projection_matrix,
    compute_iia_answer_flex,
    compute_iia_binding,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Rank-1 decomposition test")
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/rank_decomp/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lm = None
    answer_pairs, binding_pairs = None, None

    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        print(f"[{ts()}] Generating CausalToM pairs...")
        answer_pairs, binding_pairs = generate_counterfactual_pairs(args.n_samples, args.seed)

        print(f"[{ts()}] Filtering answer pairs...")
        answer_pairs = filter_on_model(lm, answer_pairs, max_size=80)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

        print(f"[{ts()}] Filtering binding pairs...")
        binding_pairs = filter_on_model(lm, binding_pairs, max_size=80)
        print(f"[{ts()}] {len(binding_pairs)} binding pairs passed filter")

    summary = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "method": "rank_decomposition",
        "method_note": (
            "Tests each SVD direction individually within each subspace. "
            "If one direction captures most of the IIA, the stated rank is inflated."
        ),
        "prediction": {
            "supports_paper": "Multiple directions contribute, no single dominant direction",
            "weakens_paper": "Direction 0 captures >80% of full-rank IIA",
        },
        "subspaces": {},
    }

    for sub_name, sub_spec in tqdm(subspaces.items(), desc="Subspaces"):
        layer = sub_spec["layer"]
        rank = sub_spec["rank"]
        lookback_type = sub_spec["lookback_type"]
        is_binding = "binding" in lookback_type
        vec_type = "state_tokens" if is_binding else "last_token"

        print(f"\n[{ts()}] {sub_name} (L{layer}, rank={rank}, {lookback_type})")

        if not args.dry_run:
            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: no SVD basis")
                summary["subspaces"][sub_name] = {
                    "skipped": True,
                    "skip_reason": "SVD basis not found",
                }
                continue

        mask_indices = sub_spec.get("mask_indices")
        if mask_indices is not None:
            indices = mask_indices
        else:
            indices = list(range(rank))

        pairs = binding_pairs if is_binding else answer_pairs

        # Individual direction IIA
        individual_iias = {}
        for i, idx in enumerate(indices):
            if args.dry_run:
                base = sub_spec["sv_iia"] / rank
                iia = float(base * (rank - i) / rank + rng.normal(0, 0.05))
                iia = max(0, min(1, iia))
            else:
                proj = build_projection_matrix(svd_basis, [idx])
                if is_binding:
                    iia = compute_iia_binding(lm, pairs, layer, proj)
                else:
                    iia = compute_iia_answer_flex(lm, pairs, layer, proj)

            individual_iias[str(idx)] = float(iia)
            print(f"  direction {idx}: IIA={iia:.4f}")

        # Cumulative IIA (top-1, top-2, ..., top-rank)
        cumulative_iias = {}
        for k in range(1, len(indices) + 1):
            cum_indices = indices[:k]
            if args.dry_run:
                iia = float(min(1.0, sub_spec["sv_iia"] * (k / rank) ** 0.5 + rng.normal(0, 0.03)))
                iia = max(0, min(1, iia))
            else:
                proj = build_projection_matrix(svd_basis, cum_indices)
                if is_binding:
                    iia = compute_iia_binding(lm, pairs, layer, proj)
                else:
                    iia = compute_iia_answer_flex(lm, pairs, layer, proj)

            cumulative_iias[str(k)] = float(iia)
            print(f"  cumulative top-{k}: IIA={iia:.4f}")

        full_iia = cumulative_iias.get(str(len(indices)), 0)
        max_individual = max(individual_iias.values()) if individual_iias else 0
        concentration_ratio = max_individual / full_iia if full_iia > 0.01 else float("inf")

        sub_result = {
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback_type,
            "original_iia": sub_spec["sv_iia"],
            "individual_direction_iias": individual_iias,
            "cumulative_iias": cumulative_iias,
            "full_rank_iia": float(full_iia),
            "max_individual_iia": float(max_individual),
            "concentration_ratio": float(concentration_ratio),
            "effectively_rank1": concentration_ratio > 0.8,
        }

        summary["subspaces"][sub_name] = sub_result

        sub_path = output_dir / f"{sub_name}.json"
        with open(sub_path, "w") as f:
            json.dump({"timestamp": summary["timestamp"], **sub_result}, f, indent=2)

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    for name, res in summary["subspaces"].items():
        if res.get("skipped"):
            continue
        cr = res["concentration_ratio"]
        status = "RANK-1 DOMINATED" if res["effectively_rank1"] else "DISTRIBUTED"
        print(f"  {name}: rank={res['rank']}, max_individual={res['max_individual_iia']:.4f}, "
              f"full={res['full_rank_iia']:.4f}, concentration={cr:.2f} — {status}")

    rank1_count = sum(
        1 for v in summary["subspaces"].values()
        if not v.get("skipped") and v.get("effectively_rank1", False)
    )
    total_tested = sum(1 for v in summary["subspaces"].values() if not v.get("skipped"))

    summary["verdict"] = f"{rank1_count}/{total_tested} subspaces are effectively rank-1"
    print(f"\nVerdict: {summary['verdict']}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Results saved to {summary_path}")


if __name__ == "__main__":
    main()
