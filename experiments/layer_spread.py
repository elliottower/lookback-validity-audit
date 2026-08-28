"""
Layer spread test: is the subspace specific to the reported layers, or
does the same SVD basis yield high IIA at adjacent layers?

If IIA is equally high at layers 33, 37, 39, 51, 54 (not reported in the
paper), the reported layers are cherry-picked peaks rather than specifically
important computational sites.

Protocol:
  1. For each of the paper's subspaces, also compute IIA at +/- 1-3 adjacent layers
  2. Use the same SVD basis (from the reported layer) applied at adjacent layers
  3. Also compute IIA using the adjacent layer's own SVD basis (if available)

Predictions:
  Sharp peak at reported layer, rapid falloff: SUPPORTS specificity claim
  Broad plateau across adjacent layers: WEAKENS claim (cherry-picking)

Usage:
    uv run python experiments/layer_spread.py --dry-run
    uv run python experiments/layer_spread.py --output results/layer_spread/
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

ADJACENT_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


def main():
    parser = argparse.ArgumentParser(description="Layer spread test")
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/layer_spread/")
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
        "method": "layer_spread",
        "method_note": (
            "Tests whether subspace IIA is specific to the reported layer "
            "or broadly present at adjacent layers. Uses the same SVD basis "
            "from the reported layer, applied at neighboring layers."
        ),
        "prediction": {
            "supports_paper": "Sharp IIA peak at reported layer, rapid falloff at +/-1",
            "weakens_paper": "Broad plateau across adjacent layers",
        },
        "subspaces": {},
    }

    for sub_name, sub_spec in tqdm(subspaces.items(), desc="Subspaces"):
        target_layer = sub_spec["layer"]
        rank = sub_spec["rank"]
        lookback_type = sub_spec["lookback_type"]
        is_binding = "binding" in lookback_type
        vec_type = "state_tokens" if is_binding else "last_token"

        print(f"\n[{ts()}] {sub_name} (target L{target_layer}, rank={rank})")

        if not args.dry_run:
            svd_basis = load_svd_basis(target_layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: no SVD basis for L{target_layer}")
                summary["subspaces"][sub_name] = {
                    "skipped": True,
                    "skip_reason": f"SVD basis not found for L{target_layer}",
                }
                continue

            mask_indices = sub_spec.get("mask_indices")
            if mask_indices is not None:
                proj = build_projection_matrix(svd_basis, mask_indices)
            else:
                proj = build_projection_matrix(svd_basis, np.arange(rank))

        layer_results = {}

        for offset in ADJACENT_OFFSETS:
            test_layer = target_layer + offset
            if test_layer < 0 or test_layer >= 80:
                continue

            if args.dry_run:
                base_iia = sub_spec["sv_iia"]
                decay = abs(offset) * 0.15
                iia = float(max(0, base_iia - decay + rng.normal(0, 0.05)))
            else:
                pairs = binding_pairs if is_binding else answer_pairs

                if is_binding:
                    iia = compute_iia_binding(lm, pairs, test_layer, proj)
                else:
                    iia = compute_iia_answer_flex(lm, pairs, test_layer, proj)

            label = f"L{test_layer}"
            if offset == 0:
                label += " (reported)"

            layer_results[str(test_layer)] = {
                "layer": test_layer,
                "offset": offset,
                "iia": float(iia),
                "is_reported_layer": offset == 0,
            }
            print(f"  L{test_layer} (offset={offset:+d}): IIA={iia:.4f}")

        iias = [v["iia"] for v in layer_results.values()]
        reported_iia = layer_results.get(str(target_layer), {}).get("iia", 0)
        adjacent_iias = [v["iia"] for v in layer_results.values() if v["offset"] != 0]

        sub_result = {
            "target_layer": target_layer,
            "rank": rank,
            "lookback_type": lookback_type,
            "original_iia": sub_spec["sv_iia"],
            "layers": layer_results,
            "reported_iia": float(reported_iia),
            "mean_adjacent_iia": float(np.mean(adjacent_iias)) if adjacent_iias else 0,
            "max_adjacent_iia": float(np.max(adjacent_iias)) if adjacent_iias else 0,
            "specificity_ratio": (
                float(reported_iia / np.mean(adjacent_iias))
                if adjacent_iias and np.mean(adjacent_iias) > 0.01
                else float("inf")
            ),
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
        ratio = res["specificity_ratio"]
        adj_max = res["max_adjacent_iia"]
        rep = res["reported_iia"]

        if ratio > 2.0:
            verdict = "SPECIFIC (sharp peak)"
        elif adj_max > 0.6 * rep and rep > 0.3:
            verdict = "BROAD (plateau)"
        else:
            verdict = "MODERATE"

        print(f"  {name}: reported={rep:.4f}, max_adjacent={adj_max:.4f}, ratio={ratio:.2f} — {verdict}")

    any_broad = any(
        v.get("specificity_ratio", float("inf")) < 1.5
        for v in summary["subspaces"].values()
        if not v.get("skipped") and v.get("reported_iia", 0) > 0.3
    )

    summary["verdict"] = (
        "Broad plateau: subspace effect is not layer-specific"
        if any_broad else
        "Sharp peaks: subspace effects are layer-specific (supports paper)"
    )
    print(f"\nVerdict: {summary['verdict']}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Results saved to {summary_path}")


if __name__ == "__main__":
    main()
