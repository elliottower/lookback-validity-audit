"""
Positive control replication with multiple seeds for proper confidence intervals.

Runs our NDIF context-swap protocol independently for each seed:
  - Each seed generates a fresh set of counterfactual pairs
  - Filters on model accuracy (80 pairs per type)
  - Computes IIA for all 6 subspaces
  - Saves per-seed results

After all seeds, aggregates: mean, SE, 95% CI per subspace.

Usage:
    uv run python experiments/positive_control_multiseed.py
    uv run python experiments/positive_control_multiseed.py --seeds 10 20 30 42 100
"""

import argparse
import json
import sys
import time
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

RESULTS_DIR = REPO_ROOT / "results" / "positive_control_multiseed"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run_single_seed(seed, subspaces, n_eval=80):
    """Run full positive control for one seed. Returns dict of results."""
    print(f"\n{'#'*60}")
    print(f"[{ts()}] SEED {seed}")
    print(f"{'#'*60}")

    seed_dir = RESULTS_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Check if this seed is already done
    final_path = seed_dir / "results.json"
    if final_path.exists():
        with open(final_path) as f:
            cached = json.load(f)
        print(f"[{ts()}] Seed {seed} already complete, resuming from disk")
        return cached

    # Generate and filter pairs
    pairs_path_answer = seed_dir / "filtered_answer_pairs.json"
    pairs_path_binding = seed_dir / "filtered_binding_pairs.json"

    if pairs_path_answer.exists() and pairs_path_binding.exists():
        with open(pairs_path_answer) as f:
            answer_pairs = json.load(f)
        with open(pairs_path_binding) as f:
            binding_pairs = json.load(f)
        print(f"[{ts()}] Loaded cached pairs: {len(answer_pairs)} answer, {len(binding_pairs)} binding")
    else:
        print(f"[{ts()}] Generating pairs with seed {seed}...")
        answer_raw, binding_raw = generate_counterfactual_pairs(
            n_samples=n_eval * 3, seed=seed
        )

        lm = setup_nnsight()
        print(f"[{ts()}] Filtering answer pairs...")
        answer_pairs = filter_on_model(lm, answer_raw, max_size=n_eval)
        with open(pairs_path_answer, "w") as f:
            json.dump(answer_pairs, f, indent=2)

        print(f"[{ts()}] Filtering binding pairs...")
        binding_pairs = filter_on_model(lm, binding_raw, max_size=n_eval)
        with open(pairs_path_binding, "w") as f:
            json.dump(binding_pairs, f, indent=2)

        print(f"[{ts()}] Filtered: {len(answer_pairs)} answer, {len(binding_pairs)} binding")

    # Compute IIA per subspace
    seed_results = {}
    for name, spec in subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        lookback = spec["lookback_type"]
        paper_iia = spec["sv_iia"]

        ckpt_path = seed_dir / f"subspace_{name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                seed_results[name] = json.load(f)
            print(f"[{ts()}] {name}: resumed IIA={seed_results[name]['our_iia']:.4f}")
            continue

        print(f"[{ts()}] {name}: computing IIA...")
        lm = setup_nnsight()

        vec_type = "last_token" if "answer" in lookback else "state_tokens"
        svd_basis = load_svd_basis(layer, vec_type)
        if svd_basis is None:
            print(f"  SKIP: no SVD basis for L{layer}/{vec_type}")
            continue

        mask_indices = spec.get("mask_indices")
        if mask_indices is not None:
            proj = build_projection_matrix(svd_basis, mask_indices)
        else:
            proj = build_projection_matrix(svd_basis, np.arange(rank))

        if "answer" in lookback:
            our_iia = compute_iia_answer(lm, answer_pairs, layer, proj)
        else:
            our_iia = compute_iia_binding(lm, binding_pairs, layer, proj)

        result = {
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback,
            "paper_iia": paper_iia,
            "our_iia": float(our_iia),
            "delta": float(abs(our_iia - paper_iia)),
            "n_pairs": len(answer_pairs if "answer" in lookback else binding_pairs),
            "seed": seed,
        }
        seed_results[name] = result

        with open(ckpt_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{ts()}] {name}: IIA={our_iia:.4f} (paper={paper_iia:.4f}, delta={abs(our_iia - paper_iia):.4f})")

    # Save full seed results
    with open(final_path, "w") as f:
        json.dump(seed_results, f, indent=2)

    return seed_results


def aggregate_results(all_results, subspaces):
    """Compute mean, SE, 95% CI across seeds for each subspace."""
    summary = {}
    for name in subspaces:
        iia_values = [r[name]["our_iia"] for r in all_results if name in r]
        paper_iia = subspaces[name]["sv_iia"]
        if not iia_values:
            continue

        mean = np.mean(iia_values)
        se = np.std(iia_values, ddof=1) / np.sqrt(len(iia_values)) if len(iia_values) > 1 else 0
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se

        summary[name] = {
            "paper_iia": paper_iia,
            "mean_iia": float(mean),
            "se": float(se),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "n_seeds": len(iia_values),
            "per_seed": iia_values,
            "is_primary": name in {"binding_addr_payload_L34", "answer_pointer_L52"},
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30, 42, 100])
    parser.add_argument("--n-eval", type=int, default=80)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subspaces = load_subspace_specs()

    print(f"[{ts()}] Multi-seed positive control")
    print(f"[{ts()}] Seeds: {args.seeds}")
    print(f"[{ts()}] N eval: {args.n_eval}")
    print(f"[{ts()}] Subspaces: {list(subspaces.keys())}")

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(seed, subspaces, n_eval=args.n_eval)
        all_results.append(result)

    summary = aggregate_results(all_results, subspaces)

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({len(args.seeds)} seeds)")
    print(f"{'='*70}")
    for name, s in summary.items():
        primary = " [PRIMARY]" if s["is_primary"] else ""
        print(f"{name}{primary}:")
        print(f"  Paper IIA:  {s['paper_iia']:.4f}")
        print(f"  Our mean:   {s['mean_iia']:.4f} +/- {s['se']:.4f}")
        print(f"  95% CI:     [{s['ci_95'][0]:.4f}, {s['ci_95'][1]:.4f}]")
        print(f"  Per seed:   {[f'{v:.4f}' for v in s['per_seed']]}")
        print()

    output = {
        "experiment": "Positive control multi-seed replication",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "protocol": "NDIF context-swap (weaker than paper's single-position)",
        "seeds": args.seeds,
        "n_eval": args.n_eval,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }

    out_path = RESULTS_DIR / "aggregate_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[{ts()}] Saved to {out_path}")


if __name__ == "__main__":
    main()
