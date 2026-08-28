"""
Amendment 7 positive control: reproduce the paper's IIA values on Llama 3.1.

Prakash et al. used Llama-3-70B-Instruct. We have Llama-3.1-70B-Instruct.
Before running any pre-registered experiments, we must show the finding
transfers across model versions.

Protocol:
  1. Load our SVD bases (extracted from 3.1) for each of the 6 subspaces
  2. Build projection P = V[:rank].T @ V[:rank] (top-r components)
  3. Generate counterfactual pairs using the paper's dataset code
  4. Filter on model accuracy (matching paper's protocol)
  5. Compute IIA via interchange intervention on NDIF
  6. Compare to paper's reported sv_iia values

Criterion (from Amendment 7):
  - Within 0.10 IIA of paper's reported values for primary subspaces
    (binding_addr_payload_L34 and answer_pointer_L52)
  - If delta > 0.10, the model substitution is not validated and we
    report this as a cross-model generalization result

This script runs BEFORE any pre-registered experiment.

Usage:
    uv run python experiments/positive_control_replication.py
    uv run python experiments/positive_control_replication.py --dry-run
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

PRIMARY_SUBSPACES = {"binding_addr_payload_L34", "answer_pointer_L52"}
REPLICATION_THRESHOLD = 0.10


def main():
    parser = argparse.ArgumentParser(
        description="Amendment 7: same-model replication positive control"
    )
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "results" / "positive_control_replication.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    subspaces = load_subspace_specs()
    print(f"[{ts()}] Loaded {len(subspaces)} subspace specs")
    print(f"[{ts()}] Primary subspaces: {sorted(PRIMARY_SUBSPACES)}")
    print(f"[{ts()}] Replication threshold: delta <= {REPLICATION_THRESHOLD}")

    if not args.dry_run:
        print(f"\n[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_pairs, binding_pairs = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed
        )

        print(f"[{ts()}] Filtering answer pairs on model accuracy...")
        answer_pairs = filter_on_model(lm, answer_pairs, max_size=args.n_eval)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed")

        print(f"[{ts()}] Filtering binding pairs on model accuracy...")
        binding_pairs = filter_on_model(lm, binding_pairs, max_size=args.n_eval)
        print(f"[{ts()}] {len(binding_pairs)} binding pairs passed")

    results = {
        "experiment": "Amendment 7 positive control replication",
        "purpose": "Validate model substitution (Llama 3.0 -> 3.1) before pre-registered experiments",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "paper_model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_eval": args.n_eval,
        "dry_run": args.dry_run,
        "replication_threshold": REPLICATION_THRESHOLD,
        "subspaces": {},
    }

    for name, spec in subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        lookback = spec["lookback_type"]
        paper_iia = spec["sv_iia"]
        is_primary = name in PRIMARY_SUBSPACES

        print(f"\n{'='*60}")
        print(f"[{ts()}] {'[PRIMARY]' if is_primary else '[secondary]'} {name}")
        print(f"  layer={layer}, rank={rank}, paper_iia={paper_iia:.4f}")
        print(f"{'='*60}")

        if args.dry_run:
            our_iia = paper_iia + np.random.default_rng(args.seed).normal(0, 0.05)
            our_iia = float(np.clip(our_iia, 0, 1))
            n_pairs_used = args.n_eval
        else:
            vec_type = "last_token" if "answer" in lookback else "state_tokens"
            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: SVD basis not found for L{layer}/{vec_type}")
                continue

            proj = build_projection_matrix(svd_basis, np.arange(rank))
            print(f"  Projection: top-{rank} of {svd_basis.shape[0]} components")
            print(f"  SVD basis shape: {svd_basis.shape}")

            if "answer" in lookback:
                pairs = answer_pairs
                our_iia = compute_iia_answer(lm, pairs, layer, proj)
            else:
                pairs = binding_pairs
                our_iia = compute_iia_binding(lm, pairs, layer, proj)

            n_pairs_used = len(pairs)

        delta = abs(our_iia - paper_iia)
        replicated = delta <= REPLICATION_THRESHOLD

        results["subspaces"][name] = {
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback,
            "paper_iia": paper_iia,
            "our_iia": float(our_iia),
            "delta": float(delta),
            "replicated": replicated,
            "is_primary": is_primary,
            "n_pairs": n_pairs_used,
            "threshold": REPLICATION_THRESHOLD,
        }

        status = "REPLICATED" if replicated else "FAILED"
        print(f"  Paper IIA:  {paper_iia:.4f}")
        print(f"  Our IIA:    {our_iia:.4f}")
        print(f"  Delta:      {delta:.4f}")
        print(f"  Status:     {status}")

    primary_results = {k: v for k, v in results["subspaces"].items()
                       if v["is_primary"]}
    all_primary_pass = all(v["replicated"] for v in primary_results.values())

    secondary_results = {k: v for k, v in results["subspaces"].items()
                         if not v["is_primary"]}
    all_secondary_pass = all(v["replicated"] for v in secondary_results.values())

    if all_primary_pass:
        verdict = ("Model substitution VALIDATED. Primary subspaces replicate "
                   "within threshold. Pre-registered experiments are interpretable on 3.1.")
    else:
        failed = [k for k, v in primary_results.items() if not v["replicated"]]
        verdict = (f"Model substitution NOT VALIDATED. Primary subspaces failed: "
                   f"{failed}. This is a cross-model generalization result (E4). "
                   f"Basis-generalization experiments (E1) become uninterpretable "
                   f"for the overfitting question. Shuffled-label and "
                   f"task-specificity controls remain valid.")

    results["summary"] = {
        "primary_all_pass": all_primary_pass,
        "secondary_all_pass": all_secondary_pass,
        "n_primary_passed": sum(1 for v in primary_results.values() if v["replicated"]),
        "n_primary_total": len(primary_results),
        "n_secondary_passed": sum(1 for v in secondary_results.values() if v["replicated"]),
        "n_secondary_total": len(secondary_results),
        "verdict": verdict,
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{ts()}] Results saved to {output_path}")


if __name__ == "__main__":
    main()
