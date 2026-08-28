"""
Amendment 7 positive control: reproduce the paper's IIA values on Llama 3.1.

Prakash et al. used Llama-3-70B-Instruct. We have Llama-3.1-70B-Instruct.
Before running any pre-registered experiments, we must show the finding
transfers across model versions.

Protocol:
  1. Load our SVD bases (extracted from 3.1) for each of the 6 subspaces
  2. Build projection P = V_sel.T @ V_sel using the paper's LEARNED MASK
     indices (not top-k; mask selects non-contiguous SVD components)
  3. Generate counterfactual pairs using the paper's dataset code
  4. Filter on model accuracy (matching paper's protocol)
  5. Compute IIA via context-swap + subspace projection on NDIF
     (NDIF limitation: single-position intervention fails; we full-swap
     context positions where prompts differ and subspace-project at
     the target positions only)
  6. Compare to paper's reported sv_iia values

Criterion (from Amendment 7):
  - Within 0.10 IIA of paper's reported values for primary subspaces
    (binding_addr_payload_L34 and answer_pointer_L52)
  - If delta > 0.10, the model substitution is not validated and we
    report this as a cross-model generalization result

Incremental saving: filtered pairs and each subspace result are saved
to results/positive_control/ as they complete, so NDIF drops don't
lose progress. Re-running resumes from the last completed stage.

Usage:
    uv run python experiments/positive_control_replication.py
    uv run python experiments/positive_control_replication.py --dry-run
    uv run python experiments/positive_control_replication.py --reset  # start fresh
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

CHECKPOINT_DIR = REPO_ROOT / "results" / "positive_control"


def save_checkpoint(name, data):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Amendment 7: same-model replication positive control"
    )
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "results" / "positive_control_replication.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoints and start fresh")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    if args.reset:
        import shutil
        if CHECKPOINT_DIR.exists():
            shutil.rmtree(CHECKPOINT_DIR)
        print(f"[{ts()}] Cleared checkpoints")

    subspaces = load_subspace_specs()
    print(f"[{ts()}] Loaded {len(subspaces)} subspace specs")
    print(f"[{ts()}] Primary subspaces: {sorted(PRIMARY_SUBSPACES)}")
    print(f"[{ts()}] Replication threshold: delta <= {REPLICATION_THRESHOLD}")

    lm = None

    if not args.dry_run:
        # --- Stage 1: Filter pairs (with checkpoint) ---
        cached_answer = load_checkpoint("filtered_answer_pairs")
        cached_binding = load_checkpoint("filtered_binding_pairs")

        if cached_answer and cached_binding:
            answer_pairs = cached_answer
            binding_pairs = cached_binding
            print(f"[{ts()}] Resumed {len(answer_pairs)} answer pairs from checkpoint")
            print(f"[{ts()}] Resumed {len(binding_pairs)} binding pairs from checkpoint")
        else:
            print(f"\n[{ts()}] Setting up nnsight + NDIF...")
            lm = setup_nnsight()

            print(f"[{ts()}] Generating counterfactual pairs...")
            answer_pairs, binding_pairs = generate_counterfactual_pairs(
                n_samples=args.n_eval * 3, seed=args.seed
            )

            if not cached_answer:
                print(f"[{ts()}] Filtering answer pairs on model accuracy...")
                answer_pairs = filter_on_model(lm, answer_pairs, max_size=args.n_eval)
                print(f"[{ts()}] {len(answer_pairs)} answer pairs passed")
                save_checkpoint("filtered_answer_pairs", answer_pairs)
                print(f"[{ts()}] Saved answer pairs checkpoint")

            if not cached_binding:
                print(f"[{ts()}] Filtering binding pairs on model accuracy...")
                binding_pairs = filter_on_model(lm, binding_pairs, max_size=args.n_eval)
                print(f"[{ts()}] {len(binding_pairs)} binding pairs passed")
                save_checkpoint("filtered_binding_pairs", binding_pairs)
                print(f"[{ts()}] Saved binding pairs checkpoint")

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

    # --- Stage 2: IIA per subspace (with per-subspace checkpoint) ---
    # Fresh NDIF connection per subspace to survive transport drops.

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

        cached_result = load_checkpoint(f"subspace_{name}")
        if cached_result and not args.dry_run:
            results["subspaces"][name] = cached_result
            print(f"  RESUMED from checkpoint:")
            print(f"  Paper IIA:  {cached_result['paper_iia']:.4f}")
            print(f"  Our IIA:    {cached_result['our_iia']:.4f}")
            print(f"  Delta:      {cached_result['delta']:.4f}")
            status = "REPLICATED" if cached_result["replicated"] else "FAILED"
            print(f"  Status:     {status}")
            continue

        if args.dry_run:
            our_iia = paper_iia + np.random.default_rng(args.seed).normal(0, 0.05)
            our_iia = float(np.clip(our_iia, 0, 1))
            n_pairs_used = args.n_eval
        else:
            print(f"  [{ts()}] Fresh NDIF connection for {name}...")
            lm = setup_nnsight()

            vec_type = "last_token" if "answer" in lookback else "state_tokens"
            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: SVD basis not found for L{layer}/{vec_type}")
                continue

            mask_indices = spec.get("mask_indices")
            if mask_indices is not None:
                proj = build_projection_matrix(svd_basis, mask_indices)
                print(f"  Projection: mask {mask_indices} (rank {len(mask_indices)}) of {svd_basis.shape[0]} components")
            else:
                proj = build_projection_matrix(svd_basis, np.arange(rank))
                print(f"  Projection: top-{rank} of {svd_basis.shape[0]} components (no mask available)")
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

        subspace_result = {
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

        results["subspaces"][name] = subspace_result

        if not args.dry_run:
            save_checkpoint(f"subspace_{name}", subspace_result)
            print(f"  Saved checkpoint for {name}")

        status = "REPLICATED" if replicated else "FAILED"
        print(f"  Paper IIA:  {paper_iia:.4f}")
        print(f"  Our IIA:    {our_iia:.4f}")
        print(f"  Delta:      {delta:.4f}")
        print(f"  Status:     {status}")

    # --- Stage 3: Verdict ---
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
