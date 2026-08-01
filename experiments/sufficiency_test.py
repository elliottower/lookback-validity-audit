"""
Sufficiency test (I2): Is the circuit enough to produce the behavior?

GRADED sufficiency: retaining ONLY a rank-r subspace in a 70B model produces
catastrophically off-distribution activations, so binary pass/fail is
unfalsifiable. Instead, we sweep k = number of top SVD directions retained
alongside the identified subspace, and report the recovery curve.

Protocol per subspace, for k in [0, 10, 25, 50, 100, 200]:
  1. Get clean output at the target layer -> CPU
  2. Build basis = identified subspace directions UNION top-k SVD directions
  3. Project onto that combined basis: x' = x @ P_combined
  4. Set projected output, read prediction, compare to clean answer
  5. Report accuracy at each k

The recovery curve tells you:
  - k=0 (subspace alone): baseline sufficiency in extreme conditions
  - k=10..200: how much context the subspace needs to function
  - Steeper recovery when subspace is included vs excluded: the subspace
    carries task-specific information beyond what top-k generics provide

Control: same sweep WITHOUT the identified subspace (top-k only).
The gap between with-subspace and without-subspace curves at each k
is the subspace's marginal contribution.

Usage:
    uv run python experiments/sufficiency_test.py --dry-run
    uv run python experiments/sufficiency_test.py
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
    build_projection_from_basis,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "sufficiency"

K_VALUES = [0, 10, 25, 50, 100, 200]


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def build_combined_projection(svd_basis, subspace_indices, k):
    """Build projection onto top-k SVD directions UNION subspace directions."""
    top_k_set = set(range(min(k, svd_basis.shape[0])))
    sub_set = set(subspace_indices) if subspace_indices is not None else set()
    combined = sorted(top_k_set | sub_set)
    if len(combined) == 0:
        return torch.zeros(svd_basis.shape[1], svd_basis.shape[1])
    basis = svd_basis[combined]
    return build_projection_from_basis(basis)


def graded_accuracy(lm, pairs, layer, projection, retries=3):
    """Measure accuracy after projecting onto a combined basis at position -1."""
    correct, total = 0, 0

    for sample in pairs:
        prompt = sample["clean_prompt"]
        target = sample["clean_ans"]

        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cl_t = cl_out.detach().cpu().float()
                projected = cl_t.clone()
                projected[-1] = cl_t[-1] @ projection

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = projected
                    pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred.item()]).lower().strip()
                correct += int(pred_tok == target.lower().strip())
                total += 1
                break

            except Exception as e:
                print(f"  Sufficiency error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Sufficiency test for lookback subspaces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()

    print(f"[{ts()}] Sufficiency test (I2)")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")

    lm = None
    answer_pairs = None
    if not args.dry_run:
        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_raw, _ = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed
        )
        lm = setup_nnsight()

        print(f"[{ts()}] Filtering answer pairs...")
        answer_pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    results = {
        "experiment": "Graded sufficiency test (I2)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval": args.n_eval,
        "k_values": K_VALUES,
        "method": (
            "Graded recovery curve: for each k in K_VALUES, retain "
            "top-k SVD directions + identified subspace, project residual "
            "stream at position -1, measure accuracy. Control: top-k only "
            "(without identified subspace). The gap between curves is the "
            "subspace's marginal contribution at each ambient dimensionality."
        ),
        "subspaces": {},
    }

    for name, spec in answer_subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        mask_indices = spec.get("mask_indices")

        print(f"\n{'='*60}")
        print(f"[{ts()}] {name}: L{layer}, rank={rank}, graded sweep")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) == args.dry_run:
                results["subspaces"][name] = cached
                print(f"  Resumed from checkpoint")
                continue

        svd_basis = load_svd_basis(layer, "last_token")
        if svd_basis is None and not args.dry_run:
            print(f"  SKIP: no SVD basis for L{layer}")
            continue

        with_subspace = {}
        without_subspace = {}

        for k in K_VALUES:
            print(f"  [{ts()}] k={k}...")

            if args.dry_run:
                base = 0.1 + 0.8 * (1 - np.exp(-k / 50))
                with_acc = float(np.clip(base + 0.15 + rng.normal(0, 0.03), 0, 1))
                without_acc = float(np.clip(base + rng.normal(0, 0.03), 0, 1))
                n = args.n_eval
            else:
                proj_with = build_combined_projection(svd_basis, mask_indices, k)
                c_with, n = graded_accuracy(lm, answer_pairs, layer, proj_with)
                with_acc = c_with / n if n > 0 else 0.0

                proj_without = build_combined_projection(svd_basis, None, k)
                c_without, n = graded_accuracy(lm, answer_pairs, layer, proj_without)
                without_acc = c_without / n if n > 0 else 0.0

            with_subspace[str(k)] = float(with_acc)
            without_subspace[str(k)] = float(without_acc)
            gap = with_acc - without_acc
            print(f"    with_subspace={with_acc:.4f}, "
                  f"without={without_acc:.4f}, gap={gap:+.4f}")

        entry = {
            "layer": layer,
            "rank": rank,
            "k_values": K_VALUES,
            "accuracy_with_subspace": with_subspace,
            "accuracy_without_subspace": without_subspace,
            "marginal_contribution": {
                str(k): with_subspace[str(k)] - without_subspace[str(k)]
                for k in K_VALUES
            },
            "n_samples": n,
            "dry_run": args.dry_run,
        }
        results["subspaces"][name] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("GRADED SUFFICIENCY SUMMARY")
    print(f"{'='*60}")
    for name, r in results["subspaces"].items():
        print(f"\n  {name}:")
        for k in K_VALUES:
            ks = str(k)
            w = r["accuracy_with_subspace"][ks]
            wo = r["accuracy_without_subspace"][ks]
            gap = r["marginal_contribution"][ks]
            print(f"    k={k:>3d}: with={w:.3f} without={wo:.3f} gap={gap:+.3f}")

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
