"""
Necessity test: are the identified subspaces NECESSARY for correct predictions?

The paper shows subspaces are sufficient (interchange intervention changes
the model's answer). This tests the converse: zero-ablate the subspace
component and check whether the model still answers correctly.

If ablated_accuracy ~ clean_accuracy: the subspace is redundant — the model
has alternative pathways. The paper found a sufficient but unnecessary readout.

If ablated_accuracy << clean_accuracy: the subspace is necessary — removing it
breaks the task. Supports the paper's claim that this is THE mechanism.

Protocol per subspace:
  For answer subspaces (L38, L52, L53):
    1. Get clean output at layer -> CPU
    2. Zero-ablate at position -1: x' = x - (x @ P)
    3. Set ablated output, read prediction, compare to clean_ans

  For binding subspaces (L34, L35, L36):
    1. Get clean output at layer -> CPU
    2. Zero-ablate at ALL positions: x'[t] = x[t] - (x[t] @ P) for all t
    3. Set ablated output, read prediction, compare to clean_ans

Usage:
    uv run python experiments/necessity_test.py
    uv run python experiments/necessity_test.py --dry-run
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
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)


def zero_ablate_accuracy(lm, pairs, layer, projection, positions="last", retries=3):
    """Measure model accuracy after zeroing the subspace component.

    Args:
        lm: nnsight LanguageModel
        pairs: filtered counterfactual pairs
        layer: intervention layer
        projection: (d_model, d_model) projection matrix P = V.T @ V
        positions: "last" (position -1 only) or "all" (full sequence)
        retries: NDIF retry count

    Returns: (clean_correct, ablated_correct, total) counts
    """
    clean_correct, ablated_correct, total = 0, 0, 0

    for sample in tqdm(pairs, desc=f"L{layer} zero-ablation ({positions})"):
        prompt = sample["clean_prompt"]
        target = sample["clean_ans"]

        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cl_t = cl_out.detach().cpu().float()
                ablated = cl_t.clone()

                if positions == "last":
                    x = cl_t[-1]
                    ablated[-1] = x - (x @ projection)
                else:
                    for t in range(cl_t.shape[0]):
                        x = cl_t[t]
                        ablated[t] = x - (x @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = ablated
                    abl_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
                abl_tok = lm.tokenizer.decode([abl_pred.item()]).lower().strip()
                tgt = target.lower().strip()

                clean_correct += int(clean_tok == tgt)
                ablated_correct += int(abl_tok == tgt)
                total += 1
                break

            except Exception as e:
                print(f"  Ablation error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return clean_correct, ablated_correct, total


def main():
    parser = argparse.ArgumentParser(description="Necessity test for lookback subspaces")
    parser.add_argument("--output", type=str, default="results/necessity/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    subspaces = load_subspace_specs()

    print(f"[{ts()}] Necessity test (zero-ablation)")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")

    lm = None
    if not args.dry_run:
        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_raw, binding_raw = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed
        )

        lm = setup_nnsight()

        print(f"[{ts()}] Filtering answer pairs...")
        answer_pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

    results = {
        "experiment": "Necessity test (zero-ablation)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval": args.n_eval,
        "method": "Zero subspace component: x' = x - P @ x. "
                  "Answer subspaces: position -1 only. "
                  "Binding subspaces: all positions.",
        "interpretation": {
            "necessary": "ablated_accuracy << clean_accuracy (subspace removal breaks task)",
            "redundant": "ablated_accuracy ~ clean_accuracy (model has other pathways)",
        },
        "subspaces": {},
    }

    rng = np.random.default_rng(args.seed)

    for name, spec in subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        lookback = spec["lookback_type"]
        mask_indices = spec.get("mask_indices")

        is_binding = "binding" in lookback
        positions = "all" if is_binding else "last"

        print(f"\n{'='*60}")
        print(f"[{ts()}] {name}: L{layer}, rank={rank}, ablation={positions}")
        print(f"{'='*60}")

        ckpt_path = output_dir / f"{name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            results["subspaces"][name] = cached
            print(f"  Resumed: clean={cached['clean_accuracy']:.4f}, ablated={cached['ablated_accuracy']:.4f}")
            continue

        if args.dry_run:
            if is_binding:
                drop = rng.uniform(0.1, 0.4)
            else:
                drop = rng.uniform(0.05, 0.3)
            clean_acc = 0.95 + rng.normal(0, 0.02)
            ablated_acc = max(0, clean_acc - drop + rng.normal(0, 0.03))
            n = args.n_eval
        else:
            vec_type = "state_tokens" if is_binding else "last_token"
            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: no SVD basis for L{layer}/{vec_type}")
                continue

            if mask_indices is not None:
                proj = build_projection_matrix(svd_basis, mask_indices)
            else:
                proj = build_projection_matrix(svd_basis, np.arange(rank))

            clean_c, ablated_c, n = zero_ablate_accuracy(
                lm, answer_pairs, layer, proj, positions=positions
            )
            clean_acc = clean_c / n if n > 0 else 0.0
            ablated_acc = ablated_c / n if n > 0 else 0.0

        drop = clean_acc - ablated_acc
        entry = {
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback,
            "ablation_positions": positions,
            "clean_accuracy": float(clean_acc),
            "ablated_accuracy": float(ablated_acc),
            "accuracy_drop": float(drop),
            "n_samples": n,
            "necessary": drop > 0.1,
        }
        results["subspaces"][name] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"  Clean accuracy:   {clean_acc:.4f}")
        print(f"  Ablated accuracy: {ablated_acc:.4f}")
        print(f"  Drop:             {drop:.4f} {'(NECESSARY)' if drop > 0.1 else '(redundant)'}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for name, r in results["subspaces"].items():
        tag = "NECESSARY" if r["necessary"] else "redundant"
        print(f"  {name}: drop={r['accuracy_drop']:.4f} ({tag})")

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
