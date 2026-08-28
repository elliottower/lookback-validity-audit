"""
Necessity test: is the subspace *required* for belief tracking, or redundant?

Zeros out the subspace component from model activations and checks whether
the model still answers belief questions correctly. If accuracy stays high,
the subspace is sufficient but not necessary — the model has redundant
pathways for belief tracking.

Protocol:
  1. Load CausalToM pairs, filter on model accuracy (clean baseline)
  2. For each subspace, zero-ablate: x' = x - (x @ P), where P = V V^T
  3. Run forward pass with ablated activations, check accuracy
  4. Compare: ablated accuracy vs clean accuracy

Predictions:
  Accuracy drops to chance (~0.5): subspace is NECESSARY (supports paper)
  Accuracy stays high (>0.8): subspace is REDUNDANT (weakens paper)

Usage:
    uv run python experiments/necessity_ablation.py --dry-run
    uv run python experiments/necessity_ablation.py --output results/necessity/
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
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

BINDING_POSITIONS = [(167, 155), (168, 156), (155, 167), (156, 168)]


def measure_clean_accuracy(lm, pairs, target_key="counterfactual_ans", retries=3):
    """Measure model accuracy on pairs without any intervention."""
    correct, total = 0, 0
    for sample in pairs:
        prompt = sample["clean_prompt"]
        target = sample[target_key].lower().strip()
        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()
                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                correct += int(pred_tok == target)
                total += 1
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1
    return correct / total if total > 0 else 0.0


def measure_ablated_accuracy_answer(lm, pairs, layer, projection, retries=3):
    """Zero-ablate subspace at given layer and measure accuracy on clean prompts.

    Uses two traces: (1) get clean activations, (2) forward with ablated.
    """
    correct, total = 0, 0
    for sample in tqdm(pairs, desc=f"Ablate L{layer}", leave=False):
        prompt = sample["clean_prompt"]
        target = sample["counterfactual_ans"].lower().strip()
        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()
                cl_t = cl_out.detach().cpu().float()

                ablated = cl_t.clone()
                x = cl_t[-1]
                ablated[-1] = x - (x @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = ablated
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                correct += int(pred_tok == target)
                total += 1
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1
    return correct / total if total > 0 else 0.0


def measure_ablated_accuracy_binding(lm, pairs, layer, projection, retries=3):
    """Zero-ablate subspace at binding token positions."""
    correct, total = 0, 0
    for sample in tqdm(pairs, desc=f"Ablate L{layer} binding", leave=False):
        prompt = sample["clean_prompt"]
        target = sample.get("target", sample.get("counterfactual_ans", "")).lower().strip()
        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()
                cl_t = cl_out.detach().cpu().float()

                ablated = cl_t.clone()
                for tgt_pos, _ in BINDING_POSITIONS:
                    pos = tgt_pos % cl_t.shape[0]
                    x = cl_t[pos]
                    ablated[pos] = x - (x @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = ablated
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                correct += int(pred_tok == target)
                total += 1
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Necessity ablation test")
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/necessity/")
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
        "method": "necessity_ablation",
        "method_note": (
            "Zeros out subspace component from clean activations, "
            "measures whether model still answers correctly. "
            "If accuracy stays high, subspace is redundant."
        ),
        "prediction": {
            "supports_paper": "Accuracy drops to chance (~0.5) after ablation",
            "weakens_paper": "Accuracy stays high (>0.8) after ablation",
        },
        "subspaces": {},
    }

    for sub_name, sub_spec in tqdm(subspaces.items(), desc="Subspaces"):
        layer = sub_spec["layer"]
        rank = sub_spec["rank"]
        lookback_type = sub_spec["lookback_type"]
        is_binding = "binding" in lookback_type

        if args.dry_run:
            clean_acc = float(rng.uniform(0.9, 1.0))
            ablated_acc = float(rng.uniform(0.3, 0.6))
        else:
            vec_type = "state_tokens" if is_binding else "last_token"
            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP {sub_name}: no SVD basis")
                summary["subspaces"][sub_name] = {
                    "skipped": True,
                    "skip_reason": "SVD basis not found",
                }
                continue

            mask_indices = sub_spec.get("mask_indices")
            if mask_indices is not None:
                proj = build_projection_matrix(svd_basis, mask_indices)
            else:
                proj = build_projection_matrix(svd_basis, np.arange(rank))

            pairs = binding_pairs if is_binding else answer_pairs
            target_key = "target" if is_binding else "counterfactual_ans"

            print(f"\n[{ts()}] {sub_name} (L{layer}, rank={rank}, {lookback_type})")

            clean_acc = measure_clean_accuracy(lm, pairs, target_key=target_key)
            print(f"  Clean accuracy: {clean_acc:.4f}")

            if is_binding:
                ablated_acc = measure_ablated_accuracy_binding(lm, pairs, layer, proj)
            else:
                ablated_acc = measure_ablated_accuracy_answer(lm, pairs, layer, proj)
            print(f"  Ablated accuracy: {ablated_acc:.4f}")

        drop = clean_acc - ablated_acc

        result = {
            "clean_accuracy": float(clean_acc),
            "ablated_accuracy": float(ablated_acc),
            "accuracy_drop": float(drop),
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback_type,
            "original_iia": sub_spec["sv_iia"],
            "necessary": drop > 0.3,
        }

        summary["subspaces"][sub_name] = result
        print(f"  {sub_name}: clean={clean_acc:.4f} ablated={ablated_acc:.4f} drop={drop:.4f}")

        sub_path = output_dir / f"{sub_name}.json"
        with open(sub_path, "w") as f:
            json.dump({"timestamp": summary["timestamp"], **result}, f, indent=2)

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    necessary_count = sum(
        1 for v in summary["subspaces"].values()
        if not v.get("skipped") and v.get("necessary", False)
    )
    total_tested = sum(1 for v in summary["subspaces"].values() if not v.get("skipped"))

    summary["verdict"] = f"{necessary_count}/{total_tested} subspaces are necessary (accuracy drops >0.3 on ablation)"

    for name, res in summary["subspaces"].items():
        if res.get("skipped"):
            continue
        status = "NECESSARY" if res["necessary"] else "REDUNDANT"
        print(f"  {name}: drop={res['accuracy_drop']:.4f} — {status}")

    print(f"\nVerdict: {summary['verdict']}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Results saved to {summary_path}")


if __name__ == "__main__":
    main()
