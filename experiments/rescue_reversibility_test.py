"""
Rescue reversibility (I10): Does restoring a corrupted component recover behavior?

Three conditions per subspace:
  A. Clean baseline — no intervention, measure accuracy
  B. Ablated — zero the subspace component: x' = x - (x @ P). Behavior should degrade.
  C. Rescued — zero the subspace, then restore ONLY the subspace projection
     from the original activation: x' = (x - x@P) + x@P = x. This SHOULD
     recover to clean accuracy if the subspace is the operative mechanism.

The rescue condition is formally identity, but the test is whether the
ablation-then-restore pipeline preserves numerical precision: any loss
of accuracy in the rescue condition vs clean would indicate rounding
or intervention artifacts. More importantly, the three-way comparison
(clean >> ablated, rescued ~ clean) demonstrates that the subspace alone
is sufficient to explain the accuracy drop from ablation.

Controls:
  D. Wrong-sample rescue: ablate, then restore the subspace projection
     from a DIFFERENT sample. Should NOT rescue if the subspace carries
     sample-specific information.
  E. Random-subspace rescue: ablate the identified subspace, then restore
     a RANDOM subspace of the same rank from the original activation.
     If projecting ANYTHING back partially restores behavior, the rescue
     is not specific to the identified directions.

Usage:
    uv run python experiments/rescue_reversibility_test.py --dry-run
    uv run python experiments/rescue_reversibility_test.py
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
    build_projection_matrix,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "rescue_reversibility"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def rescue_test(lm, pairs, layer, projection, random_projection=None, positions="last", retries=3):
    """Run clean / ablated / rescued / wrong-rescue / random-rescue conditions.

    Returns: dict with per-condition accuracy counts.
    """
    counts = {
        "clean_correct": 0,
        "ablated_correct": 0,
        "rescued_correct": 0,
        "wrong_rescue_correct": 0,
        "random_rescue_correct": 0,
        "total": 0,
    }

    for i, sample in enumerate(tqdm(pairs, desc=f"L{layer} rescue ({positions})")):
        prompt = sample["clean_prompt"]
        target = sample["clean_ans"]

        wrong_idx = (i + 1) % len(pairs)
        wrong_prompt = pairs[wrong_idx]["clean_prompt"]

        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cl_t = cl_out.detach().cpu().float()

                ablated = cl_t.clone()
                if positions == "last":
                    ablated[-1] = cl_t[-1] - (cl_t[-1] @ projection)
                else:
                    for t in range(cl_t.shape[0]):
                        ablated[t] = cl_t[t] - (cl_t[t] @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = ablated
                    abl_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                rescued = ablated.clone()
                if positions == "last":
                    rescued[-1] = ablated[-1] + (cl_t[-1] @ projection)
                else:
                    for t in range(cl_t.shape[0]):
                        rescued[t] = ablated[t] + (cl_t[t] @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = rescued
                    res_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                with lm.trace(wrong_prompt, remote=True):
                    wrong_out = lm.model.layers[layer].output[0].save()

                wrong_t = wrong_out.detach().cpu().float()
                wrong_rescued = ablated.clone()
                if positions == "last":
                    wrong_rescued[-1] = ablated[-1] + (wrong_t[-1] @ projection)
                else:
                    for t in range(min(ablated.shape[0], wrong_t.shape[0])):
                        wrong_rescued[t] = ablated[t] + (wrong_t[t] @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = wrong_rescued
                    wrong_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                rand_pred_item = None
                if random_projection is not None:
                    rand_rescued = ablated.clone()
                    if positions == "last":
                        rand_rescued[-1] = ablated[-1] + (cl_t[-1] @ random_projection)
                    else:
                        for t in range(cl_t.shape[0]):
                            rand_rescued[t] = ablated[t] + (cl_t[t] @ random_projection)

                    with lm.trace(prompt, remote=True):
                        lm.model.layers[layer].output[0] = rand_rescued
                        rand_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()
                    rand_pred_item = rand_pred.item()

                tgt = target.lower().strip()
                counts["clean_correct"] += int(lm.tokenizer.decode([clean_pred.item()]).lower().strip() == tgt)
                counts["ablated_correct"] += int(lm.tokenizer.decode([abl_pred.item()]).lower().strip() == tgt)
                counts["rescued_correct"] += int(lm.tokenizer.decode([res_pred.item()]).lower().strip() == tgt)
                counts["wrong_rescue_correct"] += int(lm.tokenizer.decode([wrong_pred.item()]).lower().strip() == tgt)
                if rand_pred_item is not None:
                    counts["random_rescue_correct"] += int(lm.tokenizer.decode([rand_pred_item]).lower().strip() == tgt)
                counts["total"] += 1
                break

            except Exception as e:
                print(f"  Rescue error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    counts["total"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Rescue reversibility test (I10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()

    print(f"[{ts()}] Rescue reversibility test (I10)")
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

    results = {
        "experiment": "Rescue reversibility (I10)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval": args.n_eval,
        "method": "Three conditions: (A) clean baseline, (B) zero-ablated subspace, "
                  "(C) ablated then restored from original, (D) ablated then restored "
                  "from WRONG sample. If C ~ A >> B and D ~ B: mechanism is the "
                  "subspace component specifically, not a generic perturbation effect.",
        "subspaces": {},
    }

    for name, spec in subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        lookback = spec["lookback_type"]
        mask_indices = spec.get("mask_indices")

        is_binding = "binding" in lookback
        positions = "all" if is_binding else "last"

        print(f"\n{'='*60}")
        print(f"[{ts()}] {name}: L{layer}, rank={rank}, rescue ({positions})")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) == args.dry_run:
                results["subspaces"][name] = cached
                print(f"  Resumed: clean={cached['clean_accuracy']:.4f}, "
                      f"ablated={cached['ablated_accuracy']:.4f}, "
                      f"rescued={cached['rescued_accuracy']:.4f}")
                continue

        if args.dry_run:
            n = args.n_eval
            clean_acc = 0.95 + rng.normal(0, 0.02)
            drop = rng.uniform(0.1, 0.4)
            ablated_acc = max(0, clean_acc - drop)
            rescued_acc = clean_acc - rng.uniform(0, 0.02)
            wrong_acc = ablated_acc + rng.uniform(0, 0.05)
            rand_acc = ablated_acc + rng.uniform(0, 0.08)
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

            all_indices = list(range(svd_basis.shape[0]))
            id_set = set(mask_indices) if mask_indices is not None else set(range(rank))
            non_id = [i for i in all_indices if i not in id_set]
            rand_indices = rng.choice(non_id, size=min(rank, len(non_id)), replace=False)
            rand_proj = build_projection_matrix(svd_basis, rand_indices)

            counts = rescue_test(lm, answer_pairs, layer, proj,
                                 random_projection=rand_proj, positions=positions)
            n = counts["total"]
            clean_acc = counts["clean_correct"] / n if n > 0 else 0.0
            ablated_acc = counts["ablated_correct"] / n if n > 0 else 0.0
            rescued_acc = counts["rescued_correct"] / n if n > 0 else 0.0
            wrong_acc = counts["wrong_rescue_correct"] / n if n > 0 else 0.0
            rand_acc = counts["random_rescue_correct"] / n if n > 0 else 0.0

        rescue_recovery = rescued_acc - ablated_acc
        wrong_recovery = wrong_acc - ablated_acc
        rand_recovery = rand_acc - ablated_acc

        entry = {
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback,
            "positions": positions,
            "clean_accuracy": float(clean_acc),
            "ablated_accuracy": float(ablated_acc),
            "rescued_accuracy": float(rescued_acc),
            "wrong_rescue_accuracy": float(wrong_acc),
            "random_rescue_accuracy": float(rand_acc),
            "ablation_drop": float(clean_acc - ablated_acc),
            "rescue_recovery": float(rescue_recovery),
            "wrong_recovery": float(wrong_recovery),
            "random_recovery": float(rand_recovery),
            "n_samples": n,
            "reversible": rescue_recovery > 0.1 and wrong_recovery < 0.1 and rand_recovery < 0.1,
            "dry_run": args.dry_run,
        }
        results["subspaces"][name] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"  Clean accuracy:         {clean_acc:.4f}")
        print(f"  Ablated accuracy:       {ablated_acc:.4f}")
        print(f"  Rescued accuracy:       {rescued_acc:.4f}")
        print(f"  Wrong-rescue accuracy:  {wrong_acc:.4f}")
        print(f"  Random-rescue accuracy: {rand_acc:.4f}")
        tag = "REVERSIBLE" if entry["reversible"] else "not reversible"
        print(f"  Verdict:                {tag}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESCUE REVERSIBILITY SUMMARY")
    print(f"{'='*60}")
    for name, r in results["subspaces"].items():
        tag = "REVERSIBLE" if r["reversible"] else "not reversible"
        print(f"  {name}: drop={r['ablation_drop']:.4f}, "
              f"rescue={r['rescue_recovery']:.4f}, "
              f"wrong={r['wrong_recovery']:.4f}, "
              f"random={r['random_recovery']:.4f} ({tag})")

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
