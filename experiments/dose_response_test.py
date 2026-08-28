"""
Dose-response test (M1): does IIA degrade proportionally with partial ablation?

E7 (necessity) is binary: zero-ablate or don't. This test ablates at graded
fractions (0%, 25%, 50%, 75%, 100%) and measures accuracy at each level.

If accuracy drops linearly with ablation fraction, the subspace contributes
proportionally — consistent with a real mechanism, not an on/off artifact.

If accuracy is flat until 100% then crashes, the signal is threshold-like —
redundant pathways compensate until complete removal.

Only tests answer subspaces (L38, L52, L53) at position -1.

Usage:
    uv run python experiments/dose_response_test.py --dry-run
    uv run python experiments/dose_response_test.py
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
    wilson_ci,
)

RESULTS_DIR = REPO_ROOT / "results" / "dose_response"

ABLATION_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def partial_ablate_accuracy(lm, pairs, layer, projection, fraction, retries=3):
    """Measure accuracy after ablating `fraction` of the subspace component.

    x' = x - fraction * (P @ x)
    fraction=0 is clean, fraction=1 is full zero-ablation.
    """
    clean_correct, ablated_correct, total = 0, 0, 0

    for sample in tqdm(pairs, desc=f"L{layer} ablation={fraction:.0%}", leave=False):
        prompt = sample["clean_prompt"]
        target = sample["clean_ans"].lower().strip()

        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cl_t = cl_out.detach().cpu().float()
                ablated = cl_t.clone()
                x = cl_t[-1]
                ablated[-1] = x - fraction * (x @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = ablated
                    abl_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
                abl_tok = lm.tokenizer.decode([abl_pred.item()]).lower().strip()

                clean_correct += int(clean_tok == target)
                ablated_correct += int(abl_tok == target)
                total += 1
                break

            except Exception as e:
                print(f"  Error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return clean_correct, ablated_correct, total


def main():
    parser = argparse.ArgumentParser(description="Dose-response test for lookback subspaces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Dose-response test (M1)")
    print(f"[{ts()}] Fractions: {ABLATION_FRACTIONS}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    lm = None
    pairs = None

    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight...")
        lm = setup_nnsight()

        pairs_path = RESULTS_DIR / "filtered_answer_pairs.json"
        if pairs_path.exists():
            with open(pairs_path) as f:
                pairs = json.load(f)
            print(f"[{ts()}] Loaded cached pairs: {len(pairs)}")
        else:
            print(f"[{ts()}] Generating pairs...")
            answer_raw, _ = generate_counterfactual_pairs(
                n_samples=args.n_eval * 3, seed=args.seed)
            print(f"[{ts()}] Filtering on model accuracy...")
            pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
            with open(pairs_path, "w") as f:
                json.dump(pairs, f, indent=2)
            print(f"[{ts()}] {len(pairs)} pairs passed filter")

    for sub_name, spec in answer_subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        mask_indices = spec.get("mask_indices")

        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            print(f"\n[{ts()}] {sub_name}: checkpoint exists, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{ts()}] {sub_name}: layer={layer}, rank={rank}")
        print(f"{'='*60}")

        if not args.dry_run:
            svd_basis = load_svd_basis(layer, "last_token")
            if svd_basis is None:
                print(f"  SKIP: no SVD basis for L{layer}")
                continue
            proj = build_projection_matrix(svd_basis, mask_indices)

        dose_results = []
        for fraction in ABLATION_FRACTIONS:
            print(f"\n  [{ts()}] Fraction: {fraction:.0%}")

            if args.dry_run:
                n = args.n_eval
                clean_acc = 0.95
                drop = fraction * rng.uniform(0.05, 0.15)
                ablated_acc = max(0, clean_acc - drop)
            else:
                clean_c, ablated_c, n = partial_ablate_accuracy(
                    lm, pairs, layer, proj, fraction)
                clean_acc = clean_c / n if n > 0 else 0.0
                ablated_acc = ablated_c / n if n > 0 else 0.0

            k_abl = int(round(ablated_acc * n))
            lo, hi = wilson_ci(k_abl, n)

            entry = {
                "fraction": fraction,
                "clean_accuracy": float(clean_acc),
                "ablated_accuracy": float(ablated_acc),
                "accuracy_drop": float(clean_acc - ablated_acc),
                "n_samples": n,
                "wilson_ci_lower": lo,
                "wilson_ci_upper": hi,
            }
            dose_results.append(entry)
            print(f"    Clean: {clean_acc:.4f}, Ablated: {ablated_acc:.4f}, "
                  f"Drop: {clean_acc - ablated_acc:.4f}, CI: [{lo:.4f}, {hi:.4f}]")

        # Compute linearity: correlation between fraction and accuracy drop
        fracs = [d["fraction"] for d in dose_results]
        drops = [d["accuracy_drop"] for d in dose_results]
        if len(set(drops)) > 1:
            correlation = float(np.corrcoef(fracs, drops)[0, 1])
        else:
            correlation = 0.0

        full_drop = dose_results[-1]["accuracy_drop"]
        half_drop = dose_results[2]["accuracy_drop"]  # fraction=0.50
        proportionality = half_drop / full_drop if full_drop > 0.01 else None

        sub_result = {
            "layer": layer,
            "rank": rank,
            "mask_indices": mask_indices,
            "dose_curve": dose_results,
            "linearity_correlation": correlation,
            "proportionality_at_50pct": proportionality,
            "interpretation": (
                "PROPORTIONAL" if correlation > 0.9
                else "THRESHOLD" if correlation < 0.5
                else "MIXED"
            ),
        }

        with open(ckpt_path, "w") as f:
            json.dump(sub_result, f, indent=2)
        print(f"  [{ts()}] Saved {ckpt_path}")
        print(f"  Linearity r={correlation:.3f} -> {sub_result['interpretation']}")

    # Write summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "fractions": ABLATION_FRACTIONS,
        "subspaces": {},
    }
    for sub_name in answer_subspaces:
        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                summary["subspaces"][sub_name] = json.load(f)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("DOSE-RESPONSE SUMMARY")
    print(f"{'='*60}")
    for sub_name, res in summary["subspaces"].items():
        print(f"\n{sub_name}: {res['interpretation']} (r={res['linearity_correlation']:.3f})")
        for d in res["dose_curve"]:
            print(f"  {d['fraction']:5.0%}: acc={d['ablated_accuracy']:.4f} "
                  f"(drop={d['accuracy_drop']:+.4f}) "
                  f"CI=[{d['wilson_ci_lower']:.4f}, {d['wilson_ci_upper']:.4f}]")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
