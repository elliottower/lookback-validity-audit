"""
Double dissociation test (I6): can the belief circuit be separated from
general language processing?

Tests whether ablating the belief subspace selectively impairs belief-tracking
while preserving general next-token prediction, and whether ablating a
control subspace of the same rank preserves belief-tracking while having
comparable impact on general prediction.

Conditions:
  A. Ablate belief subspace -> measure belief IIA AND general accuracy
  B. Ablate random subspace of same rank -> measure belief IIA AND general accuracy

If condition A shows: belief IIA drops, general accuracy preserved
And condition B shows: belief IIA preserved, general accuracy comparable
That is a single dissociation: the belief subspace is functionally specialized.

NOTE: A true double dissociation requires a second known circuit (Circuit B)
with its own task. We lack a known non-belief circuit at these layers, so
this test establishes single dissociation (selective impairment) rather
than the full double. The general-accuracy prompts are NOT matched in
token positions or answer vocabulary to CausalToM, so asymmetries could
reflect task-difficulty artifacts. This limitation is reported explicitly.

General accuracy is measured on 40 diverse prompts (factual questions,
sentence completion, simple math) where the model's clean prediction is
taken as ground truth. After ablation, we check whether the model still
produces the same top-1 token.

Usage:
    uv run python experiments/double_dissociation_test.py --dry-run
    uv run python experiments/double_dissociation_test.py
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
    compute_iia_answer_flex,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "double_dissociation"

GENERAL_PROMPTS = [
    "The capital of France is",
    "Water boils at 100 degrees",
    "The largest planet in our solar system is",
    "In 1969, the first person to walk on the moon was",
    "The chemical symbol for gold is",
    "Shakespeare wrote the play Romeo and",
    "The speed of light is approximately 300,000 kilometers per",
    "DNA stands for deoxyribonucleic",
    "The Great Wall of China was built to protect against",
    "Photosynthesis converts sunlight into",
    "The Pythagorean theorem states that a squared plus b squared equals c",
    "The human body has 206",
    "Mount Everest is located in",
    "The currency of Japan is the",
    "Einstein's most famous equation is E equals mc",
    "The Amazon River flows through South",
    "Oxygen makes up about 21 percent of Earth's",
    "The Mona Lisa was painted by Leonardo da",
    "Antibiotics are used to treat bacterial",
    "The periodic table organizes chemical",
    "A triangle has three sides and three",
    "The tallest mammal on Earth is the",
    "Mars is often called the red",
    "The first computer programmer was Ada",
    "Mitochondria are often called the powerhouse of the",
    "The Nile is the longest river in",
    "Gravity on Earth accelerates objects at 9.8 meters per second",
    "The United Nations headquarters is in New",
    "Hemoglobin carries oxygen in the",
    "The boiling point of water at sea level is 100 degrees",
    "Beethoven composed his ninth symphony while he was",
    "The primary colors are red, blue, and",
    "Carbon dioxide is exhaled by humans and absorbed by",
    "The speed of sound in air is approximately 343 meters per",
    "Insulin is produced by the",
    "The Sahara is the largest hot desert in",
    "A leap year occurs every four",
    "The smallest bone in the human body is in the",
    "Plato was a student of",
    "The pH of pure water is",
]


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def measure_general_accuracy_after_ablation(lm, prompts, layer, projection, retries=3):
    """Measure how many general prompts produce the same top-1 token after ablation.

    Returns: (n_preserved, n_total, per_prompt_results)
    """
    preserved, total = 0, 0
    per_prompt = []

    for prompt in tqdm(prompts, desc=f"General accuracy L{layer}"):
        for attempt in range(retries):
            try:
                with lm.trace(prompt, remote=True):
                    clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                with lm.trace(prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cl_t = cl_out.detach().cpu().float()
                ablated = cl_t.clone()
                x = cl_t[-1]
                ablated[-1] = x - (x @ projection)

                with lm.trace(prompt, remote=True):
                    lm.model.layers[layer].output[0] = ablated
                    abl_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
                abl_tok = lm.tokenizer.decode([abl_pred.item()]).lower().strip()

                match = clean_tok == abl_tok
                preserved += int(match)
                total += 1
                per_prompt.append({
                    "prompt": prompt[:60],
                    "clean": clean_tok,
                    "ablated": abl_tok,
                    "preserved": match,
                })
                break

            except Exception as e:
                print(f"  General accuracy error (attempt {attempt+1}): "
                      f"{type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1
                    per_prompt.append({
                        "prompt": prompt[:60],
                        "clean": None,
                        "ablated": None,
                        "preserved": False,
                    })

    return preserved, total, per_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Double dissociation test (I6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Double dissociation test (I6)")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}, "
          f"N random controls: {args.n_random}")

    lm = None
    belief_pairs = None

    if not args.dry_run:
        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_raw, _ = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed)
        lm = setup_nnsight()

        pairs_cache = RESULTS_DIR / "filtered_belief_pairs.json"
        if pairs_cache.exists():
            with open(pairs_cache) as f:
                saved = json.load(f)
            belief_pairs = saved["pairs"]
            print(f"[{ts()}] Loaded {len(belief_pairs)} cached belief pairs")
        else:
            print(f"[{ts()}] Filtering belief pairs on model accuracy...")
            belief_pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
            print(f"[{ts()}] {len(belief_pairs)} pairs passed filter")
            with open(pairs_cache, "w") as f:
                json.dump({"pairs": belief_pairs}, f, indent=2)

    summary = {
        "experiment": "Double dissociation (I6)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval": args.n_eval,
        "n_random_controls": args.n_random,
        "n_general_prompts": len(GENERAL_PROMPTS),
        "method": (
            "Condition A: ablate identified belief subspace, measure "
            "(1) belief IIA and (2) general next-token preservation rate. "
            "Condition B: ablate random subspace of same rank (averaged "
            "over n_random draws), measure same two metrics. "
            "Double dissociation: A drops belief but preserves general; "
            "B preserves belief but has comparable general impact."
        ),
        "subspaces": {},
    }

    for sub_name, spec in answer_subspaces.items():
        layer = spec["layer"]
        rank = spec["rank"]
        mask_indices = spec.get("mask_indices")

        print(f"\n{'='*60}")
        print(f"[{ts()}] {sub_name}: L{layer}, rank={rank}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{sub_name}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) == args.dry_run:
                summary["subspaces"][sub_name] = cached
                print(f"  Resumed from checkpoint")
                continue

        svd_basis = load_svd_basis(layer, "last_token")
        if svd_basis is None and not args.dry_run:
            print(f"  SKIP: no SVD basis for L{layer}")
            continue

        n_total = svd_basis.shape[0] if svd_basis is not None else 500

        if args.dry_run:
            belief_iia_identified = 0.05 + rng.normal(0, 0.02)
            general_preserved_identified = 0.92 + rng.normal(0, 0.03)

            belief_iia_randoms = [float(0.85 + rng.normal(0, 0.05))
                                  for _ in range(args.n_random)]
            general_preserved_randoms = [float(0.90 + rng.normal(0, 0.03))
                                         for _ in range(args.n_random)]
        else:
            # --- Condition A: Ablate identified belief subspace ---
            print(f"  [{ts()}] Condition A: ablating identified subspace...")
            proj_id = build_projection_matrix(svd_basis, mask_indices)

            print(f"  [{ts()}] Measuring belief IIA after ablation...")
            belief_iia_identified = compute_iia_answer_flex(
                lm, belief_pairs, layer, proj_id)
            print(f"    Belief IIA (ablated): {belief_iia_identified:.4f}")

            print(f"  [{ts()}] Measuring general accuracy after ablation...")
            gen_pres, gen_tot, gen_details = measure_general_accuracy_after_ablation(
                lm, GENERAL_PROMPTS, layer, proj_id)
            general_preserved_identified = gen_pres / gen_tot if gen_tot > 0 else 0.0
            print(f"    General preserved: {gen_pres}/{gen_tot} "
                  f"({general_preserved_identified:.4f})")

            # --- Condition B: Ablate random subspaces ---
            print(f"  [{ts()}] Condition B: ablating {args.n_random} random subspaces...")
            all_indices = list(range(n_total))
            belief_iia_randoms = []
            general_preserved_randoms = []

            for i in range(args.n_random):
                rand_indices = rng.choice(all_indices, size=rank, replace=False)
                proj_rand = build_projection_matrix(svd_basis, rand_indices)

                rand_belief_iia = compute_iia_answer_flex(
                    lm, belief_pairs, layer, proj_rand)
                belief_iia_randoms.append(float(rand_belief_iia))

                rp, rt, _ = measure_general_accuracy_after_ablation(
                    lm, GENERAL_PROMPTS, layer, proj_rand)
                rand_general = rp / rt if rt > 0 else 0.0
                general_preserved_randoms.append(float(rand_general))

                print(f"    Random {i+1}/{args.n_random}: "
                      f"belief_IIA={rand_belief_iia:.4f}, "
                      f"general={rand_general:.4f}")

        belief_drop = float(np.mean(belief_iia_randoms)) - float(belief_iia_identified)
        general_drop_id = 1.0 - float(general_preserved_identified) if not args.dry_run else 1.0 - float(general_preserved_identified)
        general_drop_rand = 1.0 - float(np.mean(general_preserved_randoms))

        entry = {
            "dry_run": args.dry_run,
            "layer": layer,
            "rank": rank,
            "condition_a_identified": {
                "belief_iia_after_ablation": float(belief_iia_identified),
                "general_preservation_rate": float(general_preserved_identified),
            },
            "condition_b_random_mean": {
                "belief_iia_after_ablation": float(np.mean(belief_iia_randoms)),
                "belief_iia_std": float(np.std(belief_iia_randoms)),
                "general_preservation_rate": float(np.mean(general_preserved_randoms)),
                "general_preservation_std": float(np.std(general_preserved_randoms)),
                "n_random": args.n_random,
                "belief_iia_all": belief_iia_randoms,
                "general_preservation_all": general_preserved_randoms,
            },
            "dissociation_metrics": {
                "belief_iia_drop_from_random": belief_drop,
                "general_drop_identified": general_drop_id,
                "general_drop_random_mean": general_drop_rand,
                "selective_belief_impairment": belief_drop > 0.3 and general_drop_id < 0.2,
            },
        }
        summary["subspaces"][sub_name] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"\n  Condition A (identified subspace ablated):")
        print(f"    Belief IIA:      {belief_iia_identified:.4f}")
        print(f"    General preserved: {general_preserved_identified:.4f}")
        print(f"  Condition B (random subspace ablated, mean of {args.n_random}):")
        print(f"    Belief IIA:      {np.mean(belief_iia_randoms):.4f} +/- "
              f"{np.std(belief_iia_randoms):.4f}")
        print(f"    General preserved: {np.mean(general_preserved_randoms):.4f} +/- "
              f"{np.std(general_preserved_randoms):.4f}")
        print(f"  Dissociation:")
        print(f"    Belief IIA drop (random - identified): {belief_drop:+.4f}")
        verdict = "YES" if belief_drop > 0.3 and general_drop_id < 0.2 else "NO"
        print(f"    Selective impairment: {verdict}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("DOUBLE DISSOCIATION SUMMARY")
    print(f"{'='*60}")
    for name, r in summary["subspaces"].items():
        ca = r["condition_a_identified"]
        cb = r["condition_b_random_mean"]
        sel = r["dissociation_metrics"]["selective_belief_impairment"]
        tag = "DISSOCIATED" if sel else "not dissociated"
        print(f"  {name}: ablate_belief_IIA={ca['belief_iia_after_ablation']:.3f}, "
              f"ablate_rand_IIA={cb['belief_iia_after_ablation']:.3f}, "
              f"general={ca['general_preservation_rate']:.3f} ({tag})")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
