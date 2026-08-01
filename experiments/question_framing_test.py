"""
Question-framing control: does the subspace encode beliefs or entity bindings?

The paper's CausalToM task always asks characters about containers they
filled, so belief = reality. A subspace achieving high IIA could encode
"what did this character put here?" without representing epistemic state.

This test uses IDENTICAL stories but swaps the question framing:
  1. Belief (original): "What does [char] believe the [container] contains?"
  2. Action recall: "What did [char] put in the [container]?"
  3. Reality state: "What is inside the [container]?"
  4. Fill completion: "[Char] filled the [container] with"

All framings yield the same correct answer. If IIA matches across
framings, the subspace encodes entity-state bindings, not beliefs.

Only tests answer subspaces (L38, L52, L53).

Usage:
    uv run python experiments/question_framing_test.py --dry-run
    uv run python experiments/question_framing_test.py
"""

import argparse
import json
import re
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

RESULTS_DIR = REPO_ROOT / "results" / "question_framing"

BELIEF_QUESTION_RE = re.compile(
    r"What does (.+?) believe the (.+?) contains\?"
)

FRAMINGS = {
    "belief": "What does {char} believe the {container} contains?",
    "action_recall": "What did {char} put in the {container}?",
    "reality_state": "What is inside the {container}?",
    "fill_completion": "{char} filled the {container} with",
}


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def extract_char_container(question_text):
    """Extract character name and container from a belief question."""
    m = BELIEF_QUESTION_RE.search(question_text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def reframe_prompt(prompt, question_text, framing_key):
    """Replace the Question line in a prompt with a new framing."""
    char, container = extract_char_container(question_text)
    if char is None:
        return None

    new_question = FRAMINGS[framing_key].format(char=char, container=container)
    old_line = f"Question: {question_text}"
    new_line = f"Question: {new_question}"

    if old_line not in prompt:
        return None
    return prompt.replace(old_line, new_line)


def reframe_pairs(pairs, framing_key):
    """Create reframed copies of all pairs for a given question framing."""
    reframed = []
    for i, sample in enumerate(pairs):
        clean_reframed = reframe_prompt(
            sample["clean_prompt"], sample["clean_question"], framing_key)
        cf_reframed = reframe_prompt(
            sample["counterfactual_prompt"],
            sample["counterfactual_question"], framing_key)

        if clean_reframed is None or cf_reframed is None:
            continue

        reframed.append({
            "clean_prompt": clean_reframed,
            "counterfactual_prompt": cf_reframed,
            "clean_ans": sample["clean_ans"],
            "counterfactual_ans": sample["counterfactual_ans"],
            "_pair_idx": i,
        })
    return reframed


def main():
    parser = argparse.ArgumentParser(
        description="Question-framing control for Lookback audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Question-framing control test")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")
    print(f"[{ts()}] Framings: {list(FRAMINGS.keys())}")

    lm = None
    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight...")
        lm = setup_nnsight()

    print(f"[{ts()}] Generating base pairs...")
    answer_raw, _ = generate_counterfactual_pairs(
        n_samples=args.n_eval * 3, seed=args.seed)
    print(f"[{ts()}] {len(answer_raw)} raw pairs generated")

    # Tag original indices before any filtering
    for i, p in enumerate(answer_raw):
        p["_pair_idx"] = i

    # For each framing, create reframed pairs and filter independently
    framing_filtered = {}
    for framing_key in FRAMINGS:
        print(f"\n[{ts()}] Framing: {framing_key}")

        cache_path = RESULTS_DIR / f"filtered_pairs_{framing_key}.json"
        if cache_path.exists() and not args.dry_run:
            with open(cache_path) as f:
                framing_filtered[framing_key] = json.load(f)
            print(f"  Loaded {len(framing_filtered[framing_key])} cached filtered pairs")
            continue

        if framing_key == "belief":
            reframed = [{**p} for p in answer_raw]
        else:
            reframed = reframe_pairs(answer_raw, framing_key)
            print(f"  {len(reframed)}/{len(answer_raw)} pairs reframed successfully")

        if not args.dry_run:
            filtered = filter_on_model(lm, reframed, max_size=args.n_eval)
            with open(cache_path, "w") as f:
                json.dump(filtered, f, indent=2)
        else:
            filtered = reframed[:args.n_eval]
        print(f"  {len(filtered)} pairs passed filter")
        framing_filtered[framing_key] = filtered

    # Compute intersection: pairs that pass ALL framings
    passing_per_framing = [
        {p["_pair_idx"] for p in framing_filtered[fk]}
        for fk in FRAMINGS
    ]
    shared_idxs = set.intersection(*passing_per_framing)
    print(f"\n[{ts()}] Intersection: {len(shared_idxs)} pairs pass all framings")

    # Build aligned subsets for each framing
    framing_shared = {}
    for framing_key in FRAMINGS:
        subset = sorted(
            [p for p in framing_filtered[framing_key] if p["_pair_idx"] in shared_idxs],
            key=lambda p: p["_pair_idx"],
        )
        framing_shared[framing_key] = subset
        assert len(subset) == len(shared_idxs), (
            f"{framing_key}: {len(subset)} != {len(shared_idxs)}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "n_raw": len(answer_raw),
        "n_shared": len(shared_idxs),
        "n_per_framing": {fk: len(framing_filtered[fk]) for fk in FRAMINGS},
        "framings": {},
    }

    for framing_key in FRAMINGS:
        print(f"\n{'='*60}")
        print(f"[{ts()}] Computing IIA for framing: {framing_key}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{framing_key}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) != args.dry_run:
                print(f"  Stale checkpoint (dry_run mismatch), recomputing")
            else:
                summary["framings"][framing_key] = cached
                print(f"  Resumed from checkpoint")
                for sub_name, iia in cached.get("subspace_iias", {}).items():
                    print(f"    {sub_name}: IIA={iia:.4f}")
                continue

        pairs = framing_shared[framing_key]
        sub_iias = {}

        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                if framing_key == "belief":
                    iia = spec["sv_iia"] + rng.normal(0, 0.02)
                else:
                    iia = spec["sv_iia"] + rng.normal(0, 0.05)
                iia = float(np.clip(iia, 0, 1))
            else:
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                proj = build_projection_matrix(svd_basis, mask_indices)
                iia = compute_iia_answer_flex(lm, pairs, layer, proj)

            sub_iias[sub_name] = float(iia)
            print(f"    {sub_name}: IIA={iia:.4f}")

        entry = {
            "dry_run": args.dry_run,
            "framing": framing_key,
            "question_template": FRAMINGS[framing_key],
            "n_pairs": len(pairs),
            "subspace_iias": sub_iias,
        }
        summary["framings"][framing_key] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"[{ts()}] Saved {ckpt_path}")

    # Compute framing transfer ratios
    belief_iias = summary["framings"].get("belief", {}).get("subspace_iias", {})
    transfer_ratios = {}
    for framing_key in FRAMINGS:
        if framing_key == "belief":
            continue
        other_iias = summary["framings"].get(framing_key, {}).get("subspace_iias", {})
        ratios = {}
        for sub_name in belief_iias:
            b = belief_iias.get(sub_name, 0)
            o = other_iias.get(sub_name, 0)
            ratios[sub_name] = float(o / b) if b > 0 else 0.0
        transfer_ratios[framing_key] = ratios
    summary["transfer_ratios"] = transfer_ratios

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("QUESTION-FRAMING SUMMARY")
    print(f"{'='*60}")
    print(f"\nShared pairs across all framings: {len(shared_idxs)}")

    for framing_key in FRAMINGS:
        fdata = summary["framings"].get(framing_key, {})
        iias = fdata.get("subspace_iias", {})
        marker = " [BASELINE]" if framing_key == "belief" else ""
        if iias:
            mean_iia = np.mean(list(iias.values()))
            print(f"\n  {framing_key}{marker}: mean IIA={mean_iia:.4f}")
            for sub_name, iia in iias.items():
                ratio_str = ""
                if framing_key != "belief" and sub_name in belief_iias:
                    ratio = transfer_ratios.get(framing_key, {}).get(sub_name, 0)
                    ratio_str = f" (ratio={ratio:.2f})"
                print(f"    {sub_name}: IIA={iia:.4f}{ratio_str}")

    print(f"\nInterpretation:")
    all_within_threshold = True
    for framing_key, ratios in transfer_ratios.items():
        for sub_name, ratio in ratios.items():
            diff = abs(belief_iias.get(sub_name, 0) -
                       summary["framings"][framing_key]["subspace_iias"].get(sub_name, 0))
            if diff > 0.10:
                all_within_threshold = False
                print(f"  {framing_key}/{sub_name}: diff={diff:.4f} > 0.10 threshold")

    if all_within_threshold:
        print("  All framings within 0.10 of belief baseline.")
        print("  -> Subspace encodes entity-state bindings, not beliefs.")
    else:
        print("  Some framings differ by > 0.10 from belief baseline.")
        print("  -> Subspace may be sensitive to epistemic question framing.")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
