"""
Distractor insertion test: does the subspace use positional drink-token
location or semantic character-drink binding?

For each model-filtered CausalToM pair, insert an irrelevant drink
mention at three positions:

1. EARLY: after "busy restaurant." — "A customer nearby was drinking [X]."
2. MID: between the two fill actions — "The smell of [X] wafted from the kitchen."
3. LATE: after both fills, before the question — "Another order called for [X]."

The distractor drink is drawn from the CausalToM drink pool, excluding
drinks used in the story. If IIA drops with distractors, the subspace
uses positional features. If IIA is preserved, the subspace tracks
semantic binding.

Only tests answer subspaces (L38, L52, L53).

Usage:
    uv run python experiments/distractor_insertion_test.py --dry-run
    uv run python experiments/distractor_insertion_test.py
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ndif_utils import (
    BELIEF_TRACKING,
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "distractor_insertion"

DISTRACTOR_TEMPLATES = {
    "early": "A customer nearby was drinking {drink}.",
    "mid": "The smell of {drink} wafted from the kitchen.",
    "late": "Another order called for {drink}.",
}

EARLY_ANCHOR = re.compile(r"(busy restaurant\.)")
MID_ANCHOR = re.compile(r"(fills it with \w+\.)\s+(Then )")


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def load_all_drinks():
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        return json.load(f)


def extract_story_drinks(prompt, all_drinks):
    """Extract drink names mentioned in the story portion of the prompt."""
    story_match = re.search(r"Story:\s*(.+?)(?:\nQuestion:)", prompt, re.DOTALL)
    if not story_match:
        return set()
    story = story_match.group(1).lower()
    return {d for d in all_drinks
            if re.search(r'\b' + re.escape(d.lower()) + r'\b', story)}


def pick_distractor(story_drinks, rng, all_drinks):
    """Pick a distractor drink not used in the story."""
    available = [d for d in all_drinks if d not in story_drinks]
    return rng.choice(available)


def insert_distractor(prompt, position, distractor_drink):
    """Insert a distractor sentence into the story portion of a prompt.

    The prompt structure is:
        Instruction: ...
        Story: [story text]
        Question: ...
        Answer:

    We modify only the story text between "Story: " and "Question:".
    """
    story_match = re.search(r"(Story:\s*)(.+?)(\nQuestion:)", prompt, re.DOTALL)
    if not story_match:
        return prompt

    prefix = story_match.group(1)
    story = story_match.group(2)
    suffix = story_match.group(3)

    sentence = DISTRACTOR_TEMPLATES[position].format(drink=distractor_drink)

    if position == "early":
        story = EARLY_ANCHOR.sub(r"\1 " + sentence, story, count=1)
    elif position == "mid":
        story = MID_ANCHOR.sub(r"\1 " + sentence + r" \2", story, count=1)
    elif position == "late":
        story = story.rstrip() + " " + sentence

    return prompt[:story_match.start()] + prefix + story + suffix + prompt[story_match.end():]


def create_distractor_pairs(pairs, position, rng, all_drinks):
    """Create distractor variants for a list of pairs."""
    distractor_pairs = []
    for sample in pairs:
        clean_drinks = extract_story_drinks(sample["clean_prompt"], all_drinks)
        cf_drinks = extract_story_drinks(sample["counterfactual_prompt"], all_drinks)
        all_story_drinks = clean_drinks | cf_drinks

        distractor = pick_distractor(all_story_drinks, rng, all_drinks)

        new_clean = insert_distractor(sample["clean_prompt"], position, distractor)
        new_cf = insert_distractor(sample["counterfactual_prompt"], position, distractor)

        distractor_pairs.append({
            "clean_prompt": new_clean,
            "counterfactual_prompt": new_cf,
            "clean_ans": sample["clean_ans"],
            "counterfactual_ans": sample["counterfactual_ans"],
            "_pair_idx": sample["_pair_idx"],
            "_distractor_drink": distractor,
            "_distractor_position": position,
        })
    return distractor_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Distractor insertion test for Lookback audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Distractor insertion test")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    all_drinks = load_all_drinks()
    lm = None

    # Generate and filter baseline pairs
    if not args.dry_run:
        lm = setup_nnsight()

    baseline_cache = RESULTS_DIR / "filtered_baseline_pairs.json"
    if baseline_cache.exists() and not args.dry_run:
        with open(baseline_cache) as f:
            baseline_pairs = json.load(f)
        print(f"[{ts()}] Loaded {len(baseline_pairs)} cached baseline pairs")
    else:
        print(f"[{ts()}] Generating template-2 pairs...")
        answer_raw, _ = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed)

        if not args.dry_run:
            print(f"[{ts()}] Filtering on model accuracy...")
            baseline_pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
            for i, p in enumerate(baseline_pairs):
                p["_pair_idx"] = i
            with open(baseline_cache, "w") as f:
                json.dump(baseline_pairs, f, indent=2)
        else:
            baseline_pairs = answer_raw[:args.n_eval]
            for i, p in enumerate(baseline_pairs):
                p["_pair_idx"] = i

    print(f"[{ts()}] {len(baseline_pairs)} baseline pairs")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "n_baseline_pairs": len(baseline_pairs),
        "conditions": {},
    }

    conditions = ["baseline", "early", "mid", "late"]

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"[{ts()}] Condition: {condition}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"{condition}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) != args.dry_run:
                print(f"  Stale checkpoint (dry_run mismatch), recomputing")
            else:
                summary["conditions"][condition] = cached
                print(f"  Resumed from checkpoint")
                for sub_name, iia in cached.get("subspace_iias", {}).items():
                    print(f"    {sub_name}: IIA={iia:.4f}")
                continue

        if condition == "baseline":
            eval_pairs = baseline_pairs
            n_shared = len(baseline_pairs)
        else:
            dist_cache = RESULTS_DIR / f"filtered_pairs_{condition}.json"
            if dist_cache.exists() and not args.dry_run:
                with open(dist_cache) as f:
                    eval_pairs = json.load(f)
                n_shared = len(eval_pairs)
                print(f"  Loaded {n_shared} cached distractor pairs")
            else:
                distractor_variants = create_distractor_pairs(
                    baseline_pairs, condition, rng, all_drinks)

                if not args.dry_run:
                    eval_pairs = filter_on_model(
                        lm, distractor_variants, max_size=len(distractor_variants))
                    n_shared = len(eval_pairs)
                    retention = n_shared / len(distractor_variants)
                    print(f"  {n_shared}/{len(distractor_variants)} distractor pairs "
                          f"pass filter (retention={retention:.2f})")
                    with open(dist_cache, "w") as f:
                        json.dump(eval_pairs, f, indent=2)
                else:
                    eval_pairs = distractor_variants
                    n_shared = len(eval_pairs)

        sub_iias = {}
        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                if condition == "baseline":
                    iia = spec["sv_iia"] + rng.normal(0, 0.02)
                else:
                    iia = spec["sv_iia"] * rng.uniform(0.85, 1.05)
                iia = float(np.clip(iia, 0, 1))
            else:
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                proj = build_projection_matrix(svd_basis, mask_indices)
                iia = compute_iia_answer_flex(lm, eval_pairs, layer, proj)

            sub_iias[sub_name] = float(iia)
            print(f"    {sub_name}: IIA={iia:.4f}")

        entry = {
            "dry_run": args.dry_run,
            "condition": condition,
            "n_pairs": n_shared,
            "subspace_iias": sub_iias,
        }
        if condition != "baseline":
            entry["distractor_example"] = eval_pairs[0]["_distractor_drink"] if eval_pairs else None

        summary["conditions"][condition] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"[{ts()}] Saved {ckpt_path}")

    # Compute distractor drops
    baseline_iias = summary["conditions"].get("baseline", {}).get("subspace_iias", {})
    print(f"\n{'='*60}")
    print("DISTRACTOR INSERTION SUMMARY")
    print(f"{'='*60}")

    for condition in conditions:
        cond_iias = summary["conditions"].get(condition, {}).get("subspace_iias", {})
        n = summary["conditions"].get(condition, {}).get("n_pairs", 0)
        print(f"\n{condition} (n={n}):")
        for sub_name in cond_iias:
            iia = cond_iias[sub_name]
            if condition != "baseline" and sub_name in baseline_iias:
                drop = baseline_iias[sub_name] - iia
                print(f"  {sub_name}: IIA={iia:.4f} (drop={drop:+.4f})")
            else:
                print(f"  {sub_name}: IIA={iia:.4f}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
