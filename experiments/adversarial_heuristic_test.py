"""
Adversarial heuristic test: does the subspace track beliefs or surface shortcuts?

Three probes:

1. TEMPLATE TRANSFER: The paper trains DAS exclusively on template 2.
   Test whether the same subspace works on templates 0, 1, 3. If IIA
   drops substantially, the subspace learned a template-specific positional
   shortcut, not a general belief-tracking mechanism.

2. RECENCY AUDIT: For each CausalToM pair, check whether the correct
   answer happens to be the most-recently-mentioned drink before the
   question. If the recency heuristic achieves >90% on the evaluation
   set, the task has confounded structure and the subspace might just
   encode "last mentioned state."

3. ENTITY-SWAP ROBUSTNESS: Take working pairs from template 2 but
   replace all character names with novel names not in the training
   entity pool. If IIA drops, the subspace memorized character-specific
   token patterns rather than encoding beliefs abstractly.

Only tests answer subspaces (L38, L52, L53) since binding subspaces
require CausalToM state token positions that differ per template.

Usage:
    uv run python experiments/adversarial_heuristic_test.py --dry-run
    uv run python experiments/adversarial_heuristic_test.py
"""

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ndif_utils import (
    BELIEF_TRACKING,
    REPO_ROOT,
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "adversarial_heuristic"

NOVEL_NAMES = [
    "Takeshi", "Ingrid", "Desmond", "Celeste", "Rajiv",
    "Brenna", "Kofi", "Astrid", "Javier", "Freya",
    "Darius", "Mei", "Arlo", "Priya", "Sven",
    "Nadira", "Otis", "Yuki", "Lennox", "Saba",
]


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def generate_pairs_for_template(template_idx, n_samples, seed):
    """Generate CausalToM pairs using a specific template index."""
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    import sys
    sys.path.insert(0, str(BELIEF_TRACKING))
    from src.dataset import Dataset, Sample

    random.seed(seed)
    np.random.seed(seed)

    clean_configs, counterfactual_configs = [], []
    for _ in range(n_samples):
        chars = random.sample(characters, 2)
        objs = random.sample(bottles, 2)
        states = random.sample(drinks, 2)

        clean_configs.append(Sample(
            template_idx=template_idx,
            characters=chars,
            objects=objs,
            states=states,
        ))

        new_states = random.sample(drinks, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(drinks, 2)

        counterfactual_configs.append(Sample(
            template_idx=template_idx,
            characters=list(reversed(chars)),
            objects=list(reversed(objs)),
            states=new_states,
        ))

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(counterfactual_configs)

    samples = []
    for idx in range(n_samples):
        rc = random.choice([0, 1])
        clean = clean_dataset.__getitem__(idx, set_container=rc, set_character=rc)
        cf = corrupt_dataset.__getitem__(idx, set_container=1 ^ rc, set_character=1 ^ rc)

        samples.append({
            "clean_prompt": clean["prompt"],
            "counterfactual_prompt": cf["prompt"],
            "clean_ans": clean["target"],
            "counterfactual_ans": cf["target"],
            "template_idx": template_idx,
        })

    return samples


def audit_recency_heuristic(pairs):
    """Check whether the correct answer is the most-recently-mentioned drink.

    Returns: fraction of pairs where recency heuristic gives correct answer.
    """
    recency_correct = 0
    total = 0
    for sample in pairs:
        prompt = sample["clean_prompt"]
        target = sample["clean_ans"].lower().strip()

        story_part = prompt.split("Question:")[0] if "Question:" in prompt else prompt
        words = story_part.lower().split()

        last_pos = -1
        for i, w in enumerate(words):
            if target in w:
                last_pos = i

        if last_pos >= 0:
            is_last_mention = True
            for i in range(last_pos + 1, len(words)):
                for other_drink in ["water", "juice", "tea", "coffee", "milk", "soda",
                                     "lemonade", "cola", "sprite", "fanta"]:
                    if other_drink in words[i] and other_drink != target:
                        is_last_mention = False
                        break
            if is_last_mention:
                recency_correct += 1
        total += 1

    return recency_correct / total if total > 0 else 0.0


def swap_character_names(pairs, rng):
    """Replace character names in prompts with novel names."""
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        original_names = json.load(f)

    name_set = sorted(set(original_names))
    swapped_pairs = []

    available_novels = list(NOVEL_NAMES)
    rng.shuffle(available_novels)
    name_map = {}

    for sample in pairs:
        prompt_clean = sample["clean_prompt"]
        prompt_cf = sample["counterfactual_prompt"]

        names_in_prompt = [n for n in name_set
                          if n in prompt_clean or n in prompt_cf]

        for name in names_in_prompt:
            if name not in name_map:
                if not available_novels:
                    raise RuntimeError(
                        f"NOVEL_NAMES exhausted after {len(name_map)} mappings; "
                        f"extend the list to cover all {len(name_set)} characters")
                name_map[name] = available_novels.pop()

        new_clean = prompt_clean
        new_cf = prompt_cf
        for original, novel in sorted(name_map.items(), key=lambda kv: -len(kv[0])):
            pattern = re.compile(rf"\b{re.escape(original)}\b")
            new_clean = pattern.sub(novel, new_clean)
            new_cf = pattern.sub(novel, new_cf)

        swapped_pairs.append({
            "clean_prompt": new_clean,
            "counterfactual_prompt": new_cf,
            "clean_ans": sample["clean_ans"],
            "counterfactual_ans": sample["counterfactual_ans"],
            "name_map": {k: v for k, v in name_map.items() if k in names_in_prompt},
        })

    return swapped_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial heuristic test for Lookback audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] Adversarial heuristic test")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    lm = None
    if not args.dry_run:
        lm = setup_nnsight()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "probes": {},
    }

    # ── Probe 1: Template transfer ──
    print(f"\n{'='*60}")
    print(f"[{ts()}] PROBE 1: Template transfer")
    print(f"{'='*60}")

    template_results = {}
    for tidx in [0, 1, 2, 3]:
        label = f"template_{tidx}" + (" [PAPER]" if tidx == 2 else "")
        print(f"\n[{ts()}] {label}")

        ckpt_path = RESULTS_DIR / f"template_{tidx}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) != args.dry_run:
                print(f"  Stale checkpoint (dry_run mismatch), recomputing")
            else:
                template_results[str(tidx)] = cached
                print(f"  Resumed from checkpoint")
                for sub_name, iia in cached.get("subspace_iias", {}).items():
                    print(f"    {sub_name}: IIA={iia:.4f}")
                continue

        try:
            pairs_raw = generate_pairs_for_template(tidx, args.n_eval * 3, args.seed)
        except Exception as e:
            print(f"  SKIP template {tidx}: {type(e).__name__}: {e}")
            template_results[str(tidx)] = {"error": str(e), "subspace_iias": {}}
            continue

        if not args.dry_run:
            pairs = filter_on_model(lm, pairs_raw, max_size=args.n_eval)
            print(f"  {len(pairs)} pairs passed filter")
        else:
            pairs = pairs_raw[:args.n_eval]

        sub_iias = {}
        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                if tidx == 2:
                    iia = spec["sv_iia"] + rng.normal(0, 0.02)
                else:
                    iia = spec["sv_iia"] * rng.uniform(0.3, 0.8)
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
            "template_idx": tidx,
            "is_paper_template": tidx == 2,
            "n_pairs": len(pairs),
            "subspace_iias": sub_iias,
        }
        template_results[str(tidx)] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

    summary["probes"]["template_transfer"] = {
        "description": "IIA on templates 0,1,3 vs paper's template 2",
        "templates": template_results,
    }

    # ── Probe 2: Recency audit ──
    print(f"\n{'='*60}")
    print(f"[{ts()}] PROBE 2: Recency heuristic audit")
    print(f"{'='*60}")

    recency_ckpt = RESULTS_DIR / "recency_audit.json"
    if recency_ckpt.exists():
        with open(recency_ckpt) as f:
            recency_result = json.load(f)
        if recency_result.get("dry_run", False) != args.dry_run:
            print(f"  Stale recency checkpoint (dry_run mismatch), recomputing")
            recency_result = None
        else:
            print(f"  Resumed: recency_accuracy={recency_result['recency_accuracy']:.4f}")
    else:
        recency_result = None
    if recency_result is None:
        t2_pairs = generate_pairs_for_template(2, args.n_eval * 3, args.seed)
        recency_acc = audit_recency_heuristic(t2_pairs)
        recency_result = {
            "dry_run": args.dry_run,
            "recency_accuracy": float(recency_acc),
            "n_checked": len(t2_pairs),
            "interpretation": (
                "CONFOUNDED (>0.9): task structure allows recency heuristic"
                if recency_acc > 0.9 else
                "CLEAN (<0.9): recency heuristic insufficient"
            ),
        }
        with open(recency_ckpt, "w") as f:
            json.dump(recency_result, f, indent=2)

    print(f"  Recency accuracy: {recency_result['recency_accuracy']:.4f}")
    print(f"  {recency_result['interpretation']}")
    summary["probes"]["recency_audit"] = recency_result

    # ── Probe 3: Entity-swap robustness ──
    print(f"\n{'='*60}")
    print(f"[{ts()}] PROBE 3: Entity-swap robustness")
    print(f"{'='*60}")

    swap_ckpt = RESULTS_DIR / "entity_swap.json"
    if swap_ckpt.exists():
        with open(swap_ckpt) as f:
            swap_result = json.load(f)
        if swap_result.get("dry_run", False) != args.dry_run:
            print(f"  Stale swap checkpoint (dry_run mismatch), recomputing")
            swap_result = None
        else:
            print(f"  Resumed from checkpoint")
    else:
        swap_result = None
    if swap_result is None:
        t2_pairs_raw = generate_pairs_for_template(2, args.n_eval * 3, args.seed)

        if not args.dry_run:
            t2_pairs = filter_on_model(lm, t2_pairs_raw, max_size=args.n_eval)
        else:
            t2_pairs = t2_pairs_raw[:args.n_eval]

        for i, p in enumerate(t2_pairs):
            p["_pair_idx"] = i

        swapped_pairs = swap_character_names(t2_pairs, rng)
        for i, p in enumerate(swapped_pairs):
            p["_pair_idx"] = t2_pairs[i]["_pair_idx"]

        if not args.dry_run:
            swapped_filtered = filter_on_model(lm, swapped_pairs, max_size=len(swapped_pairs))
            surviving_idxs = {p["_pair_idx"] for p in swapped_filtered}
            shared_original = [p for p in t2_pairs if p["_pair_idx"] in surviving_idxs]
            shared_swapped = [p for p in swapped_filtered if p["_pair_idx"] in surviving_idxs]
            assert [p["_pair_idx"] for p in shared_original] == [p["_pair_idx"] for p in shared_swapped]
            retention_rate = len(swapped_filtered) / len(swapped_pairs)
            print(f"  {len(swapped_filtered)}/{len(swapped_pairs)} swapped pairs pass filter (retention={retention_rate:.2f})")
            print(f"  {len(shared_original)} paired examples in intersection")
        else:
            swapped_filtered = swapped_pairs
            shared_original = t2_pairs
            shared_swapped = swapped_pairs
            retention_rate = 1.0

        swap_iias = {}
        original_iias = {}
        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                orig_iia = spec["sv_iia"] + rng.normal(0, 0.02)
                swap_iia = orig_iia * rng.uniform(0.7, 1.15)
            else:
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                proj = build_projection_matrix(svd_basis, mask_indices)

                orig_iia = compute_iia_answer_flex(lm, shared_original, layer, proj)
                swap_iia = compute_iia_answer_flex(lm, shared_swapped, layer, proj)

            original_iias[sub_name] = float(orig_iia)
            swap_iias[sub_name] = float(swap_iia)
            drop = float(orig_iia - swap_iia)
            print(f"  {sub_name}: original={orig_iia:.4f}, swapped={swap_iia:.4f}, drop={drop:.4f}")

        swap_result = {
            "dry_run": args.dry_run,
            "n_original_filtered": len(t2_pairs),
            "n_swapped_filtered": len(swapped_filtered),
            "n_shared_paired": len(shared_original),
            "behavioral_retention_rate": float(retention_rate),
            "original_iias": original_iias,
            "swapped_iias": swap_iias,
            "note": "IIA computed on paired intersection (both original and swapped pass filter)",
            "name_map_sample": swapped_pairs[0].get("name_map", {}) if swapped_pairs else {},
        }
        with open(swap_ckpt, "w") as f:
            json.dump(swap_result, f, indent=2)

    summary["probes"]["entity_swap"] = swap_result

    # ── Summary ──
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ADVERSARIAL HEURISTIC SUMMARY")
    print(f"{'='*60}")

    print("\nProbe 1 — Template transfer:")
    for tidx_str, tres in summary["probes"]["template_transfer"]["templates"].items():
        tidx = int(tidx_str)
        marker = " [PAPER]" if tidx == 2 else ""
        iias = tres.get("subspace_iias", {})
        if iias:
            mean_iia = np.mean(list(iias.values()))
            print(f"  Template {tidx}{marker}: mean IIA={mean_iia:.4f}")
        else:
            print(f"  Template {tidx}{marker}: no data")

    print(f"\nProbe 2 — Recency heuristic: {summary['probes']['recency_audit']['recency_accuracy']:.4f}")

    print("\nProbe 3 — Entity swap:")
    orig = summary["probes"]["entity_swap"].get("original_iias", {})
    swap = summary["probes"]["entity_swap"].get("swapped_iias", {})
    for sub_name in orig:
        drop = orig[sub_name] - swap.get(sub_name, 0)
        print(f"  {sub_name}: {orig[sub_name]:.4f} -> {swap.get(sub_name, 0):.4f} (drop={drop:.4f})")

    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
