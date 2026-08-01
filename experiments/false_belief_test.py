"""
True false-belief test: can the subspace handle genuine epistemic states?

The paper's CausalToM stories never include drink swaps — the causal_event
template field exists but is never included in generated stories. Every
evaluated pair asks the character about the container they themselves filled,
so belief = reality. The subspace may encode entity-state bindings rather
than epistemic state.

This test constructs stories with actual false beliefs: a drink is swapped
while the character is distracted, and the character doesn't notice. The
belief-correct answer is the PRE-swap drink (false belief), not the current
contents (reality).

If IIA is high: the subspace genuinely tracks epistemic state.
If IIA is low: the subspace only tracks entity bindings that happen to
align with belief in the paper's task format.

Usage:
    uv run python experiments/false_belief_test.py --dry-run
    uv run python experiments/false_belief_test.py
"""

import argparse
import json
import random
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
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "false_belief"

INSTRUCTION = (
    "1. Track the belief of each character as described in the story. "
    "2. A character's belief is formed only when they perform an action "
    "themselves or can observe the action taking place. "
    "3. A character does not have any beliefs about the container and its "
    "contents which they cannot observe. "
    "4. To answer the question, predict only what is inside the queried "
    "container, strictly based on the belief of the character, mentioned "
    "in the question. "
    "5. If the queried character has no belief about the container in "
    "question, then predict 'unknown'. "
    "6. Do not predict container or character as the final output."
)


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def make_false_belief_story(char1, char2, container1, container2,
                            state1, state2, swap_state):
    """Build a story where char1 falsely believes container1 has state1."""
    story = (
        f"{char1} and {char2} are working in a busy restaurant. "
        f"To complete an order, {char1} grabs an opaque {container1} and "
        f"fills it with {state1}. Then {char2} grabs another opaque "
        f"{container2} and fills it with {state2}. "
        f"While {char1} was attending to another task, a co-worker swapped "
        f"the {state1} in the {container1} with {swap_state}. "
        f"{char1} did not notice the swap."
    )
    return story


def make_prompt(story, question):
    return f"Instruction: {INSTRUCTION}\n\nStory: {story}\nQuestion: {question}\nAnswer:"


def generate_false_belief_pairs(n_samples, seed):
    """Generate clean/CF pairs for the false-belief condition."""
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    random.seed(seed)
    np.random.seed(seed)

    pairs = []
    for _ in range(n_samples):
        chars = random.sample(characters, 2)
        containers = random.sample(bottles, 2)
        states = random.sample(drinks, 3)
        state1, state2, swap_state = states[0], states[1], states[2]

        clean_story = make_false_belief_story(
            chars[0], chars[1], containers[0], containers[1],
            state1, state2, swap_state,
        )
        clean_question = (
            f"What does {chars[0]} believe the {containers[0]} contains?"
        )
        clean_prompt = make_prompt(clean_story, clean_question)

        cf_chars = list(reversed(chars))
        cf_containers = list(reversed(containers))
        cf_states = random.sample(drinks, 3)
        while any(s in [state1, state2, swap_state] for s in cf_states):
            cf_states = random.sample(drinks, 3)
        cf_state1, cf_state2, cf_swap = cf_states[0], cf_states[1], cf_states[2]

        cf_story = make_false_belief_story(
            cf_chars[0], cf_chars[1], cf_containers[0], cf_containers[1],
            cf_state1, cf_state2, cf_swap,
        )
        cf_question = (
            f"What does {cf_chars[0]} believe the {cf_containers[0]} contains?"
        )
        cf_prompt = make_prompt(cf_story, cf_question)

        pairs.append({
            "clean_prompt": clean_prompt,
            "counterfactual_prompt": cf_prompt,
            "clean_ans": state1,
            "counterfactual_ans": cf_state1,
            "clean_reality": swap_state,
            "cf_reality": cf_swap,
            "clean_chars": chars,
            "cf_chars": cf_chars,
        })

    return pairs


def generate_reality_pairs(n_samples, seed):
    """Generate clean/CF pairs for the reality-question control.

    Same stories, but asks "What is actually in [container]?" instead of
    "What does [char] believe [container] contains?"
    """
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    random.seed(seed)
    np.random.seed(seed)

    pairs = []
    for _ in range(n_samples):
        chars = random.sample(characters, 2)
        containers = random.sample(bottles, 2)
        states = random.sample(drinks, 3)
        state1, state2, swap_state = states[0], states[1], states[2]

        clean_story = make_false_belief_story(
            chars[0], chars[1], containers[0], containers[1],
            state1, state2, swap_state,
        )
        clean_question = f"What is actually inside the {containers[0]}?"
        clean_prompt = make_prompt(clean_story, clean_question)

        cf_chars = list(reversed(chars))
        cf_containers = list(reversed(containers))
        cf_states = random.sample(drinks, 3)
        while any(s in [state1, state2, swap_state] for s in cf_states):
            cf_states = random.sample(drinks, 3)
        cf_state1, cf_state2, cf_swap = cf_states[0], cf_states[1], cf_states[2]

        cf_story = make_false_belief_story(
            cf_chars[0], cf_chars[1], cf_containers[0], cf_containers[1],
            cf_state1, cf_state2, cf_swap,
        )
        cf_question = f"What is actually inside the {cf_containers[0]}?"
        cf_prompt = make_prompt(cf_story, cf_question)

        pairs.append({
            "clean_prompt": clean_prompt,
            "counterfactual_prompt": cf_prompt,
            "clean_ans": swap_state,
            "counterfactual_ans": cf_swap,
        })

    return pairs


def filter_on_model_with_accuracy(lm, pairs, max_size=80):
    """Filter pairs AND track behavioral accuracy separately."""
    passed = []
    correct_clean = 0
    correct_cf = 0
    total = 0

    for sample in tqdm(pairs, desc="Filtering on model accuracy"):
        clean_prompt = sample["clean_prompt"]
        cf_prompt = sample["counterfactual_prompt"]
        clean_target = sample["clean_ans"]
        cf_target = sample["counterfactual_ans"]

        try:
            with lm.trace(clean_prompt, remote=True):
                clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()
            with lm.trace(cf_prompt, remote=True):
                cf_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

            clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
            cf_tok = lm.tokenizer.decode([cf_pred.item()]).lower().strip()

            total += 1
            if clean_tok == clean_target.lower().strip():
                correct_clean += 1
            if cf_tok == cf_target.lower().strip():
                correct_cf += 1

            if (clean_tok == clean_target.lower().strip()
                    and cf_tok == cf_target.lower().strip()):
                passed.append(sample)
                if len(passed) >= max_size:
                    break
        except Exception as e:
            print(f"  Filter error: {e}")
            time.sleep(2)

    accuracy = {
        "clean_accuracy": correct_clean / total if total > 0 else 0.0,
        "cf_accuracy": correct_cf / total if total > 0 else 0.0,
        "both_correct": len(passed) / total if total > 0 else 0.0,
        "total_tested": total,
        "total_passed": len(passed),
    }
    return passed, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="True false-belief test for lookback subspaces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    answer_subspaces = {k: v for k, v in subspaces.items()
                        if "answer" in v["lookback_type"]}

    print(f"[{ts()}] True false-belief test")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")
    print(f"[{ts()}] Answer subspaces: {list(answer_subspaces.keys())}")

    lm = None
    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight...")
        lm = setup_nnsight()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "conditions": {},
    }

    # ── Condition 1: False-belief IIA ──
    print(f"\n{'='*60}")
    print(f"[{ts()}] CONDITION 1: False-belief (belief question)")
    print(f"{'='*60}")

    belief_ckpt = RESULTS_DIR / "false_belief.json"
    if belief_ckpt.exists():
        with open(belief_ckpt) as f:
            cached = json.load(f)
        if cached.get("dry_run", False) != args.dry_run:
            print(f"  Stale checkpoint (dry_run mismatch), recomputing")
        else:
            summary["conditions"]["false_belief"] = cached
            print(f"  Resumed from checkpoint")
            for sub_name, iia in cached.get("subspace_iias", {}).items():
                print(f"    {sub_name}: IIA={iia:.4f}")
            print(f"  Behavioral accuracy: {cached['behavioral_accuracy']}")
    if "false_belief" not in summary["conditions"]:
        pairs_path = RESULTS_DIR / "filtered_false_belief_pairs.json"
        if not args.dry_run and pairs_path.exists():
            with open(pairs_path) as f:
                saved = json.load(f)
            fb_pairs = saved["pairs"]
            behavioral = saved["behavioral_accuracy"]
            print(f"[{ts()}] Loaded cached pairs: {len(fb_pairs)}")
        elif not args.dry_run:
            print(f"[{ts()}] Generating false-belief pairs...")
            fb_raw = generate_false_belief_pairs(args.n_eval * 3, args.seed)
            fb_pairs, behavioral = filter_on_model_with_accuracy(
                lm, fb_raw, max_size=args.n_eval)
            with open(pairs_path, "w") as f:
                json.dump({"pairs": fb_pairs, "behavioral_accuracy": behavioral}, f, indent=2)
            print(f"[{ts()}] {len(fb_pairs)} pairs passed filter")
            print(f"[{ts()}] Behavioral accuracy: {behavioral}")
        else:
            fb_pairs = generate_false_belief_pairs(args.n_eval, args.seed)
            behavioral = {"clean_accuracy": 0.7, "cf_accuracy": 0.65,
                          "both_correct": 0.5, "total_tested": args.n_eval,
                          "total_passed": int(args.n_eval * 0.5)}

        sub_iias = {}
        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                iia = float(rng.uniform(0.05, 0.35))
            else:
                if not fb_pairs:
                    print(f"  No pairs passed filter — cannot compute IIA")
                    sub_iias[sub_name] = 0.0
                    continue
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                proj = build_projection_matrix(svd_basis, mask_indices)
                iia = compute_iia_answer_flex(lm, fb_pairs, layer, proj)

            sub_iias[sub_name] = float(iia)
            print(f"    {sub_name}: IIA={iia:.4f}")

        result = {
            "dry_run": args.dry_run,
            "n_pairs": len(fb_pairs),
            "behavioral_accuracy": behavioral,
            "subspace_iias": sub_iias,
        }
        summary["conditions"]["false_belief"] = result
        with open(belief_ckpt, "w") as f:
            json.dump(result, f, indent=2)

    # ── Condition 2: Reality-question control ──
    print(f"\n{'='*60}")
    print(f"[{ts()}] CONDITION 2: Reality question control")
    print(f"{'='*60}")

    reality_ckpt = RESULTS_DIR / "reality_question.json"
    if reality_ckpt.exists():
        with open(reality_ckpt) as f:
            cached = json.load(f)
        if cached.get("dry_run", False) != args.dry_run:
            print(f"  Stale checkpoint (dry_run mismatch), recomputing")
        else:
            summary["conditions"]["reality_question"] = cached
            print(f"  Resumed from checkpoint")
            for sub_name, iia in cached.get("subspace_iias", {}).items():
                print(f"    {sub_name}: IIA={iia:.4f}")
    if "reality_question" not in summary["conditions"]:
        rq_cache = RESULTS_DIR / "filtered_reality_pairs.json"
        if not args.dry_run and rq_cache.exists():
            with open(rq_cache) as f:
                saved = json.load(f)
            rq_pairs = saved["pairs"]
            rq_behavioral = saved["behavioral_accuracy"]
            print(f"[{ts()}] Loaded {len(rq_pairs)} cached reality pairs")
        elif not args.dry_run:
            print(f"[{ts()}] Generating reality-question pairs...")
            rq_raw = generate_reality_pairs(args.n_eval * 3, args.seed)
            rq_pairs, rq_behavioral = filter_on_model_with_accuracy(
                lm, rq_raw, max_size=args.n_eval)
            with open(rq_cache, "w") as f:
                json.dump({"pairs": rq_pairs, "behavioral_accuracy": rq_behavioral}, f, indent=2)
            print(f"[{ts()}] {len(rq_pairs)} reality pairs passed filter")
            print(f"[{ts()}] Reality behavioral accuracy: {rq_behavioral}")
        else:
            rq_pairs = generate_reality_pairs(args.n_eval, args.seed)
            rq_behavioral = {"clean_accuracy": 0.8, "cf_accuracy": 0.75,
                             "both_correct": 0.6, "total_tested": args.n_eval,
                             "total_passed": int(args.n_eval * 0.6)}

        sub_iias = {}
        for sub_name, spec in answer_subspaces.items():
            layer = spec["layer"]
            mask_indices = spec.get("mask_indices")

            if args.dry_run:
                iia = float(rng.uniform(0.05, 0.35))
            else:
                if not rq_pairs:
                    print(f"  No pairs passed filter — cannot compute IIA")
                    sub_iias[sub_name] = 0.0
                    continue
                svd_basis = load_svd_basis(layer, "last_token")
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis")
                    continue
                proj = build_projection_matrix(svd_basis, mask_indices)
                iia = compute_iia_answer_flex(lm, rq_pairs, layer, proj)

            sub_iias[sub_name] = float(iia)
            print(f"    {sub_name}: IIA={iia:.4f}")

        result = {
            "dry_run": args.dry_run,
            "n_pairs": len(rq_pairs),
            "behavioral_accuracy": rq_behavioral,
            "subspace_iias": sub_iias,
        }
        summary["conditions"]["reality_question"] = result
        with open(reality_ckpt, "w") as f:
            json.dump(result, f, indent=2)

    # ── Summary ──
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("FALSE BELIEF TEST SUMMARY")
    print(f"{'='*60}")

    for cond_name, cond in summary["conditions"].items():
        print(f"\n{cond_name}:")
        behavioral = cond.get("behavioral_accuracy", {})
        if isinstance(behavioral, dict):
            print(f"  Behavioral: clean={behavioral.get('clean_accuracy', '?'):.4f}, "
                  f"cf={behavioral.get('cf_accuracy', '?'):.4f}, "
                  f"both={behavioral.get('both_correct', '?'):.4f} "
                  f"({behavioral.get('total_passed', '?')}/{behavioral.get('total_tested', '?')} passed)")
        print(f"  Subspace IIAs:")
        for sub_name, iia in cond.get("subspace_iias", {}).items():
            print(f"    {sub_name}: {iia:.4f}")

    print(f"\n[{ts()}] Done. Results at {RESULTS_DIR}")


if __name__ == "__main__":
    main()
