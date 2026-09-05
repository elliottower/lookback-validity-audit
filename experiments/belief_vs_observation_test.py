"""E16: belief against observation history and against reality.

Registered in AMENDMENTS.md Amendment 12 (tag prereg-amendment-8), before any results.

E14 showed the subspaces transfer to false-belief stories at IIA 0.975. In every one of
those stories the character does not observe the swap, so the belief-correct answer and the
answer given by a surface rule -- a character who did not observe a change reports the state
before it -- name the same drink. IIA cannot separate them.

Adding a condition where the character is told about a real swap separates belief from
observation history, but there belief and reality coincide, which is the degeneracy this
audit identifies in CausalToM itself. Three conditions are required:

    condition      what happens                       belief  observation  reality
    untold         swap occurs, char does not see it  pre     pre          post
    told_true      swap occurs, char is told          post    pre          post
    misinformed    NO swap, char is told one did      post    pre          pre

Belief tracks (pre, post, post); observation history (pre, pre, pre); reality (post, post,
pre). Only the misinformed cell distinguishes belief from reality, and the experiment turns
on it.

Usage:
    uv run python experiments/belief_vs_observation_test.py --dry-run
    uv run python experiments/belief_vs_observation_test.py --n-samples 240
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ndif_utils import (
    REPO_ROOT,
    BELIEF_TRACKING,
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

INSTRUCTION = (
    "Read the story and answer the question. Respond with a single word."
)

ANSWER_SUBSPACES = ["answer_pointer_L38", "answer_pointer_L52", "answer_pointer_L53"]

# The three conditions differ only in the last two sentences, which are held to the same
# two-clause shape so that condition varies in content rather than in structure.
CONDITIONS = {
    "untold": {
        "event": "a co-worker swapped the {state1} in the {container1} with {swap}.",
        "report": "{char1} did not notice the swap.",
        "belief": "pre", "reality": "post",
    },
    "told_true": {
        "event": "a co-worker swapped the {state1} in the {container1} with {swap}.",
        "report": "A colleague told {char1} about the swap.",
        "belief": "post", "reality": "post",
    },
    "misinformed": {
        "event": "a co-worker left the {state1} in the {container1} untouched.",
        "report": "A colleague told {char1} it was swapped with {swap}.",
        "belief": "post", "reality": "pre",
    },
}


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def make_story(condition, char1, char2, container1, container2, state1, state2, swap):
    c = CONDITIONS[condition]
    fields = dict(char1=char1, container1=container1, state1=state1, swap=swap)
    return (
        f"{char1} and {char2} are working in a busy restaurant. "
        f"To complete an order, {char1} grabs an opaque {container1} and "
        f"fills it with {state1}. Then {char2} grabs another opaque "
        f"{container2} and fills it with {state2}. "
        f"While {char1} was attending to another task, {c['event'].format(**fields)} "
        f"{c['report'].format(**fields)}"
    )


def make_prompt(story, question):
    return f"Instruction: {INSTRUCTION}\n\nStory: {story}\nQuestion: {question}\nAnswer:"


def belief_answer(condition, state1, swap):
    return state1 if CONDITIONS[condition]["belief"] == "pre" else swap


def generate_pairs(condition, n_samples, seed):
    """Clean/CF pairs within one condition. Condition is fixed inside a pair; only the
    drink identities vary, so the interchange measures drink content and not condition."""
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    rnd = random.Random(seed)
    pairs = []
    for _ in range(n_samples):
        chars = rnd.sample(characters, 2)
        containers = rnd.sample(bottles, 2)
        s1, s2, swap = rnd.sample(drinks, 3)

        cf_chars = list(reversed(chars))
        cf_containers = list(reversed(containers))
        while True:
            c1, c2, cswap = rnd.sample(drinks, 3)
            if not set([c1, c2, cswap]) & set([s1, s2, swap]):
                break

        pairs.append({
            "condition": condition,
            "clean_prompt": make_prompt(
                make_story(condition, chars[0], chars[1], containers[0], containers[1],
                           s1, s2, swap),
                f"What does {chars[0]} believe the {containers[0]} contains?"),
            "counterfactual_prompt": make_prompt(
                make_story(condition, cf_chars[0], cf_chars[1], cf_containers[0],
                           cf_containers[1], c1, c2, cswap),
                f"What does {cf_chars[0]} believe the {cf_containers[0]} contains?"),
            "clean_ans": belief_answer(condition, s1, swap),
            "counterfactual_ans": belief_answer(condition, c1, cswap),
            "clean_reality": swap if CONDITIONS[condition]["reality"] == "post" else s1,
            "clean_observed": s1,
        })
    return pairs


def main():
    ap = argparse.ArgumentParser(description="E16 belief vs observation history vs reality")
    ap.add_argument("--n-samples", type=int, default=240,
                    help="generated per condition; registration asks for 3x the 80 target")
    ap.add_argument("--target", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--condition", action="append", choices=sorted(CONDITIONS))
    ap.add_argument("--output", default="results/belief_vs_observation/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    conditions = args.condition or sorted(CONDITIONS)

    record = {
        "experiment": "E16 belief vs observation history vs reality",
        "registered": "AMENDMENTS.md Amendment 12, tag prereg-amendment-8",
        "dry_run": args.dry_run, "seed": args.seed,
        "target_n": args.target, "generated_n": args.n_samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signatures": {
            "belief": {"untold": "pre", "told_true": "post", "misinformed": "post"},
            "observation_history": {"untold": "pre", "told_true": "pre",
                                    "misinformed": "pre"},
            "reality": {"untold": "post", "told_true": "post", "misinformed": "pre"},
        },
        "conditions": {},
    }

    def checkpoint():
        out_path.write_text(json.dumps(record, indent=2))

    all_pairs = {c: generate_pairs(c, args.n_samples, args.seed) for c in conditions}

    # Length is reported rather than assumed: the conditions are matched in shape, and any
    # residual difference in token count is a property the reader should be able to see.
    for c in conditions:
        lens = [len(p["clean_prompt"].split()) for p in all_pairs[c]]
        record["conditions"][c] = {
            "generated": len(all_pairs[c]),
            "prompt_words_mean": float(np.mean(lens)),
            "prompt_words_sd": float(np.std(lens)),
            "example_prompt": all_pairs[c][0]["clean_prompt"],
        }
    checkpoint()

    if args.dry_run:
        record["dry_run_only"] = True
        checkpoint()
        for c in conditions:
            print(f"\n=== {c} ===\n{all_pairs[c][0]['clean_prompt']}")
            print(f"belief-correct answer: {all_pairs[c][0]['clean_ans']}  "
                  f"reality: {all_pairs[c][0]['clean_reality']}  "
                  f"observed: {all_pairs[c][0]['clean_observed']}")
        print(f"\n[{ts()}] dry run; wrote {out_path}")
        return

    print(f"[{ts()}] setting up nnsight")
    lm = setup_nnsight()
    specs = load_subspace_specs()

    for c in conditions:
        pre = len(all_pairs[c])
        kept = filter_on_model(lm, all_pairs[c], max_size=args.target)
        record["conditions"][c].update({
            "pre_filter": pre, "post_filter": len(kept),
            "attrition": 1 - len(kept) / pre if pre else None,
            "rejected_sample": [p["clean_prompt"] for p in all_pairs[c][:3]
                                if p not in kept][:3],
        })
        checkpoint()
        print(f"[{ts()}] {c}: {len(kept)}/{pre} survived filtering")

        if len(kept) < 40:
            record["conditions"][c]["void"] = "fewer than 40 pairs survived filtering"
            checkpoint()
            continue

        iias = {}
        for name in ANSWER_SUBSPACES:
            spec = specs[name]; L = spec["layer"]
            basis = load_svd_basis(L, "last_token")
            if basis is None:
                iias[name] = None
                continue
            idx = spec.get("mask_indices") or list(range(spec["rank"]))
            proj = build_projection_matrix(basis, idx)
            iias[name] = float(compute_iia_answer_flex(lm, kept, L, proj))
            record["conditions"][c]["iia"] = iias
            checkpoint()
            print(f"[{ts()}]   {name}: IIA={iias[name]:.4f}")

    # The void condition the registration names: non-comparable subsets across conditions.
    ns = [record["conditions"][c].get("post_filter") for c in conditions]
    ns = [n for n in ns if n]
    if len(ns) > 1 and max(ns) / min(ns) > 2:
        record["void"] = ("post-filter n differs by more than a factor of two between "
                          "conditions; the IIAs are not computed on comparable subsets")
    checkpoint()
    print(f"[{ts()}] wrote {out_path}")


if __name__ == "__main__":
    main()
