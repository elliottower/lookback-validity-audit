"""
Surface heuristic test: does the subspace track beliefs or positional heuristics?

In CausalToM stories, the correct answer (what the character believes is in
their bottle) correlates with a surface heuristic: "the last substance
mentioned before the character left." This test constructs adversarial stories
where that heuristic gives the WRONG answer.

Example adversarial template:
  "Character sees drink X placed in their bottle. Character leaves. Someone
  pours Y into the bottle, then pours Y out and puts X back. Character
  returns. What does Character think is in the bottle?"
  - Heuristic answer (last mentioned before return): X (the replacement)
  - Belief answer: X (saw it placed originally)
  Wait — that's the same. We need the heuristic and belief to diverge.

  Better: "Character sees X placed in their bottle. Character leaves.
  Drink Y is poured in. THEN drink Z is poured in (replacing Y).
  Character returns. What does Character think is in their bottle?"
  - Belief answer: X (last thing character saw)
  - Last-substance-mentioned: Z (most recent in narrative)

If IIA tracks the heuristic (last mentioned substance) rather than the
true belief, the subspace is doing pattern matching.

Protocol:
  1. Generate adversarial stories where belief != last-mentioned substance
  2. Create counterfactual pairs by swapping the believed substance
  3. Measure IIA using the paper's subspaces
  4. Also measure which answer the model gives under intervention

Usage:
    uv run python experiments/heuristic_adversarial.py --dry-run
    uv run python experiments/heuristic_adversarial.py --output results/heuristic/
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
    build_projection_matrix,
    compute_iia_answer_flex,
    filter_on_model,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Adversarial story templates where belief != last mentioned substance.
# Structure: character sees X, leaves, Y replaces X, Z replaces Y.
# Character believes: X (what they last saw). Last mentioned: Z.

CHARACTERS = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Peter",
]

DRINKS = [
    "water", "juice", "tea", "coffee", "milk", "soda", "lemonade", "wine",
]

TEMPLATE_DOUBLE_SWAP = (
    "{char} watched as {drink_a} was poured into their bottle. "
    "{char} then left the room. "
    "While {char} was away, someone poured out the {drink_a} and "
    "replaced it with {drink_b}. Then, someone else poured out the "
    "{drink_b} and replaced it with {drink_c}. "
    "{char} returned to the room. "
    "{char} thinks their bottle contains"
)

TEMPLATE_DISCUSSED = (
    "{char} saw {drink_a} being poured into their bottle. "
    "{char} left the room shortly after. "
    "While {char} was gone, the {drink_a} was replaced with {drink_b}. "
    "Later, people in the room talked about how good {drink_c} is. "
    "{char} came back. "
    "{char} thinks their bottle contains"
)

TEMPLATE_REVERSAL = (
    "{char} was present when {drink_a} was poured into their bottle. "
    "{char} then stepped out. "
    "First, the {drink_a} was swapped for {drink_b}. "
    "Then the {drink_b} was swapped for {drink_c}. "
    "Then the {drink_c} was swapped back to {drink_b}. "
    "{char} came back into the room. "
    "{char} thinks their bottle contains"
)


def generate_adversarial_pairs(n_pairs, seed):
    """Generate adversarial CausalToM-style pairs.

    For each pair:
    - Clean story: character saw drink_a, left, drink replaced twice
    - CF story: same but character saw drink_d instead of drink_a
    - Belief answer (clean): drink_a (what character saw)
    - Belief answer (CF): drink_d
    - Heuristic answer: drink_c (last mentioned substance)

    Also generates standard CausalToM-format pairs for IIA compatibility.
    """
    rng = random.Random(seed)
    templates = [TEMPLATE_DOUBLE_SWAP, TEMPLATE_DISCUSSED, TEMPLATE_REVERSAL]
    pairs = []

    for i in range(n_pairs):
        char = rng.choice(CHARACTERS)
        drinks = rng.sample(DRINKS, 4)
        drink_a, drink_b, drink_c, drink_d = drinks
        template = templates[i % len(templates)]

        clean_story = template.format(
            char=char, drink_a=drink_a, drink_b=drink_b, drink_c=drink_c
        )
        cf_story = template.format(
            char=char, drink_a=drink_d, drink_b=drink_b, drink_c=drink_c
        )

        pairs.append({
            "clean_prompt": clean_story,
            "counterfactual_prompt": cf_story,
            "clean_ans": drink_a,
            "counterfactual_ans": drink_d,
            "heuristic_ans": drink_c if "REVERSAL" not in template.__class__.__name__ else drink_b,
            "belief_ans_clean": drink_a,
            "belief_ans_cf": drink_d,
            "template": template.split("{char}")[0][:30],
            "task_type": "adversarial_belief",
        })

    return pairs


def measure_intervention_alignment(lm, pairs, layer, projection, retries=3):
    """Measure what answer the intervention produces: belief or heuristic?

    Returns (belief_count, heuristic_count, other_count, total).
    """
    belief, heuristic, other, total = 0, 0, 0, 0

    for sample in tqdm(pairs, desc=f"Alignment L{layer}", leave=False):
        clean_prompt = sample["clean_prompt"]
        cf_prompt = sample["counterfactual_prompt"]
        target_belief = sample["counterfactual_ans"].lower().strip()
        target_heuristic = sample.get("heuristic_ans", "").lower().strip()

        for attempt in range(retries):
            try:
                with lm.trace(cf_prompt, remote=True):
                    cf_out = lm.model.layers[layer].output[0].save()
                with lm.trace(clean_prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cf_t = cf_out.detach().cpu().float()
                cl_t = cl_out.detach().cpu().float()

                patched = cl_t.clone()
                if projection is not None:
                    x = cl_t[-1]
                    patched[-1] = x - (x @ projection) + (cf_t[-1] @ projection)
                else:
                    patched[-1] = cf_t[-1]

                with lm.trace(clean_prompt, remote=True):
                    lm.model.layers[layer].output[0] = patched
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()

                if pred_tok == target_belief:
                    belief += 1
                elif target_heuristic and pred_tok == target_heuristic:
                    heuristic += 1
                else:
                    other += 1
                total += 1
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    other += 1
                    total += 1

    return belief, heuristic, other, total


def main():
    parser = argparse.ArgumentParser(description="Surface heuristic adversarial test")
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/heuristic/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)

    subspaces = load_subspace_specs()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lm = None

    print(f"[{ts()}] Generating {args.n_pairs} adversarial pairs...")
    pairs = generate_adversarial_pairs(args.n_pairs, args.seed)
    print(f"  Generated {len(pairs)} pairs")

    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        print(f"[{ts()}] Filtering on model accuracy...")
        pairs = filter_on_model(lm, pairs, max_size=80)
        print(f"[{ts()}] {len(pairs)} pairs passed filter")

    summary = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_pairs": len(pairs),
        "method": "heuristic_adversarial",
        "method_note": (
            "Adversarial CausalToM stories where belief != last-mentioned substance. "
            "If intervention tracks the heuristic (last mentioned) rather than the "
            "true belief, the subspace is doing surface pattern matching."
        ),
        "prediction": {
            "supports_paper": "Intervention tracks belief answer",
            "weakens_paper": "Intervention tracks heuristic (last-mentioned) answer",
        },
        "subspaces": {},
    }

    for sub_name, sub_spec in tqdm(subspaces.items(), desc="Subspaces"):
        layer = sub_spec["layer"]
        rank = sub_spec["rank"]
        lookback_type = sub_spec["lookback_type"]

        if "binding" in lookback_type:
            summary["subspaces"][sub_name] = {
                "skipped": True,
                "skip_reason": "binding subspace requires CausalToM state positions",
                "layer": layer,
                "rank": rank,
            }
            continue

        print(f"\n[{ts()}] {sub_name} (L{layer}, rank={rank})")

        if args.dry_run:
            iia = float(rng.uniform(0.0, 0.4))
            belief_frac = float(rng.uniform(0.3, 0.7))
            heuristic_frac = float(rng.uniform(0.1, 0.4))
            other_frac = 1.0 - belief_frac - heuristic_frac
        else:
            vec_type = "last_token"
            svd_basis = load_svd_basis(layer, vec_type)
            if svd_basis is None:
                print(f"  SKIP: no SVD basis")
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

            iia = compute_iia_answer_flex(lm, pairs, layer, proj)
            print(f"  IIA: {iia:.4f}")

            belief, heuristic, other, total = measure_intervention_alignment(
                lm, pairs, layer, proj
            )
            belief_frac = belief / total if total > 0 else 0
            heuristic_frac = heuristic / total if total > 0 else 0
            other_frac = other / total if total > 0 else 0

        result = {
            "iia": float(iia),
            "belief_fraction": float(belief_frac),
            "heuristic_fraction": float(heuristic_frac),
            "other_fraction": float(other_frac),
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback_type,
            "original_iia": sub_spec["sv_iia"],
        }

        summary["subspaces"][sub_name] = result
        print(f"  belief={belief_frac:.4f}, heuristic={heuristic_frac:.4f}, other={other_frac:.4f}")

        sub_path = output_dir / f"{sub_name}.json"
        with open(sub_path, "w") as f:
            json.dump({"timestamp": summary["timestamp"], **result}, f, indent=2)

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    for name, res in summary["subspaces"].items():
        if res.get("skipped"):
            continue
        bf = res["belief_fraction"]
        hf = res["heuristic_fraction"]
        if bf > hf * 2:
            verdict = "BELIEF-TRACKING (supports paper)"
        elif hf > bf * 2:
            verdict = "HEURISTIC (weakens paper)"
        else:
            verdict = "MIXED (inconclusive)"
        print(f"  {name}: belief={bf:.4f}, heuristic={hf:.4f} — {verdict}")

    any_heuristic = any(
        v.get("heuristic_fraction", 0) > v.get("belief_fraction", 0) * 2
        for v in summary["subspaces"].values()
        if not v.get("skipped")
    )

    summary["verdict"] = (
        "Heuristic: intervention tracks positional pattern, not belief"
        if any_heuristic else
        "Belief-tracking: intervention follows true belief (supports paper)"
    )
    print(f"\nVerdict: {summary['verdict']}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{ts()}] Results saved to {summary_path}")


if __name__ == "__main__":
    main()
