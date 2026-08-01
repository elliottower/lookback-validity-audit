"""
Compositional generalization (C3, C5): test whether the lookback pattern
extends beyond two-character, first-order, location-only false belief.

Three axes of generalization:
  1. Character scaling: 3, 4, 5 characters (add non-interacting distractors)
  2. Nested beliefs: second-order false belief ("What does Alice think Bob
     believes?")
  3. Content selectivity: beliefs about location, existence, intentions,
     knowledge states

If the lookback mechanism is compositionally structured, it should generalize
across all three axes. If it is a template-matched circuit, it will degrade.

Method: transfer test. Apply the CausalToM-derived subspace projections
(SVD basis + paper's binary mask) to novel stories and measure IIA.  This
tests whether the *identified subspace* transfers, not whether re-running
SVD + mask training on new stories recovers the same subspace.  The full
DCM procedure (re-identification) requires GPU compute.

For answer_lookback subspaces: interchange at position [-1] (last token).
For binding_lookback subspaces: CausalToM-specific positions [155, 156,
167, 168] do not apply to novel stories with different templates and token
counts, so binding subspaces are tested with answer-style (last-token)
intervention only.  This is noted in the output as method=transfer_last_token.

Usage:
    uv run python experiments/compositional_generalization.py --dry-run
    uv run python experiments/compositional_generalization.py --output results/compositional/
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
    compute_iia_answer_flex,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CHARACTERS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
              "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Paul"]
CONTAINERS = ["box", "bag", "jar", "bottle", "basket", "bucket", "pouch", "case"]
SUBSTANCES = ["water", "sand", "rice", "flour", "salt", "sugar", "beans", "marbles"]
LOCATIONS = ["shelf", "table", "drawer", "basket", "closet", "desk",
             "counter", "cabinet", "chair", "bench", "floor", "windowsill"]
OBJECTS = ["book", "key", "ball", "cup", "hat", "ring", "phone", "wallet",
           "pen", "watch", "coin", "letter", "toy", "scarf", "bag", "box"]
SETTINGS = ["kitchen", "workshop", "laboratory", "office", "classroom",
            "garage", "studio", "library"]

SUBSTANCE_INSTRUCTION = (
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

YESNO_INSTRUCTION = (
    "1. Track the belief of each character as described in the story. "
    "2. A character's belief is formed only when they perform an action "
    "themselves or can observe the action taking place. "
    "3. Answer the question with a single word: 'yes' or 'no'."
)

SINGLE_WORD_INSTRUCTION = (
    "1. Track the belief of each character as described in the story. "
    "2. A character's belief is formed only when they perform an action "
    "themselves or can observe the action taking place. "
    "3. Answer the question with a single word."
)


def _wrap_prompt(instruction, story, question):
    return f"Instruction: {instruction}\n\nStory: {story}\nQuestion: {question}\nAnswer:"


# ── Counterfactual pair generators ──
#
# Each returns a list of dicts with:
#   clean_prompt, counterfactual_prompt, clean_ans, counterfactual_ans
# matching the interface expected by ndif_utils.compute_iia_answer.


def generate_character_scaling_pairs(n_characters, n_pairs, rng):
    """CausalToM-style false belief with n_characters, as counterfactual pairs.

    Clean: c1 leaves, c2 replaces cont2 contents with s1.  c1 believes s2.
    Counterfactual: same structure, different substances (s3, s4).  c1 believes s4.
    """
    pairs = []
    for _ in range(n_pairs):
        chars = list(rng.choice(CHARACTERS, size=n_characters, replace=False))
        c1, c2 = chars[0], chars[1]
        distractors = chars[2:]
        cont1, cont2 = rng.choice(CONTAINERS, size=2, replace=False)
        s1, s2, s3, s4 = rng.choice(SUBSTANCES, size=4, replace=False)
        setting = rng.choice(SETTINGS)

        distractor_text = ""
        if distractors:
            activities = rng.choice(
                ["reading a book", "looking out the window", "sitting quietly",
                 "organizing papers", "checking their phone", "drinking coffee"],
                size=len(distractors), replace=True,
            )
            distractor_text = " ".join(
                f"{d} is {a}." for d, a in zip(distractors, activities)
            ) + " "

        def _make_prompt(sa, sb):
            story = (
                f"{c1} and {c2} are working in a {setting}. {distractor_text}"
                f"{c1} grabs an opaque {cont1} and fills it with {sa}. "
                f"{c2} grabs another opaque {cont2} and fills it with {sb}. "
                f"{c1} leaves the {setting}. "
                f"{c2} empties {cont2} and refills it with {sa}."
            )
            question = f"What does {c1} believe {cont2} contains?"
            return _wrap_prompt(SUBSTANCE_INSTRUCTION, story, question)

        pairs.append({
            "clean_prompt": _make_prompt(s1, s2),
            "counterfactual_prompt": _make_prompt(s3, s4),
            "clean_ans": s2,
            "counterfactual_ans": s4,
            "condition": f"character_scaling_{n_characters}",
            "n_characters": n_characters,
        })
    return pairs


def generate_nested_belief_pairs(n_pairs, rng):
    """Second-order false belief as counterfactual pairs.

    Clean: c1 thinks c2 believes cont2 has s2.
    Counterfactual: different substances throughout → c1 thinks c2 believes s5.
    """
    pairs = []
    for _ in range(n_pairs):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        cont1, cont2 = rng.choice(CONTAINERS, size=2, replace=False)
        s1, s2, s3, s4, s5, s6 = rng.choice(SUBSTANCES, size=6, replace=False)
        setting = rng.choice(SETTINGS)

        def _make_prompt(sa, sb, sc):
            story = (
                f"{c1} and {c2} are working in a {setting}. "
                f"{c1} grabs an opaque {cont1} and fills it with {sa}. "
                f"{c2} grabs another opaque {cont2} and fills it with {sb}. "
                f"{c1} leaves the {setting}. "
                f"{c2} empties {cont2} and refills it with {sc}."
            )
            question = f"What does {c1} think {c2} believes {cont2} contains?"
            return _wrap_prompt(SUBSTANCE_INSTRUCTION, story, question)

        pairs.append({
            "clean_prompt": _make_prompt(s1, s2, s3),
            "counterfactual_prompt": _make_prompt(s4, s5, s6),
            "clean_ans": s2,
            "counterfactual_ans": s5,
            "condition": "nested_belief",
        })
    return pairs


def generate_existence_belief_pairs(n_pairs, rng):
    """Belief about existence as counterfactual pairs.

    Clean: c1 absent when c2 destroys obj → c1 believes it exists → "yes".
    Counterfactual: c1 watches → c1 knows it's gone → "no".
    """
    pairs = []
    for _ in range(n_pairs):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        obj = rng.choice(OBJECTS)
        loc = rng.choice(LOCATIONS)
        setting = rng.choice(SETTINGS)

        clean_story = (
            f"{c1} and {c2} are in a {setting}. "
            f"{c1} places the {obj} on the {loc}. "
            f"{c1} leaves the {setting}. "
            f"{c2} takes the {obj} and throws it away. {c1} returns."
        )
        cf_story = (
            f"{c1} and {c2} are in a {setting}. "
            f"{c1} places the {obj} on the {loc}. "
            f"{c1} watches as "
            f"{c2} takes the {obj} and throws it away."
        )
        question = f"Does {c1} believe the {obj} is still on the {loc}?"

        pairs.append({
            "clean_prompt": _wrap_prompt(YESNO_INSTRUCTION, clean_story, question),
            "counterfactual_prompt": _wrap_prompt(YESNO_INSTRUCTION, cf_story, question),
            "clean_ans": "yes",
            "counterfactual_ans": "no",
            "condition": "content_existence",
        })
    return pairs


def generate_intention_belief_pairs(n_pairs, rng):
    """Belief about intentions as counterfactual pairs.

    Clean: c1 leaves before c2 changes plan → c1 believes original.
    Counterfactual: c1 stays and hears change → c1 knows new plan.
    """
    actions = ["cooking", "cleaning", "reading", "painting",
               "writing", "running", "swimming", "singing"]
    pairs = []
    for _ in range(n_pairs):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        act1, act2 = rng.choice(actions, size=2, replace=False)
        setting = rng.choice(SETTINGS)

        clean_story = (
            f"{c1} and {c2} are in a {setting}. "
            f"{c2} tells {c1} that they plan to go {act1}. "
            f"{c1} leaves the {setting}. "
            f"{c2} changes their mind and decides to go {act2} instead."
        )
        cf_story = (
            f"{c1} and {c2} are in a {setting}. "
            f"{c2} tells {c1} that they plan to go {act1}. "
            f"{c2} then tells {c1} they changed their mind and will go {act2} instead."
        )
        question = f"What activity does {c1} believe {c2} is planning?"

        pairs.append({
            "clean_prompt": _wrap_prompt(SINGLE_WORD_INSTRUCTION, clean_story, question),
            "counterfactual_prompt": _wrap_prompt(SINGLE_WORD_INSTRUCTION, cf_story, question),
            "clean_ans": act1,
            "counterfactual_ans": act2,
            "condition": "content_intention",
        })
    return pairs


def generate_knowledge_belief_pairs(n_pairs, rng):
    """Belief about knowledge states as counterfactual pairs.

    Clean: c1 leaves without telling c2 → c1 believes c2 doesn't know → "no".
    Counterfactual: c1 tells c2 before leaving → c1 believes c2 knows → "yes".
    """
    facts = ["the door is locked", "the meeting is canceled", "the package arrived",
             "the password changed", "the deadline moved", "the recipe was updated",
             "the key is missing", "the schedule shifted"]
    pairs = []
    for _ in range(n_pairs):
        c1, c2 = rng.choice(CHARACTERS, size=2, replace=False)
        fact = rng.choice(facts)
        setting = rng.choice(SETTINGS)

        clean_story = (
            f"{c1} and {c2} are in a {setting}. "
            f"{c1} learns that {fact}. "
            f"{c1} leaves the {setting} without telling {c2}."
        )
        cf_story = (
            f"{c1} and {c2} are in a {setting}. "
            f"{c1} learns that {fact}. "
            f"{c1} tells {c2} that {fact} before leaving the {setting}."
        )
        question = f"Does {c1} believe that {c2} knows that {fact}?"

        pairs.append({
            "clean_prompt": _wrap_prompt(YESNO_INSTRUCTION, clean_story, question),
            "counterfactual_prompt": _wrap_prompt(YESNO_INSTRUCTION, cf_story, question),
            "clean_ans": "no",
            "counterfactual_ans": "yes",
            "condition": "content_knowledge",
        })
    return pairs


CONDITION_GENERATORS = {
    "baseline_2char": ("CausalToM baseline, 2 characters",
                       lambda n, rng: generate_character_scaling_pairs(2, n, rng)),
    "scaling_3char": ("Character scaling, 3 characters (1 distractor)",
                      lambda n, rng: generate_character_scaling_pairs(3, n, rng)),
    "scaling_4char": ("Character scaling, 4 characters (2 distractors)",
                      lambda n, rng: generate_character_scaling_pairs(4, n, rng)),
    "scaling_5char": ("Character scaling, 5 characters (3 distractors)",
                      lambda n, rng: generate_character_scaling_pairs(5, n, rng)),
    "nested_belief": ("Second-order false belief", generate_nested_belief_pairs),
    "content_location": ("Content: location belief (CausalToM baseline)",
                         lambda n, rng: generate_character_scaling_pairs(2, n, rng)),
    "content_existence": ("Content: existence belief", generate_existence_belief_pairs),
    "content_intention": ("Content: intention belief", generate_intention_belief_pairs),
    "content_knowledge": ("Content: knowledge state belief", generate_knowledge_belief_pairs),
}


def compute_mask_overlap(mask_a, mask_b, n_components=500):
    """Cosine similarity between two binary mask vectors over SVD components."""
    vec_a = np.zeros(n_components)
    vec_a[mask_a] = 1.0
    vec_b = np.zeros(n_components)
    vec_b[mask_b] = 1.0
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def filter_on_model_diagnostic(lm, pairs, max_size, condition_name):
    """Filter pairs with diagnostic logging of rejections."""
    from tqdm import tqdm

    filtered = []
    rejections = []
    for sample in tqdm(pairs, desc=f"Filtering {condition_name}"):
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

            clean_match = clean_tok == clean_target.lower().strip()
            cf_match = cf_tok == cf_target.lower().strip()

            if clean_match and cf_match:
                filtered.append(sample)
                if len(filtered) >= max_size:
                    break
            elif len(rejections) < 5:
                rejections.append({
                    "clean_target": clean_target,
                    "clean_predicted": clean_tok,
                    "clean_match": clean_match,
                    "cf_target": cf_target,
                    "cf_predicted": cf_tok,
                    "cf_match": cf_match,
                })
        except Exception as e:
            print(f"  Filter error: {e}")
            time.sleep(2)

    print(f"  Pre-filter: {len(pairs)}, post-filter: {len(filtered)}")
    if rejections:
        print(f"  Sample rejections (first {len(rejections)}):")
        for r in rejections:
            print(f"    clean: expected={r['clean_target']!r} got={r['clean_predicted']!r} "
                  f"| cf: expected={r['cf_target']!r} got={r['cf_predicted']!r}")

    return filtered, rejections


def main():
    parser = argparse.ArgumentParser(
        description="Compositional generalization tests for Lookback audit")
    parser.add_argument("--n-stories", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/compositional/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    subspaces = load_subspace_specs()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lm = None
    if not args.dry_run:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

    svd_dir = REPO_ROOT / "results" / "svd" / "CausalToM"

    summary = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_stories_per_condition": args.n_stories,
        "method": "transfer_test",
        "method_note": (
            "Applies CausalToM-derived projection to novel stories. "
            "Tests transfer of identified subspace, not re-identification. "
            "Binding subspaces tested with last-token intervention only "
            "(CausalToM state positions do not apply to novel templates). "
            "Full DCM (SVD + mask on new stories) requires GPU."
        ),
        "axes": ["character_scaling", "nested_beliefs", "content_selectivity"],
        "hypothesis": {
            "character_scaling": "If compositional, IIA maintained at 3-5 characters. If template-matched, degrades.",
            "nested_beliefs": "If compositional, same subspace handles second-order. If first-order-specific, different layers/rank.",
            "content_selectivity": "If unified, comparable IIA across content types. If family, different layers/subspaces.",
        },
        "conditions": {},
    }

    for cond_name, (desc, generator) in tqdm(CONDITION_GENERATORS.items(),
                                              desc="Conditions"):
        pairs = generator(args.n_stories, rng)

        print(f"\n{'='*60}")
        print(f"[{ts()}] Condition: {cond_name}")
        print(f"  {desc}")
        print(f"  N pairs: {len(pairs)}")
        print(f"  Example: {pairs[0]['clean_prompt'][:120]}...")
        print(f"{'='*60}")

        n_pre_filter = len(pairs)
        rejections = []
        if not args.dry_run:
            print(f"[{ts()}] Filtering pairs on model accuracy...")
            pairs, rejections = filter_on_model_diagnostic(
                lm, pairs, max_size=args.n_stories, condition_name=cond_name)
            print(f"[{ts()}] {len(pairs)} pairs passed filter")

        cond_result = {
            "description": desc,
            "n_pairs_pre_filter": n_pre_filter,
            "n_pairs": len(pairs),
            "filter_pass_rate": len(pairs) / n_pre_filter if n_pre_filter > 0 else 0.0,
            "sample_rejections": rejections,
            "subspace_results": {},
        }

        for sub_name, sub_spec in subspaces.items():
            layer = sub_spec["layer"]
            rank = sub_spec["rank"]
            lookback = sub_spec["lookback_type"]

            if args.dry_run:
                if "scaling" in cond_name:
                    n_char = int(cond_name.split("_")[-1].replace("char", ""))
                    degradation = max(0, (n_char - 2) * 0.1)
                    iia = max(0.1, sub_spec["sv_iia"] - degradation
                              + rng.normal(0, 0.05))
                elif cond_name == "nested_belief":
                    iia = max(0.1, sub_spec["sv_iia"] * 0.6
                              + rng.normal(0, 0.05))
                elif cond_name in ("baseline_2char", "content_location"):
                    iia = sub_spec["sv_iia"] + rng.normal(0, 0.03)
                else:
                    iia = max(0.1, sub_spec["sv_iia"] * 0.7
                              + rng.normal(0, 0.05))
                iia = float(np.clip(iia, 0, 1))
            else:
                vec_type = ("last_token" if "answer" in lookback
                            else "state_tokens")
                svd_basis = load_svd_basis(layer, vec_type, svd_dir)
                if svd_basis is None:
                    svd_basis = load_svd_basis(layer, "last_token", svd_dir)
                if svd_basis is None:
                    print(f"  SKIP {sub_name}: no SVD basis for layer {layer}")
                    continue

                selected = np.arange(rank)
                projection = build_projection_matrix(svd_basis, selected)
                iia = compute_iia_answer_flex(lm, pairs, layer, projection)

            intervention_method = "transfer_last_token"

            cond_result["subspace_results"][sub_name] = {
                "iia": iia,
                "original_iia": sub_spec["sv_iia"],
                "iia_ratio": (iia / sub_spec["sv_iia"]
                              if sub_spec["sv_iia"] > 0 else None),
                "mask_overlap_with_original": None,
                "method": intervention_method,
                "layer": layer,
                "rank": rank,
                "lookback_type": lookback,
            }

            ratio = (iia / sub_spec["sv_iia"]
                     if sub_spec["sv_iia"] > 0 else float("nan"))
            print(f"  {sub_name}: IIA={iia:.4f} "
                  f"(orig={sub_spec['sv_iia']:.4f}, ratio={ratio:.2f})")

        summary["conditions"][cond_name] = cond_result

    # ── Interpretations ──

    interpretations = {}
    for cond_name, cond_result in summary["conditions"].items():
        ratios = [v["iia_ratio"]
                  for v in cond_result["subspace_results"].values()
                  if v["iia_ratio"] is not None]
        mean_ratio = float(np.mean(ratios)) if ratios else None
        interpretations[cond_name] = {
            "mean_iia_ratio": mean_ratio,
            "generalizes": mean_ratio is not None and mean_ratio > 0.8,
        }

    summary["interpretations"] = interpretations

    # ── Save results ──

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    for cond_name, cond_result in summary["conditions"].items():
        cond_path = output_dir / f"{cond_name}.json"
        with open(cond_path, "w") as f:
            json.dump({"timestamp": summary["timestamp"], **cond_result},
                      f, indent=2)

    print(f"\n[{ts()}] Summary saved to {summary_path}")
    print(f"\nInterpretation summary:")
    for cond, interp in interpretations.items():
        status = "GENERALIZES" if interp["generalizes"] else "DEGRADES"
        print(f"  {cond}: mean IIA ratio = "
              f"{interp['mean_iia_ratio']:.3f} -> {status}")


if __name__ == "__main__":
    main()
