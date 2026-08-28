"""
X2 — Mismatch test on Qwen2.5-14B-Instruct.

Reproduces Prakash et al.'s best causal design (lines 921-938) on the model
where they acknowledge subspace experiments were never run. The test patches
full residual streams at specific token positions:

  Condition 1: Patch recalled tokens only (state word positions).
               Address updated, pointer stale -> QK mismatch -> should FAIL.
  Condition 2: Patch lookback tokens only (question + answer positions).
               Pointer updated, address stale -> QK mismatch -> should FAIL.
  Condition 3: Patch BOTH recalled + lookback tokens.
               Both address and pointer coherent -> should SUCCEED.

A clean IIA signature (singles near chance, paired near ceiling) is evidence
that the dependency structure holds cross-model without subspace fitting.

Runs on Modal with a single A100 (Qwen2.5-14B fits in ~30GB fp16).

Usage:
    modal run scripts/modal_qwen_mismatch_test.py --detach               # full
    modal run scripts/modal_qwen_mismatch_test.py --dry-run --detach     # verify setup
    modal volume get lookback-qwen-mismatch results/ results/qwen_mismatch/
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nnsight==0.7.0",
        "torch==2.5.1",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "matplotlib==3.10.1",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "dataclasses-json==0.6.7",
    )
    .add_local_dir(
        "/Users/elliottower/Documents/GitHub/lookback-validity-audit/reference/belief_tracking",
        remote_path="/root/belief_tracking",
    )
)

vol = modal.Volume.from_name("lookback-qwen-mismatch", create_if_missing=True)

app = modal.App("lookback-qwen-mismatch-test")

MODEL = "Qwen/Qwen2.5-14B-Instruct"
N_EVAL = 200
N_STORIES = 600
SEED = 42

# Llama-70B binding at L34-38/80 = 42-48% depth -> Qwen-48 L20-23
# Llama-70B answer at L38-53/80 = 48-66% depth -> Qwen-48 L23-32
# Sweep wider to catch shifted mechanisms: every 2 layers across 25-75% depth
BINDING_LAYERS = list(range(12, 36, 2))
ANSWER_LAYERS = list(range(16, 40, 2))


# ── Pure helpers ──


def _generate_binding_pairs(n_samples, seed):
    """Generate binding lookback counterfactual pairs using vendored code."""
    import json
    import random
    import sys

    sys.path.insert(0, "/root/belief_tracking")
    from notebooks.causalToM_novis.utils import get_reversed_sentence_counterfacts

    with open("/root/belief_tracking/data/synthetic_entities/characters.json") as f:
        characters = json.load(f)
    with open("/root/belief_tracking/data/synthetic_entities/bottles.json") as f:
        bottles = json.load(f)
    with open("/root/belief_tracking/data/synthetic_entities/drinks.json") as f:
        drinks = json.load(f)

    random.seed(seed)
    return get_reversed_sentence_counterfacts(characters, bottles, drinks, n_samples)


def _generate_answer_pairs(n_samples, seed):
    """Generate answer lookback counterfactual pairs using vendored code."""
    import json
    import random
    import sys

    sys.path.insert(0, "/root/belief_tracking")
    from notebooks.causalToM_novis.utils import get_reversed_sent_diff_state_counterfacts

    with open("/root/belief_tracking/data/synthetic_entities/characters.json") as f:
        characters = json.load(f)
    with open("/root/belief_tracking/data/synthetic_entities/bottles.json") as f:
        bottles = json.load(f)
    with open("/root/belief_tracking/data/synthetic_entities/drinks.json") as f:
        drinks = json.load(f)

    random.seed(seed)
    return get_reversed_sent_diff_state_counterfacts(characters, bottles, drinks, n_samples)


def _find_state_positions(tokenizer, prompt, states):
    """Find token positions of state words before 'Question:' in the prompt."""
    full_ids = tokenizer.encode(prompt)
    q_idx = prompt.find("Question:")
    story_prefix = prompt[:q_idx] if q_idx > -1 else prompt
    story_prefix_ids = tokenizer.encode(story_prefix)
    story_end = len(story_prefix_ids)

    positions = []
    for state in states:
        target_ids = tokenizer.encode(f" {state}", add_special_tokens=False)
        for i in range(story_end - len(target_ids) + 1):
            if full_ids[i:i + len(target_ids)] == target_ids:
                positions.extend(range(i, i + len(target_ids)))
    return sorted(set(positions))


def _find_question_answer_positions(tokenizer, prompt):
    """Find token positions from 'Question:' onward (lookback tokens)."""
    full_ids = tokenizer.encode(prompt)
    q_idx = prompt.find("Question:")
    if q_idx == -1:
        return list(range(len(full_ids) - 5, len(full_ids)))

    prefix_before_q = prompt[:q_idx]
    prefix_ids = tokenizer.encode(prefix_before_q)
    return list(range(len(prefix_ids), len(full_ids)))


def _setup_model():
    """Load Qwen2.5-14B-Instruct locally on GPU."""
    import torch
    from nnsight import LanguageModel

    lm = LanguageModel(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return lm


def _filter_on_model(lm, pairs, max_size):
    """Keep pairs where model gets both clean and CF correct."""
    import torch
    from tqdm import tqdm

    filtered = []
    for sample in tqdm(pairs, desc="Filtering on model accuracy"):
        clean_prompt = sample["clean_prompt"]
        cf_prompt = sample["counterfactual_prompt"]
        clean_target = sample["clean_ans"]
        cf_target = sample["counterfactual_ans"]

        try:
            with torch.no_grad():
                with lm.trace(clean_prompt):
                    clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()
                with lm.trace(cf_prompt):
                    cf_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

            clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
            cf_tok = lm.tokenizer.decode([cf_pred.item()]).lower().strip()

            if clean_tok == clean_target.lower().strip() and cf_tok == cf_target.lower().strip():
                filtered.append(sample)
                if len(filtered) >= max_size:
                    break
        except Exception as e:
            print(f"  Filter error: {e}")

    return filtered


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _run_mismatch_test_binding(lm, pairs, layers, ts_fn):
    """Run three-condition mismatch test for binding lookback.

    Recalled tokens = state word positions (address + payload).
    Lookback tokens = question + answer positions (pointer).
    Checkpoints every 25 samples within each layer.
    """
    import json
    import time
    from pathlib import Path

    import torch
    from tqdm import tqdm

    results_dir = Path("/results")
    results = {}

    for layer in layers:
        ckpt_path = results_dir / f"binding_mismatch_L{layer}.json"
        partial_path = results_dir / f"binding_mismatch_L{layer}_partial.json"

        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if not cached.get("partial", False):
                results[layer] = cached
                print(f"[{ts_fn()}] L{layer}: resumed from complete checkpoint")
                continue

        counts = {
            "clean_correct": 0,
            "recalled_only_correct": 0,
            "lookback_only_correct": 0,
            "both_correct": 0,
            "total": 0,
        }
        start_idx = 0

        if partial_path.exists():
            with open(partial_path) as f:
                partial = json.load(f)
            counts = partial["counts"]
            start_idx = partial["samples_done"]
            print(f"[{ts_fn()}] L{layer}: resuming from sample {start_idx}/{len(pairs)}")

        for i in tqdm(range(start_idx, len(pairs)), desc=f"Binding mismatch L{layer}",
                      initial=start_idx, total=len(pairs)):
            sample = pairs[i]
            clean_prompt = sample["clean_prompt"]
            cf_prompt = sample["counterfactual_prompt"]
            target = sample["target"].lower().strip()

            clean_states = sample["clean_states"]
            cf_states = sample["counterfactual_states"]

            recalled_pos = _find_state_positions(lm.tokenizer, clean_prompt, clean_states)
            lookback_pos = _find_question_answer_positions(lm.tokenizer, clean_prompt)

            cf_recalled_pos = _find_state_positions(lm.tokenizer, cf_prompt, cf_states)
            cf_lookback_pos = _find_question_answer_positions(lm.tokenizer, cf_prompt)

            try:
                with torch.no_grad():
                    with lm.trace(clean_prompt):
                        cl_out = lm.model.layers[layer].output[0][0].save()
                        clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                    with lm.trace(cf_prompt):
                        cf_out = lm.model.layers[layer].output[0][0].save()

                cl_t = cl_out.detach().clone()
                cf_t = cf_out.detach().clone()

                # Condition 1: patch recalled (state) positions only
                patched_recalled = cl_t.clone()
                for cp, rp in zip(cf_recalled_pos, recalled_pos):
                    if cp < cf_t.shape[0] and rp < patched_recalled.shape[0]:
                        patched_recalled[rp] = cf_t[cp]

                with lm.trace(clean_prompt):
                    lm.model.layers[layer].output[0][0] = patched_recalled
                    recalled_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                # Condition 2: patch lookback (question/answer) positions only
                patched_lookback = cl_t.clone()
                for cp, lp in zip(cf_lookback_pos, lookback_pos):
                    if cp < cf_t.shape[0] and lp < patched_lookback.shape[0]:
                        patched_lookback[lp] = cf_t[cp]

                with lm.trace(clean_prompt):
                    lm.model.layers[layer].output[0][0] = patched_lookback
                    lookback_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                # Condition 3: patch BOTH
                patched_both = cl_t.clone()
                for cp, rp in zip(cf_recalled_pos, recalled_pos):
                    if cp < cf_t.shape[0] and rp < patched_both.shape[0]:
                        patched_both[rp] = cf_t[cp]
                for cp, lp in zip(cf_lookback_pos, lookback_pos):
                    if cp < cf_t.shape[0] and lp < patched_both.shape[0]:
                        patched_both[lp] = cf_t[cp]

                with lm.trace(clean_prompt):
                    lm.model.layers[layer].output[0][0] = patched_both
                    both_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
                recalled_tok = lm.tokenizer.decode([recalled_pred.item()]).lower().strip()
                lookback_tok = lm.tokenizer.decode([lookback_pred.item()]).lower().strip()
                both_tok = lm.tokenizer.decode([both_pred.item()]).lower().strip()

                counts["clean_correct"] += int(clean_tok == sample["clean_ans"].lower().strip())
                counts["recalled_only_correct"] += int(recalled_tok == target)
                counts["lookback_only_correct"] += int(lookback_tok == target)
                counts["both_correct"] += int(both_tok == target)
                counts["total"] += 1

            except Exception as e:
                print(f"  L{layer} sample {i} error: {type(e).__name__}: {e}")
                counts["total"] += 1
                time.sleep(1)

            if (i + 1) % 25 == 0:
                with open(partial_path, "w") as f:
                    json.dump({"counts": counts, "samples_done": i + 1}, f, indent=2)
                vol.commit()
                print(f"  [{ts_fn()}] L{layer} checkpoint: {i+1}/{len(pairs)}")

        n = counts["total"]
        entry = {
            "layer": layer,
            "lookback_type": "binding",
            "n_samples": n,
            "clean_accuracy": counts["clean_correct"] / n if n > 0 else 0,
            "recalled_only_iia": counts["recalled_only_correct"] / n if n > 0 else 0,
            "lookback_only_iia": counts["lookback_only_correct"] / n if n > 0 else 0,
            "both_iia": counts["both_correct"] / n if n > 0 else 0,
            "mismatch_signature": (
                counts["both_correct"] / n > 0.5
                and counts["recalled_only_correct"] / n < 0.3
                and counts["lookback_only_correct"] / n < 0.3
            ) if n > 0 else False,
            "wilson_ci_both": _wilson_ci(counts["both_correct"], n),
            "wilson_ci_recalled": _wilson_ci(counts["recalled_only_correct"], n),
            "wilson_ci_lookback": _wilson_ci(counts["lookback_only_correct"], n),
        }
        results[layer] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)
        if partial_path.exists():
            partial_path.unlink()
        vol.commit()

        print(f"[{ts_fn()}] L{layer}: recalled={entry['recalled_only_iia']:.3f}, "
              f"lookback={entry['lookback_only_iia']:.3f}, "
              f"both={entry['both_iia']:.3f} "
              f"{'MISMATCH' if entry['mismatch_signature'] else 'no sig'}")

    return results


def _run_mismatch_test_answer(lm, pairs, layers, ts_fn):
    """Run three-condition mismatch test for answer lookback.

    For answer lookback the recalled token is the state word and the lookback
    token is the final ":" / answer position. Patching the last token alone
    updates the pointer; patching the state token alone updates the address.
    Checkpoints every 25 samples within each layer.
    """
    import json
    import time
    from pathlib import Path

    import torch
    from tqdm import tqdm

    results_dir = Path("/results")
    results = {}

    for layer in layers:
        ckpt_path = results_dir / f"answer_mismatch_L{layer}.json"
        partial_path = results_dir / f"answer_mismatch_L{layer}_partial.json"

        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if not cached.get("partial", False):
                results[layer] = cached
                print(f"[{ts_fn()}] L{layer}: resumed from complete checkpoint")
                continue

        counts = {
            "clean_correct": 0,
            "recalled_only_correct": 0,
            "lookback_only_correct": 0,
            "both_correct": 0,
            "total": 0,
        }
        start_idx = 0

        if partial_path.exists():
            with open(partial_path) as f:
                partial = json.load(f)
            counts = partial["counts"]
            start_idx = partial["samples_done"]
            print(f"[{ts_fn()}] L{layer}: resuming from sample {start_idx}/{len(pairs)}")

        for i in tqdm(range(start_idx, len(pairs)), desc=f"Answer mismatch L{layer}",
                      initial=start_idx, total=len(pairs)):
            sample = pairs[i]
            clean_prompt = sample["clean_prompt"]
            cf_prompt = sample["counterfactual_prompt"]
            clean_target = sample["clean_ans"].lower().strip()
            cf_target = sample["counterfactual_ans"].lower().strip()

            clean_states = sample["clean_states"]
            cf_states = sample["counterfactual_states"]

            recalled_pos = _find_state_positions(lm.tokenizer, clean_prompt, clean_states)
            cf_recalled_pos = _find_state_positions(lm.tokenizer, cf_prompt, cf_states)

            try:
                with torch.no_grad():
                    with lm.trace(clean_prompt):
                        cl_out = lm.model.layers[layer].output[0][0].save()
                        clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                    with lm.trace(cf_prompt):
                        cf_out = lm.model.layers[layer].output[0][0].save()

                cl_t = cl_out.detach().clone()
                cf_t = cf_out.detach().clone()

                # Condition 1: patch recalled (state token) positions only
                patched_recalled = cl_t.clone()
                for cp, rp in zip(cf_recalled_pos, recalled_pos):
                    if cp < cf_t.shape[0] and rp < patched_recalled.shape[0]:
                        patched_recalled[rp] = cf_t[cp]

                with lm.trace(clean_prompt):
                    lm.model.layers[layer].output[0][0] = patched_recalled
                    recalled_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                # Condition 2: patch lookback (last token = answer position) only
                patched_lookback = cl_t.clone()
                last_pos = cl_t.shape[0] - 1
                cf_last_pos = cf_t.shape[0] - 1
                patched_lookback[last_pos] = cf_t[cf_last_pos]

                with lm.trace(clean_prompt):
                    lm.model.layers[layer].output[0][0] = patched_lookback
                    lookback_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                # Condition 3: patch BOTH
                patched_both = cl_t.clone()
                for cp, rp in zip(cf_recalled_pos, recalled_pos):
                    if cp < cf_t.shape[0] and rp < patched_both.shape[0]:
                        patched_both[rp] = cf_t[cp]
                patched_both[last_pos] = cf_t[cf_last_pos]

                with lm.trace(clean_prompt):
                    lm.model.layers[layer].output[0][0] = patched_both
                    both_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                clean_tok = lm.tokenizer.decode([clean_pred.item()]).lower().strip()
                recalled_tok = lm.tokenizer.decode([recalled_pred.item()]).lower().strip()
                lookback_tok = lm.tokenizer.decode([lookback_pred.item()]).lower().strip()
                both_tok = lm.tokenizer.decode([both_pred.item()]).lower().strip()

                counts["clean_correct"] += int(clean_tok == clean_target)
                counts["recalled_only_correct"] += int(recalled_tok == cf_target)
                counts["lookback_only_correct"] += int(lookback_tok == cf_target)
                counts["both_correct"] += int(both_tok == cf_target)
                counts["total"] += 1

            except Exception as e:
                print(f"  L{layer} sample {i} error: {type(e).__name__}: {e}")
                counts["total"] += 1
                time.sleep(1)

            if (i + 1) % 25 == 0:
                with open(partial_path, "w") as f:
                    json.dump({"counts": counts, "samples_done": i + 1}, f, indent=2)
                vol.commit()
                print(f"  [{ts_fn()}] L{layer} checkpoint: {i+1}/{len(pairs)}")

        n = counts["total"]
        entry = {
            "layer": layer,
            "lookback_type": "answer",
            "n_samples": n,
            "clean_accuracy": counts["clean_correct"] / n if n > 0 else 0,
            "recalled_only_iia": counts["recalled_only_correct"] / n if n > 0 else 0,
            "lookback_only_iia": counts["lookback_only_correct"] / n if n > 0 else 0,
            "both_iia": counts["both_correct"] / n if n > 0 else 0,
            "mismatch_signature": (
                counts["both_correct"] / n > 0.5
                and counts["recalled_only_correct"] / n < 0.3
                and counts["lookback_only_correct"] / n < 0.3
            ) if n > 0 else False,
            "wilson_ci_both": _wilson_ci(counts["both_correct"], n),
            "wilson_ci_recalled": _wilson_ci(counts["recalled_only_correct"], n),
            "wilson_ci_lookback": _wilson_ci(counts["lookback_only_correct"], n),
        }
        results[layer] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)
        if partial_path.exists():
            partial_path.unlink()
        vol.commit()

        print(f"[{ts_fn()}] L{layer}: recalled={entry['recalled_only_iia']:.3f}, "
              f"lookback={entry['lookback_only_iia']:.3f}, "
              f"both={entry['both_iia']:.3f} "
              f"{'MISMATCH' if entry['mismatch_signature'] else 'no sig'}")

    return results


# ── Modal function ──


@app.function(
    image=image,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100",
    timeout=86400,
    memory=65536,
)
def run_mismatch_test(dry_run: bool = False):
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    results_dir = Path("/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ts()}] X2 Mismatch test: Qwen2.5-14B-Instruct")
    print(f"[{ts()}] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[{ts()}] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"[{ts()}] Loading model...")
    lm = _setup_model()
    print(f"[{ts()}] Model loaded")

    n_stories = 20 if dry_run else N_STORIES

    print(f"[{ts()}] Generating binding pairs (n={n_stories})...")
    binding_raw = _generate_binding_pairs(n_stories, SEED)
    print(f"[{ts()}] {len(binding_raw)} raw binding pairs")

    print(f"[{ts()}] Generating answer pairs (n={n_stories})...")
    answer_raw = _generate_answer_pairs(n_stories, SEED)
    print(f"[{ts()}] {len(answer_raw)} raw answer pairs")

    n_eval = 10 if dry_run else N_EVAL

    # Filter on model accuracy
    binding_cache = results_dir / "filtered_binding_pairs.json"
    if binding_cache.exists() and not dry_run:
        with open(binding_cache) as f:
            binding_pairs = json.load(f)
        print(f"[{ts()}] Loaded {len(binding_pairs)} cached binding pairs")
    else:
        print(f"[{ts()}] Filtering binding pairs on Qwen accuracy...")
        binding_pairs = _filter_on_model(lm, binding_raw, max_size=n_eval)
        with open(binding_cache, "w") as f:
            json.dump(binding_pairs, f, indent=2)
        vol.commit()
        print(f"[{ts()}] {len(binding_pairs)} binding pairs pass filter")

    answer_cache = results_dir / "filtered_answer_pairs.json"
    if answer_cache.exists() and not dry_run:
        with open(answer_cache) as f:
            answer_pairs = json.load(f)
        print(f"[{ts()}] Loaded {len(answer_pairs)} cached answer pairs")
    else:
        print(f"[{ts()}] Filtering answer pairs on Qwen accuracy...")
        answer_pairs = _filter_on_model(lm, answer_raw, max_size=n_eval)
        with open(answer_cache, "w") as f:
            json.dump(answer_pairs, f, indent=2)
        vol.commit()
        print(f"[{ts()}] {len(answer_pairs)} answer pairs pass filter")

    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN: verifying tokenization + model inference")
        print(f"{'='*60}\n")

        for label, pairs in [("binding", binding_pairs[:3]), ("answer", answer_pairs[:3])]:
            print(f"\n--- {label} pairs ---")
            for i, p in enumerate(pairs):
                tokens = lm.tokenizer.encode(p["clean_prompt"])
                states = p["clean_states"]
                state_pos = _find_state_positions(lm.tokenizer, p["clean_prompt"], states)
                qa_pos = _find_question_answer_positions(lm.tokenizer, p["clean_prompt"])
                print(f"  Pair {i}: {len(tokens)} tokens, states={states}")
                print(f"    State positions: {state_pos}")
                for sp in state_pos:
                    print(f"      [{sp}] = '{lm.tokenizer.decode([tokens[sp]])}'")
                print(f"    Q/A positions: {qa_pos[:5]}...{qa_pos[-3:]}")
                print(f"    Clean ans: {p['clean_ans']}, CF ans: {p['counterfactual_ans']}")

        # Quick single-layer test
        test_layers = [24] if not dry_run else [24]
        print(f"\n[{ts()}] Running quick single-layer binding test (L24)...")
        binding_results = _run_mismatch_test_binding(lm, binding_pairs[:5], [24], ts)
        print(f"\n[{ts()}] Running quick single-layer answer test (L24)...")
        answer_results = _run_mismatch_test_answer(lm, answer_pairs[:5], [24], ts)

        return {
            "status": "dry_run_ok",
            "binding_pairs_filtered": len(binding_pairs),
            "answer_pairs_filtered": len(answer_pairs),
            "test_binding_L24": binding_results.get(24, {}),
            "test_answer_L24": answer_results.get(24, {}),
        }

    # Full run
    print(f"\n{'='*60}")
    print(f"[{ts()}] BINDING MISMATCH TEST ({len(binding_pairs)} pairs, {len(BINDING_LAYERS)} layers)")
    print(f"{'='*60}")
    binding_results = _run_mismatch_test_binding(lm, binding_pairs, BINDING_LAYERS, ts)

    print(f"\n{'='*60}")
    print(f"[{ts()}] ANSWER MISMATCH TEST ({len(answer_pairs)} pairs, {len(ANSWER_LAYERS)} layers)")
    print(f"{'='*60}")
    answer_results = _run_mismatch_test_answer(lm, answer_pairs, ANSWER_LAYERS, ts)

    # Summary
    summary = {
        "experiment": "X2 Mismatch test — Qwen2.5-14B-Instruct",
        "model": MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_binding_pairs": len(binding_pairs),
        "n_answer_pairs": len(answer_pairs),
        "binding_layers": BINDING_LAYERS,
        "answer_layers": ANSWER_LAYERS,
        "method": (
            "Three-condition mismatch test (Prakash et al. lines 921-938). "
            "Condition 1: patch recalled tokens only. "
            "Condition 2: patch lookback tokens only. "
            "Condition 3: patch both. "
            "Mismatch signature: singles fail (<0.3 IIA), pair succeeds (>0.5 IIA)."
        ),
        "binding_results": {str(k): v for k, v in binding_results.items()},
        "answer_results": {str(k): v for k, v in answer_results.items()},
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()

    print(f"\n{'='*60}")
    print("MISMATCH TEST SUMMARY")
    print(f"{'='*60}")

    for label, res_dict in [("BINDING", binding_results), ("ANSWER", answer_results)]:
        print(f"\n{label}:")
        for layer, r in sorted(res_dict.items()):
            sig = "MISMATCH" if r.get("mismatch_signature") else "no sig"
            print(f"  L{layer}: recalled={r['recalled_only_iia']:.3f}  "
                  f"lookback={r['lookback_only_iia']:.3f}  "
                  f"both={r['both_iia']:.3f}  ({sig})")

    binding_sig_layers = [l for l, r in binding_results.items() if r.get("mismatch_signature")]
    answer_sig_layers = [l for l, r in answer_results.items() if r.get("mismatch_signature")]
    print(f"\nBinding mismatch signature at: {binding_sig_layers or 'NONE'}")
    print(f"Answer mismatch signature at: {answer_sig_layers or 'NONE'}")

    print(f"\n[{ts()}] Done. Results at /results/")
    print(f"Download: modal volume get lookback-qwen-mismatch results/ results/qwen_mismatch/")

    return summary


@app.local_entrypoint()
def main(dry_run: bool = False):
    import json
    result = run_mismatch_test.remote(dry_run=dry_run)
    print(json.dumps(result, indent=2, default=str))
