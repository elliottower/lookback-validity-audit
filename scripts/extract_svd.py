"""
SVD extraction for the Lookback validity audit (runs locally via NDIF).

Generates CausalToM stories using the paper's dataset code, extracts residual
stream activations at target layers via NDIF (Llama 3.1 70B), computes
truncated SVD (500 components) matching the paper's basis format.

Output: per-layer .pt files of shape (n_components, 8192) in results/svd/.

Usage:
    uv run python scripts/extract_svd.py                # full extraction
    uv run python scripts/extract_svd.py --dry-run      # verify tokenization + NDIF
    uv run python scripts/extract_svd.py --n-stories 50 # quick test with fewer stories
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
BELIEF_TRACKING = REPO_ROOT / "reference" / "belief_tracking"

sys.path.insert(0, str(BELIEF_TRACKING))
from src.dataset import Dataset as StoryDataset
from src.dataset import Sample

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
N_SVD_COMPONENTS = 500
SEED = 42

LAYERS_ANSWER = [34, 35, 36, 37, 38, 39, 43, 50, 52, 53, 54]
LAYERS_BINDING = [34, 35, 36, 37, 38]
ALL_LAYERS = sorted(set(LAYERS_ANSWER + LAYERS_BINDING))


def generate_stories(n_stories, seed):
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    random.seed(seed)
    samples = []
    for _ in range(n_stories):
        samples.append(Sample(
            template_idx=2,
            characters=random.sample(characters, 2),
            objects=random.sample(bottles, 2),
            states=random.sample(drinks, 2),
        ))

    dataset = StoryDataset(samples)
    stories = []
    for i in range(len(dataset)):
        item = dataset.__getitem__(
            i,
            set_character=random.choice([0, 1]),
            set_container=random.choice([0, 1]),
        )
        stories.append({
            "prompt": item["prompt"],
            "states": samples[i].states,
            "idx": i,
        })
    return stories


def find_state_positions(tokenizer, prompt, states):
    full_ids = tokenizer.encode(prompt)

    q_char = prompt.find("Question:")
    story_prefix = prompt[:q_char] if q_char > -1 else prompt
    story_prefix_ids = tokenizer.encode(story_prefix)
    story_end = len(story_prefix_ids)

    positions = []
    for state in states:
        target_ids = tokenizer.encode(f" {state}", add_special_tokens=False)
        for i in range(story_end - len(target_ids) + 1):
            if full_ids[i:i + len(target_ids)] == target_ids:
                positions.extend(range(i, i + len(target_ids)))

    return sorted(set(positions))


def setup_nnsight():
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    os.environ.setdefault("HF_TOKEN", "")

    from nnsight import CONFIG, LanguageModel

    CONFIG.APP.REMOTE_LOGGING = False
    CONFIG.set_default_api_key(os.environ["NDIF_KEY"])

    return LanguageModel(MODEL)


def main():
    parser = argparse.ArgumentParser(description="Extract SVD basis from Llama 3.1 70B via NDIF")
    parser.add_argument("--n-stories", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "svd"))
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    lm = setup_nnsight()
    tokenizer = lm.tokenizer
    print(f"[{ts()}] Model loaded: {MODEL}")

    n_stories = 10 if args.dry_run else args.n_stories
    stories = generate_stories(n_stories, SEED)
    print(f"[{ts()}] Generated {len(stories)} stories")

    # ── Dry run ──

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN: verifying tokenization and NDIF connectivity")
        print(f"{'='*60}\n")

        for i in range(min(5, len(stories))):
            prompt = stories[i]["prompt"]
            states = stories[i]["states"]
            tokens = tokenizer.encode(prompt)

            print(f"Story {i}: states={states}, {len(tokens)} tokens")
            positions = find_state_positions(tokenizer, prompt, states)
            for pos in positions:
                print(f"  [{pos}] = '{tokenizer.decode([tokens[pos]])}'")
            if not positions:
                print("  WARNING: no state positions found!")
            print(f"  Last token [{len(tokens)-1}] = '{tokenizer.decode([tokens[-1]])}'\n")

        print(f"[{ts()}] Testing NDIF forward pass (11 layers, mixed pattern)...")
        prompt = stories[0]["prompt"]
        with lm.trace(prompt, remote=True):
            full_34 = lm.model.layers[34].output[0].save()
            full_35 = lm.model.layers[35].output[0].save()
            full_36 = lm.model.layers[36].output[0].save()
            full_37 = lm.model.layers[37].output[0].save()
            full_38 = lm.model.layers[38].output[0].save()
            last_39 = lm.model.layers[39].output[0][-1].save()
            last_43 = lm.model.layers[43].output[0][-1].save()
            last_50 = lm.model.layers[50].output[0][-1].save()
            last_52 = lm.model.layers[52].output[0][-1].save()
            last_53 = lm.model.layers[53].output[0][-1].save()
            last_54 = lm.model.layers[54].output[0][-1].save()

        print(f"  Layer 34 last token: shape={full_34[-1].shape}, norm={torch.norm(full_34[-1]).item():.2f}")
        print(f"  Layer 52 last token: shape={last_52.shape}, norm={torch.norm(last_52).item():.2f}")

        state_pos = find_state_positions(tokenizer, prompt, stories[0]["states"])
        if state_pos:
            state_34 = full_34[state_pos[0]]
            print(f"  Layer 34 state[{state_pos[0]}]: shape={state_34.shape}, norm={torch.norm(state_34).item():.2f}")

        print(f"\n[{ts()}] Dry run passed")
        return

    # ── Full extraction ──
    #
    # nnsight 0.7.0 + NDIF constraints:
    #   1. No Python loops inside trace contexts (graph builder can't handle them)
    #   2. Can't access the same layer's output[0] twice (OutOfOrderError)
    # Solution: unroll all layer accesses as explicit statements. Save full
    # output for binding layers (need multiple positions), last-token only for
    # answer-only layers. Index binding layers AFTER the trace exits.

    output_dir = Path(args.output_dir)
    last_token_acts = {layer: [] for layer in ALL_LAYERS}
    state_token_acts = {layer: [] for layer in LAYERS_BINDING}
    n_ok, n_fail = 0, 0
    failed_indices = []
    succeeded_indices = []

    print(f"\n[{ts()}] Extracting: {len(stories)} stories, {len(ALL_LAYERS)} layers")
    print(f"  last_token layers:  {ALL_LAYERS}")
    print(f"  state_token layers: {LAYERS_BINDING}")
    print(f"  Output: {output_dir}")

    for i, story in enumerate(tqdm(stories, desc="NDIF extraction")):
        prompt = story["prompt"]
        states = story["states"]
        state_positions = find_state_positions(tokenizer, prompt, states)

        retries = 0
        while retries < 3:
            try:
                with lm.trace(prompt, remote=True):
                    # Binding layers: save full output (need last + state positions)
                    full_34 = lm.model.layers[34].output[0].save()
                    full_35 = lm.model.layers[35].output[0].save()
                    full_36 = lm.model.layers[36].output[0].save()
                    full_37 = lm.model.layers[37].output[0].save()
                    full_38 = lm.model.layers[38].output[0].save()
                    # Answer-only layers: save last token only
                    last_39 = lm.model.layers[39].output[0][-1].save()
                    last_43 = lm.model.layers[43].output[0][-1].save()
                    last_50 = lm.model.layers[50].output[0][-1].save()
                    last_52 = lm.model.layers[52].output[0][-1].save()
                    last_53 = lm.model.layers[53].output[0][-1].save()
                    last_54 = lm.model.layers[54].output[0][-1].save()

                # After trace: extract positions from materialized tensors
                binding_fulls = {
                    34: full_34.detach().cpu().float(),
                    35: full_35.detach().cpu().float(),
                    36: full_36.detach().cpu().float(),
                    37: full_37.detach().cpu().float(),
                    38: full_38.detach().cpu().float(),
                }
                answer_lasts = {
                    39: last_39.detach().cpu().float(),
                    43: last_43.detach().cpu().float(),
                    50: last_50.detach().cpu().float(),
                    52: last_52.detach().cpu().float(),
                    53: last_53.detach().cpu().float(),
                    54: last_54.detach().cpu().float(),
                }

                for layer, full in binding_fulls.items():
                    last_token_acts[layer].append(full[-1])
                    for pos in state_positions:
                        state_token_acts[layer].append(full[pos])

                for layer, last in answer_lasts.items():
                    last_token_acts[layer].append(last)

                n_ok += 1
                succeeded_indices.append(i)
                break

            except Exception as e:
                retries += 1
                if retries < 3:
                    time.sleep(5 * retries)
                else:
                    n_fail += 1
                    failed_indices.append(i)
                    if n_fail <= 10:
                        print(f"\n[{ts()}] Story {i} FAILED: {e}")

        if (i + 1) % 100 == 0:
            print(f"\n[{ts()}] Progress: {i+1}/{len(stories)} ({n_ok} ok, {n_fail} fail)")

        # Save partial results every 200 stories
        if (i + 1) % 200 == 0:
            ckpt_dir = output_dir / "partial_activations"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for layer in ALL_LAYERS:
                if last_token_acts[layer]:
                    torch.save(torch.stack(last_token_acts[layer]), ckpt_dir / f"last_token_L{layer}.pt")
            for layer in LAYERS_BINDING:
                if state_token_acts[layer]:
                    torch.save(torch.stack(state_token_acts[layer]), ckpt_dir / f"state_tokens_L{layer}.pt")
            with open(ckpt_dir / "progress.json", "w") as f:
                json.dump({"stories_done": i + 1, "ok": n_ok, "fail": n_fail, "ts": ts()}, f)
            print(f"\n[{ts()}] Checkpoint saved at story {i+1}")

    print(f"\n[{ts()}] Extraction done: {n_ok}/{len(stories)} succeeded, {n_fail} failed")

    # ── Compute SVD ──

    print(f"\n[{ts()}] Computing SVDs...")
    svd_meta = {}

    for vec_type, acts_dict, layers in [
        ("last_token", last_token_acts, ALL_LAYERS),
        ("state_tokens", state_token_acts, LAYERS_BINDING),
    ]:
        for layer in layers:
            acts = acts_dict[layer]
            if len(acts) < 10:
                print(f"[{ts()}] SKIP {vec_type} L{layer}: only {len(acts)} samples")
                continue

            X = torch.stack(acts)
            n_comp = min(N_SVD_COMPONENTS, X.shape[0], X.shape[1])

            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            Vt = Vt[:n_comp]
            S = S[:n_comp]

            vecs_dir = output_dir / "CausalToM" / vec_type / "singular_vecs"
            vecs_dir.mkdir(parents=True, exist_ok=True)
            torch.save(Vt, vecs_dir / f"{layer}.pt")

            vals_dir = output_dir / "CausalToM" / vec_type / "singular_values"
            vals_dir.mkdir(parents=True, exist_ok=True)
            torch.save(S, vals_dir / f"{layer}.pt")

            total_var = (S ** 2).sum().item()
            full_var = torch.norm(X, "fro").item() ** 2
            var_expl = total_var / full_var if full_var > 0 else 0.0

            svd_meta[f"{vec_type}_L{layer}"] = {
                "n_samples": len(acts),
                "n_components": int(n_comp),
                "variance_explained": var_expl,
                "top_10_sv": S[:10].tolist(),
            }
            print(f"[{ts()}] {vec_type} L{layer}: {len(acts)} -> {n_comp} components, var={var_expl:.4f}")

    # ── Save metadata ──

    metadata = {
        "model": MODEL,
        "model_note": "Llama 3.1 (paper used Llama 3.0)",
        "n_stories_target": args.n_stories,
        "n_stories_extracted": n_ok,
        "n_stories_failed": n_fail,
        "succeeded_indices": succeeded_indices,
        "failed_indices": failed_indices[:100],
        "seed": SEED,
        "n_svd_components_target": N_SVD_COMPONENTS,
        "centered": False,
        "layers_all": ALL_LAYERS,
        "layers_binding": LAYERS_BINDING,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "svd_details": svd_meta,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n[{ts()}] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
