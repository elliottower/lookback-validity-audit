"""
SVD extraction for the Lookback validity audit.

Generates CausalToM stories using the paper's dataset code (mounted from
reference/belief_tracking/), extracts residual stream activations at target
layers via NDIF (Llama 3.1 70B), and computes truncated SVD matching the
paper's basis format: per-layer .pt files of shape (n_components, 8192).

Two vector types:
  last_token:   activation at position -1 (for answer_lookback experiments)
  state_tokens: activations at state word positions (for binding_lookback)

Usage:
    modal run scripts/modal_extract_svd.py --detach          # full extraction
    modal run scripts/modal_extract_svd.py --dry-run         # verify tokenization + NDIF
    modal volume get lookback-svd-results svd/ results/svd/  # download results
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
        "scipy==1.14.1",
        "tqdm==4.67.1",
        "dataclasses-json==0.6.7",
    )
    .add_local_dir(
        "/Users/elliottower/Documents/GitHub/lookback-validity-audit/reference/belief_tracking",
        remote_path="/root/belief_tracking",
    )
)

vol = modal.Volume.from_name("lookback-svd-results", create_if_missing=True)

app = modal.App("lookback-extract-svd")

MODAL_SECRETS = modal.Secret.from_dict({
    "NDIF_KEY": "801a2649-a054-4703-ad6d-9e93c8cbcb9a",
    "HF_TOKEN": modal.Secret.from_name("huggingface-token"),
})

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
N_STORIES = 600
N_SVD_COMPONENTS = 500
SEED = 42

LAYERS_ANSWER = [34, 35, 36, 37, 38, 39, 43, 50, 52, 53, 54]
LAYERS_BINDING = [34, 35, 36, 37, 38]
ALL_LAYERS = sorted(set(LAYERS_ANSWER + LAYERS_BINDING))


# ── Pure helpers (must run inside Modal where /root/belief_tracking is mounted) ──


def _generate_stories(n_stories, seed):
    """Generate CausalToM stories using the paper's dataset code."""
    import json
    import random
    import sys

    sys.path.insert(0, "/root/belief_tracking")
    from src.dataset import Dataset as StoryDataset
    from src.dataset import Sample

    with open("/root/belief_tracking/data/synthetic_entities/characters.json") as f:
        characters = json.load(f)
    with open("/root/belief_tracking/data/synthetic_entities/bottles.json") as f:
        bottles = json.load(f)
    with open("/root/belief_tracking/data/synthetic_entities/drinks.json") as f:
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


def _find_state_positions(tokenizer, prompt, states):
    """Find token positions of state words in the story portion of the prompt.

    Searches only before "Question:" to avoid matching answer text.
    Returns sorted list of token positions (ints).
    """
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


def _setup_nnsight():
    """Configure nnsight for NDIF remote execution and return model handle."""
    import os

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    from nnsight import LanguageModel

    try:
        from nnsight import CONFIG
        CONFIG.APP.REMOTE_LOGGING = False
        CONFIG.set_default_api_key(os.environ["NDIF_KEY"])
    except Exception:
        from nnsight import ndif as ndif_mod
        ndif_mod.api_key = os.environ["NDIF_KEY"]

    return LanguageModel(MODEL)


# ── Modal function ──


@app.function(
    image=image,
    volumes={"/results": vol},
    secrets=[MODAL_SECRETS],
    timeout=86400,
    cpu=1,
    memory=4096,
)
def extract_svd(dry_run: bool = False):
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    from tqdm import tqdm

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    lm = _setup_nnsight()
    tokenizer = lm.tokenizer
    print(f"[{ts()}] Model loaded: {MODEL}")

    n_stories = 10 if dry_run else N_STORIES
    stories = _generate_stories(n_stories, SEED)
    print(f"[{ts()}] Generated {len(stories)} stories")

    # ── Dry run: verify tokenization and one NDIF pass ──

    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN: verifying tokenization and NDIF connectivity")
        print(f"{'='*60}\n")

        for i in range(min(5, len(stories))):
            prompt = stories[i]["prompt"]
            states = stories[i]["states"]
            tokens = tokenizer.encode(prompt)

            print(f"Story {i}: states={states}, {len(tokens)} tokens")
            positions = _find_state_positions(tokenizer, prompt, states)
            for pos in positions:
                print(f"  [{pos}] = '{tokenizer.decode([tokens[pos]])}'")
            if not positions:
                print("  WARNING: no state positions found!")
            print(f"  Last token [{len(tokens)-1}] = '{tokenizer.decode([tokens[-1]])}'\n")

        print(f"[{ts()}] Testing NDIF forward pass (layers 34, 52)...")
        prompt = stories[0]["prompt"]
        state_pos = _find_state_positions(tokenizer, prompt, stories[0]["states"])
        with lm.trace(prompt, remote=True):
            full_34 = lm.model.layers[34].output[0].save()
            full_52 = lm.model.layers[52].output[0].save()

        last_34 = full_34[-1]
        last_52 = full_52[-1]
        print(f"  Layer 34 last token: shape={last_34.shape}, norm={torch.norm(last_34).item():.2f}")
        print(f"  Layer 52 last token: shape={last_52.shape}, norm={torch.norm(last_52).item():.2f}")
        if state_pos:
            state_34 = full_34[state_pos[0]]
            print(f"  Layer 34 state[{state_pos[0]}]: shape={state_34.shape}, norm={torch.norm(state_34).item():.2f}")
        print(f"\n[{ts()}] Dry run passed")

        return {"status": "dry_run_ok", "n_tokens_story0": len(tokenizer.encode(stories[0]["prompt"]))}

    # ── Full extraction ──

    results_dir = Path("/results")
    last_token_acts = {layer: [] for layer in ALL_LAYERS}
    state_token_acts = {layer: [] for layer in LAYERS_BINDING}
    n_ok, n_fail = 0, 0
    failed_indices = []

    print(f"\n[{ts()}] Extracting: {len(stories)} stories, {len(ALL_LAYERS)} layers")
    print(f"  last_token layers:  {ALL_LAYERS}")
    print(f"  state_token layers: {LAYERS_BINDING}")

    for i, story in enumerate(tqdm(stories, desc="NDIF extraction")):
        prompt = story["prompt"]
        states = story["states"]
        state_positions = _find_state_positions(tokenizer, prompt, states)

        retries = 0
        while retries < 3:
            try:
                with lm.trace(prompt, remote=True):
                    saved_full = {}
                    for layer in ALL_LAYERS:
                        saved_full[layer] = lm.model.layers[layer].output[0].save()

                for layer in ALL_LAYERS:
                    full = saved_full[layer].detach().cpu().float()
                    last_token_acts[layer].append(full[-1])
                    if layer in LAYERS_BINDING:
                        for pos in state_positions:
                            state_token_acts[layer].append(full[pos])

                n_ok += 1
                break

            except Exception as e:
                retries += 1
                if retries < 3:
                    time.sleep(5 * retries)
                else:
                    n_fail += 1
                    failed_indices.append(i)
                    if n_fail <= 10:
                        print(f"[{ts()}] Story {i} FAILED: {e}")

        if (i + 1) % 100 == 0:
            print(f"[{ts()}] Progress: {i+1}/{len(stories)} ({n_ok} ok, {n_fail} fail)")

        # Partial save every 200 stories
        if (i + 1) % 200 == 0:
            ckpt_dir = results_dir / "svd" / "partial_activations"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for layer in ALL_LAYERS:
                if last_token_acts[layer]:
                    torch.save(torch.stack(last_token_acts[layer]), ckpt_dir / f"last_token_L{layer}.pt")
            for layer in LAYERS_BINDING:
                if state_token_acts[layer]:
                    torch.save(torch.stack(state_token_acts[layer]), ckpt_dir / f"state_tokens_L{layer}.pt")
            with open(ckpt_dir / "progress.json", "w") as f:
                json.dump({"stories_done": i + 1, "ok": n_ok, "fail": n_fail, "ts": ts()}, f)
            vol.commit()
            print(f"[{ts()}] Checkpoint saved at story {i+1}")

    print(f"\n[{ts()}] Extraction done: {n_ok}/{len(stories)} succeeded, {n_fail} failed")

    # ── Compute SVD per (vector_type, layer) ──

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

            X = torch.stack(acts)  # (n, d_model)
            n_comp = min(N_SVD_COMPONENTS, X.shape[0], X.shape[1])

            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            Vt = Vt[:n_comp]
            S = S[:n_comp]

            # Save right singular vectors (the SVD basis)
            vecs_dir = results_dir / "svd" / "CausalToM" / vec_type / "singular_vecs"
            vecs_dir.mkdir(parents=True, exist_ok=True)
            torch.save(Vt, vecs_dir / f"{layer}.pt")

            # Save singular values (for diagnostics)
            vals_dir = results_dir / "svd" / "CausalToM" / vec_type / "singular_values"
            vals_dir.mkdir(parents=True, exist_ok=True)
            torch.save(S, vals_dir / f"{layer}.pt")

            vol.commit()

            total_var = (S ** 2).sum().item()
            full_var = torch.norm(X, "fro").item() ** 2
            var_expl = total_var / full_var if full_var > 0 else 0.0

            svd_meta[f"{vec_type}_L{layer}"] = {
                "n_samples": len(acts),
                "n_components": int(n_comp),
                "variance_explained": var_expl,
                "top_10_sv": S[:10].tolist(),
            }
            print(f"[{ts()}] {vec_type} L{layer}: {len(acts)} samples -> {n_comp} components, var={var_expl:.4f}")

    # ── Save metadata ──

    metadata = {
        "model": MODEL,
        "model_note": "Llama 3.1 (paper used Llama 3.0)",
        "n_stories_target": N_STORIES,
        "n_stories_extracted": n_ok,
        "n_stories_failed": n_fail,
        "failed_indices": failed_indices[:100],
        "seed": SEED,
        "n_svd_components_target": N_SVD_COMPONENTS,
        "centered": False,
        "layers_all": ALL_LAYERS,
        "layers_binding": LAYERS_BINDING,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "svd_details": svd_meta,
    }

    meta_path = results_dir / "svd" / "extraction_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    vol.commit()

    print(f"\n[{ts()}] All results saved to Modal volume 'lookback-svd-results'")
    print(f"[{ts()}] Download: modal volume get lookback-svd-results svd/ results/svd/")

    return metadata


@app.local_entrypoint()
def main(dry_run: bool = False):
    import json
    result = extract_svd.remote(dry_run=dry_run)
    print(json.dumps(result, indent=2, default=str))
