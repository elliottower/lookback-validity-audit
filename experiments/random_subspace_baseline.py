"""
Random subspace baseline for the Lookback paper's identified subspaces.

Tests whether random selections of SVD components achieve comparable IIA.
The null: select r-of-500 SVD indices uniformly at random (matching the
paper's binary-mask method). A floor control: the identified subspaces are
optimized to maximize IIA while random selections are not, so the identified
subspace beating random draws is expected.

The Lookback paper's actual method: SVD of residual stream activations
(500 components) -> learn binary mask over singular vectors via Adam + L1
-> round to {0,1}. The correct null matches this: random binary masks
over the same SVD basis, not arbitrary rotations on the Grassmannian.

Intervention protocol (from run_single_layer_patching_exps.py):
    P = V_selected.T @ V_selected   (d_model, d_model) projection matrix
    x_patched = x_org - (x_org @ P) + (x_alt @ P)
    IIA = fraction where argmax(logits_patched) == target

Uses NDIF for remote Llama 3.1-70B inference (no local GPU needed).

Usage:
    uv run python experiments/random_subspace_baseline.py --dry-run
    uv run python experiments/random_subspace_baseline.py --n-random 200
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
BELIEF_TRACKING = REPO_ROOT / "reference" / "belief_tracking"

sys.path.insert(0, str(BELIEF_TRACKING))
from src.dataset import Dataset as StoryDataset
from src.dataset import Sample


MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


def setup_nnsight():
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("HF_TOKEN", "")

    from nnsight import CONFIG, LanguageModel
    CONFIG.APP.REMOTE_LOGGING = False
    CONFIG.set_default_api_key(os.environ["NDIF_KEY"])

    return LanguageModel(MODEL)


def load_subspace_specs():
    specs_path = REPO_ROOT / "reference" / "extracted_results_llama70b.json"
    with open(specs_path) as f:
        data = json.load(f)
    return {k: v for k, v in data["experiment_scripts_should_use"].items() if not k.startswith("_")}


def mask_rng(seed, subspace, i):
    """A generator determined by (seed, subspace, index) rather than by call order.

    A single generator advanced per draw cannot be resumed: restarting reseeds it while the
    loop skips ahead, so the run redraws masks it already has. Deriving per index makes each
    mask reproducible on its own and makes resume exact.
    """
    tag = int.from_bytes(hashlib.sha256(subspace.encode()).digest()[:4], "big")
    return np.random.default_rng([seed, tag, i])


def sample_random_svd_mask(n_components, rank, rng):
    return np.sort(rng.choice(n_components, size=rank, replace=False))


def build_projection_matrix(svd_basis, selected_indices):
    """Build (d_model, d_model) projection matrix from selected SVD directions.

    Matches the paper: P = V_selected.T @ V_selected
    """
    V_sel = svd_basis[selected_indices]  # (rank, d_model)
    return V_sel.T @ V_sel  # (d_model, d_model)


def generate_counterfactual_pairs(n_samples, seed):
    """Generate clean/counterfactual prompt pairs using the paper's method.

    For answer_lookback-pointer: reversed sentences with different states.
    Uses get_reversed_sent_diff_state_counterfacts from the paper's code.
    """
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "characters.json") as f:
        characters = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "bottles.json") as f:
        bottles = json.load(f)
    with open(BELIEF_TRACKING / "data" / "synthetic_entities" / "drinks.json") as f:
        drinks = json.load(f)

    from notebooks.causalToM_novis.utils import (
        get_reversed_sent_diff_state_counterfacts,
        get_reversed_sentence_counterfacts,
    )

    random.seed(seed)
    np.random.seed(seed)

    answer_pairs = get_reversed_sent_diff_state_counterfacts(
        characters, bottles, drinks, n_samples
    )
    binding_pairs = get_reversed_sentence_counterfacts(
        characters, bottles, drinks, n_samples
    )

    return answer_pairs, binding_pairs


def filter_on_model(lm, pairs, max_size=80):
    """Keep only pairs where the model gets both clean and counterfactual correct.

    Uses two separate traces per sample (nnsight can't loop inside trace).
    """
    filtered = []
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

            if clean_tok == clean_target.lower().strip() and cf_tok == cf_target.lower().strip():
                filtered.append(sample)
                if len(filtered) >= max_size:
                    break
        except Exception as e:
            print(f"  Filter error: {e}")
            time.sleep(2)

    return filtered


def compute_iia_answer(lm, pairs, layer, projection, retries=3):
    """Compute IIA for answer_lookback via subspace interchange on NDIF.

    Intervention at the last token position only:
        x_patched = x_org - (x_org @ P) + (x_alt @ P)

    nnsight constraint: no loops inside trace, no double output access.
    """
    correct, total = 0, 0

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample.get("target", sample.get("counterfactual_ans", ""))

        for attempt in range(retries):
            try:
                with lm.trace(remote=True) as tracer:
                    with tracer.invoke(alt_prompt):
                        alt_last = lm.model.layers[layer].output[0][-1].clone()

                    with tracer.invoke(org_prompt):
                        curr = lm.model.layers[layer].output[0][-1].clone()
                        if projection is not None:
                            alt_proj = alt_last @ projection
                            org_proj = curr @ projection
                            patch = curr - org_proj + alt_proj
                        else:
                            patch = alt_last

                        lm.model.layers[layer].output[0][-1] = patch

                        logits = lm.lm_head.output[0, -1]
                        pred_id = logits.argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                is_correct = pred_tok == target.lower().strip()
                correct += int(is_correct)
                total += 1
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return correct / total if total > 0 else 0.0


def compute_iia_binding(lm, pairs, layer, projection, retries=3):
    """Compute IIA for binding_lookback via subspace interchange on NDIF.

    Intervention at state token positions [155, 156, 167, 168] with swap:
        cache=[155,156,167,168], patch=[167,168,155,156]
    Must save full layer output to access multiple positions (OutOfOrderError
    prevents multiple output[0] accesses).
    """
    correct, total = 0, 0

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample.get("target", sample.get("counterfactual_ans", ""))

        for attempt in range(retries):
            try:
                with lm.trace(remote=True) as tracer:
                    with tracer.invoke(alt_prompt):
                        alt_out = lm.model.layers[layer].output[0].save()

                    with tracer.invoke(org_prompt):
                        org_out = lm.model.layers[layer].output[0]

                        # Swap: patch[167] <- cache[155], patch[168] <- cache[156]
                        #        patch[155] <- cache[167], patch[156] <- cache[168]
                        if projection is not None:
                            for p_pos, c_pos in [(167, 155), (168, 156), (155, 167), (156, 168)]:
                                curr = org_out[p_pos].clone()
                                alt_proj = alt_out[c_pos] @ projection
                                org_proj = curr @ projection
                                org_out[p_pos] = curr - org_proj + alt_proj
                        else:
                            org_out[167] = alt_out[155]
                            org_out[168] = alt_out[156]
                            org_out[155] = alt_out[167]
                            org_out[156] = alt_out[168]

                        logits = lm.lm_head.output[0, -1]
                        pred_id = logits.argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                is_correct = pred_tok == target.lower().strip()
                correct += int(is_correct)
                total += 1
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Random subspace baseline")
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--n-eval-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "results" / "random_subspace_baseline.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run and "DRYRUN" not in str(args.output):
        sys.exit(f"refusing to write a dry run to {args.output}: put DRYRUN in the path. "
                 "A synthetic run once landed in results/ and was read as a measurement.")


    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)
    n_svd = 500

    print(f"[{ts()}] Setting up nnsight + NDIF...")
    lm = setup_nnsight()
    tokenizer = lm.tokenizer

    subspace_specs = load_subspace_specs()
    print(f"[{ts()}] Loaded {len(subspace_specs)} subspace specs")

    # Generate and filter evaluation data
    print(f"[{ts()}] Generating counterfactual pairs...")
    answer_pairs, binding_pairs = generate_counterfactual_pairs(
        n_samples=args.n_eval_samples * 3,  # oversample for filtering
        seed=args.seed,
    )

    if not args.dry_run:
        print(f"[{ts()}] Filtering answer pairs on model accuracy...")
        answer_pairs = filter_on_model(lm, answer_pairs, max_size=args.n_eval_samples)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

        print(f"[{ts()}] Filtering binding pairs on model accuracy...")
        binding_pairs = filter_on_model(lm, binding_pairs, max_size=args.n_eval_samples)
        print(f"[{ts()}] {len(binding_pairs)} binding pairs passed filter")

    # Load SVD bases
    svd_dir = REPO_ROOT / "results" / "svd" / "CausalToM"
    if not svd_dir.exists() and not args.dry_run:
        print(f"[{ts()}] ERROR: SVD bases not found at {svd_dir}")
        print("Run scripts/extract_svd.py first.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "model_note": "Llama 3.1 (paper used Llama 3.0)",
        "seed": args.seed,
        "n_random": args.n_random,
        "n_eval_samples": args.n_eval_samples,
        "dry_run": args.dry_run,
        "subspaces": {},
    }

    for name, spec in subspace_specs.items():
        layer = spec["layer"]
        rank = spec["rank"]
        lookback = spec["lookback_type"]
        concept = spec["concept"]

        print(f"\n{'='*60}")
        print(f"[{ts()}] {name}: layer={layer}, rank={rank}")
        print(f"  lookback={lookback}, concept={concept}")
        print(f"  Sampling {args.n_random} random {rank}-of-{n_svd} masks")
        print(f"{'='*60}")

        pairs = answer_pairs if "answer" in lookback else binding_pairs

        # Determine SVD vector type
        if "answer" in lookback:
            vec_type = "last_token"
        else:
            vec_type = "state_tokens"

        # Load SVD basis for this layer
        svd_path = svd_dir / vec_type / "singular_vecs" / f"{layer}.pt"
        if not args.dry_run:
            if not svd_path.exists():
                print(f"  SKIP: SVD basis not found at {svd_path}")
                continue
            svd_basis = torch.load(svd_path, weights_only=True).float()  # (n_comp, d_model)
            print(f"  SVD basis: {svd_basis.shape}")

        # Compute IIA with identified subspace (paper's result)
        identified_iia = spec["sv_iia"]

        # Pick the right IIA function based on experiment type
        if "answer" in lookback:
            compute_fn = compute_iia_answer
        else:
            compute_fn = compute_iia_binding

        # Compute IIA with random subspaces
        # Append-only shard, one line per mask. A rewritten file can be truncated by a
        # kill between open and write; an append cannot lose the lines already on disk.
        shard = output_path.parent / f"{name}.masks.jsonl"
        random_iias = []
        if shard.exists():
            for line in shard.read_text().splitlines():
                if line.strip():
                    random_iias.append(json.loads(line)["iia"])
            print(f"  resuming: {len(random_iias)} masks already on disk in {shard.name}")
        for i in tqdm(range(len(random_iias), args.n_random), desc="Random subspaces",
                      initial=len(random_iias), total=args.n_random):
            selected = sample_random_svd_mask(n_svd, rank, mask_rng(args.seed, name, i))

            if args.dry_run:
                iia = mask_rng(args.seed, name + ":iia", i).random()
            else:
                proj = build_projection_matrix(svd_basis, selected)
                iia = compute_fn(lm, pairs, layer, proj)

            with open(shard, "a") as fh:
                fh.write(json.dumps({
                    "i": i, "subspace": name, "layer": layer, "rank": rank,
                    "mask": [int(x) for x in selected], "iia": float(iia),
                    "dry_run": bool(args.dry_run),
                    "t": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
            random_iias.append(iia)

        random_arr = np.array(random_iias)
        rank_count = int(np.sum(random_arr >= identified_iia))
        p_value = (rank_count + 1) / (len(random_iias) + 1)

        results["subspaces"][name] = {
            "layer": layer,
            "rank": rank,
            "lookback_type": lookback,
            "concept": concept,
            "fraction_of_svd_basis": rank / n_svd,
            "identified_iia": identified_iia,
            "random_iia_mean": float(np.mean(random_arr)),
            "random_iia_std": float(np.std(random_arr, ddof=1)),
            "random_iia_median": float(np.median(random_arr)),
            "random_iia_max": float(np.max(random_arr)),
            "random_iia_min": float(np.min(random_arr)),
            "random_iia_p95": float(np.quantile(random_arr, 0.95)),
            "random_iia_p99": float(np.quantile(random_arr, 0.99)),
            "identified_rank": rank_count,
            "p_value_one_sided": p_value,
            "p_value_formula": "(rank + 1) / (n + 1)",
            "significant_at_001": p_value < 0.01,
        }

        print(f"  Random IIA: mean={np.mean(random_arr):.4f}, std={np.std(random_arr, ddof=1):.4f}")
        print(f"  Identified IIA: {identified_iia:.4f}")
        print(f"  Rank: {rank_count}/{args.n_random}, p={p_value:.6f}")
        print(f"  Significant at 0.01: {p_value < 0.01}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  checkpointed {len(results['subspaces'])} of {len(subspace_specs)} subspaces")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{ts()}] Results saved to {output_path}")


if __name__ == "__main__":
    main()
