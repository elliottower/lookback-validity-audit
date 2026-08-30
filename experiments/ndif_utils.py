"""
Shared NDIF utilities for all Lookback validity audit experiments.

Centralizes nnsight setup, subspace loading, projection building, IIA
computation, and counterfactual pair generation so each experiment imports
from here rather than duplicating the tricky NDIF trace patterns.

CRITICAL nnsight 0.7.0 + NDIF constraints:
  1. No Python loops inside lm.trace() contexts (graph builder fails silently)
  2. Cannot access the same layer's output[0] twice (OutOfOrderError)
  3. All layer accesses must be unrolled as explicit statements
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BELIEF_TRACKING = REPO_ROOT / "reference" / "belief_tracking"

sys.path.insert(0, str(BELIEF_TRACKING))

MODEL = "meta-llama/Llama-3.1-70B-Instruct"


def setup_nnsight():
    """Load nnsight LanguageModel with NDIF remote config."""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("HF_TOKEN", "")

    from nnsight import CONFIG, LanguageModel
    CONFIG.APP.REMOTE_LOGGING = False
    key = os.environ.get("NDIF_KEY") or os.environ.get("NDIF_API_KEY")
    if not key:
        raise RuntimeError(
            "no NDIF key in the environment: set NDIF_KEY or NDIF_API_KEY. "
            "The 1Password environment exports it as NDIF_API_KEY."
        )
    CONFIG.set_default_api_key(key)

    from nnsight.intervention.backends.remote import RemoteBackend
    RemoteBackend.CONNECT_TIMEOUT = 30.0
    RemoteBackend.READ_TIMEOUT = 120.0

    return LanguageModel(MODEL)


def load_subspace_specs():
    """Load the 6 pre-registered subspace specs from extracted results."""
    specs_path = REPO_ROOT / "reference" / "extracted_results_llama70b.json"
    with open(specs_path) as f:
        data = json.load(f)
    return {k: v for k, v in data["experiment_scripts_should_use"].items()
            if not k.startswith("_")}


def load_svd_basis(layer, vec_type, svd_dir=None):
    """Load SVD basis for a specific layer.

    Args:
        layer: layer number
        vec_type: "last_token" or "state_tokens"
        svd_dir: path to SVD results (default: results/svd/CausalToM)

    Returns: (n_components, d_model) tensor, or None if not found
    """
    if svd_dir is None:
        svd_dir = REPO_ROOT / "results" / "svd" / "CausalToM"
    path = svd_dir / vec_type / "singular_vecs" / f"{layer}.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=True).float()


def build_projection_matrix(svd_basis, selected_indices):
    """Build (d_model, d_model) projection from selected SVD directions.

    Matches the paper: P = V_selected.T @ V_selected
    """
    V_sel = svd_basis[selected_indices]  # (rank, d_model)
    return V_sel.T @ V_sel  # (d_model, d_model)


def build_projection_from_basis(basis):
    """Build (d_model, d_model) projection from a (rank, d_model) basis directly.

    P = V.T @ V — assumes rows of basis are the selected directions.
    """
    return basis.T @ basis


def generate_counterfactual_pairs(n_samples, seed):
    """Generate counterfactual prompt pairs using the paper's functions.

    Returns: (answer_pairs, binding_pairs) where each is a list of dicts
    with keys: clean_prompt, counterfactual_prompt, clean_ans, counterfactual_ans
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
    """Keep only pairs where model gets both clean and counterfactual correct.

    Uses two separate traces per sample (nnsight can't loop inside trace).
    """
    from tqdm import tqdm

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


def _build_context_swapped_output(cl_t, cf_t, projection, intervention_positions):
    """Build patched output: full swap at context, subspace proj at targets.

    On NDIF, single-position intervention produces out-of-distribution
    outputs because the model needs consistent context across positions.
    This helper full-swaps positions where clean/CF differ (context) and
    applies the subspace projection only at the target positions.

    Args:
        cl_t: (seq, d_model) clean activations
        cf_t: (seq, d_model) counterfactual activations
        projection: (d_model, d_model) or None for full swap
        intervention_positions: list of (target_pos, source_pos) tuples.
            For answer: [(-1, -1)]. For binding: [(167,155),(168,156),...]

    Returns: (seq, d_model) patched output tensor
    """
    delta_norm = (cf_t - cl_t).norm(dim=-1)
    patched = cl_t.clone()

    target_set = {p[0] % cl_t.shape[0] for p in intervention_positions}

    for t in range(cl_t.shape[0]):
        if t in target_set:
            continue
        if delta_norm[t] > 0.01:
            patched[t] = cf_t[t]

    for tgt, src in intervention_positions:
        if projection is not None:
            x = cl_t[tgt]
            patched[tgt] = x - (x @ projection) + (cf_t[src] @ projection)
        else:
            patched[tgt] = cf_t[src]

    return patched


def compute_iia_answer(lm, pairs, layer, projection, retries=3):
    """Compute IIA for answer_lookback via subspace interchange on NDIF.

    Uses three-trace context-swap protocol (required on NDIF because
    cross-invoke variable references fail with NameError):
      1. Get CF full output at layer
      2. Get clean full output at layer
      3. Build patched: context swap + subspace proj at position -1

    Args:
        lm: nnsight LanguageModel
        pairs: list of dicts with clean_prompt, counterfactual_prompt, counterfactual_ans
        layer: int, layer to intervene at
        projection: (d_model, d_model) projection matrix, or None for full swap
        retries: retry count for NDIF errors

    Returns: IIA as float in [0, 1]
    """
    correct, total = 0, 0

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample["counterfactual_ans"]

        for attempt in range(retries):
            try:
                with lm.trace(alt_prompt, remote=True):
                    cf_out = lm.model.layers[layer].output[0].save()

                with lm.trace(org_prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cf_t = cf_out.detach().cpu().float()
                cl_t = cl_out.detach().cpu().float()

                patched = _build_context_swapped_output(
                    cl_t, cf_t, projection,
                    intervention_positions=[(-1, -1)],
                )

                with lm.trace(org_prompt, remote=True):
                    lm.model.layers[layer].output[0] = patched
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                is_correct = pred_tok == target.lower().strip()
                correct += int(is_correct)
                total += 1
                break

            except Exception as e:
                print(f"  IIA answer trace error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return correct / total if total > 0 else 0.0


def compute_iia_answer_flex(lm, pairs, layer, projection, retries=3):
    """Like compute_iia_answer but handles different-length clean/CF prompts.

    For same-length pairs: uses full context-swap protocol.
    For different-length pairs: position-only intervention at -1 (no context swap).
    """
    correct, total = 0, 0

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample["counterfactual_ans"]

        for attempt in range(retries):
            try:
                with lm.trace(alt_prompt, remote=True):
                    cf_out = lm.model.layers[layer].output[0].save()

                with lm.trace(org_prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cf_t = cf_out.detach().cpu().float()
                cl_t = cl_out.detach().cpu().float()

                if cf_t.shape[0] == cl_t.shape[0]:
                    patched = _build_context_swapped_output(
                        cl_t, cf_t, projection,
                        intervention_positions=[(-1, -1)],
                    )
                else:
                    patched = cl_t.clone()
                    if projection is not None:
                        x = cl_t[-1]
                        patched[-1] = x - (x @ projection) + (cf_t[-1] @ projection)
                    else:
                        patched[-1] = cf_t[-1]

                with lm.trace(org_prompt, remote=True):
                    lm.model.layers[layer].output[0] = patched
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                correct += int(pred_tok == target.lower().strip())
                total += 1
                break

            except Exception as e:
                print(f"  IIA flex trace error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return correct / total if total > 0 else 0.0


def compute_iia_answer_per_sample(lm, pairs, layer, projection, retries=3):
    """Like compute_iia_answer but returns per-sample binary results."""
    results = []

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample["counterfactual_ans"]

        for attempt in range(retries):
            try:
                with lm.trace(alt_prompt, remote=True):
                    cf_out = lm.model.layers[layer].output[0].save()

                with lm.trace(org_prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cf_t = cf_out.detach().cpu().float()
                cl_t = cl_out.detach().cpu().float()

                patched = _build_context_swapped_output(
                    cl_t, cf_t, projection,
                    intervention_positions=[(-1, -1)],
                )

                with lm.trace(org_prompt, remote=True):
                    lm.model.layers[layer].output[0] = patched
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                results.append(int(pred_tok == target.lower().strip()))
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    results.append(0)

    return results


BINDING_POSITIONS = [(167, 155), (168, 156), (155, 167), (156, 168)]


def compute_iia_binding(lm, pairs, layer, projection, retries=3):
    """Compute IIA for binding_lookback via subspace interchange on NDIF.

    Intervention at state token positions with cross-position swap:
        patch[167]=cf[155], patch[168]=cf[156], patch[155]=cf[167], patch[156]=cf[168]

    Uses three-trace context-swap protocol (same NDIF constraint as answer).

    NOTE: binding pairs have counterfactual_ans == clean_ans (both prompts
    produce the same answer). The intervention TARGET is the 'target' field
    which is the OTHER drink (what the model should predict after swapping
    the sentence-position bindings).
    """
    correct, total = 0, 0

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample["target"]

        for attempt in range(retries):
            try:
                with lm.trace(alt_prompt, remote=True):
                    cf_out = lm.model.layers[layer].output[0].save()

                with lm.trace(org_prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cf_t = cf_out.detach().cpu().float()
                cl_t = cl_out.detach().cpu().float()

                patched = _build_context_swapped_output(
                    cl_t, cf_t, projection,
                    intervention_positions=BINDING_POSITIONS,
                )

                with lm.trace(org_prompt, remote=True):
                    lm.model.layers[layer].output[0] = patched
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                is_correct = pred_tok == target.lower().strip()
                correct += int(is_correct)
                total += 1
                break

            except Exception as e:
                print(f"  IIA binding trace error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1

    return correct / total if total > 0 else 0.0


def compute_iia_binding_per_sample(lm, pairs, layer, projection, retries=3):
    """Like compute_iia_binding but returns per-sample binary results."""
    results = []

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample["target"]

        for attempt in range(retries):
            try:
                with lm.trace(alt_prompt, remote=True):
                    cf_out = lm.model.layers[layer].output[0].save()

                with lm.trace(org_prompt, remote=True):
                    cl_out = lm.model.layers[layer].output[0].save()

                cf_t = cf_out.detach().cpu().float()
                cl_t = cl_out.detach().cpu().float()

                patched = _build_context_swapped_output(
                    cl_t, cf_t, projection,
                    intervention_positions=BINDING_POSITIONS,
                )

                with lm.trace(org_prompt, remote=True):
                    lm.model.layers[layer].output[0] = patched
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                results.append(int(pred_tok == target.lower().strip()))
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    results.append(0)

    return results


def get_model_prediction(lm, prompt, retries=3):
    """Get model's argmax prediction for a single prompt via NDIF.

    Returns: (predicted_token_str, predicted_token_id) or (None, None) on failure
    """
    for attempt in range(retries):
        try:
            with lm.trace(prompt, remote=True):
                pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()

            tok = lm.tokenizer.decode([pred_id.item()]).lower().strip()
            return tok, pred_id.item()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None, None


def collect_activations_last_token(lm, prompt, layer, retries=3):
    """Collect last-token activation at a single layer via NDIF.

    Returns: (d_model,) tensor or None on failure
    """
    for attempt in range(retries):
        try:
            with lm.trace(prompt, remote=True):
                act = lm.model.layers[layer].output[0][-1].save()
            return act.detach().cpu().float()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds. With z=1.96, this is a 95% CI.
    Returns (0.0, 1.0) if n=0.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


MIN_N_FOR_INTERPRETATION = 50


def iia_with_ci(k, n, z=1.96):
    """Compute IIA point estimate with Wilson CI and interpretability flag.

    Returns dict with: iia, n, ci_lower, ci_upper, interpretable.
    """
    iia = k / n if n > 0 else 0.0
    lo, hi = wilson_ci(k, n, z)
    return {
        "iia": iia,
        "n": n,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_width": hi - lo,
        "interpretable": n >= MIN_N_FOR_INTERPRETATION,
    }
