"""
Cross-stage mediation test: does the binding representation causally
affect the answer pointer, or are they independent processes?

Replaces the original super-additivity unity test. Super-additivity is
non-diagnostic because independent stages in a serial pipeline can
produce super-additive loss (disrupting the upstream stage deprives
the downstream stage of its input, amplifying the effect).

Protocol:
  (a) Baseline: no ablation, measure answer-pointer IIA.
  (b) Intervene on binding subspace (layer 34), measure answer-pointer
      IIA at layer 52. If binding causally feeds the pointer,
      disrupting binding should degrade pointer IIA.
  (c) Intervene on answer subspace (layer 38), measure binding IIA
      at layer 34. If they are independent, disrupting answer should
      NOT degrade binding IIA (no backward causation).

Uses mean-ablation (replace subspace component with its dataset mean)
rather than zero-ablation. All conditions measure IIA via interchange
intervention (matching the other experiments' instrument).

The mediation test: if binding -> answer is causal, then
  binding_ablation should degrade answer_IIA (forward mediation)
  answer_ablation should NOT degrade binding_IIA (no backward effect)
Bootstrap CI on the mediation effect.

Uses NDIF for remote Llama 3.1-70B inference (no local GPU needed).

Usage:
    uv run python experiments/unity_coupling_test.py --dry-run
    uv run python experiments/unity_coupling_test.py --output results/cross_stage_mediation.json
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
    REPO_ROOT,
    build_projection_from_basis,
    build_projection_matrix,
    collect_activations_last_token,
    compute_iia_answer_per_sample,
    compute_iia_binding_per_sample,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

STAGES = {
    "binding": {
        "lookback_type": "binding_lookback",
        "concept": "address_and_payload",
        "layers": {34: 3, 35: 7, 36: 8},
    },
    "answer": {
        "lookback_type": "answer_lookback",
        "concept": "pointer",
        "layers": {38: 3, 52: 18, 53: 19},
    },
}

BINDING_ABLATION_LAYER = 34
BINDING_ABLATION_RANK = 3
ANSWER_ABLATION_LAYER = 38
ANSWER_ABLATION_RANK = 3
ANSWER_MEASURE_LAYER = 52
ANSWER_MEASURE_RANK = 18
BINDING_MEASURE_LAYER = 34
BINDING_MEASURE_RANK = 3


def compute_mean_projection(lm, pairs, basis, layer):
    """Compute the mean subspace projection of last-token activations.

    For each story's clean prompt, collects the last-token activation at
    `layer`, projects onto `basis`, and averages. Returns the mean
    projection in the original d_model space.

    Args:
        lm: nnsight LanguageModel
        pairs: list of dicts with clean_prompt
        basis: (rank, d_model), orthonormal rows
        layer: int

    Returns: (d_model,) tensor — the mean of (x @ basis.T @ basis) across stories
    """
    projections = []
    P = build_projection_from_basis(basis)

    for sample in tqdm(pairs, desc=f"Mean projection L{layer}"):
        act = collect_activations_last_token(lm, sample["clean_prompt"], layer)
        if act is not None:
            proj = act @ P  # (d_model,)
            projections.append(proj)

    if not projections:
        return torch.zeros(basis.shape[1])
    return torch.stack(projections).mean(dim=0)


def measure_iia_with_ablation(lm, pairs, ablation_layer, ablation_P,
                               mean_proj, measure_layer, measure_P,
                               retries=3):
    """Measure IIA at measure_layer while mean-ablating at ablation_layer.

    For each counterfactual pair, runs a single NDIF trace that:
      1. In the alt invocation: caches the last-token activation at measure_layer
      2. In the org invocation:
         a. Mean-ablates the subspace at ablation_layer (position [-1])
         b. Applies interchange intervention at measure_layer (position [-1])
         c. Reads the predicted token

    ablation_layer and measure_layer MUST be different (OutOfOrderError otherwise).

    All layer accesses are unrolled (no loops inside trace).

    Args:
        lm: nnsight LanguageModel
        pairs: counterfactual pairs
        ablation_layer: layer to mean-ablate
        ablation_P: (d_model, d_model) projection for ablation subspace
        mean_proj: (d_model,) mean projection to replace with
        measure_layer: layer to run interchange at
        measure_P: (d_model, d_model) projection for measurement subspace
        retries: retry count

    Returns: list of 0/1 per sample
    """
    results = []

    for sample in pairs:
        alt_prompt = sample["counterfactual_prompt"]
        org_prompt = sample["clean_prompt"]
        target = sample.get("target", sample.get("counterfactual_ans", ""))

        for attempt in range(retries):
            try:
                with lm.trace(remote=True) as tracer:
                    with tracer.invoke(alt_prompt):
                        alt_last = lm.model.layers[measure_layer].output[0][-1].clone()

                    with tracer.invoke(org_prompt):
                        # Step 1: mean-ablate at ablation_layer
                        abl_out = lm.model.layers[ablation_layer].output[0]
                        abl_curr = abl_out[-1].clone()
                        abl_subspace = abl_curr @ ablation_P
                        abl_out[-1] = abl_curr - abl_subspace + mean_proj

                        # Step 2: interchange at measure_layer
                        meas_curr = lm.model.layers[measure_layer].output[0][-1].clone()
                        meas_org_proj = meas_curr @ measure_P
                        meas_alt_proj = alt_last @ measure_P
                        lm.model.layers[measure_layer].output[0][-1] = (
                            meas_curr - meas_org_proj + meas_alt_proj
                        )

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


def bootstrap_mediation_effect(baseline_iia, ablated_iia, n_bootstrap=10000, rng=None):
    """Bootstrap CI on the mediation effect: baseline_IIA - ablated_IIA.

    A positive effect means ablation degraded IIA (forward mediation).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(baseline_iia)
    effects = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        effects.append(np.mean(baseline_iia[idx]) - np.mean(ablated_iia[idx]))

    effects = np.array(effects)
    point = float(np.mean(baseline_iia) - np.mean(ablated_iia))
    ci_lower = float(np.quantile(effects, 0.025))
    ci_upper = float(np.quantile(effects, 0.975))
    return point, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description="Cross-stage mediation test for Lookback audit")
    parser.add_argument("--output", type=str, default="results/cross_stage_mediation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-eval-samples", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    conditions = {
        "baseline": {
            "ablate": None,
            "measure": "answer",
            "description": "No ablation, measure answer-pointer IIA at L52",
        },
        "ablate_binding_measure_answer": {
            "ablate": "binding",
            "measure": "answer",
            "description": f"Mean-ablate binding (L{BINDING_ABLATION_LAYER}), measure answer-pointer IIA at L{ANSWER_MEASURE_LAYER}",
        },
        "ablate_answer_measure_binding": {
            "ablate": "answer",
            "measure": "binding",
            "description": f"Mean-ablate answer (L{ANSWER_ABLATION_LAYER}), measure binding IIA at L{BINDING_MEASURE_LAYER}",
        },
        "baseline_binding": {
            "ablate": None,
            "measure": "binding",
            "description": f"No ablation, measure binding IIA at L{BINDING_MEASURE_LAYER} (for backward comparison)",
        },
    }

    results = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "ablation_method": "mean-ablation (not zero-ablation)",
        "measurement_instrument": "interchange intervention (matching other experiments)",
        "representative_layers": {
            "binding_ablation": BINDING_ABLATION_LAYER,
            "binding_measure": BINDING_MEASURE_LAYER,
            "answer_ablation": ANSWER_ABLATION_LAYER,
            "answer_measure": ANSWER_MEASURE_LAYER,
        },
        "stages": {k: v for k, v in STAGES.items()},
        "hypothesis": {
            "forward_mediation": "Ablating binding degrades answer IIA (binding causally feeds answer)",
            "no_backward_effect": "Ablating answer does NOT degrade binding IIA (no reverse causation)",
        },
        "conditions": {},
    }

    per_story_iias = {}

    if args.dry_run:
        n_stories = 80
        for cond_name, cond_spec in conditions.items():
            print(f"\nCondition: {cond_name}")
            print(f"  {cond_spec['description']}")

            if cond_name == "baseline":
                stories_iia = rng.beta(8, 2, size=n_stories)
            elif cond_name == "ablate_binding_measure_answer":
                stories_iia = rng.beta(4, 3, size=n_stories)
            elif cond_name == "ablate_answer_measure_binding":
                stories_iia = rng.beta(7, 2, size=n_stories)
            else:
                stories_iia = rng.beta(8, 2, size=n_stories)

            per_story_iias[cond_name] = stories_iia
            mean_iia = float(np.mean(stories_iia))

            results["conditions"][cond_name] = {
                "description": cond_spec["description"],
                "ablated_stage": cond_spec["ablate"],
                "measured_stage": cond_spec["measure"],
                "iia_mean": mean_iia,
                "iia_std": float(np.std(stories_iia, ddof=1)),
            }
            print(f"  IIA: {mean_iia:.4f} +/- {float(np.std(stories_iia, ddof=1)):.4f}")

    else:
        print(f"[{ts()}] Setting up nnsight + NDIF...")
        lm = setup_nnsight()

        # Generate and filter counterfactual pairs
        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_pairs, binding_pairs = generate_counterfactual_pairs(
            n_samples=args.n_eval_samples * 3,
            seed=args.seed,
        )

        print(f"[{ts()}] Filtering answer pairs on model accuracy...")
        answer_pairs = filter_on_model(lm, answer_pairs, max_size=args.n_eval_samples)
        print(f"[{ts()}] {len(answer_pairs)} answer pairs passed filter")

        print(f"[{ts()}] Filtering binding pairs on model accuracy...")
        binding_pairs = filter_on_model(lm, binding_pairs, max_size=args.n_eval_samples)
        print(f"[{ts()}] {len(binding_pairs)} binding pairs passed filter")

        # Load SVD bases and build projections
        svd_dir = REPO_ROOT / "results" / "svd" / "CausalToM"

        binding_basis = load_svd_basis(BINDING_ABLATION_LAYER, "state_tokens", svd_dir)
        if binding_basis is None:
            print(f"ERROR: No SVD basis for binding L{BINDING_ABLATION_LAYER}. Run extract_svd.py first.")
            return
        binding_basis_selected = binding_basis[:BINDING_ABLATION_RANK]
        binding_P = build_projection_from_basis(binding_basis_selected)

        answer_basis = load_svd_basis(ANSWER_ABLATION_LAYER, "last_token", svd_dir)
        if answer_basis is None:
            print(f"ERROR: No SVD basis for answer L{ANSWER_ABLATION_LAYER}. Run extract_svd.py first.")
            return
        answer_basis_selected = answer_basis[:ANSWER_ABLATION_RANK]
        answer_P = build_projection_from_basis(answer_basis_selected)

        measure_answer_basis = load_svd_basis(ANSWER_MEASURE_LAYER, "last_token", svd_dir)
        if measure_answer_basis is None:
            print(f"ERROR: No SVD basis for answer measure L{ANSWER_MEASURE_LAYER}.")
            return
        measure_answer_P = build_projection_matrix(measure_answer_basis, np.arange(ANSWER_MEASURE_RANK))

        measure_binding_basis = load_svd_basis(BINDING_MEASURE_LAYER, "state_tokens", svd_dir)
        if measure_binding_basis is None:
            print(f"ERROR: No SVD basis for binding measure L{BINDING_MEASURE_LAYER}.")
            return
        measure_binding_P = build_projection_matrix(measure_binding_basis, np.arange(BINDING_MEASURE_RANK))

        # Compute mean projections for ablation
        print(f"[{ts()}] Computing mean projection for binding ablation (L{BINDING_ABLATION_LAYER})...")
        binding_mean = compute_mean_projection(lm, answer_pairs, binding_basis_selected, BINDING_ABLATION_LAYER)

        print(f"[{ts()}] Computing mean projection for answer ablation (L{ANSWER_ABLATION_LAYER})...")
        answer_mean = compute_mean_projection(lm, binding_pairs, answer_basis_selected, ANSWER_ABLATION_LAYER)

        # Condition 1: baseline (no ablation, measure answer IIA at L52)
        print(f"\n[{ts()}] Condition: baseline")
        baseline_iia = compute_iia_answer_per_sample(
            lm, answer_pairs, ANSWER_MEASURE_LAYER, measure_answer_P
        )
        per_story_iias["baseline"] = np.array(baseline_iia, dtype=float)
        results["conditions"]["baseline"] = {
            "description": conditions["baseline"]["description"],
            "ablated_stage": None,
            "measured_stage": "answer",
            "iia_mean": float(np.mean(baseline_iia)),
            "iia_std": float(np.std(baseline_iia, ddof=1)),
        }
        print(f"  IIA: {np.mean(baseline_iia):.4f} +/- {np.std(baseline_iia, ddof=1):.4f}")

        # Condition 2: ablate binding, measure answer
        print(f"\n[{ts()}] Condition: ablate_binding_measure_answer")
        abl_bind_iia = measure_iia_with_ablation(
            lm, answer_pairs,
            ablation_layer=BINDING_ABLATION_LAYER,
            ablation_P=binding_P,
            mean_proj=binding_mean,
            measure_layer=ANSWER_MEASURE_LAYER,
            measure_P=measure_answer_P,
        )
        per_story_iias["ablate_binding_measure_answer"] = np.array(abl_bind_iia, dtype=float)
        results["conditions"]["ablate_binding_measure_answer"] = {
            "description": conditions["ablate_binding_measure_answer"]["description"],
            "ablated_stage": "binding",
            "measured_stage": "answer",
            "iia_mean": float(np.mean(abl_bind_iia)),
            "iia_std": float(np.std(abl_bind_iia, ddof=1)),
        }
        print(f"  IIA: {np.mean(abl_bind_iia):.4f} +/- {np.std(abl_bind_iia, ddof=1):.4f}")

        # Condition 3: ablate answer, measure binding
        # NOTE: binding measurement uses binding pairs (not answer pairs)
        # and the ablation happens AFTER the measurement layer (L38 > L34),
        # so ablation cannot affect binding IIA (it's downstream). This is
        # the backward-causation sanity check.
        print(f"\n[{ts()}] Condition: ablate_answer_measure_binding")
        # Since L38 (ablation) > L34 (measurement), the ablation at L38
        # happens AFTER L34 in the forward pass and cannot affect L34's
        # output. We still run the full ablation+measurement trace to
        # confirm this empirically.
        abl_ans_iia = measure_iia_with_ablation(
            lm, binding_pairs,
            ablation_layer=ANSWER_ABLATION_LAYER,
            ablation_P=answer_P,
            mean_proj=answer_mean,
            measure_layer=BINDING_MEASURE_LAYER,
            measure_P=measure_binding_P,
        )
        per_story_iias["ablate_answer_measure_binding"] = np.array(abl_ans_iia, dtype=float)
        results["conditions"]["ablate_answer_measure_binding"] = {
            "description": conditions["ablate_answer_measure_binding"]["description"],
            "ablated_stage": "answer",
            "measured_stage": "binding",
            "iia_mean": float(np.mean(abl_ans_iia)),
            "iia_std": float(np.std(abl_ans_iia, ddof=1)),
        }
        print(f"  IIA: {np.mean(abl_ans_iia):.4f} +/- {np.std(abl_ans_iia, ddof=1):.4f}")

        # Condition 4: baseline binding (no ablation)
        print(f"\n[{ts()}] Condition: baseline_binding")
        baseline_bind_iia = compute_iia_binding_per_sample(
            lm, binding_pairs, BINDING_MEASURE_LAYER, measure_binding_P
        )
        per_story_iias["baseline_binding"] = np.array(baseline_bind_iia, dtype=float)
        results["conditions"]["baseline_binding"] = {
            "description": conditions["baseline_binding"]["description"],
            "ablated_stage": None,
            "measured_stage": "binding",
            "iia_mean": float(np.mean(baseline_bind_iia)),
            "iia_std": float(np.std(baseline_bind_iia, ddof=1)),
        }
        print(f"  IIA: {np.mean(baseline_bind_iia):.4f} +/- {np.std(baseline_bind_iia, ddof=1):.4f}")

    # Mediation analysis
    fwd_point, fwd_lo, fwd_hi = bootstrap_mediation_effect(
        per_story_iias["baseline"],
        per_story_iias["ablate_binding_measure_answer"],
        n_bootstrap=args.n_bootstrap, rng=rng,
    )

    bwd_point, bwd_lo, bwd_hi = bootstrap_mediation_effect(
        per_story_iias["baseline_binding"],
        per_story_iias["ablate_answer_measure_binding"],
        n_bootstrap=args.n_bootstrap, rng=rng,
    )

    fwd_significant = fwd_lo > 0
    bwd_significant = bwd_lo > 0

    if fwd_significant and not bwd_significant:
        verdict = "Forward mediation supported: binding causally feeds answer but not vice versa"
    elif fwd_significant and bwd_significant:
        verdict = "Bidirectional coupling: both stages affect each other (shared representation?)"
    elif not fwd_significant and not bwd_significant:
        verdict = "Independent processes: neither stage mediates the other"
    else:
        verdict = "Backward-only effect: answer affects binding (unexpected, check for confounds)"

    results["mediation_analysis"] = {
        "forward_effect": {
            "description": "Binding ablation effect on answer IIA",
            "point_estimate": fwd_point,
            "ci_95_lower": fwd_lo,
            "ci_95_upper": fwd_hi,
            "significant": fwd_significant,
        },
        "backward_effect": {
            "description": "Answer ablation effect on binding IIA",
            "point_estimate": bwd_point,
            "ci_95_lower": bwd_lo,
            "ci_95_upper": bwd_hi,
            "significant": bwd_significant,
        },
        "verdict": verdict,
    }

    print(f"\n{'='*60}")
    print(f"Forward mediation (binding -> answer):")
    print(f"  Effect: {fwd_point:.4f}, 95% CI [{fwd_lo:.4f}, {fwd_hi:.4f}]")
    print(f"  Significant: {fwd_significant}")
    print(f"Backward effect (answer -> binding):")
    print(f"  Effect: {bwd_point:.4f}, 95% CI [{bwd_lo:.4f}, {bwd_hi:.4f}]")
    print(f"  Significant: {bwd_significant}")
    print(f"Verdict: {verdict}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
