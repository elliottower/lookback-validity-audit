"""
Random subspace baseline for the Lookback paper's identified subspaces.

Tests whether random subspaces of matched rank achieve comparable IIA.
A floor control: if random subspaces match the identified subspace, the
mask-selection procedure is uninformative. See shuffled_label_control.py
for the matched-capacity trained control that tests belief-specificity.

The Lookback paper's actual method: SVD of residual stream activations
(500 components) → learn binary mask over singular vectors via Adam + L1
→ round to {0,1}. NOT DCM (Distributed Causal Model) optimization.

Requires: nnsight, torch, numpy, tqdm
Model: Llama-3-70B-Instruct (meta-llama/Llama-3-70b-Instruct)

Cost estimate (n=200, 6 per-layer subspaces, 80 examples):
    200 * 6 * 80 = 96,000 forward passes on 70B.
    At ~0.5s/pass on A100-80GB: ~13 GPU-hours.

Usage:
    python random_subspace_baseline.py --n-random 200 --output results/random_subspace.json
    python random_subspace_baseline.py --n-random 5 --dry-run --output results/DRYRUN_random_subspace.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def sample_random_subspace(d_model: int, d_sub: int, dtype: torch.dtype = torch.bfloat16,
                           device: str = "cpu") -> torch.Tensor:
    """Sample a random d_sub-dimensional subspace of R^d_model uniformly from the Grassmannian.

    Returns an orthonormal basis (d_sub, d_model) via QR decomposition of a random Gaussian matrix.
    Uses float64 for QR numerical stability, then casts to target dtype.
    """
    G = torch.randn(d_model, d_sub, dtype=torch.float64)
    Q, _ = torch.linalg.qr(G)
    basis = Q[:, :d_sub].T  # (d_sub, d_model)
    return basis.to(dtype=dtype, device=device)


def project_to_subspace(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project x onto the subspace spanned by basis rows.

    Args:
        x: (..., d_model)
        basis: (d_sub, d_model), orthonormal rows

    Returns:
        projected x: (..., d_model)
    """
    coeffs = x @ basis.T  # (..., d_sub)
    return coeffs @ basis  # (..., d_model)


def compute_iia_with_subspace(
    model,
    base_inputs,
    source_inputs,
    layer: int,
    token_pos: int,
    basis: torch.Tensor,
    target_token_id: int,
):
    """Compute IIA using interchange intervention restricted to a subspace.

    Swaps only the component of the residual stream that lies in the given subspace.
    basis must match model dtype (bf16) and device (cuda).

    Returns: float, fraction of examples where the model's top-1 prediction matches
    the counterfactual target.
    """
    raise NotImplementedError(
        "Implementation requires NNsight hooks specific to the CausalToM setup. "
        "See the original paper's code for the intervention protocol."
    )


def main():
    parser = argparse.ArgumentParser(description="Random subspace baseline for Lookback subspace identification")
    parser.add_argument("--n-random", type=int, default=200,
                        help="Number of random subspaces per identified subspace (200 gives p-floor=0.005)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="results/random_subspace.json")
    parser.add_argument("--dry-run", action="store_true", help="Test pipeline with synthetic data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Per-layer subspace ranks from the Lookback paper's SVD + mask results.
    # Each layer has its own rank (number of selected singular vectors).
    # These are placeholders — extract actual ranks from their released results/.json files.
    # Source: https://github.com/Nix07/belief_tracking/results/
    subspaces_per_layer = {
        "answer_pointer_L52": {"layer": 52, "rank": 18, "tokens": "final"},
        "answer_pointer_L53": {"layer": 53, "rank": 10, "tokens": "final"},
        "answer_pointer_L54": {"layer": 54, "rank": 8, "tokens": "final"},
        "binding_L35": {"layer": 35, "rank": 7, "tokens": "state"},
        "binding_L36": {"layer": 36, "rank": 5, "tokens": "state"},
        "binding_L38": {"layer": 38, "rank": 3, "tokens": "state"},
    }
    n_svd_components = 500  # SVD truncation used by the paper
    d_model = 8192  # Llama-3-70B

    results = {
        "synthetic": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_random": args.n_random,
        "d_model": d_model,
        "subspaces": {},
    }

    for name, spec in subspaces_per_layer.items():
        d_sub = spec["rank"]
        print(f"\n{'='*60}")
        print(f"Testing {name}: rank={d_sub}, layer={spec['layer']}")
        print(f"Sampling {args.n_random} random {d_sub}-dim subspaces from {n_svd_components}-component SVD basis")
        print(f"{'='*60}")

        random_iias = []
        for i in tqdm(range(args.n_random), desc=f"Random subspaces for {name}"):
            # Sample from the 500-component SVD basis, not from full R^d_model.
            # In dry-run mode, sample from R^d_model as a placeholder.
            basis = sample_random_subspace(
                n_svd_components if not args.dry_run else d_model,
                d_sub,
                dtype=torch.bfloat16 if not args.dry_run else torch.float32,
                device=args.device if not args.dry_run else "cpu",
            )

            if args.dry_run:
                iia = torch.rand(1).item()
            else:
                raise NotImplementedError(
                    "Full IIA computation requires model loading and CausalToM dataset. "
                    "See compute_iia_with_subspace()."
                )

            random_iias.append(iia)

        random_iias_arr = np.array(random_iias)

        # TODO: extract from released results at
        # https://github.com/Nix07/belief_tracking/results/{dataset}/{model}/{lookback_type}/...
        identified_iia = None
        if identified_iia is not None:
            rank = int(np.sum(random_iias_arr >= identified_iia))
            p_value = (rank + 1) / (len(random_iias) + 1)
        else:
            rank = None
            p_value = None

        results["subspaces"][name] = {
            "rank": d_sub,
            "layer": spec["layer"],
            "fraction_of_svd_basis": d_sub / n_svd_components,
            "identified_iia": identified_iia,
            "random_iia_mean": float(np.mean(random_iias_arr)),
            "random_iia_std": float(np.std(random_iias_arr, ddof=1)),
            "random_iia_median": float(np.median(random_iias_arr)),
            "random_iia_max": float(np.max(random_iias_arr)),
            "random_iia_min": float(np.min(random_iias_arr)),
            "random_iia_p95": float(np.quantile(random_iias_arr, 0.95)),
            "random_iia_p99": float(np.quantile(random_iias_arr, 0.99)),
            "identified_rank": rank,
            "p_value_one_sided": p_value,
            "p_value_formula": "(rank + 1) / (n + 1)",
            "significant_at_001": p_value < 0.01 if p_value is not None else None,
        }

        print(f"Random IIA: mean={results['subspaces'][name]['random_iia_mean']:.4f}, "
              f"std={results['subspaces'][name]['random_iia_std']:.4f}, "
              f"max={results['subspaces'][name]['random_iia_max']:.4f}")
        if identified_iia is not None:
            print(f"Identified IIA: {identified_iia:.4f}, rank: {rank}/{args.n_random}, p={p_value:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
