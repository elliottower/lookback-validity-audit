"""
Convergent validity (C3): Do multiple independent methods agree?

Trains a linear probe on the same layers identified by DAS/IIA, and compares
probe accuracy ranking to IIA ranking. If both methods identify the same layers
as important, that's convergent validity from two independent evidence families.

Method:
  1. Collect last-token activations at each target layer + control layers via NDIF
  2. Train logistic regression on each layer's activations to predict the answer
  3. Compare probe accuracy ranking vs known IIA ranking (from positive control)
  4. Report Spearman correlation between the two rankings

Usage:
    uv run python experiments/convergent_validity_probe.py --dry-run
    uv run python experiments/convergent_validity_probe.py
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from ndif_utils import (
    REPO_ROOT,
    collect_activations_last_token,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    setup_nnsight,
)

RESULTS_DIR = REPO_ROOT / "results" / "convergent_validity"

TARGET_LAYERS = [34, 35, 36, 38, 52, 53]
CONTROL_LAYERS = [5, 15, 25, 45, 65]

IIA_RANKING = {
    34: 0.988,
    35: 0.650,
    36: 0.625,
    38: 1.000,
    52: 1.000,
    53: 1.000,
}


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def collect_activations_for_layer(lm, pairs, layer, retries=3):
    """Collect last-token activations for all pairs at one layer."""
    activations = []
    labels = []

    for sample in tqdm(pairs, desc=f"L{layer} activations"):
        act = collect_activations_last_token(lm, sample["clean_prompt"], layer, retries=retries)
        if act is not None:
            activations.append(act.numpy())
            labels.append(sample["clean_ans"])

    return np.stack(activations), labels


def train_probe(activations, labels):
    """Train logistic regression and return cross-validated accuracy."""
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])

    if len(unique_labels) < 2:
        return 0.0, 0

    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        return 0.0, len(unique_labels)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, activations, y, cv=n_folds, scoring="accuracy")
    return float(np.mean(scores)), len(unique_labels)


def main():
    parser = argparse.ArgumentParser(description="Convergent validity: linear probe vs DAS/IIA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_layers = sorted(set(TARGET_LAYERS + CONTROL_LAYERS))

    print(f"[{ts()}] Convergent validity probe (C3)")
    print(f"[{ts()}] Target layers: {TARGET_LAYERS}")
    print(f"[{ts()}] Control layers: {CONTROL_LAYERS}")
    print(f"[{ts()}] Seed: {args.seed}, N eval: {args.n_eval}")

    lm = None
    pairs = None
    if not args.dry_run:
        print(f"[{ts()}] Generating counterfactual pairs...")
        answer_raw, _ = generate_counterfactual_pairs(
            n_samples=args.n_eval * 3, seed=args.seed
        )
        lm = setup_nnsight()

        print(f"[{ts()}] Filtering on model accuracy...")
        pairs = filter_on_model(lm, answer_raw, max_size=args.n_eval)
        print(f"[{ts()}] {len(pairs)} pairs passed filter")

    layer_results = {}

    for layer in all_layers:
        print(f"\n{'='*60}")
        print(f"[{ts()}] Layer {layer}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f"layer_{layer}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                cached = json.load(f)
            if cached.get("dry_run", False) == args.dry_run:
                layer_results[layer] = cached
                print(f"  Resumed: probe_accuracy={cached['probe_accuracy']:.4f}")
                continue

        if args.dry_run:
            if layer in TARGET_LAYERS:
                probe_acc = 0.5 + rng.uniform(0.1, 0.4)
            else:
                probe_acc = 0.5 + rng.uniform(-0.05, 0.1)
            n_classes = 10
            n_samples = args.n_eval
        else:
            activations, labels = collect_activations_for_layer(lm, pairs, layer)
            n_samples = len(labels)
            probe_acc, n_classes = train_probe(activations, labels)

        entry = {
            "layer": layer,
            "is_target": layer in TARGET_LAYERS,
            "probe_accuracy": float(probe_acc),
            "n_classes": n_classes,
            "n_samples": n_samples,
            "iia": IIA_RANKING.get(layer),
            "dry_run": args.dry_run,
        }
        layer_results[layer] = entry

        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"  Probe accuracy: {probe_acc:.4f}")
        if layer in IIA_RANKING:
            print(f"  Known IIA:      {IIA_RANKING[layer]:.4f}")

    target_probe = [layer_results[l]["probe_accuracy"] for l in TARGET_LAYERS]
    target_iia = [IIA_RANKING[l] for l in TARGET_LAYERS]

    from scipy.stats import spearmanr
    rho, pval = spearmanr(target_probe, target_iia)

    control_probe = [layer_results[l]["probe_accuracy"] for l in CONTROL_LAYERS]
    target_mean = float(np.mean(target_probe))
    control_mean = float(np.mean(control_probe))

    summary = {
        "experiment": "Convergent validity: linear probe vs DAS/IIA (C3)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dry_run": args.dry_run,
        "n_eval": args.n_eval,
        "target_layers": TARGET_LAYERS,
        "control_layers": CONTROL_LAYERS,
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "target_probe_mean": target_mean,
        "control_probe_mean": control_mean,
        "separation": target_mean - control_mean,
        "convergent_validity": rho > 0.5 and target_mean > control_mean + 0.05,
        "layer_results": {str(l): r for l, r in layer_results.items()},
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("CONVERGENT VALIDITY SUMMARY")
    print(f"{'='*60}")
    print(f"  Spearman rho (probe vs IIA): {rho:.4f} (p={pval:.4f})")
    print(f"  Target layers mean probe:    {target_mean:.4f}")
    print(f"  Control layers mean probe:   {control_mean:.4f}")
    print(f"  Separation:                  {target_mean - control_mean:.4f}")
    print(f"  Convergent validity:         {'YES' if summary['convergent_validity'] else 'NO'}")
    print(f"\n[{ts()}] Saved to {summary_path}")


if __name__ == "__main__":
    main()
