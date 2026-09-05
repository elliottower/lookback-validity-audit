"""E17: joint ablation of a subspace group, against orthogonal-complement controls.

Registered in AMENDMENTS.md Amendment 12 (tag prereg-amendment-8), before any results.

Single-subspace zero-ablation costs 0.00-0.12 accuracy while the same subspaces carry
belief content under interchange at IIA near 1.0. Four readings are live: a dormant pathway
recruited by the intervention, an off-distribution artifact of zero-ablation, genuine
dispensability, and redundancy across a parallel group where no member is necessary and each
is sufficient. This tests the last one.

Protocol:
  1. Ablate a whole group at once - binding (L34, L35, L36) or answer (L38, L52, L53).
  2. Two ablation types: zero (x - xP) and mean (x - xP + mean_xP), the second staying on
     the data manifold.
  3. Controls draw random subspaces of the same total rank from the ORTHOGONAL COMPLEMENT
     of the identified directions in the SVD basis, so a control cannot overlap what it
     controls for. 200 draws.
  4. Report the observed drop as an exact quantile of the control distribution, not against
     a threshold.

Unlike the single-subspace test this intervenes live in one trace rather than capturing
clean activations first: under joint ablation each layer must see the effect of the ablation
applied below it, which a capture-then-inject pass cannot reproduce.

Usage:
    uv run python experiments/joint_ablation_test.py --dry-run
    uv run python experiments/joint_ablation_test.py --group answer --n-draws 200
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
    BINDING_POSITIONS,
    build_projection_matrix,
    filter_on_model,
    generate_counterfactual_pairs,
    load_subspace_specs,
    load_svd_basis,
    setup_nnsight,
)

GROUPS = {
    "binding": ["binding_addr_payload_L34", "binding_addr_payload_L35",
                "binding_addr_payload_L36"],
    "answer": ["answer_pointer_L38", "answer_pointer_L52", "answer_pointer_L53"],
}


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def complement_indices(n_components, used):
    """SVD rows are orthonormal, so any row outside `used` is orthogonal to the subspace."""
    return sorted(set(range(n_components)) - set(int(i) for i in used))


def draw_control(rng, n_components, used, rank):
    pool = complement_indices(n_components, used)
    if len(pool) < rank:
        raise RuntimeError(f"complement holds {len(pool)} directions, need {rank}")
    return list(rng.choice(pool, size=rank, replace=False))


def subspace_means(lm, pairs, layers, projections, is_binding, retries=3):
    """Dataset mean of the projected component per layer, for mean-ablation."""
    sums = {L: None for L in layers}
    counts = {L: 0 for L in layers}
    for sample in tqdm(pairs, desc="means", leave=False):
        for attempt in range(retries):
            try:
                with lm.trace(sample["clean_prompt"], remote=True):
                    outs = {L: lm.model.layers[L].output[0].save() for L in layers}
                for L in layers:
                    t = outs[L].detach().cpu().float()
                    positions = ([p % t.shape[0] for p, _ in BINDING_POSITIONS]
                                 if is_binding else [t.shape[0] - 1])
                    for pos in positions:
                        comp = t[pos] @ projections[L]
                        sums[L] = comp if sums[L] is None else sums[L] + comp
                        counts[L] += 1
                break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
    return {L: (sums[L] / counts[L] if counts[L] else None) for L in layers}


def joint_ablated_accuracy(lm, pairs, layers, projections, is_binding,
                           means=None, retries=3):
    """Ablate every layer in the group within a single forward pass."""
    correct, total = 0, 0
    for sample in tqdm(pairs, desc="joint", leave=False):
        target = sample.get("target", sample.get("counterfactual_ans", "")).lower().strip()
        for attempt in range(retries):
            try:
                with lm.trace(sample["clean_prompt"], remote=True):
                    for L in layers:
                        h = lm.model.layers[L].output[0]
                        P = projections[L]
                        n = h.shape[0]
                        positions = ([p % n for p, _ in BINDING_POSITIONS]
                                     if is_binding else [-1])
                        for pos in positions:
                            comp = h[pos] @ P
                            h[pos] = h[pos] - comp
                            if means is not None and means[L] is not None:
                                h[pos] = h[pos] + means[L]
                    pred_id = lm.lm_head.output[0, -1].argmax(dim=-1).save()
                pred = lm.tokenizer.decode([pred_id.item()]).lower().strip()
                correct += int(pred == target)
                total += 1
                break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    total += 1
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description="E17 joint ablation")
    ap.add_argument("--group", choices=sorted(GROUPS), default="answer")
    ap.add_argument("--n-samples", type=int, default=160)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/joint_ablation/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.group}.json"

    specs = load_subspace_specs()
    names = GROUPS[args.group]
    is_binding = args.group == "binding"
    layers = [specs[n]["layer"] for n in names]

    record = {
        "experiment": "E17 joint ablation",
        "registered": "AMENDMENTS.md Amendment 12, tag prereg-amendment-8",
        "group": args.group, "layers": layers, "dry_run": args.dry_run,
        "seed": args.seed, "n_draws": args.n_draws,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "control": "random subspaces from the orthogonal complement, matched total rank",
        "note": "live single-trace intervention; each layer sees the ablation applied below it",
    }

    def checkpoint():
        out_path.write_text(json.dumps(record, indent=2))

    checkpoint()

    if args.dry_run:
        record["dry_run_only"] = True
        checkpoint()
        print(f"[{ts()}] dry run; wrote {out_path}")
        return

    print(f"[{ts()}] setting up nnsight")
    lm = setup_nnsight()
    answer_pairs, binding_pairs = generate_counterfactual_pairs(args.n_samples, args.seed)
    pairs = filter_on_model(lm, binding_pairs if is_binding else answer_pairs, max_size=80)
    record["n_pairs"] = len(pairs); checkpoint()
    print(f"[{ts()}] {len(pairs)} pairs after filtering")

    bases, projections, used = {}, {}, {}
    for n in names:
        spec = specs[n]; L = spec["layer"]
        vec_type = "state_tokens" if is_binding else "last_token"
        basis = load_svd_basis(L, vec_type)
        if basis is None:
            raise SystemExit(f"no SVD basis for L{L} ({vec_type})")
        idx = spec.get("mask_indices") or list(range(spec["rank"]))
        bases[L] = basis; used[L] = idx
        projections[L] = build_projection_matrix(basis, idx)
    total_rank = sum(len(v) for v in used.values())
    record["total_rank"] = total_rank; checkpoint()

    clean = joint_ablated_accuracy(lm, pairs, layers,
                                   {L: torch.zeros_like(projections[L]) for L in layers},
                                   is_binding)
    record["clean_accuracy"] = clean; checkpoint()
    print(f"[{ts()}] clean accuracy {clean:.4f}")

    means = subspace_means(lm, pairs, layers, projections, is_binding)

    for kind, mm in (("zero", None), ("mean", means)):
        acc = joint_ablated_accuracy(lm, pairs, layers, projections, is_binding, means=mm)
        drop = clean - acc
        headroom = max(clean - 0.5, 1e-9)
        record[f"joint_{kind}"] = {
            "ablated_accuracy": acc, "drop": drop,
            "fraction_of_headroom_removed": drop / headroom,
        }
        checkpoint()
        print(f"[{ts()}] joint {kind}: acc {acc:.4f}, drop {drop:.4f}, "
              f"{drop/headroom:.3f} of headroom")

        control = []
        for d in tqdm(range(args.n_draws), desc=f"control {kind}"):
            cproj = {}
            for L in layers:
                cidx = draw_control(rng, bases[L].shape[0], used[L], len(used[L]))
                cproj[L] = build_projection_matrix(bases[L], cidx)
            cmeans = (subspace_means(lm, pairs, layers, cproj, is_binding)
                      if kind == "mean" else None)
            cacc = joint_ablated_accuracy(lm, pairs, layers, cproj, is_binding, means=cmeans)
            control.append(clean - cacc)
            if (d + 1) % 10 == 0:
                arr = np.array(control)
                record[f"control_{kind}"] = {
                    "n_draws_done": len(control), "drops": [float(x) for x in control],
                    "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
                    "quantile_of_observed": float((arr < drop).mean()),
                }
                checkpoint()

        arr = np.array(control)
        record[f"control_{kind}"] = {
            "n_draws_done": len(control), "drops": [float(x) for x in control],
            "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
            "quantile_of_observed": float((arr < drop).mean()),
        }
        checkpoint()
        print(f"[{ts()}] control {kind}: mean drop {arr.mean():.4f}, "
              f"observed at quantile {(arr < drop).mean():.3f}")

    checkpoint()
    print(f"[{ts()}] wrote {out_path}")


if __name__ == "__main__":
    main()
