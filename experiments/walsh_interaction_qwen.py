"""Exact order-2 Walsh interaction between the two token groups of the mismatch test.

The cross-model mismatch test (scripts/modal_qwen_mismatch_test.py) patches the
recalled tokens alone, the lookback tokens alone, and both. With the unpatched
condition those are all four cells of a 2-player coalition function, so the
Walsh decomposition is exact: no sparse recovery, no regularization, no
held-out set.

Conventions follow epistatic-circuits:
  - normalized WHT, w_S = 2^{-n} sum_x v(x) chi_S(x)   (src/walsh.py)
  - masking matrix M = -w_ij                            (scripts/v13_complementation.py:54)
  - positive M = masking / sub-additive, negative M = synergy / super-additive
    (experiments/E9_complementation/PREREG.md:84)

Player 0 = recalled tokens patched.  Player 1 = lookback tokens patched.
v = interchange intervention accuracy against the counterfactual answer.

v(00) = 0 exactly: pairs are filtered so the clean prompt returns clean_ans and
IIA counts matches against cf_target, and clean_ans != cf_target by construction
of the counterfactual pair.
"""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "qwen_mismatch" / "summary.json"
OUT_DIR = REPO / "results" / "walsh_interaction"

N_PLAYERS = 2
# v in [0, 1] with v(00) pinned to 0 bounds w_2 = (v00 - v10 - v01 + v11) / 4.
W2_FLOOR = -0.5   # v10 = v01 = 1, v11 = 0: maximal masking
W2_CEIL = 0.25    # v10 = v01 = 0, v11 = 1: maximal synergy
ADDITIVE_TOL = 0.01  # |w2| below this is reported as additive


def wht(values):
    """Normalized Walsh-Hadamard transform, matching epistatic-circuits/src/walsh.py."""
    v = np.asarray(values, dtype=np.float64).copy()
    h = 1
    while h < len(v):
        for i in range(0, len(v), h * 2):
            a = v[i:i + h].copy()
            b = v[i + h:i + 2 * h].copy()
            v[i:i + h] = a + b
            v[i + h:i + 2 * h] = a - b
        h *= 2
    return v / len(v)


def coalition_values(cell):
    """Order the four IIA measurements by coalition bitmask.

    bit 0 = recalled tokens patched, bit 1 = lookback tokens patched.
    """
    return np.array([
        0.0,
        cell["recalled_only_iia"],
        cell["lookback_only_iia"],
        cell["both_iia"],
    ])


def classify(w2):
    if abs(w2) < ADDITIVE_TOL:
        return "additive"
    return "synergy" if w2 > 0 else "masking"


def bootstrap_w2(cell, n_draws, rng):
    """Resample each patched cell as an independent binomial.

    The three patched conditions were measured on the same filtered pairs, but
    modal_qwen_mismatch_test.py stores counts rather than per-pair outcomes, so
    the pairing cannot be preserved. Ignoring it inflates the variance, making
    this interval an upper bound on the width rather than the exact one.
    """
    n = cell["n_samples"]
    if n == 0:
        return None
    recalled = rng.binomial(n, cell["recalled_only_iia"], n_draws) / n
    lookback = rng.binomial(n, cell["lookback_only_iia"], n_draws) / n
    both = rng.binomial(n, cell["both_iia"], n_draws) / n
    w2 = (0.0 - recalled - lookback + both) / 4.0
    return w2


def analyze_cell(cell, n_draws, rng):
    v = coalition_values(cell)
    w = wht(v)
    o1 = {i: float(w[1 << i]) for i in range(N_PLAYERS)}
    o2 = {(i, j): float(w[(1 << i) | (1 << j)])
          for i, j in itertools.combinations(range(N_PLAYERS), 2)}
    w2 = o2[(0, 1)]

    nonconstant = float(sum(w[k] ** 2 for k in range(1, 1 << N_PLAYERS)))
    order1_energy = float(o1[0] ** 2 + o1[1] ** 2)
    order2_energy = float(w2 ** 2)

    boot = bootstrap_w2(cell, n_draws, rng)
    if boot is None:
        ci = None
        sign_stability = None
    else:
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        sign_stability = float(np.mean(np.sign(boot) == np.sign(w2))) if w2 != 0 else None

    return {
        "layer": cell["layer"],
        "lookback_type": cell["lookback_type"],
        "n_samples": cell["n_samples"],
        "coalition_values": {
            "neither": 0.0,
            "recalled_only": cell["recalled_only_iia"],
            "lookback_only": cell["lookback_only_iia"],
            "both": cell["both_iia"],
        },
        "order1": {"recalled": o1[0], "lookback": o1[1]},
        "order2": w2,
        "masking_M": -w2,
        "verdict": classify(w2),
        "saturation": float(w2 / W2_FLOOR) if w2 < 0 else float(w2 / W2_CEIL),
        "order1_energy_frac": order1_energy / nonconstant if nonconstant > 0 else 0.0,
        "order2_energy_frac": order2_energy / nonconstant if nonconstant > 0 else 0.0,
        "order2_ci95_unpaired": ci,
        "sign_stability_unpaired": sign_stability,
        "mismatch_signature_reported": cell["mismatch_signature"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-draws", type=int, default=10000)
    args = parser.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(f"mismatch summary not found: {args.summary}")

    with open(args.summary) as fh:
        summary = json.load(fh)

    rng = np.random.default_rng(args.seed)
    cells = []
    for block in ("binding_results", "answer_results"):
        for layer in sorted(summary[block], key=int):
            cells.append(analyze_cell(summary[block][layer], args.n_draws, rng))

    out = {
        "analysis": "Exact order-2 Walsh interaction, mismatch-test token groups",
        "generated": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "summary": str(args.summary),
            "source_experiment": summary["experiment"],
            "model": summary["model"],
            "source_timestamp": summary["timestamp"],
            "source_seed": summary["seed"],
        },
        "seed": args.seed,
        "n_bootstrap_draws": args.n_draws,
        "conventions": {
            "transform": "normalized WHT, w_S = 2^-n sum_x v(x) chi_S(x)",
            "reference": "epistatic-circuits/src/walsh.py, scripts/v13_complementation.py:54",
            "masking_matrix": "M = -w_ij; positive M = masking, negative M = synergy",
            "players": {"0": "recalled tokens patched", "1": "lookback tokens patched"},
            "empty_coalition": "v(00) = 0 exactly; clean_ans != cf_target by pair construction",
            "w2_bounds": {"floor": W2_FLOOR, "ceiling": W2_CEIL},
            "additive_tolerance": ADDITIVE_TOL,
        },
        "limitations": {
            "reliability_gate": (
                "Split-half reliability (prereg_v13 Spearman-Brown >= 0.6) cannot be "
                "computed: modal_qwen_mismatch_test.py stores counts, not per-pair outcomes."
            ),
            "bootstrap": (
                "Cells resampled as independent binomials. The three patched conditions "
                "share the same filtered pairs, so the true paired interval is narrower; "
                "the reported interval is an upper bound on width."
            ),
            "perturbation_semantics": (
                "The Benzer complementation reading was defined for loss-of-function "
                "perturbations. Here each perturbation is a patch-in (gain of function), "
                "so the same-unit / different-unit verdict does not carry over unchanged "
                "and is not asserted."
            ),
        },
        "cells": cells,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "qwen_mismatch_walsh.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"wrote {dest}")
    print(f"{'mech':<9}{'L':>4}{'recall':>8}{'lookbk':>8}{'both':>7}"
          f"{'w2':>9}{'M':>9}{'sat':>7}  verdict")
    for c in cells:
        cv = c["coalition_values"]
        print(f"{c['lookback_type']:<9}{c['layer']:>4}{cv['recalled_only']:>8.3f}"
              f"{cv['lookback_only']:>8.3f}{cv['both']:>7.3f}"
              f"{c['order2']:>+9.4f}{c['masking_M']:>+9.4f}"
              f"{c['saturation']:>7.2f}  {c['verdict']}")


if __name__ == "__main__":
    main()
