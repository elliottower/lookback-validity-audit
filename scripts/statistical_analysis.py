"""Statistical analysis of extracted IIA results from Prakash et al. (ICLR 2026).

Computes Wilson confidence intervals, BH-corrected significance tests,
and generates figures for the validity audit paper.

Run: uv run --python 3.12 python scripts/statistical_analysis.py
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "reference" / "extracted_results_llama70b.json"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
N = 80

LOOKBACK_TYPES = [
    "answer_lookback_pointer",
    "answer_lookback_payload",
    "binding_lookback_address_and_payload",
    "visibility_lookback_source",
]

DISPLAY_NAMES = {
    "answer_lookback_pointer": "Answer (pointer)",
    "answer_lookback_payload": "Answer (payload)",
    "binding_lookback_address_and_payload": "Binding (addr+payload)",
    "visibility_lookback_source": "Visibility (source)",
}

COLORS = {
    "answer_lookback_pointer": "#d62728",
    "answer_lookback_payload": "#ff7f0e",
    "binding_lookback_address_and_payload": "#1f77b4",
    "visibility_lookback_source": "#2ca02c",
}


def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def standard_error(p, n):
    return math.sqrt(p * (1 - p) / n)


def binomial_pvalue(k, n, p0=0.5):
    from math import comb
    if k / n <= p0:
        return 1.0
    pval = 0.0
    for i in range(k, n + 1):
        pval += comb(n, i) * p0**i * (1 - p0)**(n - i)
    return pval


def benjamini_hochberg(pvalues, alpha=0.05):
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    rejected = [False] * m
    adjusted = [1.0] * m
    prev_adj = 1.0
    for rank_minus_1 in range(m - 1, -1, -1):
        orig_idx, pval = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj = min(prev_adj, pval * m / rank)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj
    for i in range(m):
        rejected[i] = adjusted[i] <= alpha
    return rejected, adjusted


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)

    records = []
    for lt in LOOKBACK_TYPES:
        section = raw[lt]
        for key, vals in section.items():
            if key.startswith("_"):
                continue
            layer = int(key[1:])
            records.append({
                "lookback_type": lt,
                "layer": layer,
                "sv_iia": vals["sv_iia"],
                "full_rank_iia": vals["full_rank_iia"],
                "rank": vals["rank"],
            })
    return records


def compute_stats(records):
    results = []
    for r in records:
        for metric in ("sv_iia", "full_rank_iia"):
            p = r[metric]
            k = round(p * N)
            lo, hi = wilson_ci(p, N)
            se = standard_error(p, N)
            pval = binomial_pvalue(k, N, p0=0.5)
            results.append({
                "lookback_type": r["lookback_type"],
                "layer": r["layer"],
                "metric": metric,
                "iia": p,
                "k": k,
                "se": se,
                "ci_lo": lo,
                "ci_hi": hi,
                "pvalue_vs_chance": pval,
                "rank": r["rank"],
            })
    return results


def apply_bh(stats):
    pvals = [s["pvalue_vs_chance"] for s in stats]
    rejected, adjusted = benjamini_hochberg(pvals)
    for s, rej, adj in zip(stats, rejected, adjusted):
        s["bh_rejected"] = rej
        s["bh_adjusted_p"] = adj
    return stats


def print_summary(stats):
    sv_stats = [s for s in stats if s["metric"] == "sv_iia"]
    sv_stats.sort(key=lambda s: (s["lookback_type"], s["layer"]))

    print(f"\n{'='*90}")
    print(f"{'Lookback Type':<35} {'Layer':>5} {'IIA':>6} {'95% CI':>16} {'p(BH)':>10} {'Sig':>5}")
    print(f"{'='*90}")

    current_lt = None
    for s in sv_stats:
        if s["lookback_type"] != current_lt:
            current_lt = s["lookback_type"]
            print(f"\n  {DISPLAY_NAMES[current_lt]}")
            print(f"  {'-'*84}")

        sig = "***" if s["bh_adjusted_p"] < 0.001 else "**" if s["bh_adjusted_p"] < 0.01 else "*" if s["bh_adjusted_p"] < 0.05 else "ns"
        print(f"  {'':30} L{s['layer']:>3}  {s['iia']:>5.3f}  [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  {s['bh_adjusted_p']:>9.2e}  {sig:>5}")

    n_total = len(sv_stats)
    n_sig = sum(1 for s in sv_stats if s["bh_rejected"])
    print(f"\n{'='*90}")
    print(f"sv_iia: {n_sig}/{n_total} survive BH correction at alpha=0.05 vs chance (0.5)")

    fr_stats = [s for s in stats if s["metric"] == "full_rank_iia"]
    n_fr_sig = sum(1 for s in fr_stats if s["bh_rejected"])
    print(f"full_rank_iia: {n_fr_sig}/{len(fr_stats)} survive BH correction at alpha=0.05 vs chance (0.5)")
    print()


def fig1_iia_by_layer(records):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, lt in enumerate(LOOKBACK_TYPES):
        ax = axes[idx]
        subset = sorted([r for r in records if r["lookback_type"] == lt], key=lambda r: r["layer"])
        layers = [r["layer"] for r in subset]

        for metric, marker, ls, label in [
            ("sv_iia", "o", "-", "Selected subspace"),
            ("full_rank_iia", "s", "--", "Full 500-component basis"),
        ]:
            vals = [r[metric] for r in subset]
            ci_lo = [wilson_ci(v, N)[0] for v in vals]
            ci_hi = [wilson_ci(v, N)[1] for v in vals]
            yerr_lo = [v - lo for v, lo in zip(vals, ci_lo)]
            yerr_hi = [hi - v for v, hi in zip(vals, ci_hi)]

            ax.errorbar(layers, vals, yerr=[yerr_lo, yerr_hi],
                        fmt=marker + ls, color=COLORS[lt], markersize=5,
                        capsize=3, label=label,
                        alpha=1.0 if metric == "sv_iia" else 0.5)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (0.5)")
        ax.set_title(DISPLAY_NAMES[lt], fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("IIA")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)

    fig.suptitle("IIA by Layer with 95% Wilson CIs (n=80)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_iia_by_layer.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'fig1_iia_by_layer.png'}")


def fig2_rank_by_layer(records):
    fig, ax = plt.subplots(figsize=(12, 6))

    for lt in LOOKBACK_TYPES:
        subset = sorted([r for r in records if r["lookback_type"] == lt], key=lambda r: r["layer"])
        layers = [r["layer"] for r in subset]
        ranks = [r["rank"] for r in subset]
        ax.plot(layers, ranks, "o-", color=COLORS[lt], label=DISPLAY_NAMES[lt], markersize=6)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Subspace Rank (# selected SVD components)", fontsize=11)
    ax.set_title("Identified Subspace Rank by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_rank_by_layer.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'fig2_rank_by_layer.png'}")


def fig3_visibility_anomaly(records):
    vis = sorted([r for r in records if r["lookback_type"] == "visibility_lookback_source"],
                 key=lambda r: r["layer"])
    layers = [r["layer"] for r in vis]
    sv = [r["sv_iia"] for r in vis]
    fr = [r["full_rank_iia"] for r in vis]
    diff = [s - f for s, f in zip(sv, fr)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1], sharex=True)

    ax1.plot(layers, sv, "o-", color=COLORS["visibility_lookback_source"],
             label="Selected subspace (sv_iia)", markersize=7)
    ax1.plot(layers, fr, "s--", color=COLORS["visibility_lookback_source"],
             alpha=0.5, label="Full 500-component basis (full_rank_iia)", markersize=7)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("IIA", fontsize=11)
    ax1.set_title("Visibility Lookback: Selected Subspace Exceeds Full Basis", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(-0.05, 1.1)

    colors = ["#2ca02c" if d > 0 else "#d62728" for d in diff]
    ax2.bar(layers, diff, color=colors, alpha=0.7, width=0.6)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("sv_iia - full_rank_iia", fontsize=11)
    ax2.set_title("Difference (positive = subspace beats full basis)", fontsize=10)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_visibility_anomaly.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'fig3_visibility_anomaly.png'}")


def save_results(stats):
    for s in stats:
        s["ci_lo"] = round(s["ci_lo"], 4)
        s["ci_hi"] = round(s["ci_hi"], 4)
        s["se"] = round(s["se"], 4)
        s["pvalue_vs_chance"] = float(s["pvalue_vs_chance"])
        s["bh_adjusted_p"] = float(s["bh_adjusted_p"])

    out = {
        "n": N,
        "alpha": 0.05,
        "correction": "Benjamini-Hochberg",
        "null_hypothesis": "IIA = 0.5 (chance for binary outcome)",
        "ci_method": "Wilson score interval",
        "results": stats,
    }
    outpath = RESULTS_DIR / "statistical_analysis_results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {outpath}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records = load_data()
    print(f"Loaded {len(records)} layer-concept entries across {len(LOOKBACK_TYPES)} lookback types")

    stats = compute_stats(records)
    stats = apply_bh(stats)

    print_summary(stats)

    fig1_iia_by_layer(records)
    fig2_rank_by_layer(records)
    fig3_visibility_anomaly(records)

    save_results(stats)
    print("\nDone.")


if __name__ == "__main__":
    main()
