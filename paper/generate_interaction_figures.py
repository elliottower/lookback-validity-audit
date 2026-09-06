"""Figures for the interaction-structure paper."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results" / "walsh_interaction" / "qwen_mismatch_walsh.json"
OUT = REPO / "paper" / "figures"

W2_FLOOR = -0.5
W2_CEIL = 0.25


def load():
    with open(DATA) as fh:
        d = json.load(fh)
    binding = [c for c in d["cells"] if c["lookback_type"] == "binding"]
    answer = [c for c in d["cells"] if c["lookback_type"] == "answer"]
    return binding, answer


def fig_coefficient(binding, answer):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for cells, color, label in ((binding, "#b2182b", "Binding lookback"),
                                (answer, "#2166ac", "Answer lookback")):
        layers = [c["layer"] for c in cells]
        w2 = [c["order2"] for c in cells]
        lo = [c["order2_ci95_unpaired"][0] for c in cells]
        hi = [c["order2_ci95_unpaired"][1] for c in cells]
        ax.plot(layers, w2, "o-", color=color, label=label, markersize=4)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.2, linewidth=0)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(W2_FLOOR, color="#b2182b", linestyle=":", linewidth=1)
    ax.axhline(W2_CEIL, color="#2166ac", linestyle=":", linewidth=1)
    ax.text(38.4, W2_FLOOR + 0.012, "maximal masking", fontsize=7,
            color="#b2182b", ha="right")
    ax.text(38.4, W2_CEIL + 0.012, "maximal synergy", fontsize=7,
            color="#2166ac", ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"order-2 Walsh coefficient  $w_{2}$")
    ax.set_ylim(-0.56, 0.31)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"interaction_coefficient.{ext}", dpi=200)
    plt.close(fig)


def fig_cells(binding, answer):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, cells, title in ((axes[0], binding, "Binding lookback"),
                             (axes[1], answer, "Answer lookback")):
        layers = [c["layer"] for c in cells]
        cv = [c["coalition_values"] for c in cells]
        ax.plot(layers, [v["recalled_only"] for v in cv], "o-",
                color="#f4a582", markersize=3.5, label="recalled only")
        ax.plot(layers, [v["lookback_only"] for v in cv], "s-",
                color="#92c5de", markersize=3.5, label="lookback only")
        ax.plot(layers, [v["both"] for v in cv], "^-",
                color="#1a1a1a", markersize=4, label="both")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("IIA")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"interaction_cells.{ext}", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    binding, answer = load()
    fig_coefficient(binding, answer)
    fig_cells(binding, answer)
    print(f"wrote {OUT}/interaction_coefficient.pdf and interaction_cells.pdf")


if __name__ == "__main__":
    main()
