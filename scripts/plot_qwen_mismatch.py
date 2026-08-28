"""Plot X2 Qwen mismatch test results — two-panel figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "qwen_mismatch"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


def load_results():
    with open(RESULTS_DIR / "summary.json") as f:
        return json.load(f)


def plot_mismatch(ax, results, lookback_type, layers, title):
    recalled = []
    lookback = []
    both = []

    for layer in layers:
        r = results[str(layer)]
        recalled.append(r["recalled_only_iia"])
        lookback.append(r["lookback_only_iia"])
        both.append(r["both_iia"])

    x = np.array(layers)

    ax.plot(x, recalled, "o-", color="#e74c3c", label="Recalled only", linewidth=2, markersize=7)
    ax.plot(x, lookback, "s-", color="#3498db", label="Lookback only", linewidth=2, markersize=7)
    ax.plot(x, both, "D-", color="#2ecc71", label="Both", linewidth=2, markersize=8)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("IIA", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(x)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)


def main():
    data = load_results()

    binding_layers = data["binding_layers"]
    answer_layers = data["answer_layers"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    plot_mismatch(
        ax1, data["binding_results"], "binding", binding_layers,
        "Binding lookback (anti-mismatch)"
    )
    plot_mismatch(
        ax2, data["answer_results"], "answer", answer_layers,
        "Answer lookback (mismatch signature)"
    )

    ax2.annotate(
        "Mismatch\nsignature",
        xy=(32, 1.0), xytext=(22, 0.7),
        fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5),
        color="#2ecc71", fontweight="bold",
    )

    ax1.annotate(
        "Each alone works,\nboth together fails",
        xy=(30, 0.0), xytext=(30, 0.45),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
        color="grey",
    )

    fig.suptitle(
        "Cross-model mismatch test: Qwen2.5-14B-Instruct ($n = 200$)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"fig7_qwen_mismatch.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
