"""Generate all figures for the lookback validity audit paper."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#878787"
ORANGE = "#e08214"
GREEN = "#1b7837"
TEAL = "#35978f"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_iia(d, subspace_key):
    """Extract IIA from a result dict, handling nested and flat formats."""
    if "subspace_iias" in d:
        val = d["subspace_iias"].get(subspace_key)
        if isinstance(val, dict):
            iia = val.get("iia")
            return float(iia) if iia is not None else np.nan
        if isinstance(val, (int, float)):
            return float(val)
    if "subspaces" in d:
        entry = d["subspaces"].get(subspace_key, {})
        for k in ("our_iia", "iia"):
            if k in entry and entry[k] is not None:
                return float(entry[k])
    return np.nan


def fig1_iia_heatmap():
    """Figure 1: IIA across experiments and subspaces (answer layers only)."""
    experiments = [
        ("Replication (E1)", "positive_control_replication.json", "replication"),
        ("False belief (E14)", "false_belief/false_belief.json", None),
        ("Observability (E15)", "observability/observed_other_container.json", None),
        ("Framing: belief (E12)", "question_framing/belief.json", None),
        ("Framing: action (E12)", "question_framing/action_recall.json", None),
        ("Framing: reality (E12)", "question_framing/reality_state.json", None),
        ("Cross-template (E10)", None, "cross_template"),
        ("Factual recall (E11)", "cross_task/factual_recall.json", None),
        ("Distractor: early (E13)", "distractor_insertion/early.json", None),
    ]

    subspaces = ["answer_pointer_L38", "answer_pointer_L52", "answer_pointer_L53"]
    col_labels = ["L38", "L52", "L53"]

    data = np.full((len(experiments), 3), np.nan)
    ns = np.full((len(experiments), 3), 0, dtype=int)

    for i, (name, path, mode) in enumerate(experiments):
        if mode == "cross_template":
            for j, lyr in enumerate(["L38", "L52", "L53"]):
                try:
                    d2 = load_json(RESULTS / f"surface_heuristic/standard_answer_pointer_{lyr}.json")
                    data[i, j] = d2.get("iia", np.nan)
                    ns[i, j] = d2.get("n_pairs", 0)
                except FileNotFoundError:
                    pass
            continue

        try:
            d = load_json(RESULTS / path)
            n = d.get("n_pairs", 0)
            if mode == "replication":
                for j, sub in enumerate(subspaces):
                    if sub in d.get("subspaces", {}):
                        data[i, j] = d["subspaces"][sub].get("our_iia", np.nan)
                        ns[i, j] = d["subspaces"][sub].get("n_pairs", 0)
            else:
                for j, sub in enumerate(subspaces):
                    data[i, j] = extract_iia(d, sub)
                    ns[i, j] = n
        except (FileNotFoundError, KeyError) as e:
            print(f"  Skip {path}: {e}")

    fig, ax = plt.subplots(figsize=(5.5, 5))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect=0.6)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            n = ns[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8,
                        color=GREY, fontstyle="italic")
            else:
                color = "white" if val < 0.25 or val > 0.75 else "black"
                txt = f"{val:.2f}"
                if n > 0 and n != 80:
                    txt += f"\nn={n}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        fontweight="bold" if val > 0.9 else "normal", color=color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels([e[0] for e in experiments], fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("IIA", fontsize=10)

    ax.set_title("IIA Across Experiments\n(answer subspaces, n=80 unless noted)",
                 fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_iia_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_iia_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 1: IIA heatmap saved")


def fig2_framing_dissociation():
    """Figure 2: Framing dissociation bar chart with Wilson CIs."""
    framings = ["belief", "action_recall", "reality_state", "fill_completion"]
    framing_labels = ["Belief\n\"What does X\nbelieve...\"",
                      "Action recall\n\"What did X\nput...\"",
                      "Reality state\n\"What is\ninside...\"",
                      "Fill completion\n\"X filled\n...with\""]
    subspaces = ["answer_pointer_L38", "answer_pointer_L52", "answer_pointer_L53"]
    sub_labels = ["L38", "L52", "L53"]
    colors = [BLUE, ORANGE, TEAL]

    iias = np.zeros((4, 3))
    ci_lo = np.zeros((4, 3))
    ci_hi = np.zeros((4, 3))
    has_ci = np.zeros((4, 3), dtype=bool)

    for i, framing in enumerate(framings):
        try:
            d = load_json(RESULTS / f"question_framing/{framing}.json")
            for j, sub in enumerate(subspaces):
                iias[i, j] = d["subspace_iias"].get(sub, 0)
                if "subspace_cis" in d and sub in d["subspace_cis"]:
                    ci_lo[i, j] = d["subspace_cis"][sub]["ci_lower"]
                    ci_hi[i, j] = d["subspace_cis"][sub]["ci_upper"]
                    has_ci[i, j] = True
        except (FileNotFoundError, KeyError):
            pass

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(4)
    width = 0.22

    for j in range(3):
        offsets = x + (j - 1) * width
        yerr_lo = np.where(has_ci[:, j], np.clip(iias[:, j] - ci_lo[:, j], 0, None), 0)
        yerr_hi = np.where(has_ci[:, j], np.clip(ci_hi[:, j] - iias[:, j], 0, None), 0)
        bars = ax.bar(offsets, iias[:, j], width, label=sub_labels[j], color=colors[j],
                       edgecolor="white", linewidth=0.5, zorder=3)
        # Only draw error bars where there's a nonzero value and CI data
        for k in range(4):
            if has_ci[k, j] and iias[k, j] > 0.01:
                ax.errorbar(offsets[k], iias[k, j],
                           yerr=[[yerr_lo[k]], [yerr_hi[k]]],
                           fmt="none", capsize=3, color="black", linewidth=1, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(framing_labels, fontsize=8.5, linespacing=1.1)
    ax.set_ylabel("IIA", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(3.6, 0.52, "chance", fontsize=7, color=GREY, alpha=0.7)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    ax.annotate("n = 35 per condition", xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=8, color=GREY, va="top")

    ax.set_title("Question Framing Dissociation", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_framing_dissociation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_framing_dissociation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 2: Framing dissociation saved")


def fig3_evalues():
    """Figure 3: E-value robustness as horizontal bar chart."""
    d = load_json(RESULTS / "confounding_sensitivity" / "summary.json")
    evalues = d["all_evalues"]

    positive = [e for e in evalues if e["iia"] > e["baseline_iia"]]
    positive.sort(key=lambda x: x["evalue"], reverse=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    ys = range(len(positive))
    colors_bar = [GREEN if e["evalue"] >= 2.0 else RED for e in positive]

    ax.barh(ys, [e["evalue"] for e in positive], color=colors_bar,
            edgecolor="white", linewidth=0.5, height=0.7)

    ax.axvline(2.0, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(1.0, color=GREY, linestyle=":", linewidth=0.8, alpha=0.4)

    labels = []
    for e in positive:
        sub = e["subspace"].replace("answer_pointer_", "").replace("binding_addr_payload_", "bind ")
        src = e["source"].replace("_", " ")
        cond = e["condition"].replace("_", " ")
        labels.append(f"{sub} | {cond}")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("E-value", fontsize=11)
    ax.set_xlim(0, 3.8)
    ax.invert_yaxis()

    robust_n = sum(1 for e in positive if e["evalue"] >= 2.0)
    fragile_n = sum(1 for e in positive if e["evalue"] < 2.0)
    robust_patch = mpatches.Patch(color=GREEN, label=f"Robust (E ≥ 2.0): {robust_n}/{len(positive)}")
    fragile_patch = mpatches.Patch(color=RED, label=f"Fragile (E < 2.0): {fragile_n}/{len(positive)}")
    ax.legend(handles=[robust_patch, fragile_patch], fontsize=8, loc="lower right")

    ax.set_title("Confounding Sensitivity Analysis (E-values)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_evalues.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_evalues.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 3: E-values saved")


def fig4_replication():
    """Figure 4: Replication comparison (paper vs our IIA)."""
    d = load_json(RESULTS / "positive_control_replication.json")

    names = []
    paper_iia = []
    our_iia = []
    for name, sub in d["subspaces"].items():
        short = name.replace("answer_pointer_", "").replace("binding_addr_payload_", "bind ")
        names.append(short)
        paper_iia.append(sub["paper_iia"])
        our_iia.append(sub["our_iia"])

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(len(names))
    width = 0.3

    ax.bar(x - width / 2, paper_iia, width, label="Prakash et al.", color=GREY, edgecolor="white")
    ax.bar(x + width / 2, our_iia, width, label="This work (Llama 3.1)", color=BLUE, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("IIA", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Replication: Prakash et al. vs. This Work", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(len(names)):
        delta = our_iia[i] - paper_iia[i]
        sign = "+" if delta >= 0 else ""
        ax.text(x[i] + width / 2, our_iia[i] + 0.02, f"{sign}{delta:.2f}",
                ha="center", va="bottom", fontsize=7, color=GREEN if delta >= 0 else RED)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_replication.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_replication.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 4: Replication comparison saved")


def fig5_crosstask():
    """Figure 5: Cross-task transfer showing L38 vs L52 dissociation."""
    tasks = {
        "Belief\n(CausalToM)": {"L38": 1.0, "L52": 1.0},
        "False belief": {"L38": 0.975, "L52": 0.975},
        "Observability": {"L38": 0.988, "L52": 0.975},
        "Cross-template\n(n=23)": {"L38": 0.957, "L52": 0.783},
        "Action recall": {"L38": 0.886, "L52": 0.400},
        "Factual recall": {"L38": 0.529, "L52": 0.0},
        "Color property\n(n=6)": {"L38": 0.333, "L52": 0.0},
        "Reality state": {"L38": 0.0, "L52": 0.0},
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    task_names = list(tasks.keys())
    x = np.arange(len(task_names))

    l38 = [tasks[t]["L38"] for t in task_names]
    l52 = [tasks[t]["L52"] for t in task_names]

    ax.plot(x, l38, "o-", color=BLUE, linewidth=2, markersize=8, label="L38 (answer pointer)", zorder=3)
    ax.plot(x, l52, "s-", color=ORANGE, linewidth=2, markersize=8, label="L52 (answer pointer)", zorder=3)

    ax.fill_between(x, l38, l52, alpha=0.1, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=8, ha="center")
    ax.set_ylabel("IIA", fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")

    ax.annotate("L38 partially transfers;\nL52 drops to zero",
                xy=(5, 0.529), xytext=(5.5, 0.72),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1),
                fontsize=8, color=GREY, ha="center")

    ax.set_title("Cross-Task Dissociation: L38 vs L52", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_crosstask.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig5_crosstask.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 5: Cross-task dissociation saved")


def fig6_validity_grid():
    """Figure 6: Validity criteria assessment grid."""
    categories = ["Construct\n(C1--C5)", "Measurement\n(M1--M7)", "Internal\n(I1--I11)",
                   "External\n(E1--E6)", "Interpretive\n(V1--V5)"]
    confirmed = [1, 0, 1, 0, 0]
    partial =   [2, 2, 2, 2, 2]
    not_met =   [2, 5, 8, 4, 1]
    fails =     [0, 0, 0, 0, 2]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(categories))
    width = 0.18

    ax.bar(x - 1.5 * width, confirmed, width, label="Confirmed", color=GREEN, edgecolor="white")
    ax.bar(x - 0.5 * width, partial, width, label="Partial", color="#fed976", edgecolor="white")
    ax.bar(x + 0.5 * width, not_met, width, label="Not met", color=ORANGE, edgecolor="white")
    ax.bar(x + 1.5 * width, fails, width, label="Fails", color=RED, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Number of criteria", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Validity Assessment: 35 Criteria Across 5 Categories", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig6_validity_grid.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig6_validity_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 6: Validity grid saved")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_iia_heatmap()
    fig2_framing_dissociation()
    fig3_evalues()
    fig4_replication()
    fig5_crosstask()
    fig6_validity_grid()
    print("Done.")
