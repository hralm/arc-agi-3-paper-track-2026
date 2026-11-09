#!/usr/bin/env python3
"""Figures 3 and 4 of the paper, as vector PDFs (Figures 1 and 2 are diagrams: make_diagrams.py).

No new numbers: all come from Section 5, Table 3, and Table 4 of the paper. Figure 3 uses the raw phase-A score of
each run (mean of final_score in the kernel's benchmark.json) and rounds only for display (the mean of rounded
values would differ).
  Figure 3 · phase-A runs per arm, in two panels: A, the first batch (2 November; Section 5, "Three modes of
  presence"); B, the second batch (7 November; Table 3), on the same dfranzen base, games, and caps.
  Figure 4 · H5 bench wins with and without a true instance in the winning frame (Table 4; see bench/table4.py).
Palette: two categorical colors checked for color-vision deficiency (Delta E 24.7) and contrast (>= 3:1); ink and axes muted.
Run: python3 figures/make_figures.py (writes the PDFs to the current folder)
"""
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

FIRST_BATCH = [  # (label, valid runs, read by the rule?); raw scores of 2 November 2026, rounded only for display
    ("Base\n(unmodified)", [77.42775455, 74.98429688, 72.31784112], True),
    ("Advisor\n(mode 2: notes + pruning)", [68.84885436, 59.41682416, 76.85650530], True),
    ("Advice text only\n(no pruning)", [42.09410735, 53.54878733, 65.57438923], True),
    ("K plays\n(mode 1, n = 2)", [17.08414318, 38.15733441], False),
]
# Second batch, 7 November 2026: the four public notebooks of the main README, one Kaggle kernel version per run,
# scores as the run log records them (two decimals). The order follows Table 3. One last-resort run was excluded
# before any reading (nine exceptions in our layer in vc33) and replaced; it is not listed. An arm with fewer than
# three valid runs would be descriptive only.
SECOND_BATCH = [
    ("Control\n(unmodified base)", [41.28, 41.35, 52.38, 74.18], True),
    ("K plays\n(mode 1)", [21.03, 64.06, 61.35], True),
    ("Advisor\n(mode 2)", [50.80, 29.06, 74.04], True),
    ("Last resort\n(mode 3, 50.75 s)", [39.51, 81.64, 39.78], True),
]


def panel(ax, arms, title, ymax, ref) -> None:
    base = arms[0][1]
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for i, (label, v, measured) in enumerate(arms):
        xs = [i + d for d in (-0.12, -0.04, 0.04, 0.12)][:len(v)] if len(v) == 4 else \
             [i + d for d in (-0.08, 0.0, 0.08)][:len(v)]
        ax.scatter(xs, v, s=34, zorder=3, facecolor=BLUE if measured else "none",
                   edgecolor=BLUE if measured else MUTED, linewidth=1.4)
        if not v:
            continue
        m = mean(v)
        ax.plot([i - 0.22, i + 0.22], [m, m], color=INK, linewidth=1.6, zorder=4)
        ax.annotate(f"mean {m:.2f}", (i + 0.25, m), va="center", fontsize=8, color=INK2)
        # reading rule, fixed before the runs (paper, Section 5): an effect only if |diff| > 2 x the larger
        # sample SD of the pair
        if measured and i > 0 and len(base) >= 3:
            diff, bar = abs(mean(base) - m), 2 * max(stdev(base), stdev(v))
            verdict = "no measurable effect" if diff <= bar else "effect"
            sign = "<" if diff <= bar else ">"
            ax.annotate(f"vs {ref}:\n|Δ| {diff:.1f} {sign} bar {bar:.1f}\n{verdict}", (i, 4), ha="center",
                        fontsize=7.5, color=INK2)
        elif not measured:
            ax.annotate(f"descriptive only\n(n = {len(v)}, under the\nthree-round rule)", (i, 4), ha="center",
                        fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([b[0] for b in arms], fontsize=8)
    ax.set_xlim(-0.5, len(arms) - 0.35)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def figure_1() -> None:
    """Two stacked panels: A, the first batch (2 November); B, the second batch (7 November, Table 3)."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 6.8))
    panel(a, FIRST_BATCH, "A. First batch, 2 November", 100, "base")
    panel(b, SECOND_BATCH, "B. Second batch, 7 November (same base, games, and caps)", 100, "control")
    for ax in (a, b):
        ax.set_ylabel("phase-A score (points, 3 public games)")
    fig.tight_layout(h_pad=2.0)
    fig.savefig("fig1-modes-phase-a.pdf")
    plt.close(fig)


def figure_2() -> None:
    arms = [  # (arm, wins, no true instance), Table 4; K0 is left out (no H5)
        ("default (O3, G1, G2, G4)", 66, 50), ("without O3", 66, 54), ("without G1", 67, 52),
        ("without G2", 64, 59), ("without G4", 66, 49), ("randomized order", 63, 48), ("default + O1", 63, 19),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for j, (label, wins, no_true) in enumerate(reversed(arms)):
        ax.barh(j, no_true, color=BLUE, height=0.62, edgecolor="white", linewidth=1.0)
        ax.barh(j, wins - no_true, left=no_true, color=ORANGE, height=0.62, edgecolor="white", linewidth=1.0)
        ax.annotate(f"{no_true} of {wins}", (wins + 1, j), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([b[0] for b in reversed(arms)], fontsize=8)
    ax.set_xlim(0, 78)
    ax.set_xlabel("levels won (four seeds summed)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    h1 = plt.Rectangle((0, 0), 1, 1, color=BLUE)
    h2 = plt.Rectangle((0, 0), 1, 1, color=ORANGE)
    ax.legend([h1, h2], ["no true hypothesis instance in the winning frame", "at least one true instance"],
              fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig("fig2-h5-identification.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_1()
    figure_2()
    print("fig1-modes-phase-a.pdf, fig2-h5-identification.pdf")
