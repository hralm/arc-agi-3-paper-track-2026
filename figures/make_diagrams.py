#!/usr/bin/env python3
"""Figures 1 and 2 of the paper (diagrams, no data), as vector PDFs.

  Figure 1 · the behavioral classification of systems in Gharajedaghi (2011, p. 36, Table 2.1), after Ackoff,
  with the reading of Section 1 beside it: where the agents of ARC-AGI-3 sit and where the paper's working
  definition of a general intelligence sits.
  Figure 2 · the no-effect certificate and the hidden-state mark (Section 3.2), and where K acts in each of its
  three modes of presence (Sections 3.1 and 5), read with Gharajedaghi's distinction between control and
  influence (2011, p. 31).
Same palette and type as figures/make_figures.py.
Run: python3 figures/make_diagrams.py [--gallery] [--page] (writes the PDFs, and the PNGs if asked, to the current folder)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_BG, ORANGE_BG = "#eaf2fc", "#fdeee7"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
META = {"Creator": None, "Producer": None, "CreationDate": None}
IN = 0.5     # inches per data unit, both axes; the box text is 7 pt
OVER = []    # boxes whose text does not fit; the script stops if any


def box(ax, x, y, w, h, text, fc="white", ec=AXIS, ls="-", lw=0.9, size=7.0, color=INK, weight="normal",
        ha="center", pad=0.09):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, ls=ls,
                                lw=lw, zorder=2))
    tx = x + w / 2 if ha == "center" else x + pad
    t = ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color, weight=weight, zorder=3,
                linespacing=1.25)
    fit(ax, t, w - 2 * pad, h - 2 * pad)


def fit(ax, t, w, h):
    """Fail if the rendered text does not fit its box (data units)."""
    r = ax.figure.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    if x1 - x0 > w or y1 - y0 > h:
        OVER.append(f"{t.get_text()[:30]!r}: needs {x1 - x0:.2f} x {y1 - y0:.2f}, has {w:.2f} x {h:.2f}")


def arrow(ax, a, b, color=MUTED, ls="-", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=1))


CELLS = [   # Gharajedaghi (2011, p. 36, Table 2.1): behavior and process, means and structure, end and function
    ("Passive\ntools", "No choice; one\nstructure in all\nenvironments", "No choice; one\nfunction in all\nenvironments"),
    ("Reactive\nstate-maintaining", "No choice;\nvariable but\ndetermined", "No choice; one\nfunction in all\ndifferent\nenvironments"),
    ("Responsive\ngoal-seeking", "Choice of means;\nvariable and chosen", "No choice of ends;\nvariable but\ndetermined"),
    ("Active\npurposeful", "Choice of means;\nvariable and chosen", "Choice of ends;\nvariable and chosen"),
]
NOTES_NARROW = [
    "", "",
    "The agents of ARC-AGI-3, ours\nincluded. Each game's rules fix the\nend, its win condition; the score\n"
    "charges the means, actions per level\nagainst a human reference. Across\ngames, at most multi-goal-seeking:\n"
    "the game, not the agent, sets which\ngoal (Ackoff & Emery).",
    "The working definition of a general\nintelligence adopted here: a\npurposeful system that selects its\n"
    "goals as well as its means. Not tested.",
]
NOTES_WIDE = [
    "", "",
    "The agents of ARC-AGI-3, ours included. Each game's rules\nfix the end, its win condition; the score charges the\n"
    "means, actions per level against a human reference. Across\ngames, at most multi-goal-seeking: the game, not the\n"
    "agent, sets which goal (Ackoff & Emery).",
    "The working definition of a general intelligence adopted\nhere: a purposeful system that selects its goals as well\n"
    "as its means. Not tested.",
]
SOURCE = "Left three columns: Gharajedaghi (2011, p. 36, Table 2.1), after Ackoff."


def figure_classification():
    heights = [0.86, 1.1, 1.98, 1.1]
    top = sum(heights) + 0.06 * len(heights) + 0.45
    xmax = 10.9
    fig = plt.figure(figsize=(IN * xmax, IN * (top + 0.75))); ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(-0.1, xmax), ylim=(-0.1, top + 0.75)); ax.axis("off")
    cols = [(0.0, 1.95), (2.0, 2.3), (4.35, 2.3), (6.7, 4.2)]
    head = ["Behavior\nProcess", "Means\nStructure", "End\nFunction", "Reading in Section 1"]
    for (x, w), t in zip(cols, head):
        ax.text(x + w / 2, top + 0.38, t, ha="center", va="center", fontsize=7.6, color=INK2, linespacing=1.2)
    ax.plot([0, xmax], [top + 0.05, top + 0.05], color=AXIS, lw=0.8)
    rows = [cells + (note, accent) for cells, note, accent in zip(CELLS, NOTES_NARROW, (None, None, BLUE, ORANGE))]
    y = top
    for (b, m, e, note, accent), h in zip(rows, heights):
        y -= h
        bg = {BLUE: BLUE_BG, ORANGE: ORANGE_BG}.get(accent, "white")
        for (x, w), t in zip(cols[:3], (b, m, e)):
            box(ax, x, y + 0.03, w, h - 0.06, t, fc=bg, ec=accent or AXIS,
                weight="bold" if (accent and t is b) else "normal")
        if note:
            box(ax, cols[3][0], y + 0.03, cols[3][1], h - 0.06, note, fc="white", ec=accent, ha="left", color=INK2)
            arrow(ax, (cols[2][0] + cols[2][1], y + h / 2), (cols[3][0], y + h / 2), color=accent)
        y -= 0.06
    ax.text(0, 0.05, SOURCE, fontsize=6.6, color=MUTED)
    assert not OVER, OVER
    fig.savefig("diagram-behavioral-classification.pdf", bbox_inches="tight", metadata=META)
    plt.close(fig)


def panel_certificate(ax, dx=0.0):
    """Panel A: main flow, top to bottom; no return path (no certificate is issued on a hidden-state level)."""
    X, W = 0.0 + dx, 4.0
    main = [
        (5.65, 0.9, "a single action outside a batch, on a\nlevel not marked hidden-state; frame\nunchanged outside the display band", "white", AXIS),
        (4.2, 0.9, "certificate issued; key masked from\nthe model's list (unless the list\nwould be empty)", BLUE_BG, BLUE),
        (2.7, 0.9, "a later execution contradicts it:\ncertificate falsified and revoked;\nthe level's other certificates voided", "white", ORANGE),
        (0.85, 0.9, "level marked hidden-state: no further\ncertificates on it; no-op notes in the\nprompt downgraded to \u201cnot a proof\u201d", ORANGE_BG, ORANGE),
    ]
    for y, h, t, fc, ec in main:
        box(ax, X, y, W, h, t, fc=fc, ec=ec)
    for (y0, h0, *_), (y1, h1, *_) in zip(main, main[1:]):
        arrow(ax, (X + W / 2, y0), (X + W / 2, y1 + h1))
    ax.text(X + W / 2 + 0.1, 2.22, "trigger (i)", fontsize=6.6, color=ORANGE, va="center")
    # side: deaths that mark the level; revocation without falsification
    SX, SW = 4.25 + dx, 3.3
    box(ax, SX, 0.8, SW, 1.0, "a death after a no-op (i), or\non an action K had considered\nsafe, or at the life budget (ii)", ec=ORANGE)
    arrow(ax, (SX, 1.3), (X + W, 1.3), color=ORANGE)
    box(ax, SX, 5.1, SW, 1.0, "K reopens tested actions:\nthe level's certificates\nrevoked, keys back in the list", ls="--")
    box(ax, SX, 3.7, SW, 0.9, "K relearns the display mask\nand reindexes: all\ncertificates revoked", ls="--")
    arrow(ax, (SX, 5.6), (X + W, 4.95), ls="--")
    arrow(ax, (SX, 4.15), (X + W, 4.65), ls="--")
    ax.text(SX, 3.4, "revocation without falsification;\nthe level is not marked", fontsize=6.4, color=MUTED, va="center")


MODES = [
    ("Mode 1, playing", "K takes turns on named triggers:\nit controls part of the play", "solid"),
    ("Mode 2, advising", "K certifies, prunes, and annotates;\nit only influences the model's choice", "advice"),
    ("Mode 3, last resort", "prompt and list as in the base;\nK plays only in a final window", "dashed"),
]


def mode_cell(ax, x, y, title, note, kind, gap=1.8, game_w=1.25):
    """One mode: title and note above; LLM agent and K on the left, the game on the right."""
    GX = x + 1.5 + gap
    ax.text(x, y + 1.98, title, fontsize=7.4, color=INK, weight="bold", va="center")
    ax.text(x, y + 1.5, note, fontsize=6.6, color=INK2, va="center", linespacing=1.2)
    box(ax, x, y + 0.7, 1.5, 0.45, "LLM agent", fc=BLUE_BG, ec=BLUE)
    box(ax, x, y + 0.0, 1.5, 0.45, "K", fc=ORANGE_BG, ec=ORANGE)
    box(ax, GX, y + 0.0, game_w, 1.15, "game")
    arrow(ax, (x + 1.5, y + 0.92), (GX, y + 0.92), color=BLUE)
    if kind == "solid":
        arrow(ax, (x + 1.5, y + 0.22), (GX, y + 0.22), color=ORANGE)
    elif kind == "dashed":
        arrow(ax, (x + 1.5, y + 0.22), (GX, y + 0.22), color=ORANGE, ls="--")
    else:
        arrow(ax, (GX, y + 0.22), (x + 1.5, y + 0.22), color=MUTED, lw=0.8)
        ax.text((x + 1.5 + GX) / 2, y + 0.12, "observes", fontsize=6.2, color=MUTED, ha="center", va="top")
        arrow(ax, (x + 0.75, y + 0.45), (x + 0.75, y + 0.7), color=ORANGE, ls=":", lw=1.2)


def panel_modes(ax, x=8.0):
    """Panel B: the three modes stacked."""
    for y, (title, note, kind) in zip((4.6, 2.35, 0.1), MODES):
        mode_cell(ax, x, y, title, note, kind)
    ax.text(x, -0.15, CONTROL, fontsize=6.4, color=MUTED)


CONTROL = "Control and influence as in Gharajedaghi (2011, p. 31)."


def figure_mechanism():
    xmax, ymax = 12.6, 7.3
    fig = plt.figure(figsize=(IN * xmax, IN * ymax)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(-0.1, xmax), ylim=(-0.4, ymax)); ax.axis("off")
    ax.text(0, 6.9, "A. No-effect certificate (Section 3.2)", fontsize=8.4, color=INK, va="center")
    ax.text(8.0, 6.9, "B. Where K acts (Sections 3.1 and 5)", fontsize=8.4, color=INK, va="center")
    ax.plot([7.75, 7.75], [-0.2, 6.75], color=GRID, lw=0.8)
    panel_certificate(ax)
    panel_modes(ax)
    assert not OVER, OVER
    fig.savefig("diagram-certificate-and-modes.pdf", bbox_inches="tight", metadata=META)
    plt.close(fig)


# ---- other layouts: the Kaggle media gallery (1280 x 580 px, the preview's crop) and the project page (narrow column) ----
GALLERY = (6.4, 2.9)   # inches at 200 dpi


def canvas(w, h, xmax, ymax, x0=-0.1, y0=-0.1):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(x0, x0 + xmax), ylim=(y0, y0 + ymax)); ax.axis("off")
    return fig, ax


def save_png(fig, name, tight):
    assert not OVER, OVER
    if tight:
        fig.savefig(name, dpi=300, bbox_inches="tight", pad_inches=0.08, metadata={"Software": None})
    else:
        fig.savefig(name, dpi=200, metadata={"Software": None})
    plt.close(fig)


def classification_table(ax, cols, heights, top, notes, note_col=None, below=None):
    """The table, with the notes in a fourth column (note_col) or in boxes below it (below = (x, w, heights))."""
    for (x, w), t in zip(cols, ["Behavior\nProcess", "Means\nStructure", "End\nFunction"]):
        ax.text(x + w / 2, top + 0.38, t, ha="center", va="center", fontsize=7.6, color=INK2, linespacing=1.2)
    if note_col:
        ax.text(note_col[0] + note_col[1] / 2, top + 0.38, "Reading in Section 1", ha="center", va="center",
                fontsize=7.6, color=INK2)
    right = (note_col[0] + note_col[1]) if note_col else cols[-1][0] + cols[-1][1]
    ax.plot([0, right], [top + 0.05, top + 0.05], color=AXIS, lw=0.8)
    y = top
    for cells, note, accent, h in zip(CELLS, notes, (None, None, BLUE, ORANGE), heights):
        y -= h
        bg = {BLUE: BLUE_BG, ORANGE: ORANGE_BG}.get(accent, "white")
        for (x, w), t in zip(cols, cells):
            box(ax, x, y + 0.03, w, h - 0.06, t, fc=bg, ec=accent or AXIS,
                weight="bold" if (accent and t is cells[0]) else "normal")
        if note and note_col:
            box(ax, note_col[0], y + 0.03, note_col[1], h - 0.06, note, ec=accent, ha="left", color=INK2)
            arrow(ax, (cols[2][0] + cols[2][1], y + h / 2), (note_col[0], y + h / 2), color=accent)
        y -= 0.06
    if below:
        x, w, hs = below
        for cells, note, accent in zip(CELLS[2:], notes[2:], (BLUE, ORANGE)):
            h = hs.pop(0)
            y -= h + 0.1
            label = cells[0].replace("\n", ", ") + ":\n"
            box(ax, x, y, w, h, label + note, ec=accent, ha="left", color=INK2)
    return y


def gallery_classification():
    fig, ax = canvas(*GALLERY, 12.8, 5.8)
    cols = [(0.0, 1.9), (1.95, 2.25), (4.25, 2.25)]
    classification_table(ax, cols, [0.86, 1.1, 1.3, 0.95], 4.9, NOTES_WIDE, note_col=(6.55, 6.1))
    ax.text(0, -0.02, SOURCE, fontsize=6.6, color=MUTED)
    save_png(fig, "gallery-diagram-1-behavioral-classification.png", tight=False)


def gallery_certificate():
    fig, ax = canvas(*GALLERY, 12.8, 5.8)
    ax.text(0, 5.45, "A. No-effect certificate (Section 3.2)", fontsize=8.4, color=INK, va="center")
    W, H, Y = 2.92, 1.3, 2.1
    main = [
        ("a single action outside\na batch, on a level not\nmarked hidden-state;\nframe unchanged outside\nthe display band", "white", AXIS),
        ("certificate issued; key\nmasked from the model's\nlist (unless the list\nwould be empty)", BLUE_BG, BLUE),
        ("a later execution\ncontradicts it:\ncertificate falsified\nand revoked; the level's\nother certificates voided", "white", ORANGE),
        ("level marked hidden-state:\nno further certificates\non it; no-op notes\ndowngraded to\n\u201cnot a proof\u201d", ORANGE_BG, ORANGE),
    ]
    xs = [0.0, 3.24, 6.48, 9.72]
    for x, (t, fc, ec) in zip(xs, main):
        box(ax, x, Y, W, H, t, fc=fc, ec=ec)
    for x0, x1 in zip(xs, xs[1:]):
        arrow(ax, (x0 + W, Y + H / 2), (x1, Y + H / 2))
    ax.text((xs[2] + W + xs[3]) / 2, Y - 0.12, "trigger (i)", fontsize=6.4, color=ORANGE, ha="center", va="top")
    box(ax, 0.9, 3.9, 3.15, 1.0, "K reopens tested actions:\nthe level's certificates\nrevoked, keys back in the list", ls="--")
    box(ax, 4.25, 3.9, 3.15, 1.0, "K relearns the display mask\nand reindexes: all\ncertificates revoked", ls="--")
    arrow(ax, (2.45, 3.9), (xs[1] + 0.9, Y + H), ls="--")
    arrow(ax, (5.8, 3.9), (xs[1] + 2.0, Y + H), ls="--")
    ax.text(7.65, 4.4, "revocation without falsification;\nthe level is not marked", fontsize=6.4, color=MUTED, va="center")
    box(ax, 9.72, 0.15, W, 1.15, "a death after a no-op (i),\nor on an action K had\nconsidered safe, or at the\nlife budget (ii)", ec=ORANGE)
    arrow(ax, (9.72 + W / 2, 1.3), (9.72 + W / 2, Y), color=ORANGE)
    ax.text(0, 0.3, "No certificate is issued on a level once it is marked.", fontsize=6.6, color=MUTED)
    save_png(fig, "gallery-diagram-2a-certificate.png", tight=False)


def gallery_modes():
    fig, ax = canvas(*GALLERY, 12.8, 5.8)
    ax.text(0, 5.45, "B. Where K acts (Sections 3.1 and 5)", fontsize=8.4, color=INK, va="center")
    for x, (title, note, kind) in zip((0.0, 4.3, 8.6), MODES):
        mode_cell(ax, x, 1.7, title, note, kind, gap=1.05, game_w=1.1)
    ax.text(0, 0.3, CONTROL, fontsize=6.6, color=MUTED)
    save_png(fig, "gallery-diagram-2b-modes.png", tight=False)


def page_classification():
    cols = [(0.0, 1.95), (2.0, 2.3), (4.35, 2.3)]
    fig, ax = canvas(IN * 6.8, IN * 7.9, 6.8, 7.9)
    y = classification_table(ax, cols, [0.86, 1.1, 0.95, 0.75], 7.0, NOTES_WIDE, below=(0.0, 6.65, [1.45, 1.1]))
    ax.text(0, y - 0.35, SOURCE, fontsize=6.6, color=MUTED)
    save_png(fig, "diagram-behavioral-classification.png", tight=True)


def page_certificate():
    fig, ax = canvas(IN * 7.7, IN * 7.3, 7.7, 7.3, y0=-0.1)
    ax.text(0, 6.9, "A. No-effect certificate (Section 3.2)", fontsize=8.4, color=INK, va="center")
    panel_certificate(ax)
    save_png(fig, "diagram-certificate.png", tight=True)


def page_modes():
    fig, ax = canvas(IN * 4.9, IN * 7.7, 4.9, 7.7, y0=-0.4)
    ax.text(0, 7.15, "B. Where K acts (Sections 3.1 and 5)", fontsize=8.4, color=INK, va="center")
    panel_modes(ax, x=0.0)
    save_png(fig, "diagram-modes.png", tight=True)


if __name__ == "__main__":
    import sys
    figure_classification()
    figure_mechanism()
    if "--gallery" in sys.argv:   # PNGs for the Kaggle media gallery
        gallery_classification(); gallery_certificate(); gallery_modes()
    if "--page" in sys.argv:      # PNGs for the project page (docs/)
        page_classification(); page_certificate(); page_modes()
    print("diagram-behavioral-classification.pdf, diagram-certificate-and-modes.pdf")
