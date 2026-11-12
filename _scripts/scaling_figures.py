"""Figures for the post "How to Think About Scaling a Foundation Model for Multiomics".

Redrawn (not copied) from ideas in Hoffmann et al. (2022), "Training Compute-Optimal
Large Language Models" (Chinchilla), arXiv:2203.15556.

  chinchilla-isoflop.png      Fig 1: loss vs. model size at fixed compute budgets
  chinchilla-projections.png  Fig 2: compute-optimal model size vs. compute
  chinchilla-cosine.png       Fig 3: schematic of learning-rate schedule length
  lr-scaling.png              Learning-rate sweeps at several budgets, and the best
                              learning rate vs. compute (DeepSeek LLM, Bi et al. 2024)

Figs 1 and 2 are computed from the paper's fitted loss (its Approach 3):
    L(N, D) = E + A / N**alpha + B / D**beta
with compute C ~= 6 * N * D (N = parameters, D = training tokens).
Fig 3 is a toy model for illustration only; the paper gives no formula for it.

lr-scaling.png uses the fitted optimum from DeepSeek LLM (arXiv:2401.02954, Eq. 1):
    eta_opt = 0.3118 * C**(-0.1250)
fitted on budgets from 1e17 to 2e19 FLOPs and checked at 1e20. In the left panel only
the positions of the minima come from that fit; the U shapes are illustrative.

Run from the repo root (needs numpy and matplotlib):
    python _scripts/scaling_figures.py
Output goes to images/posts/scaling/. The leading underscore keeps Jekyll from
publishing this folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "images" / "posts" / "scaling"

# Chinchilla fitted constants (Hoffmann et al. 2022, Approach 3).
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28
KAPLAN_SLOPE = 0.73  # N_opt ∝ C^0.73 (Kaplan et al. 2020), as reported by Hoffmann et al.

# Palette: neutral chrome plus a one-hue blue ramp and two categorical slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})

FIGSIZE = (7.2, 4.5)  # inches; at 200 dpi this is 1440 px wide
DPI = 200


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


def n_opt(c):
    """Compute-optimal model size from the fitted loss, with C = 6 N D."""
    g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    a = BETA / (ALPHA + BETA)
    return g * (c / 6) ** a


def fmt_flops(c):
    exp = int(np.floor(np.log10(c)))
    mant = c / 10**exp
    return f"{mant:g}×10$^{{{exp}}}$"


def credit(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def fig_isoflop():
    budgets = [6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21]
    colors = BLUE_RAMP[: len(budgets)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for c, col in zip(budgets, colors):
        best = n_opt(c)
        n = np.logspace(np.log10(best / 25), np.log10(best * 25), 200)
        ax.plot(n, loss(n, c / (6 * n)), color=col, label=fmt_flops(c))
        ax.plot(best, loss(best, c / (6 * best)), "o", ms=7, color=col,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("Training loss")
    ax.set_title("For each compute budget, there is a best model size")
    ax.legend(title="Compute budget (FLOPs)", title_fontsize=9.5, fontsize=9,
              loc="upper right", ncol=3)
    ax.get_legend().get_title().set_color(INK_2)
    ax.annotate("too small:\nnot enough capacity", xy=(0.04, 0.10), xycoords="axes fraction",
                fontsize=9, color=MUTED)
    ax.annotate("too big:\nnot enough data", xy=(0.84, 0.45), xycoords="axes fraction",
                fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    credit(fig, "Computed from the fitted loss in Hoffmann et al. (2022). Dots mark each valley's minimum.")
    fig.savefig(OUT / "chinchilla-isoflop.png", dpi=DPI)
    plt.close(fig)


def fig_projections():
    # Both lines use the paper's own numbers (Hoffmann et al. 2022, Appendix D.4):
    # at 1e21 FLOPs, Kaplan et al. predict 4.68B parameters and Approach 1 predicts 2.86B.
    # Slopes: 0.73 (Kaplan, Table 2) and 0.50 (Approach 1, Table 2).
    c = np.logspace(18, 25, 200)
    chin = 2.86e9 * (c / 1e21) ** 0.50
    kap = 4.68e9 * (c / 1e21) ** KAPLAN_SLOPE

    # Gopher and Chinchilla used about the same compute, 5.76e23 FLOPs.
    gopher = (5.76e23, 280e9)
    chinchilla = (5.76e23, 70e9)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(c, kap, color=SERIES_2)
    ax.plot(c, chin, color=SERIES_1)
    ax.text(c[-1], kap[-1], "  Kaplan et al. (2020)\n  $N \\propto C^{0.73}$", color=INK_2,
            fontsize=9.5, va="center")
    ax.text(c[-1], chin[-1], "  Hoffmann et al. (2022)\n  $N \\propto C^{0.50}$", color=INK_2,
            fontsize=9.5, va="center")
    for (x, y), name in [(gopher, "Gopher (280B)"), (chinchilla, "Chinchilla (70B)")]:
        ax.plot(x, y, "o", ms=8, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget (FLOPs)")
    ax.set_ylabel("Compute-optimal model size (parameters)")
    ax.set_title("Chinchilla: grow the model more slowly, and the data faster")
    ax.set_xlim(1e18, 1e25)
    fig.tight_layout(rect=(0, 0.07, 0.86, 1))
    credit(fig, "Lines use the slopes and 10$^{21}$-FLOP predictions reported by Hoffmann et al. (2022).\n"
                "Adapted from Hoffmann et al. (2022), Figure 1.")
    fig.savefig(OUT / "chinchilla-projections.png", dpi=DPI)
    plt.close(fig)


def fig_cosine():
    """Toy model: progress grows with the summed learning rate, and a high
    learning rate at the end leaves a noise penalty. Illustrative only."""
    steps = 1000
    t = np.arange(steps)
    multipliers = [1, 1.5, 2, 5]
    colors = [BLUE_RAMP[9], BLUE_RAMP[7], BLUE_RAMP[4], BLUE_RAMP[1]]

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    for k, col in zip(multipliers, colors):
        lr = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / (k * steps)))
        progress = np.cumsum(lr)
        toy_loss = 2.0 + 3.0 * (1 + progress / 30) ** -0.5 + 1.0 * lr
        label = "matches training length" if k == 1 else f"{k:g}× longer than training"
        ax_lr.plot(t, lr, color=col, label=label)
        ax_loss.plot(t, toy_loss, color=col)
        ax_loss.text(steps * 1.01, toy_loss[-1], f"{k:g}×", color=INK_2, fontsize=9.5, va="center")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_yticks([])
    ax_lr.set_title("Matching the learning-rate schedule to the training length")
    ax_lr.legend(title="Schedule length", title_fontsize=9.5, fontsize=9, loc="lower left")
    ax_lr.get_legend().get_title().set_color(INK_2)
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_yticks([])
    ax_loss.set_xlabel("Training step")
    ax_loss.set_xticks([])
    ax_loss.set_ylim(2.2, 3.9)
    ax_loss.annotate("a schedule that ends on time\nreaches the lowest loss",
                     xy=(steps * 0.99, 2.79), xytext=(steps * 0.35, 2.32),
                     fontsize=9, color=INK_2, ha="left",
                     arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1})
    fig.tight_layout(rect=(0, 0.04, 0.96, 1))
    credit(fig, "Schematic, not data. Based on the finding in Hoffmann et al. (2022), Figure A1.")
    fig.savefig(OUT / "chinchilla-cosine.png", dpi=DPI)
    plt.close(fig)


def fig_lr_scaling():
    def eta_opt(c):
        return 0.3118 * c ** -0.1250

    budgets = [1e17, 1e18, 1e19, 1e20]
    colors = [BLUE_RAMP[1], BLUE_RAMP[4], BLUE_RAMP[7], BLUE_RAMP[9]]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.9))

    # (a) learning-rate sweeps: minima from the DeepSeek fit, curve shapes illustrative
    for i, (c, col) in enumerate(zip(budgets, colors)):
        best = eta_opt(c)
        lr = np.logspace(np.log10(best) - 0.7, np.log10(best) + 0.7, 200)
        y = 3.2 - 0.25 * i + 0.9 * (np.log10(lr) - np.log10(best)) ** 2
        ax_a.plot(lr, y, color=col, label=fmt_flops(c))
        ax_a.plot(best, 3.2 - 0.25 * i, "o", ms=7, color=col,
                  markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax_a.set_xscale("log")
    ax_a.set_yticks([])
    ax_a.set_xlabel("Learning rate")
    ax_a.set_ylabel("Training loss")
    ax_a.set_title("(a) Sweep at each budget")
    ax_a.legend(title="Compute (FLOPs)", title_fontsize=9, fontsize=8.5, loc="upper center",
                ncol=2)
    ax_a.get_legend().get_title().set_color(INK_2)
    ax_a.set_ylim(2.3, 4.35)

    # (b) best learning rate vs. compute: fit on small budgets, extrapolate, check
    c_fit = np.logspace(17, np.log10(2e19), 50)
    c_ext = np.logspace(np.log10(2e19), 21, 50)
    ax_b.plot(c_fit, eta_opt(c_fit), color=SERIES_1)
    ax_b.plot(c_ext, eta_opt(c_ext), color=SERIES_1, linestyle=(0, (2, 2.5)))
    sweep = [1e17, 1e18, 1e19]
    ax_b.plot(sweep, [eta_opt(c) for c in sweep], "o", ms=7, color=SERIES_1,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax_b.plot(1e20, eta_opt(1e20), "*", ms=13, color=INK, markeredgecolor=SURFACE, zorder=3)
    ax_b.text(0.06, 0.18, "solid: fit on small budgets\ndashed: extrapolation", transform=ax_b.transAxes,
              fontsize=9, color=INK_2, ha="left")
    ax_b.annotate("checked at\na larger budget", (1e20, eta_opt(1e20)), xytext=(12, 22),
                  textcoords="offset points", fontsize=9, color=INK_2, ha="left")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Compute budget (FLOPs)")
    ax_b.set_ylabel("Best learning rate")
    ax_b.set_title("(b) Fit how the best value scales")
    ax_b.set_xlim(5e16, 1e21)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    credit(fig, "Best learning rates follow the fit reported in DeepSeek LLM (Bi et al., 2024), "
                "Figure 3.\nIn (a), only the positions of the minima come from that fit; "
                "the curve shapes are illustrative.")
    fig.savefig(OUT / "lr-scaling.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_isoflop()
    fig_projections()
    fig_cosine()
    fig_lr_scaling()
    print(f"Saved figures to {OUT}")
