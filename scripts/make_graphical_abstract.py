#!/usr/bin/env python3
"""Build the graphical abstract.

Ecological Informatics wants a single panel of about 13 by 5 cm that survives
being viewed at thumbnail size, so this carries exactly two claims and nothing
else:

  1. Adapting the encoder moves accuracy several times further than choosing
     between encoders does.
  2. The adapted embedding absorbs the otolith morphometric metadata, so the
     model needs only the image.

Every number is read from the archived result files rather than typed in, so
the figure cannot drift away from the manuscript. Run after `summarize_results.py`
has been fetched from the results repo.

    uv run python scripts/make_graphical_abstract.py
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

RESULTS = Path("results")
OUT = Path("outputs/paper_figures/graphical_abstract.png")

#: A young and an old otolith, chosen because the difference the model has to
#: read is actually visible in them at thumbnail size: the age-2 section is
#: nearly featureless, the age-10 section is densely banded.
OTOLITHS = [(2, Path("../otolith_data/otoliths_v1/2/23254826.jpg")),
            (10, Path("../otolith_data/otoliths_v1/10/15575452.jpg"))]

ENCODERS = ["clip", "siglip2", "dinov2", "dinov3"]
LABELS = {"clip": "CLIP", "siglip2": "SigLIP2", "dinov2": "DINOv2", "dinov3": "DINOv3"}

FROZEN, LORA, TUNED = "#4C72B0", "#C44E52", "#3F7D58"
INK, MUTED = "#22252A", "#6B7280"


def _find(pattern):
    hits = sorted(RESULTS.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def accuracy(run):
    path = _find(f"**/{run}/bootstrap/results.json")
    return json.loads(path.read_text())["aggregated_results"]["accuracy"]["mean"]


def tuned():
    """The best configuration from the squeeze experiments, or None.

    Read from the analysis output rather than typed in, on the same principle
    as every other number in this figure. Returns the mean over the three
    seeds, not the best of them: the seed-42 value is a point and a half
    higher, and putting that on a graphical abstract would be picking the
    luckiest run to advertise.
    """
    path = Path("outputs/squeeze.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())["augment_concat"]["accuracy"]


def forward_selection(run):
    """F1 with the embedding alone, and with every metadata feature added."""
    path = _find(f"**/{run}/bootstrap/feature_importance/feature_importance.json")
    steps = json.loads(path.read_text())["forward_selection"]
    return steps[0]["f1_mean"], steps[-1]["f1_mean"]


def clahe_preview(path, clip_limit=3.0, tile=25):
    """The image as the encoder sees it, which is also where the rings show."""
    image = np.array(Image.open(path).convert("L"))
    enhanced = cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=(tile, tile)).apply(image)
    return enhanced


def _header(ax, text):
    """Left-aligned panel headers. Centred titles collide at this aspect ratio."""
    ax.set_title(text, fontsize=7, color=INK, pad=5, loc="left",
                 fontweight="semibold")


def _spread(values, minimum_gap):
    """Nudge label positions apart, preserving order.

    CLIP and DINOv2 land 0.0015 apart after adaptation, which is the paper's
    point but makes their labels illegible. One upward pass then one downward
    pass is enough for four items.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + minimum_gap)
    return out


def panel_otolith(axes):
    """Two sections at opposite ends of the age range.

    Drawn with aspect="equal". These crops are roughly 2.1:1 and squashing them
    to fill a taller box would misrepresent the increment spacing, which is the
    one thing the reader is being asked to look at, and which Section 3.5 of the
    paper shows the model is sensitive to.
    """
    for ax, (age, path) in zip(axes, OTOLITHS):
        ax.imshow(clahe_preview(path), cmap="gray", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.985, 0.06, f"age {age}", transform=ax.transAxes, fontsize=5.8,
                color="white", ha="right", va="bottom")

    _header(axes[0], "8,619 cod otoliths")
    axes[1].set_xlabel("image only, no measurements", fontsize=5.8,
                       color=MUTED, labelpad=3)


def panel_gain(ax, frozen, lora, best=None):
    """Slope chart: every encoder makes the same large move."""
    for encoder in ENCODERS:
        ax.plot([0, 1], [frozen[encoder], lora[encoder]], "-o",
                color="#94A3B8", markersize=2.6, linewidth=1.0, zorder=2,
                markerfacecolor="white", markeredgewidth=1.0)

    values = [lora[e] for e in ENCODERS]
    for encoder, y in zip(ENCODERS, _spread(values, 0.0092)):
        ax.plot([1.03, 1.10], [lora[encoder], y], color="#CBD5E1",
                linewidth=0.5, zorder=1, clip_on=False)
        ax.text(1.13, y, LABELS[encoder], fontsize=5.6, color=INK, va="center")

    lo_f, hi_f = min(frozen.values()), max(frozen.values())
    lo_l, hi_l = min(lora.values()), max(lora.values())
    for x, lo, hi, colour in ((0, lo_f, hi_f, FROZEN), (1, lo_l, hi_l, LORA)):
        ax.plot([x, x], [lo, hi], color=colour, linewidth=4.5, alpha=0.35,
                solid_capstyle="round", zorder=1)

    # The comparison the panel exists to make: the vertical move against the
    # vertical scatter it has to beat.
    gain = np.mean([lora[e] - frozen[e] for e in ENCODERS])
    mid_f, mid_l = np.mean(list(frozen.values())), np.mean(list(lora.values()))
    ax.annotate("", xy=(0.46, mid_l), xytext=(0.46, mid_f),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(0.50, (mid_f + mid_l) / 2, f"+{100 * gain:.1f} pp\nadapting",
            fontsize=6.4, color=INK, va="center", ha="left", linespacing=1.2)
    ax.text(-0.06, (lo_f + hi_f) / 2,
            f"{100 * (hi_f - lo_f):.1f} pp\nchoosing", fontsize=6.4,
            color=FROZEN, va="center", ha="right", linespacing=1.2)

    # Drawn as a rule above the two columns rather than as a third column: the
    # panel is four centimetres wide and a third x position would collide with
    # the encoder labels. A rule also reads correctly, since this configuration
    # is one encoder pair rather than a fourth measurement of all four.
    if best is not None:
        ax.plot([-0.12, 1.10], [best, best], linestyle=(0, (3, 2)),
                color=TUNED, linewidth=0.9, zorder=3, clip_on=False)
        ax.text(-0.12, best + 0.0035,
                f"{best:.3f}  augmentation + second encoder",
                fontsize=5.4, color=TUNED, va="bottom", ha="left")

    ax.set_xlim(-0.5, 1.12)
    ax.set_ylim(0.552, 0.694)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["frozen", "LoRA"], fontsize=6.8, color=INK)
    ax.set_ylabel("accuracy", fontsize=6.8, color=INK, labelpad=2)
    ax.tick_params(axis="y", labelsize=5.8, colors=MUTED, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=2)
    _header(ax, "Adapting beats choosing")
    _clean(ax)


def panel_absorption(ax, fs_frozen, fs_lora):
    """Forward selection: what the metadata is still worth after adaptation."""
    alone_f, all_f = fs_frozen
    alone_l, all_l = fs_lora

    x = np.array([0.0, 1.0])
    width = 0.30
    ax.bar(x - 0.20, [alone_f, alone_l], width, color=[FROZEN, LORA],
           zorder=2)
    ax.bar(x + 0.20, [all_f, all_l], width, color=[FROZEN, LORA],
           alpha=0.28, zorder=2)

    for xi, a, b in zip(x, [alone_f, alone_l], [all_f, all_l]):
        ax.text(xi - 0.20, a + 0.008, f"{a:.3f}", ha="center",
                fontsize=6, color=INK)
        ax.text(xi + 0.20, b + 0.008, f"{b:.3f}", ha="center",
                fontsize=6, color=MUTED)
        ax.text(xi + 0.20, b - 0.032, f"+{100 * (b - a):.1f}", ha="center",
                fontsize=5.4, color="white")

    # The single most quotable comparison in the paper, placed over the frozen
    # group so it sits directly above the end of the line it is describing.
    ax.axhline(all_f, color=INK, linestyle=(0, (2.5, 2)), linewidth=0.8, zorder=3)
    ax.text(-0.37, 0.741, "adapted image alone\n> frozen + everything",
            fontsize=5.6, color=INK, va="top", ha="left", linespacing=1.35)

    # Bars carry their own labels on a minor tick row. A legend would have to
    # sit inside the plotting area at this size and there is no space for one.
    ax.set_xticks(x)
    ax.set_xticklabels(["frozen", "LoRA"], fontsize=6.8, color=INK)
    ax.set_xticks(np.concatenate([x - 0.20, x + 0.20]), minor=True)
    ax.set_xticklabels(["image", "image", "+ meta", "+ meta"], minor=True,
                       fontsize=5.0, color=MUTED)
    ax.tick_params(axis="x", which="minor", length=0, pad=1)
    ax.tick_params(axis="x", which="major", length=0, pad=11)
    ax.set_ylabel("macro F1", fontsize=6.8, color=INK, labelpad=2)
    ax.set_ylim(0.44, 0.745)
    ax.set_xlim(-0.40, 1.40)
    ax.tick_params(axis="y", labelsize=5.8, colors=MUTED, length=2, pad=1)
    _header(ax, "The embedding absorbs the metadata")
    _clean(ax)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D1D5DB")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    frozen = {e: accuracy(f"{e}-frozen") for e in ENCODERS}
    lora = {e: accuracy(f"{e}_lora_r16a32_s42_clahe") for e in ENCODERS}
    fs_frozen = forward_selection("siglip2-frozen")
    fs_lora = forward_selection("siglip2_lora_r16a32_s42_clahe")

    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=400)
    grid = GridSpec(1, 3, width_ratios=[0.86, 1.12, 1.30], figure=fig,
                    wspace=0.55, left=0.02, right=0.985, top=0.87, bottom=0.19)
    stack = grid[0].subgridspec(2, 1, hspace=0.12)

    panel_otolith([fig.add_subplot(stack[0]), fig.add_subplot(stack[1])])
    panel_gain(fig.add_subplot(grid[1]), frozen, lora, best=tuned())
    panel_absorption(fig.add_subplot(grid[2]), fs_frozen, fs_lora)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight",
                pad_inches=0.04)
    print(f"wrote {OUT}  ({Image.open(OUT).size[0]} x {Image.open(OUT).size[1]} px)")


if __name__ == "__main__":
    main()
