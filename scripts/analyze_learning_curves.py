#!/usr/bin/env python3
"""Learning curves with error bars, and a saturating fit with its uncertainty.

The first pass measured one run per point, which supports a picture but not an
error bar. This reads every (encoder, fraction, seed) run present in `results/`,
so it works on whatever has landed so far, and does three things:

  1. Plots accuracy against labelled images with the seed spread shown.
  2. Fits the standard scaling form  acc(n) = a - b * n**(-c),  where `a` is the
     accuracy the method would approach given unlimited labels.
  3. Reports a one-sided lower bound on `a` by resampling seeds within each
     fraction, and reports how often the fit runs to its upper bound.

The one-sided bound is deliberate, and `check_asymptote_identifiability.py` is
the evidence for it. Asymptotes anywhere between 0.68 and 0.90 fit the measured
points to within a residual standard deviation an order of magnitude smaller
than the run-to-run standard deviation, so the observations cannot separate
them, and a simulated refit of this exact design runs to the upper bound in a
quarter to a half of replicates whatever the truth is. Every observation sits at
n <= 5,860, which is inside the regime where the curve is still close to linear
in log n, so the asymptote is a property of the part of the curve nobody has
seen. What the data do support is a floor: the method reaches at least a certain
accuracy given enough labels. This script therefore prints that floor and
refuses to print a sample size for reaching peak performance.

The crossing point, the number of adapted labels that matches the frozen
encoder trained on everything, is better behaved but not automatically safe: on
the first pass it landed below the smallest run, so it too was an extrapolation.
The script checks and, when that happens, reports the inequality (fewer than the
smallest measured size) instead of a fitted figure.

    uv run python scripts/analyze_learning_curves.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

RESULTS = Path("results")
FIGURES = Path("outputs/paper_figures")
SUMMARY = Path("outputs/learning_curves.json")

FT_TRAIN = 5860          # images available for adaptation at fraction 1.0
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(20260811)

COLORS = {"siglip2": "#C44E52", "clip": "#4C72B0",
          "dinov2": "#55A868", "dinov3": "#8172B2"}
NAMES = {"siglip2": "SigLIP2", "clip": "CLIP",
         "dinov2": "DINOv2", "dinov3": "DINOv3"}

RUN = re.compile(r"^(?P<enc>clip|siglip2|dinov2|dinov3)_lora_r16a32"
                 r"(?:_f(?P<frac>[\d.]+))?_s(?P<seed>\d+)_clahe$")


def collect():
    """Every LoRA run at rank 16, keyed by encoder, then training-set size."""
    points = defaultdict(lambda: defaultdict(dict))
    for path in RESULTS.glob("**/bootstrap/results.json"):
        m = RUN.match(path.parent.parent.name)
        if not m:
            continue
        frac = float(m["frac"]) if m["frac"] else 1.0
        n = int(round(frac * FT_TRAIN))
        acc = json.loads(path.read_text())["aggregated_results"]["accuracy"]["mean"]
        points[m["enc"]][n][int(m["seed"])] = acc
    return points


def frozen_baselines():
    out = {}
    for encoder in NAMES:
        hits = sorted(RESULTS.glob(f"**/{encoder}-frozen/bootstrap/results.json"))
        if hits:
            out[encoder] = json.loads(
                hits[0].read_text())["aggregated_results"]["accuracy"]["mean"]
    return out


def saturating(n, a, b, c):
    """Accuracy approaches `a` from below as n grows; b, c set how fast."""
    return a - b * np.power(n, -c)


def fit(ns, accs):
    """Least squares with bounds that keep the parameters interpretable."""
    try:
        popt, _ = curve_fit(
            saturating, np.asarray(ns, float), np.asarray(accs, float),
            p0=[0.72, 1.0, 0.35],
            bounds=([max(accs), 1e-6, 0.01], [1.0, 1e4, 3.0]),
            maxfev=20000)
        return popt
    except Exception:
        return None


def bootstrap_asymptote(by_n):
    """Resample seeds within each training-set size, refit, collect `a`.

    Seeds are the unit of replication here: the split is held fixed across runs,
    so what varies between them is training stochasticity, and that is the
    uncertainty the interval should carry.
    """
    sizes = sorted(by_n)
    draws = []
    for _ in range(N_BOOTSTRAP):
        ns, accs = [], []
        for n in sizes:
            values = list(by_n[n].values())
            ns.append(n)
            accs.append(RNG.choice(values))
        popt = fit(ns, accs)
        if popt is not None:
            draws.append(popt[0])
    return np.array(draws)


def fit_quality(by_n, popt):
    """Residual spread of the fitted curve against the per-point means.

    Worth reporting separately from the parameter uncertainty. The measured CLIP
    curve is not smoothly concave: it moves in steps, with pairs of adjacent
    training-set sizes landing on top of each other and jumps between them. A
    saturating curve cannot reproduce that, so the residuals are structured
    rather than random, and a fit can look confident while systematically
    missing the shape. If the residual spread here is comparable to or larger
    than the seed spread, the functional form is the limitation and the
    asymptote should not be quoted at all.
    """
    sizes = sorted(by_n)
    means = np.array([np.mean(list(by_n[n].values())) for n in sizes])
    predicted = saturating(np.array(sizes, float), *popt)
    residual_sd = float(np.sqrt(np.mean((means - predicted) ** 2)))
    within = [np.ptp(list(by_n[n].values())) for n in sizes if len(by_n[n]) > 1]
    return {
        "residual_sd_vs_point_means": residual_sd,
        "mean_within_point_range": float(np.mean(within)) if within else None,
        "residuals": (means - predicted).tolist(),
    }


def crossing(popt, target, lo=10, hi=10_000_000):
    """Smallest n whose fitted accuracy reaches `target`."""
    a, b, c = popt
    if a <= target:
        return None
    return float(np.power(b / (a - target), 1.0 / c))


def main():
    points, frozen = collect(), frozen_baselines()
    if not points:
        raise SystemExit("no learning-curve runs found under results/")

    report = {}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    order = list(NAMES)
    for encoder in sorted(points, key=lambda e: order.index(e)
                          if e in order else 99):
        by_n = points[encoder]
        sizes = sorted(by_n)
        means = np.array([np.mean(list(by_n[n].values())) for n in sizes])
        # Full range across seeds, not a standard deviation: at three runs a
        # standard deviation is a noisy estimate and the range is honest about
        # how little is behind it.
        lo = np.array([min(by_n[n].values()) for n in sizes])
        hi = np.array([max(by_n[n].values()) for n in sizes])
        counts = [len(by_n[n]) for n in sizes]

        colour = COLORS.get(encoder, "#666666")
        name = NAMES.get(encoder, encoder)
        ax.errorbar(sizes, means, yerr=[means - lo, hi - means], marker="o",
                    capsize=3, lw=1.6, color=colour, label=f"{name} + LoRA",
                    zorder=3)

        flat = [(n, a) for n in sizes for a in by_n[n].values()]
        popt = fit([n for n, _ in flat], [a for _, a in flat])
        entry = {"n": sizes, "mean": means.tolist(), "min": lo.tolist(),
                 "max": hi.tolist(), "runs_per_point": counts}

        # The bootstrap resamples seeds, so it is meaningless until most points
        # have more than one run. With one run each every replicate is the same
        # fit, and the interval collapses to a point that looks like precision.
        replicated = sum(1 for n in sizes if len(by_n[n]) > 1) >= len(sizes) - 1

        if popt is not None and len(sizes) >= 4 and replicated:
            # Drawn only across the measured range. An earlier version ran the
            # fitted curve out to 10^5 images, which draws the eye to exactly
            # the extrapolation the identifiability check says is unsupported:
            # a reader would take the visual convergence of those tails for a
            # measurement of where each encoder ends up.
            # The fit is not drawn. It is still computed, because the crossing
            # point comes from it, but plotting it would put a smooth model on
            # top of data the model demonstrably does not describe, and the
            # asymptote it implies is the one thing this analysis declines to
            # report. The measured points and their spread carry the figure.
            draws = bootstrap_asymptote(by_n)
            at_bound = float(np.mean(draws > 0.995)) if len(draws) else float("nan")
            entry["fit"] = {
                "asymptote_point_estimate": float(popt[0]),
                "asymptote_lower_bound_95": float(np.percentile(draws, 5))
                                            if len(draws) else None,
                "fraction_of_fits_at_upper_bound": at_bound,
                "asymptote_upper_bound_identified": bool(at_bound < 0.05),
                "b": float(popt[1]), "c": float(popt[2]),
                "n_bootstrap_converged": int(len(draws)),
                **fit_quality(by_n, popt),
            }
            if encoder in frozen:
                n_match = crossing(popt, frozen[encoder])
                entry["fit"]["n_to_match_frozen"] = n_match
                # Say so when this falls below the smallest run: the curve was
                # never measured there, so the figure is a shape assumption
                # rather than an observation, and only the inequality is safe.
                entry["fit"]["n_to_match_frozen_extrapolated"] = \
                    bool(n_match is not None and n_match < min(sizes))

        if encoder in frozen:
            # Four separate legend entries for the frozen baselines pushed the
            # legend over the data without saying anything the colours do not.
            ax.axhline(frozen[encoder], ls="--", lw=1.0, color=colour, alpha=0.55,
                       zorder=1)
            entry["frozen"] = frozen[encoder]
        report[encoder] = entry

    ax.set_xscale("log")
    ticks = sorted({n for e in report for n in report[e]["n"]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Labelled images used for adaptation (log scale)")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="--", lw=1.0, color="#888888"))
    labels.append(f"same encoder frozen, all {FT_TRAIN:,}")
    ax.legend(handles, labels, fontsize=7.5, loc="lower right", frameon=True,
              framealpha=0.92, ncols=2, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "learning_curves.png", dpi=300)
    plt.close(fig)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(report, indent=2))

    for encoder, entry in report.items():
        print(f"\n{NAMES.get(encoder, encoder)}")
        for n, m, c in zip(entry["n"], entry["mean"], entry["runs_per_point"]):
            print(f"  n={n:5d}  acc={m:.4f}  ({c} run{'s' if c != 1 else ''})")
        f = entry.get("fit")
        if not f:
            singles = sum(1 for c in entry["runs_per_point"] if c == 1)
            print(f"  no fit: needs at least 4 training-set sizes with more than"
                  f" one run each ({len(entry['n'])} sizes, {singles} of them"
                  " single-run)")
            continue
        rsd, within = f["residual_sd_vs_point_means"], f["mean_within_point_range"]
        if within and rsd > within:
            print(f"  the saturating form does not describe this curve: residual"
                  f" sd {rsd:.4f} exceeds the mean within-point range {within:.4f},"
                  "\n    so the misfit is shape, not noise. Asymptote not reported.")
            continue
        bound = f["asymptote_lower_bound_95"]
        print(f"  asymptote: at least {bound:.3f} with 95% confidence"
              if bound is not None else "  asymptote: bootstrap did not converge")
        if not f["asymptote_upper_bound_identified"]:
            print(f"  no upper bound: {100 * f['fraction_of_fits_at_upper_bound']:.0f}%"
                  " of bootstrap fits ran to the constraint boundary,"
                  "\n    so the data are consistent with substantially more headroom")
        if f.get("n_to_match_frozen"):
            smallest = min(entry["n"])
            if f.get("n_to_match_frozen_extrapolated"):
                print(f"  adapting matches the frozen encoder trained on all "
                      f"{FT_TRAIN:,} images at fewer than {smallest:,} images"
                      f"\n    (fitted {f['n_to_match_frozen']:,.0f}, but that is "
                      "below the smallest run, so report the inequality)")
            else:
                print(f"  adapting matches the frozen encoder trained on all "
                      f"{FT_TRAIN:,} images at n = {f['n_to_match_frozen']:,.0f}")


if __name__ == "__main__":
    main()
