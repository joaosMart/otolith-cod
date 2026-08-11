#!/usr/bin/env python3
"""Can a saturating fit tell us how many otoliths would saturate the method?

Short answer: it can put a floor under the achievable accuracy and it cannot put
a ceiling on it, and no affordable number of extra runs changes that. This
script is the evidence for that claim, kept in the repository because the
negative result is the reason the paper reports a bound rather than a number.

Two experiments:

  1. Fix a candidate asymptote, fit the remaining two parameters to the four
     measured SigLIP2 points, and look at the residuals. Asymptotes anywhere
     from 0.68 to 0.90 fit to within a residual standard deviation of roughly
     0.5 to 2 parts per thousand, against a run-to-run standard deviation of
     16 parts per thousand. The candidate curves differ by less than the noise
     on a single run, so the observations cannot separate them.

  2. Simulate the design that was actually run, six training-set fractions at
     three seeds each, from each of those candidate truths, and refit. The
     recovered 95% interval runs to the upper bound in every case.

The reason is structural rather than a shortage of data. Every observation sits
at or below 5,860 images, which is inside the regime where the curve is still
close to linear in log n, and the asymptote is a property of the part of the
curve that has not been observed. Extrapolating it requires assuming the
functional form is exactly right far outside the measured range.

    uv run python scripts/check_asymptote_identifiability.py
"""

import numpy as np
from scipy.optimize import curve_fit

# The four SigLIP2 points from the first learning-curve pass.
OBS_N = np.array([586.0, 1465.0, 2930.0, 5860.0])
OBS_ACC = np.array([0.6118, 0.6295, 0.6411, 0.6546])

SEED_SD = 0.0161          # measured across five seeds at the full training set
FT_TRAIN = 5860
FRACTIONS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS_PER_POINT = 3
N_SIMULATIONS = 400
CANDIDATES = (0.68, 0.70, 0.72, 0.75, 0.80, 0.90)


def saturating(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_given_asymptote(a):
    """Best (b, c) for a fixed asymptote, and the residual spread it leaves."""
    curve = lambda n, b, c: a - b * np.power(n, -c)  # noqa: E731
    (b, c), _ = curve_fit(curve, OBS_N, OBS_ACC, p0=[1.0, 0.4], maxfev=20000)
    residual_sd = float(np.sqrt(np.mean((curve(OBS_N, b, c) - OBS_ACC) ** 2)))
    return b, c, residual_sd


def recover(truth, rng):
    """Fit the queued design once, simulated from `truth`."""
    ns, accs = [], []
    for fraction in FRACTIONS:
        n = int(fraction * FT_TRAIN)
        for _ in range(SEEDS_PER_POINT):
            ns.append(n)
            accs.append(saturating(n, *truth) + rng.normal(0, SEED_SD))
    try:
        popt, _ = curve_fit(saturating, np.array(ns, float), np.array(accs, float),
                            p0=[0.72, 1.0, 0.35],
                            bounds=([max(accs), 1e-6, 0.01], [1.0, 1e4, 3.0]),
                            maxfev=20000)
        return popt[0]
    except Exception:
        return None


def main():
    rng = np.random.default_rng(11)

    print("Fit to the four measured points, holding the asymptote fixed:")
    print(f"  (run-to-run standard deviation for reference: {1000 * SEED_SD:.1f}e-3)")
    truths = {}
    for a in CANDIDATES:
        b, c, sd = fit_given_asymptote(a)
        truths[a] = (a, b, c)
        print(f"  asymptote {a:.2f}:  b={b:6.2f}  c={c:.3f}   "
              f"residual sd {1000 * sd:5.2f}e-3")

    print(f"\nRefitting the queued design ({len(FRACTIONS)} fractions x "
          f"{SEEDS_PER_POINT} seeds), {N_SIMULATIONS} simulations each:")
    for a, truth in truths.items():
        draws = [d for d in (recover(truth, rng) for _ in range(N_SIMULATIONS))
                 if d is not None]
        draws = np.array(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"  true {a:.2f} -> median {np.median(draws):.3f}, "
              f"95% [{lo:.3f}, {hi:.3f}], "
              f"{100 * np.mean(draws > 0.995):.0f}% run to the bound")

    print("\nConclusion: the lower end of the interval is stable and informative;"
          "\nthe upper end is the optimiser's bound in every scenario. Report the"
          "\nfloor, not a point estimate, and do not quote a sample size for"
          "\nreaching peak performance.")


if __name__ == "__main__":
    main()
