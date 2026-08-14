#!/usr/bin/env python3
"""What can still be had downstream of the encoder, without retraining one.

Everything here operates on embeddings already extracted, so it costs nothing
to run and nothing to repeat. That is the point: the interventions below are
each worth a fraction of the 1.6-point seed spread, which makes them
unaffordable to test one GPU run at a time and nearly free to test here.

Five things, in the order they are tried:

  baseline   Reproduce the published pipeline exactly, as a correctness check.
             If this does not match the archived number to the fourth decimal,
             nothing below it can be trusted.
  corn       Score the ordinal head that training produced and the pipeline
             then discarded. It is a linear map on the same pooled embedding
             the Ridge is fitted to, so it can be evaluated here directly. A
             missing baseline rather than an improvement.
  stack      Append the ordinal head's conditional logits to the Ridge input.
  concat     Concatenate two adapted encoders trained from different
             pretraining objectives, on the theory that their errors are
             partly decorrelated.
  cutpoints  Stop rounding the continuous Ridge output at the halfway marks.
             Those nine thresholds are convention, not a fit, and the paper's
             weakest per-class result (age 10 losing F1 to a precision drop)
             is the shape of problem a threshold can fix.

Every variant is scored on the same fixed test split as the paper, and the
cut-point fit uses a validation split carved out of training data only, never
the test set.

    uv run python scripts/analyze_squeeze.py
"""

import json
import sys
import warnings
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.splits import load_split_by_ids
from src.features.metadata import augment_embeddings

# numpy raises divide-by-zero, overflow and invalid-value warnings from inside
# RidgeCV's matmul on this data, thousands of them, and they are spurious: the
# same fit in float64 emits them too and returns bit-identical predictions and
# the same selected alpha, and the baseline below reproduces the archived
# accuracy to four decimals. The archived pipeline emitted them as well. They
# are silenced here so a real warning would be visible.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

EMB = Path("outputs/embeddings")
SPLIT = Path("outputs/splits/main_split.json")
METADATA = "cod_otolith_age_final_with_scale.csv"
SUMMARY = Path("outputs/squeeze.json")

ALPHAS = np.logspace(1.5, 3, 30)
CV = 5
AGE = (1, 10)
TABULAR = ["length", "month_sin", "month_cos", "shot_latitude", "shot_longitude",
           "is_survey", "seg_width_px", "seg_height_px", "seg_aspect_ratio"]
SEEDS = (42, 7, 13)


def load(name):
    """Embeddings plus the nine metadata columns, exactly as the paper builds them."""
    d = np.load(EMB / f"{name}.npz", allow_pickle=True)
    features = augment_embeddings(d["features"], d["measurement_ids"],
                                  METADATA, TABULAR)
    return features, d["labels"], d["measurement_ids"]


def fit_predict(features, labels, train, test):
    """The published two-stage readout: standardise, RidgeCV, and predict.

    Returns the continuous prediction rather than the rounded one, because
    every variant below needs the underlying value: rounding is the step this
    script is partly trying to replace.
    """
    scaler = StandardScaler()
    model = RidgeCV(alphas=ALPHAS, cv=CV)
    model.fit(scaler.fit_transform(features[train]), labels[train])
    return model.predict(scaler.transform(features[test]))


def score(y_true, y_pred):
    y_pred = np.clip(np.round(y_pred).astype(int), *AGE) if y_pred.dtype.kind == "f" else y_pred
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "accuracy_pm1": float((np.abs(y_pred - y_true) <= 1).mean()),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(np.abs(y_pred - y_true).mean()),
    }


def corn_logits(run_name):
    """Apply the saved ordinal head to the embeddings it was actually fitted on.

    The head is one linear layer over the pooled vector, so no encoder forward
    pass is needed. The catch is which pooled vector. `OrdinalHead.forward`
    calls `embed_images(..., normalize=False)` on purpose, so that the head can
    use embedding magnitude, while `extract_embeddings.py` normalises. Feeding
    the head a normalised vector is therefore a scale mismatch, and it does not
    fail loudly: the first version of this function did exactly that and
    returned a confident 3.5% accuracy, which reads like a finding about the
    ordinal head and is really a finding about my own bug.

    So this insists on the `unnorm` cache and returns None otherwise, rather
    than producing a number that would be wrong in a plausible-looking way.
    The cache comes from `extract_embeddings.py --no-normalize`.

    Note it takes the embedding without the tabular columns: the head never saw
    those, by design, since keeping metadata out of the encoder is what makes
    the attribution comparison interpretable.
    """
    head = Path("outputs/runs") / run_name / "corn_head.pt"
    cache = EMB / f"siglip2_clahe_{run_name}_unnorm.npz"
    if not head.exists() or not cache.exists():
        return None
    features = np.load(cache, allow_pickle=True)["features"]
    # weights_only: the file holds a two-tensor state dict and nothing else, so
    # there is no reason to let the unpickler construct arbitrary objects.
    state = torch.load(head, map_location="cpu", weights_only=True)
    return features @ state["weight"].float().numpy().T + state["bias"].float().numpy()


def corn_to_age(logits):
    """CORN decodes to a rank by multiplying conditional probabilities.

    Task k predicts P(y > k | y >= k), so the unconditional P(y > k) is the
    running product, and the predicted rank is how many of those exceed a half.
    This is the rank-consistent decoding; thresholding each task independently
    is not, and would let the model claim y > 5 while denying y > 3.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    cumulative = np.cumprod(probs, axis=1)
    return (cumulative > 0.5).sum(axis=1) + AGE[0]


def fit_cutpoints(raw, y, objective="f1"):
    """Choose the nine boundaries that map a continuous score onto ages.

    Rounding puts them at the halfway marks, which assumes the score is
    unbiased and equally dense on both sides of every boundary. Neither holds
    for a ridge fit on an unbalanced ordinal target: regression to the mean
    pulls predictions toward the middle ages, so the outer boundaries sit in
    the wrong place and the rare classes pay for it.

    Fitted coordinate-wise, each boundary scanned against the objective with
    the others held fixed, repeated until nothing moves. Monotonicity is
    enforced by construction, since each boundary is only allowed to move
    inside the interval its neighbours leave it.
    """
    cuts = np.arange(AGE[0], AGE[1]) + 0.5

    def apply(cuts, values):
        return np.searchsorted(cuts, values) + AGE[0]

    def objective_value(cuts):
        pred = apply(cuts, raw)
        return (f1_score(y, pred, average="macro", zero_division=0)
                if objective == "f1" else float((pred == y).mean()))

    best = objective_value(cuts)
    for _ in range(6):
        improved = False
        for i in range(len(cuts)):
            low = cuts[i - 1] + 1e-3 if i else raw.min() - 1.0
            high = cuts[i + 1] - 1e-3 if i + 1 < len(cuts) else raw.max() + 1.0
            if high <= low:
                continue
            for candidate in np.linspace(low, high, 60):
                trial = cuts.copy()
                trial[i] = candidate
                value = objective_value(trial)
                if value > best + 1e-9:
                    best, cuts, improved = value, trial, True
        if not improved:
            break
    return cuts


def main():
    split = None
    report = {}
    per_seed = {k: [] for k in ("baseline", "corn", "stack", "concat",
                                "concat_same_encoder",
                                "cutpoints_f1", "cutpoints_acc")}

    for seed in SEEDS:
        run = f"siglip2_lora_r16a32_s{seed}_clahe"
        features, labels, ids = load(f"siglip2_clahe_{run}")
        split = load_split_by_ids(str(SPLIT), ids)
        train, test = split.train_indices, split.test_indices
        y_test, y_train = labels[test], labels[train]

        continuous = fit_predict(features, labels, train, test)
        per_seed["baseline"].append(score(y_test, continuous))

        # --- the discarded ordinal head, and the same logits as extra features
        logits = corn_logits(run)
        if logits is not None:
            per_seed["corn"].append(score(y_test, corn_to_age(logits[test])))
            stacked = np.hstack([features, logits])
            per_seed["stack"].append(
                score(y_test, fit_predict(stacked, labels, train, test)))

        # --- two encoders side by side. DINOv3 is self-distilled where SigLIP2
        # is contrastive vision-language, so their errors should be partly
        # decorrelated and the pair should beat either alone.
        other = EMB / "dinov3_clahe_dinov3_lora_r16a32_s42_clahe.npz"
        if other.exists():
            d = np.load(other, allow_pickle=True)
            assert np.array_equal(d["measurement_ids"], ids), \
                "encoders list images in different orders; concatenation would misalign"
            joined = np.hstack([features, d["features"]])
            per_seed["concat"].append(
                score(y_test, fit_predict(joined, labels, train, test)))

        # --- the control that makes the above interpretable. Concatenating two
        # SigLIP2 runs doubles the width and the inference cost in exactly the
        # same way, but adds no pretraining diversity, only a different seed.
        # If this matches `concat` then the gain is width and averaging rather
        # than the thing the pairing was chosen for, and the interesting claim
        # evaporates.
        twin = next((t for t in SEEDS if t != seed), None)
        twin_path = EMB / f"siglip2_clahe_siglip2_lora_r16a32_s{twin}_clahe.npz"
        if twin_path.exists():
            t = np.load(twin_path, allow_pickle=True)
            assert np.array_equal(t["measurement_ids"], ids)
            per_seed["concat_same_encoder"].append(
                score(y_test, fit_predict(np.hstack([features, t["features"]]),
                                          labels, train, test)))

        # --- cut-points, fitted on held-out training data only.
        # Both objectives are reported because they genuinely disagree: moving
        # a boundary to win a rare class costs common-class accuracy, so a fit
        # against macro F1 and a fit against accuracy land in different places
        # and quoting whichever won would be picking after the fact.
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        sub_train, sub_val = next(inner.split(np.zeros(len(train)), y_train))
        val_pred = fit_predict(features, labels, train[sub_train], train[sub_val])
        for objective, key in (("f1", "cutpoints_f1"), ("accuracy", "cutpoints_acc")):
            cuts = fit_cutpoints(val_pred, y_train[sub_val], objective=objective)
            per_seed[key].append(
                score(y_test, np.searchsorted(cuts, continuous) + AGE[0]))
            report.setdefault(f"{key}_example", cuts.round(3).tolist())

    header = f"{"variant":<20} {'n':>2} {'accuracy':>9} {'d acc':>9} {'macro F1':>9} {'d F1':>8} {'MAE':>7}"
    print(header)
    print("-" * len(header))
    base_acc = base_f1 = None
    for name, rows in per_seed.items():
        if not rows:
            print(f"{name:<20} {'-':>2}  not computable from the caches present")
            continue
        acc = np.array([r["accuracy"] for r in rows])
        f1 = np.array([r["f1"] for r in rows])
        mae = np.array([r["mae"] for r in rows])
        if base_acc is None:
            base_acc, base_f1 = acc.mean(), f1.mean()
        d_acc = "reference" if name == "baseline" else f"{100 * (acc.mean() - base_acc):+.2f} pp"
        d_f1 = "" if name == "baseline" else f"{100 * (f1.mean() - base_f1):+.2f}"
        print(f"{name:<20} {len(rows):>2} {acc.mean():>9.4f} {d_acc:>9} "
              f"{f1.mean():>9.4f} {d_f1:>8} {mae.mean():>7.3f}")
        report[name] = {"n": len(rows), "accuracy": acc.mean(),
                        "accuracy_sd": float(acc.std(ddof=1)) if len(acc) > 1 else None,
                        "f1": f1.mean(), "mae": mae.mean(),
                        "delta_acc_pp": 100 * (acc.mean() - base_acc),
                        "delta_f1_pp": 100 * (f1.mean() - base_f1)}
    # Paired deltas, which are the right comparison for everything here. Each
    # variant is scored on the same seeds, the same split and the same test
    # images as the baseline it is measured against, so the 1.6-point marginal
    # seed spread is the wrong yardstick: it includes run-to-run variation that
    # both sides of the comparison share and which therefore cancels. What
    # matters is whether the sign is consistent across seeds.
    print("\nPaired against the same seed's baseline (accuracy, pp):")
    baselines = [r["accuracy"] for r in per_seed["baseline"]]
    for name, rows in per_seed.items():
        if name == "baseline" or not rows:
            continue
        deltas = [100 * (r["accuracy"] - b) for r, b in zip(rows, baselines)]
        agree = sum(d > 0 for d in deltas)
        print(f"  {name:<20} " + "  ".join(f"{d:+5.2f}" for d in deltas)
              + f"   mean {np.mean(deltas):+5.2f}   {agree}/{len(deltas)} positive")
        report[name]["paired_deltas_pp"] = deltas
        report[name]["seeds_favouring"] = agree

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWritten to {SUMMARY}")


if __name__ == "__main__":
    raise SystemExit(main())
