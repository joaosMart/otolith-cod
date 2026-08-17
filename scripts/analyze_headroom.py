#!/usr/bin/env python3
"""How much room is left, and is the test set honest?

Two questions that the learning curves raised but could not answer, both
answerable from results already on disk.

**Where the remaining errors are.** The curves are flat over their last segment
for every encoder and for full fine-tuning, so the method has stopped
responding to labelled otoliths. That invites the reading that the approach has
failed to reach some attainable accuracy. It has not: the errors that remain
are almost all adjacent-year, and adjacent-year is where the readers who
produced the labels disagree with each other. This script quantifies both
halves, exact against within-one-year, and positions the result against the
ICES exchange figures rather than against a hypothetical perfect reader.

**Whether slide identity leaks.** Images are crops from photographs holding
several otoliths, and fish caught at one station share a photograph, an
illumination, a mounting and a section thickness. The split is stratified by
age at the fish level, so those groups straddle it. A model could in principle
read the station rather than the otolith. The check here is not whether groups
straddle the split, which they do, but how much age information group identity
carries at all: a predictor given the exact station and the mean age of its
training members is an upper bound on what any model could extract from that
channel, and it is barely above the marginal baseline.

    uv run python scripts/analyze_headroom.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
SUMMARY = Path("outputs/headroom.json")
METADATA = Path("cod_otolith_age_final_with_scale.csv")
AGE_RANGE = (1, 10)

#: Verified from the full text of each source. Percent agreement with modal age,
#: which is the only human comparator that exists for cod. Campana 2001 contains
#: no percent-agreement figure for cod and argues against the statistic; the
#: Campana 2023 Icelandic numbers are coefficients of variation, not agreement.
#: No within-one-year agreement figure could be found for Atlantic cod in any
#: ICES exchange, so the model's within-one-year number is reported without a
#: human comparator rather than against an invented one.
ICES = {
    "North Sea 2005/06, 21 readers, 108 otoliths": 0.74,
    "North Sea 2009/10, 17 readers, 120 otoliths": 0.66,
    "North Sea 2009/10, 10 assessment readers": 0.76,
    "Baltic SD 22 2015, 4 readers, 180 otoliths": 0.919,
}
ICES_PER_READER_RANGE = (0.47, 0.76)   # 2009/10 exchange, per-reader spread
ICES_TARGET = 0.90                     # the exchange's own stated target

CONDITIONS = {
    "SigLIP2 frozen": r"siglip2-frozen",
    "SigLIP2 + LoRA": r"siglip2_lora_r16a32_s\d+_clahe",
    "SigLIP2 + full fine-tuning": r"siglip2_full_s\d+_clahe",
    "DINOv3 + LoRA": r"dinov3_lora_r16a32_s\d+_clahe",
}


def agreement():
    """Exact and within-one-year accuracy for each condition, over its seeds."""
    out = {}
    for label, pattern in CONDITIONS.items():
        exact, pm1 = [], []
        for path in RESULTS.glob("**/bootstrap/results.json"):
            if not re.fullmatch(pattern, path.parent.parent.name):
                continue
            metrics = json.loads(path.read_text())["aggregated_results"]
            exact.append(metrics["accuracy"]["mean"])
            pm1.append(metrics["accuracy_pm1"]["mean"])
        if exact:
            out[label] = {
                "n_runs": len(exact),
                "exact": float(np.mean(exact)),
                "exact_sd": float(np.std(exact, ddof=1)) if len(exact) > 1 else None,
                "within_1yr": float(np.mean(pm1)),
                "within_1yr_sd": float(np.std(pm1, ddof=1)) if len(pm1) > 1 else None,
            }
    return out


def split_file():
    hits = sorted(RESULTS.glob("**/splits/main_split.json"))
    if not hits:
        raise SystemExit("no split file found under results/")
    return json.loads(hits[0].read_text())


def grouping():
    """How much age information slide and station identity carry.

    The bound is the important part. `icc` says how clustered age is within a
    group, but a high value is not by itself evidence of leakage: it also rises
    when groups are small, and it says nothing about whether a model could
    recover group identity from the pixels. `groupmate_mean_*` is the operative
    number. It hands a predictor the exact group, which is strictly more than
    any model gets, and lets it predict the mean age of that group's training
    members. Whatever that scores is the ceiling on this channel.
    """
    split = split_file()
    test_ids = set(split["test_measurement_ids"])
    keep = test_ids | set(split["train_measurement_ids"])

    df = pd.read_csv(METADATA)
    df = df[df.measurement_id.isin(keep)].copy()
    df["age"] = df.age.clip(*AGE_RANGE)
    df["split"] = np.where(df.measurement_id.isin(test_ids), "test", "train")
    train, test = df[df.split == "train"], df[df.split == "test"]

    # Always predicting the training mode is the floor any real signal must beat.
    mode = train.age.mode()[0]
    report = {
        "n_train": len(train), "n_test": len(test),
        "marginal_mode": int(mode),
        "marginal_accuracy": float((test.age == mode).mean()),
    }

    for column, name in (("image__content__url", "photograph"),
                         ("station_id", "station")):
        # Singleton groups have zero within-group variance by construction, so
        # an ICC over all groups is inflated by however many groups hold one
        # fish. Restricting to groups of two or more is the honest version.
        sizes = df.groupby(column).size()
        multi = df[df[column].isin(sizes[sizes > 1].index)]
        within = multi.groupby(column)["age"].transform("mean")
        icc = 1 - ((multi.age - within) ** 2).mean() / multi.age.var()

        straddling = df.groupby(column)["split"].nunique()
        shared = set(straddling[straddling > 1].index)

        means = train.groupby(column)["age"].mean()
        predicted = test[column].map(means)
        covered = predicted.notna()
        hit = np.clip(predicted[covered].round(), *AGE_RANGE)
        actual = test.age[covered]

        report[name] = {
            "n_groups": int(df[column].nunique()),
            "median_group_size": float(sizes.median()),
            "max_group_size": int(sizes.max()),
            "icc_multi_member_groups": float(icc),
            "test_sharing_a_group_with_train": float(test[column].isin(shared).mean()),
            "groupmate_mean_coverage": float(covered.mean()),
            "groupmate_mean_accuracy": float((hit == actual).mean()),
            "groupmate_mean_mae": float(np.abs(hit - actual).mean()),
        }
    return report


def main():
    report = {"agreement": agreement(), "grouping": grouping(),
              "ices_reference": ICES}

    print("Exact and within-one-year accuracy")
    print(f"  {'condition':<28} {'runs':>5} {'exact':>8} {'+-1 yr':>9}")
    for label, row in report["agreement"].items():
        print(f"  {label:<28} {row['n_runs']:>5} {row['exact']:>8.4f} "
              f"{row['within_1yr']:>9.4f}")

    lora = report["agreement"].get("SigLIP2 + LoRA")
    frozen = report["agreement"].get("SigLIP2 frozen")
    if lora and frozen:
        print(f"\n  Adaptation moves exact agreement by "
              f"{100 * (lora['exact'] - frozen['exact']):+.1f} points but "
              f"within-one-year by only "
              f"{100 * (lora['within_1yr'] - frozen['within_1yr']):+.1f}.")
        print(f"  {100 * (1 - lora['within_1yr']):.1f}% of test otoliths are "
              "missed by more than a year, so essentially all the remaining\n"
              "  headroom sits inside the band where readers disagree with each "
              "other.")
        low, high = ICES_PER_READER_RANGE
        inside = low <= lora["exact"] <= high
        print(f"  Exact agreement {lora['exact']:.3f} is "
              f"{'inside' if inside else 'outside'} the ICES per-reader range "
              f"{low:.2f} to {high:.2f}, and below the {ICES_TARGET:.0%} target.")

    g = report["grouping"]
    print(f"\nGroup structure (marginal baseline {g['marginal_accuracy']:.3f}, "
          f"always age {g['marginal_mode']})")
    for name in ("photograph", "station"):
        row = g[name]
        print(f"  {name}: {row['n_groups']:,} groups, max {row['max_group_size']}; "
              f"{row['test_sharing_a_group_with_train']:.1%} of test shares one "
              "with train")
        print(f"    ICC {row['icc_multi_member_groups']:.3f}, but knowing the "
              f"group exactly scores only {row['groupmate_mean_accuracy']:.3f} "
              f"({row['groupmate_mean_accuracy'] - g['marginal_accuracy']:+.3f} "
              "on the baseline)")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {SUMMARY}")


if __name__ == "__main__":
    raise SystemExit(main())
