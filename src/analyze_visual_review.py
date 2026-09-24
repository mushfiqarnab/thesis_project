"""Analyze blinded visual-review ratings without fabricating incomplete evidence."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


RATING_COLUMNS = [
    "anatomical_plausibility",
    "boundary_naturalness",
    "pigmentation_or_vascular_naturalness",
    "texture_and_maturation_consistency",
    "unintended_change_free",
    "overall_visual_plausibility",
]


def validate_ratings(frame: pd.DataFrame) -> None:
    required = {"rater_id", "review_id", *RATING_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ratings file missing columns: {missing}")
    if frame[["rater_id", "review_id"]].isna().any().any():
        raise ValueError("Ratings contain missing rater_id or review_id values.")
    if frame.duplicated(["rater_id", "review_id"]).any():
        raise ValueError("A rater rated the same review_id more than once.")
    for column in RATING_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(1, 5).all():
            raise ValueError(f"{column} must contain complete integer ratings from 1 to 5.")
        if not np.all(values == values.astype(int)):
            raise ValueError(f"{column} contains non-integer ratings.")

    raters = frame["rater_id"].nunique()
    items = frame["review_id"].nunique()
    if raters < 2:
        raise ValueError("At least two independent raters are required for agreement analysis.")
    counts = frame.groupby("review_id")["rater_id"].nunique()
    if not (counts == raters).all():
        raise ValueError("Every review_id must be rated by every rater.")
    if items < 2:
        raise ValueError("At least two review items are required.")


def pairwise_agreement(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    results = {}
    for column in RATING_COLUMNS:
        pair_values = []
        for left, right in combinations(sorted(frame["rater_id"].unique()), 2):
            left_frame = frame[frame["rater_id"] == left].set_index("review_id")[column]
            right_frame = frame[frame["rater_id"] == right].set_index("review_id")[column]
            pair_values.append(cohen_kappa_score(left_frame, right_frame, weights="quadratic"))
        results[column] = {
            "mean_quadratic_weighted_kappa": float(np.mean(pair_values)),
            "min_quadratic_weighted_kappa": float(np.min(pair_values)),
            "pair_count": len(pair_values),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--answer_key", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings)
    answer_key = pd.read_csv(args.answer_key)
    validate_ratings(ratings)
    required_key = {"review_id", "morphology", "split"}
    missing_key = sorted(required_key - set(answer_key.columns))
    if missing_key:
        raise ValueError(f"Answer key missing columns: {missing_key}")
    if set(ratings["review_id"]) != set(answer_key["review_id"]):
        raise ValueError("Ratings and answer key review_id sets do not match.")

    numeric = ratings.copy()
    for column in RATING_COLUMNS:
        numeric[column] = pd.to_numeric(numeric[column])
    item_summary = numeric.groupby("review_id")[RATING_COLUMNS].agg(["mean", "std"])
    joined = numeric.merge(answer_key[["review_id", "morphology", "split"]], on="review_id", validate="many_to_one")
    morphology_summary = joined.groupby("morphology")[RATING_COLUMNS].mean()
    split_summary = joined.groupby("split")[RATING_COLUMNS].mean()

    report = {
        "rater_count": int(ratings["rater_id"].nunique()),
        "item_count": int(ratings["review_id"].nunique()),
        "rating_scale": "1-5 ordinal",
        "agreement": pairwise_agreement(numeric),
        "overall_means": numeric[RATING_COLUMNS].mean().to_dict(),
        "morphology_means": morphology_summary.to_dict(orient="index"),
        "split_means": split_summary.to_dict(orient="index"),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    item_summary.to_csv(output.with_name(output.stem + "_item_summary.csv"))
    print(f"Saved review report: {output}")
    print(f"Raters: {report['rater_count']}")
    print(f"Items: {report['item_count']}")
    print("Agreement computed with quadratic-weighted Cohen kappa.")


if __name__ == "__main__":
    main()