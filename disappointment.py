"""Construct individualized disappointment gaps and user trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def count_positive_runs(values: np.ndarray) -> tuple[int, int]:
    positive = values > 0
    starts = positive & ~np.r_[False, positive[:-1]]
    run_ids = np.cumsum(starts)
    lengths = [int(np.sum(run_ids[positive] == run)) for run in np.unique(run_ids[positive])]
    return int(starts.sum()), max(lengths, default=0)


def summarize_user(posts: pd.DataFrame) -> pd.Series:
    posts = posts.sort_values("post_time")
    gap = (posts["expected_comments"] - posts["observed_comments"]).to_numpy(float)
    positive = np.clip(gap, 0, None)
    run_count, longest_run = count_positive_runs(gap)
    slope = float(np.polyfit(np.arange(len(gap)), gap, 1)[0]) if len(gap) > 1 else 0.0
    return pd.Series(
        {
            "disappointment_mean": gap.mean(),
            "disappointment_median": np.median(gap),
            "disappointment_maximum": gap.max(),
            "disappointment_cumulative_positive": positive.sum(),
            "disappointment_positive_share": (gap > 0).mean(),
            "disappointment_positive_runs": run_count,
            "disappointment_longest_positive_run": longest_run,
            "disappointment_slope": slope,
            "disappointment_variability": gap.std(ddof=0),
        }
    )


def build_trajectories(posts: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "post_time", "observed_comments", "expected_comments"}
    missing = required - set(posts.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return posts.groupby("user_id", sort=False).apply(summarize_user).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    trajectories = build_trajectories(pd.read_csv(args.input_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

