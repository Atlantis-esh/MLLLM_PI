"""Controlled LightGBM comparison with and without disappointment trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
    }


def compare(data: pd.DataFrame) -> pd.DataFrame:
    excluded = {"user_id", "churned"}
    trajectory = [column for column in data if column.startswith("disappointment_")]
    baseline = [column for column in data if column not in excluded and column not in trajectory]
    if not trajectory:
        raise ValueError("No disappointment_ trajectory columns were found")

    y = data["churned"].to_numpy(int)
    folds = StratifiedKFold(n_splits=5, shuffle=True)
    rows = []
    for fold, (train_index, test_index) in enumerate(folds.split(data, y), start=1):
        for specification, columns in {
            "baseline": baseline,
            "baseline_plus_disappointment": baseline + trajectory,
        }.items():
            x = data[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
            model = lgb.LGBMClassifier(objective="binary", verbosity=-1)
            model.fit(x.iloc[train_index], y[train_index])
            probability = model.predict_proba(x.iloc[test_index])[:, 1]
            rows.append({"fold": fold, "specification": specification, **metrics(y[test_index], probability)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    result = compare(pd.read_csv(args.input_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.groupby("specification").agg(["mean", "std"]))


if __name__ == "__main__":
    main()

