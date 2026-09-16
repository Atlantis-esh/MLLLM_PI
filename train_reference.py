"""Train the contextual seven-day comment reference described in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = {"user_id", "comments_7d"}


def fit_reference(data: pd.DataFrame):
    feature_names = [column for column in data.columns if column not in ID_COLUMNS]
    x = data[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = data["comments_7d"].to_numpy()

    train_index, validation_index = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2).split(
            x, y, groups=data["user_id"]
        )
    )
    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "gradient_boosting": GradientBoostingRegressor(),
        "random_forest": RandomForestRegressor(n_jobs=-1),
    }

    scores = {}
    for name, model in candidates.items():
        model.fit(x.iloc[train_index], y[train_index])
        prediction = model.predict(x.iloc[validation_index])
        scores[name] = {
            "spearman": float(spearmanr(y[validation_index], prediction)[0]),
            "r2": float(r2_score(y[validation_index], prediction)),
            "mae": float(mean_absolute_error(y[validation_index], prediction)),
        }

    selected_name = max(scores, key=lambda name: scores[name]["spearman"])
    selected_model = candidates[selected_name].fit(x, y)
    return {
        "model": selected_model,
        "feature_names": feature_names,
        "selected_model": selected_name,
        "validation_scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    artifact = fit_reference(pd.read_csv(args.input_csv))
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_model)
    print(json.dumps(artifact["validation_scores"], indent=2))


if __name__ == "__main__":
    main()
