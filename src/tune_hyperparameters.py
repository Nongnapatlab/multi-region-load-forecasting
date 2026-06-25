"""
Hyperparameter tuning for XGBoost and LightGBM, per zone, using Optuna.

WHY THIS IS A SEPARATE SCRIPT (not part of main.py):
Tuning runs dozens of training trials per zone per model -- far too slow
to run on every scheduled daily pipeline run. This script is meant to be
run occasionally (e.g. every few months, or whenever you suspect the
data's underlying patterns have shifted enough to warrant re-tuning), and
its output is a JSON file of best-found parameters per zone. main.py does
NOT call this script automatically; you decide when to re-tune and when
to apply the results to config.py.

WHY TIME-BASED CROSS-VALIDATION (not a single train/val split):
A single split (e.g. "last 20% of training data as validation") is what
caused the LSTM regression we found and fixed earlier in this project --
load trends upward year over year, so a single recent validation slice
sits at a different demand level than the rest of training, and tuning
against it would optimize for "whatever made that one slice's loss low"
rather than genuinely better generalization. TimeSeriesSplit instead
creates multiple folds, each training on an expanding window of the past
and validating on the *next* contiguous block (never shuffled, never
looking into the future), and averages the validation score across all
folds. This is slower than tuning on the real test set directly, but the
real test set only has ~22 realized rows per zone -- far too few and too
easy to overfit to if used as the tuning objective (this would also be a
direct form of data leakage: using the test set to choose
hyperparameters that affect test-set performance).

WHY PER-ZONE (not one shared hyperparameter set):
feature_importance.csv showed zones behave quite differently already
(e.g. CAC relies heavily on weekly lags and has unusually weak
temperature correlation vs. other zones, which lean on daily lags and
respond much more to weather). A single hyperparameter set is a
reasonable default but is unlikely to be optimal for every zone
simultaneously.

Usage:
    python src/tune_hyperparameters.py --zone CAC --model xgb --trials 50
    python src/tune_hyperparameters.py --zone all --model both --trials 50
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from config import CACHE_DIR, LOGS_DIR, RANDOM_STATE, ZONES
from data_loader import load_zone_data
from feature_engineering import build_features
from preprocess import prepare_features

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

N_CV_SPLITS = 5  # number of expanding-window folds; see module docstring


def time_series_cv_mape(model_class, params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    """Average MAPE across N_CV_SPLITS expanding-window time-series folds.
    Never shuffles; each fold's validation block is strictly after its
    training block in time."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    fold_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_fit, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fit, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**params)
        model.fit(X_fit, y_fit)
        preds = model.predict(X_val)

        # Guard against near-zero actuals in a validation fold blowing up
        # MAPE the same way we guard for it in metrics.py for the real
        # test set.
        valid_mask = np.abs(y_val) >= 1.0
        if valid_mask.sum() == 0:
            continue
        score = mean_absolute_percentage_error(y_val[valid_mask], preds[valid_mask]) * 100
        fold_scores.append(score)

    if not fold_scores:
        return float("inf")
    return float(np.mean(fold_scores))


def make_xgb_objective(X: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        return time_series_cv_mape(XGBRegressor, params, X, y)
    return objective


def make_lgbm_objective(X: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "objective": "regression",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }
        return time_series_cv_mape(LGBMRegressor, params, X, y)
    return objective


def tune_zone_model(zone_name: str, model_name: str, n_trials: int) -> dict:
    logging.info("Loading data for %s...", zone_name)
    train_df, test_df = load_zone_data(zone_name, ZONES[zone_name], CACHE_DIR)
    train_df = build_features(train_df)
    test_df = build_features(test_df)
    X_train, _, y_train, _, feature_cols = prepare_features(train_df, test_df)

    logging.info(
        "Tuning %s/%s: %d rows, %d features, %d CV folds, %d trials",
        zone_name, model_name, len(X_train), len(feature_cols), N_CV_SPLITS, n_trials,
    )

    if model_name == "xgb":
        objective = make_xgb_objective(X_train, y_train)
    elif model_name == "lgbm":
        objective = make_lgbm_objective(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logging.info(
        "%s/%s best CV MAPE: %.4f%% (trial %d/%d)",
        zone_name, model_name, study.best_value, study.best_trial.number, n_trials,
    )
    return {
        "zone": zone_name,
        "model": model_name,
        "best_cv_mape": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "n_cv_splits": N_CV_SPLITS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", default="all", help="Zone code (CAC/MAC/NAC/NEC/SAC) or 'all'")
    parser.add_argument("--model", default="both", choices=["xgb", "lgbm", "both"])
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per zone per model")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: output/logs/hyperparameter_tuning_results.json)",
    )
    args = parser.parse_args()

    zones = list(ZONES.keys()) if args.zone == "all" else [args.zone]
    models = ["xgb", "lgbm"] if args.model == "both" else [args.model]
    output_path = Path(args.output) if args.output else (LOGS_DIR / "hyperparameter_tuning_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for zone_name in zones:
        for model_name in models:
            result = tune_zone_model(zone_name, model_name, args.trials)
            results.append(result)
            # Write incrementally so a crash partway through doesn't lose
            # already-completed zone/model combinations.
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    logging.info("All tuning complete. Results written to %s", output_path)
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['zone']:4s} {r['model']:5s}  CV MAPE: {r['best_cv_mape']:.4f}%")


if __name__ == "__main__":
    main()