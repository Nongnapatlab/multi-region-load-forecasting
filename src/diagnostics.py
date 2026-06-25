import pandas as pd

from config import DATE_COL, TARGET_COL


def build_diagnostics(zone: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    rows.append({"zone": zone, "check": "train_rows", "value": len(train_df)})
    rows.append({"zone": zone, "check": "test_rows", "value": len(test_df)})
    rows.append({"zone": zone, "check": "feature_count", "value": len(feature_cols)})
    rows.append({"zone": zone, "check": "train_missing_total", "value": int(train_df.isna().sum().sum())})
    rows.append({"zone": zone, "check": "test_missing_total", "value": int(test_df.isna().sum().sum())})

    if DATE_COL in train_df.columns:
        rows.append({"zone": zone, "check": "train_duplicate_dates", "value": int(train_df[DATE_COL].duplicated().sum())})
    if DATE_COL in test_df.columns:
        rows.append({"zone": zone, "check": "test_duplicate_dates", "value": int(test_df[DATE_COL].duplicated().sum())})
    if TARGET_COL in train_df.columns:
        rows.append({"zone": zone, "check": "train_target_zero_count", "value": int((train_df[TARGET_COL] == 0).sum())})
    if TARGET_COL in test_df.columns:
        rows.append({"zone": zone, "check": "test_target_zero_count", "value": int((test_df[TARGET_COL] == 0).sum())})

    return pd.DataFrame(rows)


def build_distribution_shift_check(
    zone: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the target's distribution between train and the realized
    portion of test, plus same-calendar-window history, to help separate
    'normal seasonal effect' from 'something changed that the model
    doesn't know about yet'.

    z_score_vs_train: how many train-set standard deviations away the
    test mean is. As a rule of thumb: |z| < 2 is unremarkable noise,
    2-3 worth watching, > 3 is a real shift worth investigating (new
    weather extreme, sensor/meter change, a new large customer, etc.)
    rather than ordinary day-to-day variation.

    z_score_vs_same_month_history: same idea, but the baseline is only
    prior years' data for the *same calendar month* as the test period,
    so a hot June being compared against a June baseline (not an annual
    average that includes cool months) — this is what actually answers
    'is this normal seasonal variation or not'.
    """
    from config import TARGET_COL, DATE_COL
    from metrics import ZERO_ACTUAL_THRESHOLD

    rows = []

    train_target = train_df[TARGET_COL].dropna()
    test_realized = test_df[test_df[TARGET_COL].abs() >= ZERO_ACTUAL_THRESHOLD][TARGET_COL].dropna()

    train_mean, train_std = train_target.mean(), train_target.std()

    rows.append({"zone": zone, "metric": "train_mean", "value": round(train_mean, 2)})
    rows.append({"zone": zone, "metric": "train_std", "value": round(train_std, 2)})

    if len(test_realized) > 0:
        test_mean = test_realized.mean()
        rows.append({"zone": zone, "metric": "test_realized_mean", "value": round(test_mean, 2)})
        rows.append({"zone": zone, "metric": "test_realized_n_rows", "value": len(test_realized)})

        z_vs_train = (test_mean - train_mean) / train_std if train_std > 0 else float("nan")
        rows.append({"zone": zone, "metric": "z_score_vs_train", "value": round(z_vs_train, 3)})

        # Same-calendar-month baseline: prior years only, same month(s) as
        # the realized test window, so a hot June isn't compared against
        # an annual average that includes cooler months.
        if DATE_COL in train_df.columns and DATE_COL in test_df.columns:
            test_months = pd.to_datetime(
                test_df.loc[test_df[TARGET_COL].abs() >= ZERO_ACTUAL_THRESHOLD, DATE_COL]
            ).dt.month.unique()
            train_dates = pd.to_datetime(train_df[DATE_COL])
            same_month_mask = train_dates.dt.month.isin(test_months)
            same_month_target = train_df.loc[same_month_mask, TARGET_COL].dropna()

            if len(same_month_target) > 1:
                sm_mean, sm_std = same_month_target.mean(), same_month_target.std()
                rows.append({"zone": zone, "metric": "same_month_history_mean", "value": round(sm_mean, 2)})
                z_vs_sm = (test_mean - sm_mean) / sm_std if sm_std > 0 else float("nan")
                rows.append({"zone": zone, "metric": "z_score_vs_same_month_history", "value": round(z_vs_sm, 3)})

    return pd.DataFrame(rows)


def build_feature_importance(zone: str, model, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    """Extract feature importance from a fitted tree model (XGBoost/LightGBM
    expose .feature_importances_ as a raw 'gain'-style score per feature).
    Returned as a tidy long-format table — one row per (zone, model,
    feature) — with a normalized importance_pct column so different zones
    and models are comparable side by side regardless of each model's raw
    score scale."""
    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        # Defensive: if a model dropped/reordered features internally for
        # any reason, skip rather than silently mis-attributing scores to
        # the wrong feature names.
        return pd.DataFrame()

    total = importances.sum()
    rows = [
        {
            "zone": zone,
            "model": model_name,
            "feature": feature,
            "importance": float(score),
            "importance_pct": float(score) / total * 100 if total > 0 else 0.0,
        }
        for feature, score in zip(feature_cols, importances)
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)