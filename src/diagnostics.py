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
