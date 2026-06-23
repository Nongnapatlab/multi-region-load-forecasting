import pandas as pd

from config import TARGET_COL


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL in df.columns:
        df["target_roll_mean_3"] = df[TARGET_COL].rolling(3, min_periods=1).mean()
        df["target_roll_mean_6"] = df[TARGET_COL].rolling(6, min_periods=1).mean()
        df["target_roll_std_6"] = df[TARGET_COL].rolling(6, min_periods=1).std().fillna(0)
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL in df.columns:
        df["target_diff_1"] = df[TARGET_COL].diff().fillna(0)
        df["target_pct_change_1"] = df[TARGET_COL].pct_change().replace([float("inf"), float("-inf")], 0).fillna(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rolling_features(df)
    df = add_delta_features(df)
    return df
