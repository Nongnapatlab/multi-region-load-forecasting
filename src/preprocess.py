import pandas as pd
from sklearn.impute import SimpleImputer

from config import DATE_COL, TARGET_COL


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if DATE_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df["year_auto"] = df[DATE_COL].dt.year
        df["month_auto"] = df[DATE_COL].dt.month
        df["day_auto"] = df[DATE_COL].dt.day
        df["hour_auto"] = df[DATE_COL].dt.hour
        df["dayofweek_auto"] = df[DATE_COL].dt.dayofweek
    return df


def normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def encode_object_columns(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude_cols]
    for col in object_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    train_df = add_basic_time_features(train_df)
    test_df = add_basic_time_features(test_df)

    train_df = normalize_boolean_columns(train_df)
    test_df = normalize_boolean_columns(test_df)

    train_df = encode_object_columns(train_df, exclude_cols=[DATE_COL, TARGET_COL, "zone"])
    test_df = encode_object_columns(test_df, exclude_cols=[DATE_COL, TARGET_COL, "zone"])

    feature_cols = [
        c for c in train_df.columns
        if c not in [DATE_COL, TARGET_COL, "zone"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    common_feature_cols = [c for c in feature_cols if c in test_df.columns]

    X_train = train_df[common_feature_cols].copy()
    X_test = test_df[common_feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=common_feature_cols, index=train_df.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=common_feature_cols, index=test_df.index)

    return X_train, X_test, y_train, y_test, common_feature_cols
