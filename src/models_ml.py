"""Thin wrapper classes for tree-based models.

Keeps train_xgb.py and train_lgbm.py as simple functional helpers while
exposing a unified sklearn-style interface (fit / predict / feature_importances_)
that pipeline.py can work with polymorphically.
"""

import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import XGB_PARAMS, LGBM_PARAMS


class XGBModel:
    """Wrapper around XGBRegressor with feature importance support."""

    name = "XGBoost"

    def __init__(self):
        self._model = XGBRegressor(**XGB_PARAMS)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBModel":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    @property
    def feature_names_in_(self):
        return self._model.feature_names_in_


class LGBMModel:
    """Wrapper around LGBMRegressor with feature importance support."""

    name = "LightGBM"

    def __init__(self):
        self._model = LGBMRegressor(**LGBM_PARAMS)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LGBMModel":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    @property
    def feature_names_in_(self):
        return getattr(self._model, "feature_name_", None)