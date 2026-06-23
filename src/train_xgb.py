from xgboost import XGBRegressor

from config import XGB_PARAMS


def train_xgb_model(X_train, y_train):
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model
