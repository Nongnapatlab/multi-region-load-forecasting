from lightgbm import LGBMRegressor

from config import LGBM_PARAMS


def train_lgbm_model(X_train, y_train):
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    return model
