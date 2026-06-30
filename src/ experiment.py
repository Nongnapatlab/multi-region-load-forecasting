import sys, warnings
sys.path.insert(0, 'multi-region-load-forecasting/src')
warnings.filterwarnings('ignore')

import pandas as pd
from data_loader import ensure_datetime
from feature_engineering import build_features
from preprocess import add_basic_time_features, normalize_boolean_columns, encode_object_columns
from config import XGB_PARAMS, LGBM_PARAMS, DATE_COL, TARGET_COL
from metrics import evaluate_predictions
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

def load(zone, split, suffix):
    df = pd.read_csv(f'multi-region-load-forecasting/data/cache/{zone}_{split}_{suffix}_cached.csv')
    df = ensure_datetime(df)
    df['zone'] = zone
    return df

def prep(train_df, test_df, excluded_cols):
    train_df = add_basic_time_features(train_df)
    test_df = add_basic_time_features(test_df)
    train_df = normalize_boolean_columns(train_df)
    test_df = normalize_boolean_columns(test_df)
    train_df = encode_object_columns(train_df, exclude_cols=[DATE_COL, TARGET_COL, 'zone'])
    test_df = encode_object_columns(test_df, exclude_cols=[DATE_COL, TARGET_COL, 'zone'])
    feature_cols = [c for c in train_df.columns if c not in [DATE_COL, TARGET_COL, 'zone']
                    and c not in excluded_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    common = [c for c in feature_cols if c in test_df.columns]
    all_nan = {c for c in common if train_df[c].isna().all() or test_df[c].isna().all()}
    common = [c for c in common if c not in all_nan]
    X_train, X_test = train_df[common].copy(), test_df[common].copy()
    y_train, y_test = train_df[TARGET_COL].copy(), test_df[TARGET_COL].copy()
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=common, index=train_df.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=common, index=test_df.index)
    return X_train, X_test, y_train, y_test, common

CONFIGS = {
    'A_current (keep LAG_3D, proxy-filled)':       ['REQUIREMENT_LAG_1D', 'REQUIREMENT_LAG_2D', 'REQUIREMENT_LAG_4D'],
    'B_exclude_LAG_3D (rely on REQ_LAST*H only)':   ['REQUIREMENT_LAG_1D', 'REQUIREMENT_LAG_2D', 'REQUIREMENT_LAG_4D', 'REQUIREMENT_LAG_3D'],
}

for zone, suffix in [('CAC', 'V3'), ('MAC', 'V3'), ('NAC', 'V3'), ('NEC', 'V4'), ('SAC', 'V3')]:
    print(f'\n========== ZONE {zone} ==========')
    train_raw = load(zone, 'train', suffix)
    test_raw = load(zone, 'test', suffix)
    original_actual = test_raw[TARGET_COL].copy()

    train_feat = build_features(train_raw.copy())
    test_feat = build_features(test_raw.copy())

    pred_store = {}
    for name, excl in CONFIGS.items():
        X_train, X_test, y_train, y_test, cols = prep(train_feat.copy(), test_feat.copy(), excl)
        xgb = XGBRegressor(**XGB_PARAMS).fit(X_train, y_train)
        lgbm = LGBMRegressor(**LGBM_PARAMS).fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        pred_lgbm = lgbm.predict(X_test)
        pred_ens = 0.5 * pred_xgb + 0.5 * pred_lgbm

        res = test_feat[[DATE_COL]].copy()
        res['actual'] = original_actual.values
        res['pred_ens'] = pred_ens
        pred_store[name] = res

        m = evaluate_predictions(res['actual'], res['pred_ens'])
        future = res[res['actual'].abs() < 1.0]
        print(f'-- {name} --')
        print(f'   scored MAPE={m["MAPE"]:.3f}  MAE={m["MAE"]:.1f}  BIAS={m["BIAS"]:.1f}  n_scored={m["n_rows_scored"]}')
        print(f'   future(unrealized) pred_ens: min={future["pred_ens"].min():.0f} mean={future["pred_ens"].mean():.0f} max={future["pred_ens"].max():.0f}')
        if 'REQUIREMENT_LAG_3D' in cols:
            imp = dict(zip(cols, xgb.feature_importances_))
            print(f'   XGB importance of REQUIREMENT_LAG_3D: {imp["REQUIREMENT_LAG_3D"]*100:.2f}%  (rank {sorted(imp.values(), reverse=True).index(imp["REQUIREMENT_LAG_3D"])+1}/{len(imp)})')

    # daily peak progression across full 6-day future horizon
    a = pred_store['A_current (keep LAG_3D, proxy-filled)']
    b = pred_store['B_exclude_LAG_3D (rely on REQ_LAST*H only)']
    merged = a[[DATE_COL, 'actual', 'pred_ens']].rename(columns={'pred_ens': 'pred_A'}).merge(
        b[[DATE_COL, 'pred_ens']].rename(columns={'pred_ens': 'pred_B'}), on=DATE_COL)
    merged['_date'] = pd.to_datetime(merged[DATE_COL]).dt.date
    daily = merged.groupby('_date').agg(peak_A=('pred_A', 'max'), peak_B=('pred_B', 'max')).reset_index()
    print('   --- daily peak forecast across the 6-day horizon ---')
    print(daily.to_string(index=False))