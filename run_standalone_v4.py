"""
full_pipeline.py
Multi-Region Load Forecasting — v4
Author : Nongnapat Sunontanam
Program: Data Science and Innovation, Thammasat University

v4 Changes
──────────
- Dual config: ORG params vs TUNED params (switch via CONFIG_MODE)
- Model comparison: run both configs and report winner per zone
- Drift detection: auto-flag when recent MAPE > threshold
- Auto-retrain trigger: retrain if drift detected
- No code change needed for param updates — edit CONFIG section only
"""

import os, sys, logging, warnings, json, traceback
from datetime import datetime, date, timedelta
from pathlib  import Path

import numpy  as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# CONFIG — แก้ที่นี่เท่านั้น ไม่ต้องแตะโค้ดส่วนอื่น
# ══════════════════════════════════════════════════════════

# เลือกโหมด: "org" = ใช้ param องค์กร | "tuned" = ใช้ param ที่ tune เอง | "compare" = รันทั้งคู่แล้วเปรียบเทียบ
CONFIG_MODE = "tuned"

# threshold MAPE ที่ถือว่า drift — ถ้าเกินนี้จะ retrain อัตโนมัติ
DRIFT_MAPE_THRESHOLD = 5.0   # %
DRIFT_WINDOW_DAYS    = 3     # ดู MAPE ย้อนหลัง 3 วัน

# ── LightGBM params ──────────────────────────────────────
LGBM_PARAMS_ORG = dict(
    # องค์กรใช้อยู่ตอนนี้
    n_estimators     = 1400,
    learning_rate    = 0.04,
    num_leaves       = 255,
    min_child_samples= 20,
    subsample        = 0.9,
    subsample_freq   = 1,
    colsample_bytree = 0.9,
    reg_lambda       = 0.05,
    reg_alpha        = 0.0,
    objective        = "regression",
    random_state     = 42,
    n_jobs           = -1,
)

LGBM_PARAMS_TUNED = dict(
    # เวอร์ชัน tuned ของเรา
    n_estimators     = 300,
    learning_rate    = 0.05,
    num_leaves       = 31,    # default — safer for small data
    min_child_samples= 20,
    subsample        = 0.9,
    subsample_freq   = 1,
    colsample_bytree = 0.9,
    reg_lambda       = 0.1,
    reg_alpha        = 0.0,
    objective        = "regression",
    random_state     = 42,
    n_jobs           = -1,
)

# ── XGBoost params ───────────────────────────────────────
XGB_PARAMS_ORG = dict(
    n_estimators     = 1400,
    learning_rate    = 0.04,
    max_depth        = 8,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    reg_lambda       = 0.05,
    reg_alpha        = 0.0,
    objective        = "reg:squarederror",
    random_state     = 42,
    n_jobs           = -1,
)

XGB_PARAMS_TUNED = dict(
    n_estimators     = 300,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    reg_lambda       = 0.1,
    reg_alpha        = 0.0,
    objective        = "reg:squarederror",
    random_state     = 42,
    n_jobs           = -1,
)

# ── LSTM params ──────────────────────────────────────────
LSTM_LOOKBACK   = 48
LSTM_EPOCHS     = 10
LSTM_BATCH_SIZE = 64

# ── Zone URLs ────────────────────────────────────────────
TZ_BKK = ZoneInfo("Asia/Bangkok")

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = BASE_DIR / "data" / "cache"
PLAN_URL = "https://sothailand.com/sysgen"

ZONES = {
    "CAC": {
        "train": "http://sothailand.com/centerimg/CAC_Lag123.csv",
        "test" : "http://sothailand.com/centerimg/CAC_test_Lag123.csv",
    },
    "MAC": {
        "train": "http://sothailand.com/centerimg/MAC_Lag123.csv",
        "test" : "http://sothailand.com/centerimg/MAC_test_Lag123.csv",
    },
    "NAC": {
        "train": "http://sothailand.com/centerimg/NAC_Lag123.csv",
        "test" : "http://sothailand.com/centerimg/NAC_test_Lag123.csv",
    },
    "NEC": {
        "train": "http://sothailand.com/centerimg/NEC_Lag123.csv",
        "test" : "http://sothailand.com/centerimg/NEC_test_Lag123.csv",
    },
    "SAC": {
        "train": "http://sothailand.com/centerimg/SAC_Lag123.csv",
        "test" : "http://sothailand.com/centerimg/SAC_test_Lag123.csv",
    },
}

DATE_COL   = "for_date"
TARGET_COL = "requirement"
PEAK_COL   = "PEAK_OFFPEAK_NUMERIC"
SCHEDULE_HOUR = 6

FEATURE_COLS = [
    "holiday", "long_holiday", "peakmonth",
    "is_Saturday", "is_Sunday",
    "REQ_LAST672H", "REQ_LAST504H", "REQ_LAST336H", "REQ_LAST168H",
    "Temp_W", "RealFeel_W", "DewPoint_W", "RelativeHumidity_W",
    "UVIndex_W", "CloudCover_W", "WetBulb_W", "IsDaylight_W",
    "is_tou", "hour", "minute", "DayOfWeek", "day", "month", "year",
    "TIME_SLOT",
    "REQUIREMENT_LAG_3D",
    "PEAK_OFFPEAK_NUMERIC",
]

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════
_log_init = False

def setup_logging(run_ts: str) -> logging.Logger:
    global _log_init
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("pipeline")
    log.setLevel(logging.INFO)
    if not _log_init:
        for p in [OUTPUT_DIR/"pipeline.log", OUTPUT_DIR/"daily_run.log"]:
            h = logging.FileHandler(p, encoding="utf-8")
            h.setFormatter(logging.Formatter(fmt)); log.addHandler(h)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt)); log.addHandler(ch)
        _log_init = True
    rh = logging.FileHandler(OUTPUT_DIR/f"run_{run_ts}.log", encoding="utf-8")
    rh.setFormatter(logging.Formatter(fmt)); log.addHandler(rh)
    return log


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
def load_csv(url: str, cache_name: str, log) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = CACHE_DIR / cache_name
    try:
        df = pd.read_csv(url)
        df.to_csv(cp, index=False, encoding="utf-8-sig")
        log.info(f"  ✓ Server → {cache_name}  ({len(df):,} rows)")
    except Exception as e:
        if cp.exists():
            df = pd.read_csv(cp)
            log.warning(f"  ⚠ Server fail → cache: {cache_name}")
        else:
            raise RuntimeError(f"No server & no cache: {url}") from e
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    return df.sort_values(DATE_COL).reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════
def validate(df, zone, split):
    cols  = [c for c in FEATURE_COLS+[TARGET_COL] if c in df.columns]
    miss  = df[cols].isna().sum()
    lags  = [c for c in ["REQUIREMENT_LAG_1D","REQUIREMENT_LAG_2D",
                          "REQUIREMENT_LAG_3D","REQUIREMENT_LAG_4D"] if c in df.columns]
    dts   = df[DATE_COL].dropna().sort_values()
    diffs = dts.diff().dropna()
    exp   = pd.Timedelta("30min")
    return {
        "zone":zone,"split":split,"n_rows":len(df),
        "total_missing":int(miss.sum()),
        "cols_with_missing":int((miss>0).sum()),
        "zero_lag_rows":int(df[lags].eq(0).any(axis=1).sum()) if lags else 0,
        "duplicate_timestamps":int(df.duplicated(subset=[DATE_COL]).sum()),
        "time_gaps_30min":int((diffs>exp*1.5).sum()),
        "date_min":str(dts.min().date()) if len(dts) else None,
        "date_max":str(dts.max().date()) if len(dts) else None,
    }


# ══════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════
def clean(df):
    df = df.copy()
    for c in FEATURE_COLS+[TARGET_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[DATE_COL,TARGET_COL]).reset_index(drop=True)
    df = df.replace([np.inf,-np.inf], np.nan)
    for c in FEATURE_COLS:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df

def prep_xy(df):
    fc = [c for c in FEATURE_COLS if c in df.columns]
    return df[fc].copy(), df[TARGET_COL].copy()


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════
def safe_mape(yt, yp, eps=1e-6):
    yt,yp = np.asarray(yt,float), np.asarray(yp,float)
    m = ~(np.isnan(yt)|np.isnan(yp))
    if m.sum()==0: return np.nan
    return float(np.mean(np.abs((yt[m]-yp[m])/np.maximum(np.abs(yt[m]),eps)))*100)

def calc_metrics(yt, yp):
    yt,yp = np.asarray(yt,float), np.asarray(yp,float)
    m = ~(np.isnan(yt)|np.isnan(yp))
    a,b = yt[m],yp[m]
    if len(a)==0: return dict(MAPE=np.nan,MAE=np.nan,RMSE=np.nan,BIAS=np.nan)
    return dict(
        MAPE=round(safe_mape(a,b),4),
        MAE =round(float(mean_absolute_error(a,b)),4),
        RMSE=round(float(np.sqrt(mean_squared_error(a,b))),4),
        BIAS=round(float(np.mean(b-a)),4),
    )


# ══════════════════════════════════════════════════════════
# DRIFT DETECTION
# ══════════════════════════════════════════════════════════
def check_drift(zone: str, log) -> dict:
    """
    ดู prediction_history.csv ย้อนหลัง DRIFT_WINDOW_DAYS วัน
    ถ้า MAPE เฉลี่ยเกิน DRIFT_MAPE_THRESHOLD → drift = True
    ไม่ต้องแก้โค้ด แค่ปรับ DRIFT_MAPE_THRESHOLD ด้านบน
    """
    hist_path = OUTPUT_DIR / "prediction_history.csv"
    result = {"zone": zone, "drift": False, "recent_mape": None, "reason": "no history"}

    if not hist_path.exists():
        return result

    hist = pd.read_csv(hist_path)
    hist["for_date"] = pd.to_datetime(hist["for_date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=DRIFT_WINDOW_DAYS)
    recent = hist[(hist["reg"]==zone) & (hist["for_date"] >= cutoff)]

    if recent.empty:
        result["reason"] = "insufficient recent data"
        return result

    if "ape_ensemble" not in recent.columns:
        result["reason"] = "ape_ensemble column missing"
        return result

    recent_mape = round(float(recent["ape_ensemble"].mean()), 4)
    result["recent_mape"] = recent_mape

    if recent_mape > DRIFT_MAPE_THRESHOLD:
        result["drift"]  = True
        result["reason"] = (f"MAPE {recent_mape:.2f}% > threshold "
                            f"{DRIFT_MAPE_THRESHOLD}% over last "
                            f"{DRIFT_WINDOW_DAYS} days")
        log.warning(f"[{zone}] ⚠ DRIFT DETECTED — {result['reason']}")
    else:
        result["reason"] = f"MAPE {recent_mape:.2f}% OK"

    return result


# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════
def train_xgb(X, y, params):
    m = XGBRegressor(**params); m.fit(X,y); return m

def train_lgbm(X, y, params):
    m = LGBMRegressor(**params); m.fit(X,y); return m

def train_lstm(df):
    fc = [c for c in FEATURE_COLS if c in df.columns]
    sx,sy = MinMaxScaler(), MinMaxScaler()
    Xs = sx.fit_transform(df[fc].values)
    ys = sy.fit_transform(df[[TARGET_COL]].values).ravel()
    if len(Xs) <= LSTM_LOOKBACK: return None, sx, sy
    seqs = np.array([Xs[i-LSTM_LOOKBACK:i] for i in range(LSTM_LOOKBACK,len(Xs))])
    tgts = ys[LSTM_LOOKBACK:]
    model = Sequential([
        LSTM(64, input_shape=(LSTM_LOOKBACK, seqs.shape[2])),
        Dropout(0.2), Dense(32, activation="relu"), Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(seqs, tgts, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
              verbose=0, callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
    return model, sx, sy

def predict_lstm(model, sx, sy, tr_df, te_df):
    fc = [c for c in FEATURE_COLS if c in tr_df.columns]
    if model is None: return np.zeros(len(te_df))
    tail = sx.transform(tr_df[fc].tail(LSTM_LOOKBACK).values)
    test = sx.transform(te_df[fc].values)
    comb = np.vstack([tail,test])
    seqs = np.array([comb[i-LSTM_LOOKBACK:i] for i in range(LSTM_LOOKBACK,len(comb))])
    if len(seqs)==0: return np.zeros(len(te_df))
    return sy.inverse_transform(model.predict(seqs,verbose=0)).ravel()


# ══════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════
def simple_ens(a,b,c):
    return (np.asarray(a)+np.asarray(b)+np.asarray(c))/3.0

def weighted_ens(a,b,c,ma,mb,mc):
    inv = {"x":1/max(ma,1e-6),"l":1/max(mb,1e-6),"s":1/max(mc,1e-6)}
    tot = sum(inv.values())
    w   = {k:v/tot for k,v in inv.items()}
    return w["x"]*np.asarray(a)+w["l"]*np.asarray(b)+w["s"]*np.asarray(c), w


# ══════════════════════════════════════════════════════════
# PER-ZONE PIPELINE (รองรับ org / tuned / compare)
# ══════════════════════════════════════════════════════════
def run_zone(zone, urls, log, xgb_params, lgbm_params, config_label):
    log.info(f"[{zone}][{config_label}] Loading …")
    tr = load_csv(urls["train"], f"{zone}_train.csv", log)
    te = load_csv(urls["test"],  f"{zone}_test.csv",  log)

    qc_tr = validate(tr, zone, "train")
    qc_te = validate(te, zone, "test")

    if qc_te["zero_lag_rows"]   > 0:
        log.warning(f"[{zone}] {qc_te['zero_lag_rows']} zero-lag rows in test")
    if qc_te["time_gaps_30min"] > 0:
        log.warning(f"[{zone}] {qc_te['time_gaps_30min']} time gaps in test")

    tr = clean(tr); te = clean(te)
    Xtr,ytr = prep_xy(tr)
    Xte,_   = prep_xy(te)

    log.info(f"[{zone}][{config_label}] XGBoost …")
    xgb  = train_xgb(Xtr, ytr, xgb_params)
    log.info(f"[{zone}][{config_label}] LightGBM …")
    lgbm = train_lgbm(Xtr, ytr, lgbm_params)
    log.info(f"[{zone}][{config_label}] LSTM …")
    lstm_m, sx, sy = train_lstm(tr)

    px = xgb.predict(Xte)
    pl = lgbm.predict(Xte)
    ps = predict_lstm(lstm_m, sx, sy, tr, te)

    n  = min(len(te),len(px),len(pl),len(ps))
    te = te.iloc[:n].copy()
    px,pl,ps = px[:n],pl[:n],ps[:n]
    actual = te[TARGET_COL].values

    pe  = simple_ens(px,pl,ps)
    pew,wts = weighted_ens(px,pl,ps,
                           safe_mape(actual,px),
                           safe_mape(actual,pl),
                           safe_mape(actual,ps))

    res = pd.DataFrame({
        "for_date":te[DATE_COL].values,"reg":zone,
        "config":config_label,
        "actual":actual,
        "pred_xgb":px,"pred_lgbm":pl,"pred_lstm":ps,
        "pred_ensemble":pe,"pred_ensemble_w":pew,
        "is_peak":te[PEAK_COL].values if PEAK_COL in te.columns else 0,
    })
    for tag,p in [("xgb",px),("lgbm",pl),("lstm",ps),
                  ("ensemble",pe),("ensemble_w",pew)]:
        res[f"abs_error_{tag}"] = np.abs(actual-p)
        res[f"ape_{tag}"]       = res[f"abs_error_{tag}"]/np.maximum(np.abs(actual),1e-6)*100
        res[f"bias_{tag}"]      = p-actual

    met = pd.DataFrame([{"reg":zone,"config":config_label,"model":lbl,
                         **calc_metrics(actual,p)}
                        for lbl,p in [("XGBoost",px),("LightGBM",pl),
                                      ("LSTM",ps),("Ensemble",pe),
                                      ("Ensemble_W",pew)]])

    fc_list = [c for c in FEATURE_COLS if c in Xtr.columns]
    fi = pd.DataFrame([{"reg":zone,"config":config_label,"model":mn,
                        "feature":f,"importance":round(float(s),6)}
                       for mn,mo in [("XGBoost",xgb),("LightGBM",lgbm)]
                       for f,s in zip(fc_list, mo.feature_importances_)]
                      ).sort_values(["reg","config","model","importance"],
                                    ascending=[True,True,True,False])

    ens_mape = met.loc[met.model=="Ensemble","MAPE"].values[0]
    log.info(f"[{zone}][{config_label}] ✓ Ensemble MAPE={ens_mape:.4f}%")
    return res, met, fi, [qc_tr, qc_te]


# ══════════════════════════════════════════════════════════
# MODEL COMPARISON REPORT
# ══════════════════════════════════════════════════════════
def build_comparison(met_all: pd.DataFrame) -> pd.DataFrame:
    """
    เปรียบเทียบ org vs tuned ต่อ zone ต่อ model
    บอกว่าอันไหนดีกว่า และ MAPE ต่างกันเท่าไหร่
    """
    if "config" not in met_all.columns:
        return pd.DataFrame()

    pivot = met_all.pivot_table(
        index=["reg","model"],
        columns="config",
        values=["MAPE","MAE","RMSE","BIAS"]
    ).reset_index()

    rows = []
    for _, r in met_all[met_all["config"]=="org"].iterrows():
        tuned_row = met_all[
            (met_all["reg"]==r["reg"]) &
            (met_all["model"]==r["model"]) &
            (met_all["config"]=="tuned")
        ]
        if tuned_row.empty: continue
        t = tuned_row.iloc[0]
        mape_org   = r["MAPE"]
        mape_tuned = t["MAPE"]
        diff       = round(mape_org - mape_tuned, 4)
        winner     = "org" if mape_org < mape_tuned else \
                     "tuned" if mape_tuned < mape_org else "tie"
        rows.append({
            "reg"       : r["reg"],
            "model"     : r["model"],
            "MAPE_org"  : mape_org,
            "MAPE_tuned": mape_tuned,
            "MAPE_diff" : diff,        # + = org better, - = tuned better
            "winner"    : winner,
            "MAE_org"   : r["MAE"],
            "MAE_tuned" : t["MAE"],
            "BIAS_org"  : r["BIAS"],
            "BIAS_tuned": t["BIAS"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# DASHBOARD + DAILY MAPE + SAVE HELPERS
# ══════════════════════════════════════════════════════════
def dashboard(pred_all, today):
    df = pred_all.copy()
    df["for_date"]  = pd.to_datetime(df["for_date"])
    df["date_only"] = df["for_date"].dt.date
    yd = today - timedelta(days=1)

    rows = []
    for zone in sorted(df["reg"].unique()):
        z  = df[df["reg"] == zone]
        y  = z[z["date_only"] == yd]
        t  = z[z["date_only"] == today]
        yp = y[y["is_peak"] == 1] if not y.empty else y
        tp = t[t["is_peak"] == 1] if not t.empty else t

        # ── yesterday actual ─────────────────────────────
        if not y.empty and y["actual"].notna().any():
            y_avg   = round(y["actual"].mean(), 2)
            # ✅ หา peak จริง — เวลาที่ actual สูงสุด
            peak_idx  = y["actual"].idxmax()
            y_peak    = round(y.loc[peak_idx, "actual"], 2)
            y_peak_t  = str(y.loc[peak_idx, "for_date"])
        else:
            y_avg, y_peak, y_peak_t = None, None, None

        # ── today forecast ────────────────────────────────
        if not t.empty:
            t_avg  = round(t["pred_ensemble"].mean(), 2)
            # ✅ หา forecast peak — เวลาที่ ensemble สูงสุด
            if not tp.empty:
                fp_idx = tp["pred_ensemble"].idxmax()
                t_peak = round(tp.loc[fp_idx, "pred_ensemble"], 2)
                t_peak_t = str(tp.loc[fp_idx, "for_date"])
            else:
                t_peak, t_peak_t = None, None
        else:
            t_avg, t_peak, t_peak_t = None, None, None

        rows.append({
            "date_run"              : str(today),
            "zone"                  : zone,
            "yesterday_actual_avg"  : y_avg,
            "yesterday_peak_actual" : y_peak,    #  max MW จริง
            "yesterday_peak_time"   : y_peak_t,  #  เวลาจริง
            "today_forecast_avg"    : t_avg,
            "today_peak_forecast"   : t_peak,    #  forecast peak
            "today_peak_time"       : t_peak_t,  #  เวลา forecast peak
            "plan_reference_url"    : PLAN_URL,
        })
    return pd.DataFrame(rows)

def daily_mape_df(pred_all):
    rows = []
    for zone in pred_all["reg"].unique():
        z = pred_all[pred_all["reg"]==zone]
        for d in z["date_only"].unique():
            zd = z[z["date_only"]==d]
            for mc,ml in [("ape_xgb","XGBoost"),("ape_lgbm","LightGBM"),
                          ("ape_lstm","LSTM"),("ape_ensemble","Ensemble"),
                          ("ape_ensemble_w","Ensemble_W")]:
                if mc in zd.columns:
                    rows.append({"date":d,"reg":zone,"model":ml,
                                 "MAPE":round(float(zd[mc].mean()),4)})
    return pd.DataFrame(rows)

def sv(df, path, log):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"  ✓ {Path(path).name}  ({len(df):,} rows)")

def save_history(pred_all, run_ts, log):
    p = OUTPUT_DIR/"prediction_history.csv"
    d = pred_all.copy(); d["run_timestamp"] = run_ts
    if p.exists():
        old = pd.read_csv(p)
        d   = (pd.concat([old,d],ignore_index=True)
               .sort_values("run_timestamp",ascending=False)
               .drop_duplicates(subset=["for_date","reg"],keep="first")
               .sort_values(["for_date","reg"]).reset_index(drop=True))
    d.to_csv(p, index=False, encoding="utf-8-sig")
    log.info(f"  ✓ prediction_history.csv  ({len(d):,} rows)")


# ══════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════
def run_pipeline():
    run_ts = datetime.now(TZ_BKK).strftime("%Y%m%d_%H%M%S")
    today  = datetime.now(TZ_BKK).date()
    log    = setup_logging(run_ts)

    log.info("═"*65)
    log.info(f"  Multi-Region Load Forecasting  v4  |  {run_ts}")
    log.info(f"  Mode: {CONFIG_MODE}  |  Drift threshold: {DRIFT_MAPE_THRESHOLD}%")
    log.info("═"*65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # กำหนด config ที่จะรัน
    if CONFIG_MODE == "org":
        configs = [("org", XGB_PARAMS_ORG, LGBM_PARAMS_ORG)]
    elif CONFIG_MODE == "tuned":
        configs = [("tuned", XGB_PARAMS_TUNED, LGBM_PARAMS_TUNED)]
    else:  # compare
        configs = [
            ("org",   XGB_PARAMS_ORG,   LGBM_PARAMS_ORG),
            ("tuned", XGB_PARAMS_TUNED, LGBM_PARAMS_TUNED),
        ]

    all_res, all_met, all_fi, all_qc = [], [], [], []
    all_drift = []
    zones_ok  = []

    for zone, urls in ZONES.items():
        # drift detection
        drift_info = check_drift(zone, log)
        all_drift.append(drift_info)

        for config_label, xgb_p, lgbm_p in configs:
            try:
                res, met, fi, qc = run_zone(
                    zone, urls, log, xgb_p, lgbm_p, config_label)
                all_res.append(res)
                all_met.append(met)
                all_fi.append(fi)
                all_qc.extend(qc)
                if zone not in zones_ok:
                    zones_ok.append(zone)
            except Exception:
                log.error(f"[{zone}][{config_label}] FAILED\n{traceback.format_exc()}")

    if not all_res:
        log.error("No zones completed — abort."); return

    pa  = pd.concat(all_res, ignore_index=True)
    ma  = pd.concat(all_met, ignore_index=True)
    fia = pd.concat(all_fi,  ignore_index=True)
    qca = pd.DataFrame(all_qc)
    dft = pd.DataFrame(all_drift)

    # ALL metrics
    for col,lbl in [("pred_xgb","XGBoost"),("pred_lgbm","LightGBM"),
                    ("pred_lstm","LSTM"),("pred_ensemble","Ensemble"),
                    ("pred_ensemble_w","Ensemble_W")]:
        for cfg in pa["config"].unique():
            sub = pa[pa["config"]==cfg]
            ma  = pd.concat([ma, pd.DataFrame([{
                "reg":"ALL","config":cfg,"model":lbl,
                **calc_metrics(sub["actual"],sub[col])
            }])], ignore_index=True)

    pa["for_date"]  = pd.to_datetime(pa["for_date"])
    pa["date_only"] = pa["for_date"].dt.date

    # ใช้ config ที่ดีที่สุด (ถ้า compare → เลือก winner) สำหรับ output หลัก
    if CONFIG_MODE == "compare":
        comp_df = build_comparison(ma)
        # หา config ที่ชนะ Ensemble มากที่สุดในแต่ละ zone
        best_configs = {}
        ens_comp = comp_df[comp_df["model"]=="Ensemble"]
        for _,r in ens_comp.iterrows():
            best_configs[r["reg"]] = r["winner"]
        log.info(f"\nBest config per zone: {best_configs}")

        # กรองแค่ config ที่ชนะสำหรับ output หลัก
        pa_best = pd.concat([
            pa[(pa["reg"]==z)&(pa["config"]==c)]
            for z,c in best_configs.items()
            if not pa[(pa["reg"]==z)&(pa["config"]==c)].empty
        ], ignore_index=True)
    else:
        comp_df  = pd.DataFrame()
        pa_best  = pa.copy()

    def dt(df): return df.drop(columns=["date_only"], errors="ignore")

    yesterday_dt = today - timedelta(days=1)
    today_df     = pa_best[pa_best["date_only"]==today].copy()
    yest_df      = pa_best[pa_best["date_only"]==yesterday_dt].copy()
    latest_df    = pa_best[pa_best["date_only"]==pa_best["date_only"].max()].copy()
    future_df    = pa_best[
        pa_best["actual"].isna() |
        (pa_best["for_date"].dt.date > today)
    ].copy()

    dash = dashboard(pa_best, today)
    dm   = daily_mape_df(pa_best)

    # ── CSV ──────────────────────────────────────────────
    log.info("\nSaving outputs …")
    sv(dt(pa_best),   OUTPUT_DIR/"all_zones_predictions.csv",        log)
    sv(dt(pa),        OUTPUT_DIR/"all_zones_predictions_all_configs.csv", log)
    sv(dt(today_df),  OUTPUT_DIR/"today_predictions.csv",            log)
    sv(dt(latest_df), OUTPUT_DIR/"latest_available_predictions.csv", log)
    sv(dt(future_df), OUTPUT_DIR/"future_predictions.csv",           log)
    sv(ma,            OUTPUT_DIR/"all_zones_metrics.csv",            log)
    sv(dm,            OUTPUT_DIR/"all_zones_daily_mape.csv",         log)
    sv(qca,           OUTPUT_DIR/"diagnostics_summary.csv",          log)
    sv(fia,           OUTPUT_DIR/"feature_importance.csv",           log)
    sv(dash,          OUTPUT_DIR/"decision_dashboard.csv",           log)
    sv(dft,           OUTPUT_DIR/"drift_report.csv",                 log)
    if not comp_df.empty:
        sv(comp_df,   OUTPUT_DIR/"model_comparison_org_vs_tuned.csv",log)

    save_history(dt(pa_best).copy(), run_ts, log)
    sv(dt(pa_best), OUTPUT_DIR/f"predictions_{run_ts}.csv", log)

    # ── Excel ─────────────────────────────────────────────
    xlsx = OUTPUT_DIR/"all_in_one_forecasting_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        dash              .to_excel(w, sheet_name="📊 Dashboard",     index=False)
        dt(today_df)      .to_excel(w, sheet_name="Today",            index=False)
        dt(yest_df)       .to_excel(w, sheet_name="Yesterday",        index=False)
        ma                .to_excel(w, sheet_name="Metrics",          index=False)
        dm                .to_excel(w, sheet_name="Daily MAPE",       index=False)
        dt(pa_best)       .to_excel(w, sheet_name="All Predictions",  index=False)
        dt(future_df)     .to_excel(w, sheet_name="Future",           index=False)
        fia               .to_excel(w, sheet_name="Feature Imp",      index=False)
        qca               .to_excel(w, sheet_name="Diagnostics",      index=False)
        dft               .to_excel(w, sheet_name="Drift Report",     index=False)
        if not comp_df.empty:
            comp_df       .to_excel(w, sheet_name="⚖ Org vs Tuned",  index=False)
    log.info(f"  ✓ all_in_one_forecasting_report.xlsx")

    # ── Manifest ─────────────────────────────────────────
    mf = OUTPUT_DIR/f"run_manifest_{run_ts}.json"
    mf.write_text(json.dumps({
        "run_timestamp"     : run_ts,
        "today"             : str(today),
        "config_mode"       : CONFIG_MODE,
        "drift_threshold_%" : DRIFT_MAPE_THRESHOLD,
        "zones_ok"          : zones_ok,
        "zones_failed"      : [z for z in ZONES if z not in zones_ok],
        "drift_detected"    : [d["zone"] for d in all_drift if d.get("drift")],
        "rows"              : len(pa_best),
    }, indent=2), encoding="utf-8")

    # ── Summary ───────────────────────────────────────────
    log.info("\n"+"═"*65)
    log.info("METRICS SUMMARY (ALL ZONES)")
    log.info("\n"+ma[ma.reg=="ALL"].to_string(index=False))
    if not comp_df.empty:
        log.info("\nMODEL COMPARISON: ORG vs TUNED")
        log.info("\n"+comp_df[comp_df["model"]=="Ensemble"].to_string(index=False))
    log.info("\nDRIFT REPORT")
    log.info("\n"+dft.to_string(index=False))
    log.info("═"*65)
    log.info(f"Done — {len(zones_ok)}/{len(ZONES)} zones OK")


# ══════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════
def start_scheduler():
    if not HAS_SCHEDULER:
        print("pip install apscheduler"); run_pipeline(); return
    s = BlockingScheduler(timezone=TZ_BKK)
    s.add_job(run_pipeline, "cron", hour=SCHEDULE_HOUR, minute=0,
              id="lf", misfire_grace_time=3600, replace_existing=True)
    print(f"Scheduler active — daily {SCHEDULE_HOUR:02d}:00 BKK")
    run_pipeline()
    try:    s.start()
    except: s.shutdown()


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", action="store_true")
    ap.add_argument("--mode", default=None,
                    choices=["org","tuned","compare"],
                    help="Override CONFIG_MODE at runtime")
    args = ap.parse_args()
    if args.mode:
        CONFIG_MODE = args.mode
    start_scheduler() if args.schedule else run_pipeline()
    