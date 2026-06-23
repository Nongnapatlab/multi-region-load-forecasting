# multi-region-load-forecasting

Electricity load forecasting across five regions using XGBoost, LightGBM, LSTM, and ensemble methods.

## Project structure

```text
project/
├─ data/
│  ├─ raw/
│  └─ cache/
├─ output/
│  ├─ predictions/
│  ├─ metrics/
│  └─ logs/
├─ src/
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ preprocess.py
│  ├─ feature_engineering.py
│  ├─ train_xgb.py
│  ├─ train_lgbm.py
│  ├─ train_lstm.py
│  ├─ ensemble.py
│  ├─ metrics.py
│  ├─ diagnostics.py
│  ├─ export_results.py
│  └─ main.py
├─ requirements.txt
├─ run_daily.bat
└─ README.md
```

## Objective

Build a reusable forecasting pipeline in VS Code that:

- downloads train/test CSV data for all 5 zones from URL
- trains XGBoost, LightGBM, and LSTM models
- calculates MAPE, MAE, RMSE, and bias
- builds ensemble predictions
- exports one combined predictions file and one combined metrics file
- supports daily scheduled execution

## Zones

The default zones configured in `src/config.py` are:

- CAC
- MAC
- NAC
- NEC
- SAC

Each zone includes:

- train URL
- test URL
- cache file names
- zone prefix

## Baseline implementation strategy

Phase 1 in this repo is designed to be practical and stable:

- XGBoost
- LightGBM
- simple average ensemble
- CSV exports
- diagnostics summary

Phase 2 extends the same pipeline with:

- LSTM sequence model
- weighted ensemble
- richer diagnostics and charts

## Installation

Create a virtual environment first, then install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally in VS Code

```bash
python src/main.py
```

## Outputs

After a successful run, the pipeline writes:

- `output/predictions/all_zones_predictions.csv`
- `output/metrics/all_zones_metrics.csv`
- `output/metrics/all_zones_daily_mape.csv`
- `output/logs/pipeline.log`
- `output/logs/diagnostics_summary.csv`

## Expected input schema

The code assumes these columns exist in each zone CSV:

- `for_date`
- `requirement`

Other columns are treated as candidate features. Non-numeric columns are encoded or excluded safely in preprocessing.

## Recommended next steps after first successful run

1. Confirm all actual column names match the assumption.
2. Inspect diagnostics output for missing values, duplicates, and time gaps.
3. Compare XGBoost vs LightGBM by zone.
4. Add LSTM tuning only after the tree-based baseline is stable.
5. If needed, switch ensemble from simple average to weighted average.

## Daily scheduling

Use `run_daily.bat` together with Windows Task Scheduler.

Example scheduler flow:

1. activate environment
2. run `python src/main.py`
3. save outputs into the `output/` folder
4. review pipeline log if any error occurs

## Notes

- If real column names differ, update them in `src/config.py`.
- If some features are unavailable in a zone, the code keeps only columns that exist.
- LSTM is implemented in a safe baseline form and can be disabled from config if TensorFlow setup is heavy on the first run.


Multi-Region Load Forecasting
- ดึงข้อมูลไฟฟ้าจาก server ผ่าน URL
- รันพยากรณ์โหลดไฟฟ้า 5 ภาค
- ใช้ XGBoost, LightGBM, LSTM
- รวมผลด้วย Ensemble
- วัดผลด้วย MAPE, MAE, RMSE, BIAS
- Export ผลลัพธ์เป็น CSV รวมทุกภาค