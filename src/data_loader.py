from pathlib import Path
import logging
import requests
import pandas as pd

from config import DATE_COL


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL in df.columns:
        # Source CSVs use DD/MM/YYYY HH:MM:SS. Parsing without an explicit
        # format lets pandas guess MM/DD/YYYY instead, which silently turns
        # most rows into NaT (e.g. "22/06/2026" has no valid month-22).
        # NaT.year is nan (a float), which later crashes the holidays
        # library deep inside its internal `range(start_year, year)` call
        # with "TypeError: 'float' object cannot be interpreted as an
        # integer" — so this isn't just a data-quality nicety, it's load-bearing.
        parsed = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if parsed.isna().all() and len(df) > 0:
            # Fallback for if the upstream format ever changes shape.
            parsed = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        df[DATE_COL] = parsed
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def download_csv(url: str, destination: Path, timeout: int = 60) -> Path:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def load_csv_with_cache(url: str, cache_path: Path) -> pd.DataFrame:
    try:
        download_csv(url, cache_path)
        logging.info("Downloaded fresh file: %s", url)
    except Exception as exc:
        if cache_path.exists():
            logging.warning("Using cached file for %s because download failed: %s", url, exc)
        else:
            raise RuntimeError(f"Download failed and no cache exists for {url}: {exc}") from exc

    df = pd.read_csv(cache_path)
    return ensure_datetime(df)


def load_zone_data(zone_name: str, zone_config: dict, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cache = cache_dir / zone_config["cache_train"]
    test_cache = cache_dir / zone_config["cache_test"]

    train_df = load_csv_with_cache(zone_config["train"], train_cache)
    test_df = load_csv_with_cache(zone_config["test"], test_cache)

    train_df["zone"] = zone_name
    test_df["zone"] = zone_name
    return train_df, test_df
