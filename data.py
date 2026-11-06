"""
M5 data (Walmart, 2011-2016): daily unit sales per product and store, weekly prices and a
calendar with events and SNAP days (US food-stamp payout days). This project uses one
store, CA_1, from 2014 onwards: 3,049 products, about 2.6 million product-days.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"    # mirror of the Kaggle files
RAW = Path("data/m5")
FILES = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]
STORE = "CA_1"
START = "2014-01-01"


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in archive.namelist():
        if Path(name).name in FILES:
            (RAW / Path(name).name).write_bytes(archive.read(name))


def calendar() -> pd.DataFrame:
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal["d"] = [f"d_{i}" for i in range(1, len(cal) + 1)]           # the mirror drops the d column
    return cal


def build(store: str = STORE, start: str = START) -> pd.DataFrame:
    cal = calendar()
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[sales["store_id"] == store].drop(columns=["state_id", "store_id"])
    long = sales.melt(id_vars=["item_id", "dept_id", "cat_id"], var_name="d", value_name="sales")
    long = long.merge(cal[["d", "date", "wm_yr_wk", "event_type_1", f"snap_{store[:2]}"]], on="d")
    long = long.rename(columns={f"snap_{store[:2]}": "snap", "event_type_1": "event_type"})

    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"] == store].drop(columns="store_id")
    long = long.merge(prices, on=["item_id", "wm_yr_wk"], how="left")
    # a product without a price was not on the shelf yet: those zeros are not demand
    long = long[long["sell_price"].notna() & (long["date"] >= start)]

    long = long.drop(columns=["d", "wm_yr_wk"]).sort_values(["item_id", "date"], ignore_index=True)
    for column in ["item_id", "dept_id", "cat_id", "event_type"]:
        long[column] = long[column].astype("category")
    long["sales"] = long["sales"].astype("float32")
    long["sell_price"] = long["sell_price"].astype("float32")
    return long


def load() -> pd.DataFrame:
    cache = Path(f"data/{STORE}.parquet")
    if not cache.exists():
        if not all((RAW / f).exists() for f in FILES):
            download()
        build().to_parquet(cache)
    return pd.read_parquet(cache)
