"""
Features for forecasting up to 28 days ahead in one go.

Every sales feature uses sales from at least 28 days before the day being predicted, so
the same model can forecast all 28 days without feeding its own forecasts back in. Prices
and the calendar are planned in advance, so they are used for the forecast day itself.
"""
import pandas as pd

HORIZON = 28
CATEGORICAL = ["item_id", "dept_id", "cat_id", "event_type"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """df: one row per item and day, sorted by item and date, without gaps."""
    out = df.copy()
    by_item = out.groupby("item_id", observed=True)
    shifted = by_item["sales"].shift(HORIZON)
    grouped = shifted.groupby(out["item_id"], observed=True)

    for lag in [28, 35, 42, 49, 364]:                       # same weekday 4-7 weeks and a year back
        out[f"lag_{lag}"] = by_item["sales"].shift(lag)
    for window in [7, 28, 56, 112]:
        out[f"mean_{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    out["std_28"] = grouped.transform(lambda s: s.rolling(28, min_periods=7).std())
    out["zero_share_56"] = (shifted == 0).astype("float32").groupby(out["item_id"], observed=True).transform(
        lambda s: s.rolling(56, min_periods=7).mean())
    out["days_on_sale"] = by_item.cumcount()

    out["price_vs_max"] = out["sell_price"] / by_item["sell_price"].cummax()
    out["price_change_week"] = out["sell_price"] / by_item["sell_price"].shift(7) - 1
    out["price_vs_dept"] = out["sell_price"] / out.groupby(["dept_id", "date"], observed=True)["sell_price"].transform("mean")

    out["weekday"] = out["date"].dt.day_of_week
    out["day_of_month"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["is_event"] = out["event_type"].notna().astype(int)
    for column in [c for c in out.columns if out[c].dtype == "float64"]:
        out[column] = out[column].astype("float32")
    return out


FEATURES = [
    "item_id", "dept_id", "cat_id", "event_type", "is_event", "snap", "weekday", "day_of_month", "month",
    "sell_price", "price_vs_max", "price_change_week", "price_vs_dept",
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_364", "mean_7", "mean_28", "mean_56", "mean_112", "std_28",
    "zero_share_56", "days_on_sale",
]
