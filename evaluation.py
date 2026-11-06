"""Baselines, error metrics and the demand-pattern classes used in the error analysis."""
import numpy as np
import pandas as pd

ID = "item_id"


def seasonal_naive(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Each product sells the same as on the same weekday of the last week before the forecast."""
    last_week = history[history["date"] > history["date"].max() - pd.Timedelta(days=7)]
    lookup = last_week.assign(weekday=last_week["date"].dt.day_of_week).set_index([ID, "weekday"])["sales"]
    keys = pd.MultiIndex.from_arrays([future[ID], future["date"].dt.day_of_week])
    return lookup.reindex(keys).fillna(0).to_numpy()


def moving_average(history: pd.DataFrame, future: pd.DataFrame, days: int = 28) -> np.ndarray:
    """Each product sells its average of the last 28 days, every day."""
    recent = history[history["date"] > history["date"].max() - pd.Timedelta(days=days)]
    return future[ID].map(recent.groupby(ID, observed=True)["sales"].mean()).fillna(0).to_numpy(dtype=float)


def rmsse_scale(history: pd.DataFrame) -> pd.Series:
    """M5 scale per product: mean squared day-to-day change over its history (from its first sale)."""
    def scale(s):
        s = s.to_numpy()
        started = s[np.argmax(s > 0):] if (s > 0).any() else s
        return np.mean(np.diff(started) ** 2) if len(started) > 1 else np.nan
    return history.groupby(ID, observed=True)["sales"].apply(scale)


def scores(actual, forecast, items, scale: pd.Series) -> dict:
    """
    wape:  total absolute error / total sales, the number a planner understands
    rmsse: M5's scaled error per product (1.0 = as good as a naive one-step forecast in-sample), averaged
    bias:  total forecast / total sales - 1 (positive = over-forecast)
    """
    frame = pd.DataFrame({"y": np.asarray(actual, float), "f": np.asarray(forecast, float), ID: np.asarray(items)})
    frame["sq"] = (frame["y"] - frame["f"]) ** 2
    per_item = frame.groupby(ID, observed=True)["sq"].mean() / scale.reindex(frame[ID].unique()).to_numpy()
    per_item = np.sqrt(per_item.replace([np.inf], np.nan).dropna())
    return {
        "wape": float((frame["y"] - frame["f"]).abs().sum() / frame["y"].sum()),
        "rmsse": float(per_item.mean()),
        "bias": float(frame["f"].sum() / frame["y"].sum() - 1),
    }


def demand_pattern(history: pd.DataFrame) -> pd.Series:
    """
    Syntetos-Boylan classes from the last year of sales: how often a product sells (ADI, average
    days between sales) and how much the amount varies when it does (CV squared).
    """
    recent = history[history["date"] > history["date"].max() - pd.Timedelta(days=365)]

    def classify(s):
        nonzero = s[s > 0]
        if len(nonzero) < 2:
            return "rarely sold"
        adi = len(s) / len(nonzero)
        cv2 = (nonzero.std() / nonzero.mean()) ** 2
        if adi < 1.32:
            return "smooth" if cv2 < 0.49 else "erratic"
        return "intermittent" if cv2 < 0.49 else "lumpy"
    return recent.groupby(ID, observed=True)["sales"].apply(classify)
