import numpy as np
import pandas as pd
import pytest

import evaluation
import features


def series(sales_by_item: dict[str, list[float]], start="2016-01-01") -> pd.DataFrame:
    frames = []
    for item, sales in sales_by_item.items():
        frames.append(pd.DataFrame({
            "item_id": item, "dept_id": "FOODS_1", "cat_id": "FOODS", "event_type": None, "snap": 0,
            "date": pd.date_range(start, periods=len(sales)), "sales": np.asarray(sales, "float32"),
            "sell_price": np.float32(2.0),
        }))
    df = pd.concat(frames, ignore_index=True)
    for column in features.CATEGORICAL:
        df[column] = df[column].astype("category")
    return df


# ─── Features ──────────────────────────────────────────────────────────────────

def test_features_only_use_sales_from_28_days_back():
    rng = np.random.default_rng(0)
    sales = rng.poisson(3, 120).astype(float)
    changed = sales.copy()
    changed[-27:] += 100                                   # the 27 days before the last day
    last = lambda df: features.build(df).iloc[-1][features.FEATURES].drop(features.CATEGORICAL)
    a, b = last(series({"A": sales})), last(series({"A": changed}))
    pd.testing.assert_series_equal(a, b)


def test_lags_and_means():
    f = features.build(series({"A": list(range(1, 61))}))
    row = f.iloc[59]                                       # sales 60 on day 60
    assert row["lag_28"] == 32 and row["lag_35"] == 25
    assert row["mean_7"] == pytest.approx(np.mean(range(26, 33)))   # 7 days ending 28 days back


def test_lags_do_not_cross_products():
    f = features.build(series({"A": [5] * 40, "B": [1] * 40}))
    b = f[f["item_id"] == "B"]
    assert b["lag_28"].iloc[:28].isna().all() and (b["lag_28"].iloc[28:] == 1).all()


def test_price_features():
    df = series({"A": [1] * 14})
    df.loc[10:, "sell_price"] = 1.5
    f = features.build(df)
    assert f["price_vs_max"].iloc[9] == 1.0 and f["price_vs_max"].iloc[13] == 0.75      # price cut 2.00 -> 1.50
    assert f["price_change_week"].iloc[10] == pytest.approx(-0.25)


# ─── Baselines ─────────────────────────────────────────────────────────────────

def test_seasonal_naive_repeats_last_week():
    history = series({"A": [9] * 7 + list(range(1, 8))})              # last week: 1..7
    future = series({"A": [0] * 10}, start=str((history["date"].max() + pd.Timedelta(days=1)).date()))
    assert evaluation.seasonal_naive(history, future).tolist() == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]


def test_moving_average():
    history = series({"A": [100] * 10 + [2] * 28, "B": [0] * 38})
    future = series({"A": [0, 0], "B": [0, 0]}, start="2016-02-08")
    assert evaluation.moving_average(history, future).tolist() == [2, 2, 0, 0]


# ─── Metrics ───────────────────────────────────────────────────────────────────

def test_rmsse_scale_starts_at_first_sale():
    history = series({"A": [0, 0, 0, 2, 4, 2]})
    assert evaluation.rmsse_scale(history)["A"] == pytest.approx(np.mean([4, 4]))   # diffs of 2,4,2


def test_scores():
    scale = pd.Series({"A": 4.0, "B": 1.0})
    s = evaluation.scores([2, 2, 1, 1], [4, 2, 1, 2], ["A", "A", "B", "B"], scale)
    assert s["wape"] == pytest.approx(3 / 6) and s["bias"] == pytest.approx(9 / 6 - 1)
    assert s["rmsse"] == pytest.approx(np.mean([np.sqrt(2 / 4), np.sqrt(0.5 / 1)]))


@pytest.mark.parametrize("sales, pattern", [
    ([5, 6, 5, 6] * 91, "smooth"),
    ([1, 20, 2, 30] * 91, "erratic"),
    ([0, 0, 5, 0] * 91, "intermittent"),
    ([0, 0, 1, 0, 0, 40] * 61, "lumpy"),
    ([0] * 364 + [3], "rarely sold"),
])
def test_demand_patterns(sales, pattern):
    assert evaluation.demand_pattern(series({"A": sales}))["A"] == pattern
