"""
Forecasts daily unit sales for every product in store CA_1, 28 days ahead.

1. Backtest on three earlier 28-day windows (each trained only on data before it):
   two baselines and LightGBM with three loss functions.
2. Pick the loss function with the lowest average error over the backtest.
3. Forecast the final 28 days (25 April - 22 May 2016) once, and break the errors down
   by department, demand pattern, days ahead and event days.

    python forecast.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import data
import evaluation
import features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HORIZON = features.HORIZON
COMMON = dict(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
LOSSES = {
    "tweedie": dict(objective="tweedie", tweedie_variance_power=1.1),
    "poisson": dict(objective="poisson"),
    "squared error": dict(objective="regression"),
}


def windows(df: pd.DataFrame) -> tuple[list[pd.Timestamp], pd.Timestamp]:
    """Forecast origins (last day of known sales): three backtest windows, then the final test."""
    test_origin = df["date"].max() - pd.Timedelta(days=HORIZON)
    return [test_origin - pd.Timedelta(days=HORIZON * k) for k in (3, 2, 1)], test_origin


def split(feat: pd.DataFrame, origin: pd.Timestamp):
    train = feat[feat["date"] <= origin]
    future = feat[(feat["date"] > origin) & (feat["date"] <= origin + pd.Timedelta(days=HORIZON))]
    return train, future


def lightgbm(train: pd.DataFrame, loss: str) -> LGBMRegressor:
    model = LGBMRegressor(**COMMON, **LOSSES[loss])
    return model.fit(train[features.FEATURES], train["sales"], categorical_feature=features.CATEGORICAL)


def forecasts(train, future, losses) -> dict[str, np.ndarray]:
    out = {"same as last week": evaluation.seasonal_naive(train, future),
           "28-day average": evaluation.moving_average(train, future)}
    for loss in losses:
        out[f"lightgbm ({loss})"] = np.clip(lightgbm(train, loss).predict(future[features.FEATURES]), 0, None)
    return out


def breakdown(future: pd.DataFrame, forecast: np.ndarray, scale, patterns, origin) -> pd.DataFrame:
    frame = future.assign(forecast=forecast)
    frame["demand pattern"] = frame["item_id"].map(patterns).astype(str)
    frame["week ahead"] = ((frame["date"] - origin).dt.days - 1) // 7 + 1
    frame["event day"] = np.where(frame["is_event"] == 1, "event", "normal day")
    rows = []
    for column in ["dept_id", "demand pattern", "week ahead", "event day"]:
        for value, part in frame.groupby(column, observed=True):
            rows.append({"segment": column, "value": str(value), "products": part["item_id"].nunique(),
                         "units": float(part["sales"].sum()),
                         **evaluation.scores(part["sales"], part["forecast"], part["item_id"], scale)})
    return pd.DataFrame(rows)


def plot_store_total(future, results, path):
    totals = future.assign(**{k: v for k, v in results.items()}).groupby("date")[["sales", *results]].sum()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(totals.index, totals["sales"], color="#111827", linewidth=2, label="Actual")
    styles = {"same as last week": ("#9ca3af", ":"), "28-day average": ("#d1d5db", "--")}
    for name in results:
        color, style = styles.get(name, ("#2563eb", "-"))
        ax.plot(totals.index, totals[name], color=color, linestyle=style, label=name)
    ax.set_ylabel("Units per day, whole store")
    ax.set_title("Final 28 days: store CA_1, sum of 3,049 product forecasts", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(model, path):
    gain = pd.Series(model.booster_.feature_importance("gain"), index=features.FEATURES).nlargest(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(gain.index, gain / gain.sum() * 100, color="#2563eb")
    ax.set_xlabel("Share of total gain (%), top 12 features")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = data.load()
    feat = features.build(df)
    backtest_origins, test_origin = windows(df)
    Path("results").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    rows = []
    for fold, origin in enumerate(backtest_origins, start=1):
        train, future = split(feat, origin)
        scale = evaluation.rmsse_scale(train)
        for name, forecast in forecasts(train, future, LOSSES).items():
            rows.append({"window": fold, "origin": origin.date().isoformat(), "model": name,
                         **evaluation.scores(future["sales"], forecast, future["item_id"], scale)})
            print(rows[-1])
    backtest = pd.DataFrame(rows)
    backtest.to_csv("results/backtest.csv", index=False)
    mean = backtest.groupby("model")[["wape", "rmsse", "bias"]].mean().sort_values("rmsse")
    print(mean.round(4))

    best = mean.index[0].removeprefix("lightgbm (").removesuffix(")") if mean.index[0].startswith("lightgbm") \
        else min(LOSSES, key=lambda loss: mean.loc[f"lightgbm ({loss})", "rmsse"])

    train, future = split(feat, test_origin)
    scale = evaluation.rmsse_scale(train)
    model = lightgbm(train, best)
    results = {"same as last week": evaluation.seasonal_naive(train, future),
               "28-day average": evaluation.moving_average(train, future),
               f"lightgbm ({best})": np.clip(model.predict(future[features.FEATURES]), 0, None)}
    test = {name: evaluation.scores(future["sales"], f, future["item_id"], scale) for name, f in results.items()}
    totals = future.assign(**results).groupby("date")[["sales", *results]].sum()
    for name in results:
        test[name]["store_total_wape"] = float((totals[name] - totals["sales"]).abs().sum() / totals["sales"].sum())

    patterns = evaluation.demand_pattern(train)
    detail = breakdown(future, results[f"lightgbm ({best})"], scale, patterns, test_origin)
    detail.to_csv("results/breakdown.csv", index=False)
    plot_store_total(future, results, "docs/store_total.png")
    plot_importance(model, "docs/importance.png")

    summary = {"store": data.STORE, "products": int(df["item_id"].nunique()), "rows": len(df),
               "test_window": [str((test_origin + pd.Timedelta(days=1)).date()), str(df["date"].max().date())],
               "chosen_loss": best, "backtest_mean": mean.reset_index().to_dict("records"), "test": test,
               "demand_patterns": patterns.value_counts().to_dict()}
    Path("results/results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(test, indent=2))
    print(detail.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
