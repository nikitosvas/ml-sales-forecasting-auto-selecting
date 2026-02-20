
import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)

def baseline_simple_expon_forecast(
    df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    """
    ✅ Simple Exponential Smoothing baseline
    Только уровень (без тренда)
    """

    history = df[df["DDATE"] < forecast_start_date].sort_values("DDATE")
    train = history.tail(train_window_days)

    y = train["METRIC_VALUE"].values

    model = SimpleExpSmoothing(y).fit()

    horizon = (forecast_end_date - forecast_start_date).days + 1
    preds = model.forecast(horizon)

    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    return pd.DataFrame({
        "DDATE": forecast_dates,
        "FORECAST": preds
    })


def baseline_holt_forecast(
    df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    """
    ✅ Holt Linear Trend baseline
    Уровень + тренд
    """

    history = df[df["DDATE"] < forecast_start_date].sort_values("DDATE")
    train = history.tail(train_window_days)

    y = train["METRIC_VALUE"].values

    model = Holt(y).fit()

    horizon = (forecast_end_date - forecast_start_date).days + 1
    preds = model.forecast(horizon)

    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    return pd.DataFrame({
        "DDATE": forecast_dates,
        "FORECAST": preds
    })


def baseline_holt_winters_forecast(
    df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    """
    ✅ Holt-Winters baseline
    Уровень + тренд + сезонность (weekly = 7)
    """

    history = df[df["DDATE"] < forecast_start_date].sort_values("DDATE")
    train = history.tail(train_window_days)

    y = train["METRIC_VALUE"].values

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=7
    ).fit()

    horizon = (forecast_end_date - forecast_start_date).days + 1
    preds = model.forecast(horizon)

    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    return pd.DataFrame({
        "DDATE": forecast_dates,
        "FORECAST": preds
    })

