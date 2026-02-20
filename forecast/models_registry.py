# forecast/model_registry.py

"""
Единый реестр моделей.

Каждая модель должна возвращать DataFrame вида:

    [DDATE, FORECAST]

Нужно чтобы автоматически без лишнего комментирования сравнивать ЛЮБЫЕ модели, которые хочу
"""

from forecast.baseline_month import baseline_forecast
from forecast.baseline_exponential_holt_winters_forecast import (
    baseline_simple_expon_forecast,
    baseline_holt_forecast,
    baseline_holt_winters_forecast
)

from forecast.direct_catboost_forecast_month import catboost_forecast_direct_to_month_end
from forecast.recursive_catboost_forecast_month import recursive_catboost_forecast_to_month_end
from forecast.direct_lightgbm_forecast_month import lightgbm_forecast_to_month_end
from forecast.recursive_lightGBM_forecast import recursive_lightGBM_forecast_to_month_end
from forecast.random_forest_forecast_month import random_forest_forecast_direct_to_month_end
from forecast.xgboost_forecast_direct_month import xgboost_forecast_direct_to_month_end


# =========================================================
# ✅ BASELINES
# =========================================================

def model_baseline_ols(df, start, end, window, **kwargs):
    # Baseline OLS Weekly Naive + Trend
    return baseline_forecast(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

def model_baseline_simple_smooth(df, start, end, window, **kwargs):
    # Simple Exponential Smoothing
    return baseline_simple_expon_forecast(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

def model_baseline_holt_smooth(df, start, end, window, **kwargs):
    # Holt двойное экспоненц сглаживание
    return baseline_holt_forecast(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

def model_baseline_holt_winters(df, start, end, window, **kwargs):
    # Holt-Winters тройное экспоненц сглаживание
    return baseline_holt_winters_forecast(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

# =========================================================
# ✅ RANDOM FOREST
# =========================================================
def model_random_forest(df, start, end, window, **kwargs):
    # Random Forest Direct Forecast (МЛ моделька Слечайные лес, строит несколько деревьев и берет среднее значение по их результатам)
    return random_forest_forecast_direct_to_month_end(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

# =========================================================
# ✅ CATBOOST
# =========================================================
def model_catboost_recursive(df, start, end, window, full_sign=None, metric_name=None, **kwargs):
    # ✅ CatBoost рекурсивно (прогноз подставляется как факт)
    return recursive_catboost_forecast_to_month_end(
        df=df,
        full_sign=full_sign,
        metric_name=metric_name,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

def model_catboost_direct(df, start, end, window, **kwargs):
    # ✅ CatBoost сразу на весь месяц
    return catboost_forecast_direct_to_month_end(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

 # =========================================================
# ✅ LIGHTGBM
# =========================================================
def model_lightgbm_recursive(df, start, end, window, full_sign=None, metric_name=None, **kwargs):
    # ✅ CatBoost рекурсивно (прогноз подставляется как факт)
    return recursive_lightGBM_forecast_to_month_end(
        df=df,
        full_sign=full_sign,
        metric_name=metric_name,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

def model_lightgbm_direct(df, start, end, window, **kwargs):
    # ✅ CatBoost сразу на весь месяц
    return lightgbm_forecast_to_month_end(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )

# =========================================================
# ✅ XGBOOST
# =========================================================
def model_xgb_direct(df, start, end, window, **kwargs):
    # ✅ CatBoost сразу на весь месяц
    return xgboost_forecast_direct_to_month_end(
        df=df,
        forecast_start_date=start,
        forecast_end_date=end,
        train_window_days=window
    )


MODEL_REGISTRY = {

    # Baselines
    "BASELINE_OLS": model_baseline_ols,
    "BASELINE_EXPON": model_baseline_simple_smooth,
    "BASELINE_HOLT": model_baseline_holt_smooth,
    "BASELINE_HOLT_WINTERS": model_baseline_holt_winters,

    # ML models
    "BASELINE_RF": model_random_forest,

    "CATBOOST_RECURSIVE": model_catboost_recursive,
    "CATBOOST_DIRECT": model_catboost_direct,

    "LIGHTGBM_RECURSIVE": model_lightgbm_recursive,
    "LIGHTGBM_DIRECT": model_lightgbm_direct,

    "XGB_DIRECT": model_xgb_direct,
}

