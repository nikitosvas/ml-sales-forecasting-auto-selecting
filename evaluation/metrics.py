# evaluation/metrics.py
import pandas as pd

# ========== мЕТРИКИ для сравнения ==============================
def calc_metrics(bt_df):

    mae = bt_df["abs_error"].mean()

    wmape = (
        bt_df["abs_error"].sum() /
        bt_df["y_true"].abs().sum()
    )

    bias = (
        bt_df["error"].sum() /
        bt_df["y_true"].abs().sum()
    )

    return {
        "MAE": mae,
        "WMAPE": wmape,
        "BIAS": bias
    }

# ============== Подсчёт месячных метрик ===================
def calc_month_metrics(
    fact_df: pd.DataFrame,
    forecast_df: pd.DataFrame
):

    fact_month_sum = fact_df["METRIC_VALUE"].sum()
    forecast_month_sum = forecast_df["FORECAST"].sum()

    error = forecast_month_sum - fact_month_sum
    abs_error = abs(error)

    if fact_month_sum == 0:
        return None

    return {
        "FACT_MONTH": fact_month_sum,
        "FORECAST_MONTH": forecast_month_sum,
        "ERROR": error,
        "MAPE_MONTH": abs_error / abs(fact_month_sum),
        "WMAPE_MONTH": abs_error / abs(fact_month_sum),
        "BIAS_MONTH": error / abs(fact_month_sum)
    }