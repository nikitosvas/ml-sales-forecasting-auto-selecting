# utils/printers.py
import pandas as pd
from datetime import timedelta

def print_month_metrics(
    full_sign: str,
    metric: str,
    forecast_start_date: pd.Timestamp,
    month_metrics: dict
):

    # Чтобы вывести дату начала прогноза и последнее число месяца как дату конца прогноза
    predict_period_start = forecast_start_date.strftime("%Y-%m-%d")

    predict_period_end = (
        forecast_start_date.replace(day=28) + timedelta(days=4)
    ).replace(day=1) - timedelta(days=1)
    predict_period_end = predict_period_end.strftime("%Y-%m-%d")

    print(f"\n{full_sign} | {metric} | {predict_period_start} : {predict_period_end}")
    print("-" * 50)

    if month_metrics is None:
        print("Нет данных для расчёта метрик")
        return

    print(f"FACT_MONTH     : {round(month_metrics['FACT_MONTH'], 3):,.0f}")
    print(f"FORECAST_MONTH : {round(month_metrics['FORECAST_MONTH'], 3):,.0f}")
    print(f"ERROR          : {round(month_metrics['ERROR'], 3):,.0f}")
    print(f"MAPE           : {round(month_metrics['MAPE_MONTH'], 3):.4f}")
    print(f"WMAPE          : {round(month_metrics['WMAPE_MONTH'], 3):.4f}")
    print(f"BIAS           : {round(month_metrics['BIAS_MONTH'], 3):.4f}")