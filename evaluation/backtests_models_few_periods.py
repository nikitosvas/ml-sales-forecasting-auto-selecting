
import pandas as pd

from evaluation.metrics import calc_month_metrics
from utils.printers import print_month_metrics

from forecast.baseline_month import baseline_forecast
from forecast.recursive_catboost_forecast_month import recursive_catboost_forecast_to_month_end
from forecast.direct_catboost_forecast_month import catboost_forecast_direct_to_month_end
from forecast.random_forest_forecast_month import random_forest_forecast_direct_to_month_end
from forecast.direct_lightgbm_forecast_month import lightgbm_forecast_to_month_end
from forecast.xgboost_forecast_direct_month import xgboost_forecast_direct_to_month_end
from forecast.recursive_lightGBM_forecast import recursive_lightGBM_forecast_to_month_end
from forecast.baseline_exponential_holt_winters_forecast import (
    baseline_simple_expon_forecast,
    baseline_holt_forecast,
    baseline_holt_winters_forecast
)
from forecast.models_registry import MODEL_REGISTRY



def run_monthly_backtests(
        df: pd.DataFrame,
        forecast_backtest_dates: list[pd.Timestamp],
        full_sign: str,
        metric_name: str,
        train_window_days: int,
        models_to_run: list[str]
) -> pd.DataFrame:

    results_backtest = []

    for start_date in forecast_backtest_dates:

        end_date = start_date + pd.offsets.MonthEnd(0)

        print('\n' + "=" * 60 )
        print(f'Месяц бэктеста: {start_date.date()} -> {end_date.date()}')
        print(f"Канал продаж: {full_sign}")
        print(f"Метрика: {metric_name}")
        print(f"Окно : {train_window_days} days")
        print("=" * 60)


        # ✅ Факт месяца
        fact_df = df[
            (df['DDATE'] >= start_date) &
            (df['DDATE'] <= end_date)
        ].copy()

        row = {
            "START_DATE": start_date.date(),
            "TRAIN_WINDOW_DAYS": train_window_days,
            "FULL_SIGN": full_sign,
            "METRIC_NAME": metric_name,
            "FACT_MONTH": fact_df['METRIC_VALUE'].sum()
        }

        wmape_values = {}

        forecast_dict = {}

        # Цикл по выбранным моделям
        for model_key in models_to_run:

            print(f'\n MODEL: {model_key}')

            model_func = MODEL_REGISTRY[model_key]

            forecast_df = model_func(
                df=df,
                start=start_date,
                end=end_date,
                window=train_window_days,
                full_sign=full_sign,
                metric_name=metric_name
            )

            metrics = calc_month_metrics(
                fact_df=fact_df,
                forecast_df=forecast_df
            )

            wmape = round(metrics["WMAPE_MONTH"], 4)

            row[f"{model_key}_WMAPE"] = wmape
            wmape_values[model_key] = wmape

            # печать по каждой модели
            print_month_metrics(
                full_sign=full_sign,
                metric=metric_name,
                forecast_start_date=start_date,
                month_metrics=metrics
            )

        # WINNER по моделям
        best_model = min(wmape_values, key=wmape_values.get)
        row["WINNER"] = best_model

        print("\n🏆 WINNER:", best_model)
        print("=" * 70)

        results_backtest.append(row)

    return pd.DataFrame(results_backtest)
