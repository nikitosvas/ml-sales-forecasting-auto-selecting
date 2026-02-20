import json
from ast import fix_missing_locations

import pandas as pd
from forecast.models_registry import MODEL_REGISTRY
from utils.finish_formating_dframe import long_to_wide_forecast


def load_latest_policy_for_forecast(
        filename='policy_model.json'
):

    with open(filename, 'r', encoding='utf-8') as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    df['RUN_DATE'] = pd.to_datetime(df['RUN_DATE'])

    latest_policy = (
        df
        .sort_values('RUN_DATE')
        .groupby(["FULL_SIGN", "METRIC_NAME"])
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_policy

def forecast_current_month_by_policy(
        df: pd.DataFrame,
        policy_df: pd.DataFrame,
        forecast_start_date: pd.Timestamp
):
    """
        Считает актуальный прогноз текущего месяца по policy.

        Возвращает long df:
            DDATE | FULL_SIGN | METRIC_NAME | FORECAST
    """

    result = []

    for _, row in policy_df.iterrows():

        full_sign = row['FULL_SIGN']
        metric = row['METRIC_NAME']
        window = row['BEST_WINDOW']
        model_name = row['BEST_MODEL'].replace("_WMAPE", "")

        print(
            f"Актуальный прогноз: \n"
            f"{full_sign} | {metric} | {model_name} | {window}"
        )

        work_df = df[
            (df["FULL_SIGN"] == full_sign) &
            (df["METRIC_NAME"] == metric)
        ].copy()

        model_func = MODEL_REGISTRY[model_name]

        forecast_end_date = forecast_start_date + pd.offsets.MonthEnd(0)

        actual_forecast_df = model_func(
            df=work_df,
            start=forecast_start_date,
            end=forecast_end_date,
            window=window,
            full_sign=full_sign,
            metric_name=metric
        )

        actual_forecast_df['FULL_SIGN'] = full_sign
        actual_forecast_df['METRIC_NAME'] = metric

        result.append(actual_forecast_df)

    return pd.concat(result, ignore_index=True)


def run_policy_current_month_forecast(
        df,
        policy_file,
        forecast_start_date
):

    policy_df = load_latest_policy_for_forecast(policy_file)

    forecast_current_long = forecast_current_month_by_policy(
        df=df,
        policy_df=policy_df,
        forecast_start_date=forecast_start_date
    )

    finish_forecast_df = long_to_wide_forecast(forecast_current_long)

    print(finish_forecast_df)

    group_df = finish_forecast_df.groupby(["SALES_SUBSPECIES"]).agg({
        "SUM_SNDS": "sum",
        "SUM_PROFIT": "sum",
        "SUM_PROFIT_NO_KSP": "sum"
    })

    print(group_df)

    return finish_forecast_df








