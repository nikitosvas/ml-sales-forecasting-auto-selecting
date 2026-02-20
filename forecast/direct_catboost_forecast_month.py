# forecast/direct_catboost_forecast_month.py

import pandas as pd

from features.lag_features import add_lags_means_for_model
from evaluation.backtest import split_X_y
from models.catboost_model import train_catboost

def catboost_forecast_direct_to_month_end(
    df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    """
       ✅ Direct Multi-Step Forecast (без рекурсии)

       Логика:

       1) Строим лаги и rolling признаки на всей истории
       2) Берём train только ДО даты прогноза
       3) Обучаем CatBoost один раз
       4) Делаем прогноз сразу на весь месяц (одним predict)
       5) Ошибка не накапливается, как в рекурсии
    """


    # ✅ строим лаги один раз
    df_model = add_lags_means_for_model(df)

    # ✅ train = только история до старта
    train_df = df_model[df_model["DDATE"] < forecast_start_date]
    train_df = train_df.tail(train_window_days)

    # ✅ future даты месяца
    future_dates = pd.date_range(
        start=forecast_start_date,
        end=forecast_end_date
    )

    # ✅ будущие строки
    future_df = df_model[df_model["DDATE"].isin(future_dates)].copy()

    # ✅ train split
    X_train, y_train = split_X_y(train_df)

    # ✅ обучаем один раз
    model = train_catboost(X_train, y_train)

    # ✅ прогноз сразу на месяц
    X_future, _ = split_X_y(future_df)

    preds = model.predict(X_future)

    forecast_df = pd.DataFrame({
        "DDATE": future_df["DDATE"],
        "FORECAST": preds
    })

    return forecast_df

