
import pandas as pd

from features.lag_features import add_lags_means_for_model
from evaluation.backtest import split_X_y
from models.light_gbm import train_lightgbm

def lightgbm_forecast_to_month_end(
        df: pd.DataFrame,
        forecast_start_date: pd.Timestamp,
        forecast_end_date: pd.Timestamp,
        train_window_days: int
):
    """
        Direct прогноз LightGBM:
        - обучаем модель один раз
        - прогнозируем сразу все дни месяца
        (без рекурсии)
    """

    # ✅ создаём фичи
    df_model = add_lags_means_for_model(df)

    # ✅ train/test split
    train_df = df_model[df_model["DDATE"] < forecast_start_date].tail(train_window_days)

    # ✅ создаём фичи
    df_model = add_lags_means_for_model(df)

    # ✅ train/test split
    train_df = df_model[df_model["DDATE"] < forecast_start_date].tail(train_window_days)

    test_df = df_model[
        (df_model["DDATE"] >= forecast_start_date) &
        (df_model["DDATE"] <= forecast_end_date)
        ]

    # ✅ split X/y
    X_train, y_train = split_X_y(train_df)
    X_test, _ = split_X_y(test_df)

    # ✅ обучение LightGBM
    model = train_lightgbm(X_train, y_train)

    y_pred = model.predict(X_test)

    forecast_df = pd.DataFrame({
        "DDATE": test_df["DDATE"].values,
        "FORECAST": y_pred
    })

    return forecast_df

