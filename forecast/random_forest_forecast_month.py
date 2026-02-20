
import pandas as pd

from features.lag_features import add_lags_means_for_model
from evaluation.backtest import split_train_and_test_data, split_X_y
from models.random_forest_regressor import train_rand_forest_reggr

def random_forest_forecast_direct_to_month_end(
    df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    """
    Direct-прогноз до конца месяца с помощью RandomForest.

    Отличие:
    - обучаем модель 1 раз
    - прогнозируем сразу все даты месяца
    - нет рекурсии
    """


    # ✅ лаги + rolling
    df_model = add_lags_means_for_model(df)

    # ✅ обучаемся только на истории
    train_df = df_model[df_model["DDATE"] < forecast_start_date].tail(train_window_days)

    # ✅ тестовый горизонт
    test_df = df_model[
        (df_model["DDATE"] >= forecast_start_date) &
        (df_model["DDATE"] <= forecast_end_date)
    ]

    if len(test_df) == 0:
        return pd.DataFrame()

    # split
    X_train, y_train = split_X_y(train_df)
    X_test, _ = split_X_y(test_df)

    # ✅ train model
    model = train_rand_forest_reggr(X_train, y_train)

    # ✅ predict all days at once
    y_pred = model.predict(X_test)

    forecast_df = pd.DataFrame({
        "DDATE": test_df["DDATE"].values,
        "FORECAST": y_pred
    })

    return forecast_df
