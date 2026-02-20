# forecast/xgboost_forecast_direct_month.py
import pandas as pd

from features.lag_features import add_lags_means_for_model
from evaluation.backtest import split_train_and_test_data, split_X_y
from models.xgboost_model import train_xgboost

# =========== ИСПРАВИТЬ НЕВЕРНО СТРОИТ ==================


def xgboost_forecast_direct_to_month_end(
    df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    """
    Direct Forecast через XGBoost.

    Отличие от Recursive CatBoost:
    - модель обучается один раз на train_window_days
    - сразу прогнозирует все даты месяца
    - без подстановки прогнозов назад в ряд

    Возвращает DataFrame:
        [DDATE, FORECAST]
    """

    df = df.sort_values("DDATE").copy()

    # ✅ создаём лаги
    df_model = add_lags_means_for_model(df)

    # =====================================================
    # ✅ 2) Train window
    # =====================================================
    train_df = df_model[
        df_model["DDATE"] < forecast_start_date
        ].tail(train_window_days)

    # =====================================================
    # ✅ 3) Будущий горизонт (все даты месяца)
    # =====================================================
    test_df = df_model[
        (df_model["DDATE"] >= forecast_start_date) &
        (df_model["DDATE"] <= forecast_end_date)
        ].copy()

    X_train, y_train = split_X_y(train_df)
    X_test, _ = split_X_y(test_df)

    # ✅ обучаем XGB
    model = train_xgboost(X_train, y_train)

    # ✅ прогноз сразу на весь месяц
    y_pred = model.predict(X_test)

    forecast_df = pd.DataFrame({
        "DDATE": test_df["DDATE"].values,
        "FORECAST": y_pred
    })


    return forecast_df.reset_index(drop=True)