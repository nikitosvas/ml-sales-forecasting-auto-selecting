

import pandas as pd

from features.lag_features import add_lags_means_for_model

from evaluation.backtest import split_train_and_test_data, split_X_y, predict_one_day
from models.light_gbm import train_lightgbm


# ============= Прогноз до конца месяца ====================
def recursive_lightGBM_forecast_to_month_end(
    df: pd.DataFrame,
    full_sign: str,
    metric_name: str,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    train_window_days: int
) -> pd.DataFrame:
    '''
        Строит рекурсивный ML-прогноз до конца месяца.

        Алгоритм:

        1. Генерируем список дат от forecast_start_date до конца месяца.
        2. Для каждого дня:
           - считаем лаги и rolling признаки
           - выделяем train/test по окну train_window_days
           - обучаем CatBoost
           - прогнозируем значение на текущий день
           - сохраняем прогноз
           - подставляем прогноз как факт в work_df

        Результат:
        - прогнозный DataFrame по всем датам месяца

        Важно:
        - праздники НЕ зануляются вручную
        - модель должна учиться эффекту праздников через IS_HOLIDAY
    '''

    forecast_dates = pd.date_range(
        start=forecast_start_date,
        end=forecast_end_date
    ) # Даты без выходных, их добавим ниже

    work_df = df.copy()

    forecasts = []
    importance_list = []

    for d in forecast_dates:

        if d not in work_df["DDATE"].values:
            print("Нет строки на дату", d)

        # ✅ лаги и rolling
        df_model = add_lags_means_for_model(
            df=work_df
        )

        # train/test split
        train_df, test_df = split_train_and_test_data(
            df=df_model,
            forecast_date=d,
            train_window_days=train_window_days
        )

        X_train, y_train = split_X_y(train_df)
        X_test, _ = split_X_y(test_df)

        model = train_lightgbm(X_train, y_train)
        y_pred = predict_one_day(model, X_test)

        # # IMPORTANCE FEATURES
        # imp = model.get_feature_importance()
        # importance_list.append(imp)

        forecasts.append({
            "DDATE": d,
            "FORECAST": y_pred
        })

        mask = (
                (work_df["DDATE"] == d) &
                (work_df["FULL_SIGN"] == full_sign) &
                (work_df["METRIC_NAME"] == metric_name)
        )

        work_df.loc[mask, "METRIC_VALUE"] = y_pred

    forecast_df = pd.DataFrame(forecasts)


    return forecast_df