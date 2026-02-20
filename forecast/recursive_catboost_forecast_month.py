# forecast/forecast_month.py
"""
        Модуль ML-прогноза продаж до конца месяца с помощью CatBoost.

        Используется рекурсивный подход:

        1. Для каждого дня прогнозного периода пересчитываются лаговые признаки:
           - LAG_7D, LAG_14D, LAG_28D
           - rolling средние

        2. Модель обучается на скользящем окне train_window_days

        3. Делается прогноз на следующий день

        4. Прогноз подставляется обратно в данные как факт,
           чтобы использоваться в будущих лагах (recursive forecasting)

        Важно:
        - прогноз строится по всем календарным дням месяца
        - праздники и выходные НЕ зануляются вручную
        - модель должна учитывать такие эффекты через IS_HOLIDAY feature

        :return: DataFrame [DDATE, FORECAST]
"""

import pandas as pd
import numpy as np

from features.lag_features import add_lags_means_for_model

from evaluation.backtest import split_train_and_test_data, split_X_y, predict_one_day
from models.catboost_model import train_catboost

def recursive_catboost_forecast_to_month_end(
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
    )  # Даты без выходных, их добавим ниже # Даты без выходных, их добавим ниже

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

        model = train_catboost(X_train, y_train)
        y_pred = predict_one_day(model, X_test)

        # IMPORTANCE FEATURES
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


    # = === MEAN IMPORTANCE LIST ====
    # mean_importance = np.mean(importance_list, axis=0)
    #
    # feature_importance_df = pd.DataFrame({
    #     "feature": X_train.columns,
    #     "mean_importance": mean_importance
    # }).sort_values("mean_importance", ascending=False)
    #
    # print("\n✅ MEAN Feature Importance:")
    # print(feature_importance_df.head(20).reset_index(drop=True))


    forecast_df = pd.DataFrame(forecasts)


    return forecast_df
