# evaluation/backtest.py
"""
    Модуль бэктестинга и временного разбиения данных.

    Отвечает за:
    - корректное временное разделение train / test
    - обучение модели на скользящем окне
    - прогноз одного дня
    - rolling backtest по историческому периоду

    Используется:
    - для оценки качества моделей
    - для сравнения baseline и ML-подходов
"""

import pandas as pd
from models.catboost_model import train_catboost

def split_train_and_test_data(
        df: pd.DataFrame,
        forecast_date: pd.Timestamp,
        train_window_days: int
):
    """
        Разбивает модельный датафрейм на обучающую и тестовую выборки
        для прогнозирования одного конкретного дня.

        Логика:
        - train_df: последние train_window_days ДО forecast_date
        - test_df : ровно одна строка на дату forecast_date

        Важно:
        - используется строго временное разделение
        - утечка будущего полностью исключена

        :param df: датафрейм с готовыми фичами (лаги, rolling и т.д.)
        :param forecast_date: дата, на которую делаем прогноз
        :param train_window_days: размер окна обучения (в днях)
        :return: train_df, test_df
    """

    train_end = forecast_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=train_window_days - 1)

    train_df = df[
        (df['DDATE'] >= train_start) &
        (df['DDATE'] <= train_end)
    ].copy()

    test_df = df[
        df['DDATE'] == forecast_date
    ].copy() # В данном случае кусок для теста будет 1 день, далее рекурсивно до конца мес можно считать

    return train_df, test_df

def split_X_y(
        df,
        target_col="METRIC_VALUE"
):
    """
        Разделяет датафрейм на признаки (X) и таргет (y).

        Из X исключаются:
        - таргет
        - дата
        - идентификаторы каналов и метрик

        :param df: датафрейм с фичами
        :param target_col: имя таргет-колонки
        :return: (X, y)
    """

    X = df.drop(
        columns=[
            "METRIC_VALUE",
            "DDATE",
            "FULL_SIGN",
            "SALES_SUBSPECIES",
            "SIGN_IRIS",
            "METRIC_NAME"
        ]
    )
    y = df[target_col]

    return X, y

def predict_one_day(model, X_test: pd.DataFrame):

    """
        Делает прогноз одного дня.

        Выделена в отдельную функцию для:
        - унификации интерфейса прогнозирования
        - упрощения замены модели в будущем

        :param model: обученная модель
        :param X_test: признаки для одного дня
        :return: числовое значение прогноза
    """

    predict_one_day = model.predict(X_test)[0]

    return predict_one_day


# ========================= Один шаг бэктеста ============================
def backtest_one_day(
        df_model: pd.DataFrame,
        forecast_date: pd.Timestamp,
        train_window_days: int
):
    """
        Выполняет один шаг бэктеста (прогноз одного дня).

        Последовательность:
        1. Разделение данных на train / test
        2. Обучение модели
        3. Прогноз одного дня
        4. Расчёт ошибок

        Используется как базовый блок
        для rolling backtest и оценки качества модели.

        :param df_model: датафрейм с фичами для одной связки (канал + метрика)
        :param forecast_date: дата прогноза
        :param train_window_days: размер обучающего окна
        :return: словарь с фактом, прогнозом и ошибками
    """

    train_df, test_df = split_train_and_test_data(
        df=df_model,
        forecast_date=forecast_date,
        train_window_days=train_window_days
    )

    # if len(train_df) < train_window_days:
    #     return f'Недостаточно данных обучения. Окно для трени: {train_window_days}, данных в трен датафрейме: {len(train_df)}'

    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)

    model = train_catboost(X_train, y_train)
    y_pred = predict_one_day(model, X_test)

    return {
        "date": forecast_date,
        "y_true": y_test.values[0],
        "y_pred": y_pred,
        "error": y_pred - y_test.values[0],
        "abs_error": abs(y_pred - y_test.values[0])
    }

# =============== Бэктест за определенный период по КАЖДОМУ ДНЮ !!!! =============
def run_backtest(
    df_model: pd.DataFrame,
    start_forecast_date: pd.Timestamp,
    n_months: int,
    train_window_days: int
):
    """
        Выполняет rolling backtest за исторический период.

        Логика:
        - берём период: start_forecast_date - n_months
        - для каждого рабочего дня:
            - обучаем модель
            - прогнозируем 1 день
            - сохраняем ошибки

        Используется для:
        - расчёта MAE / WMAPE / BIAS
        - сравнения моделей между собой

        :param df_model: датафрейм с фичами одной связки
        :param start_forecast_date: дата начала прогнозирования
        :param n_months: глубина бэктеста (в месяцах)
        :param train_window_days: размер обучающего окна
        :return: DataFrame с результатами бэктеста
    """

    result = []

    # Опередляем начальную дату с которой будем проводить бэктесты.
    # Дата с которой прогнозируем - кол-во месяцев.
    start_date = start_forecast_date - pd.DateOffset(months=n_months)

    # Генерируем список дат бэктеста между первой и последней
    backtest_dates = pd.date_range(
        start=start_date,
        end=start_forecast_date - pd.Timedelta(days=1),
        freq="D"
    )

    backtest_dates = backtest_dates[
        (backtest_dates.dayofweek) < 5
        # (~backtest_dates.isin(holidays)) Отрезаются праздники
    ]

    # Перебираем каждую дату и на НЕЕ прогнозируем
    for d in backtest_dates:
        res = backtest_one_day(
            df_model,
            forecast_date=d,
            train_window_days=train_window_days
        )
        if res is not None:
            result.append(res)

    return pd.DataFrame(result)
# =====================================================================
