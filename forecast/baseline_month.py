# forecast/baseline_month.py
"""
    Модуль baseline-прогноза до конца месяца.

    Используется простая стратегия:

    1) Weekly Naive:
       прогноз строится как среднее значение продаж
       по каждому дню недели (понедельник, вторник и т.д.)

    2) OLS Trend:
       дополнительно рассчитывается коэффициент тренда,
       основанный на недельных суммах продаж.

    Baseline используется для:
    - сравнения с ML моделью
    - быстрого ориентира качества
    - проверки, что CatBoost действительно улучшает прогноз

    Важно:
    - праздники и выходные не зануляются вручную
    - baseline строит прогноз на все календарные дни месяца
"""

import numpy as np
import pandas as pd

# TREND_METHOD_OLS = "ols"

def calc_trend_coef_weekly(
        y_daily: pd.Series,
        horizont_weeks: int,
        min_coef: int=0.7
) -> float:
    """
        Рассчитывает коэффициент тренда на основе недельных сумм продаж.

        Шаги:

        1. Исходный дневной ряд агрегируется в недельные суммы.
        2. По недельным значениям строится линейный тренд (OLS).
        3. Рассчитывается мультипликативный коэффициент,
           который масштабирует прогноз на будущий горизонт.

        Преимущества недельного тренда:
        - меньше шума, чем при дневном тренде (так как в выходные и праздники продажи 0 могут быть, искажается тренд)
        - лучше отражает общий рост/падение

        Формула:
            trend_coef = 1 + (slope * horizont_weeks) / mean(weekly_sum)

        :param y_daily: дневной ряд значений метрики
        :param horizont_weeks: горизонт прогноза в неделях
        :return: коэффициент тренда (min = 0.7)
    """

    y_daily = y_daily.dropna()

    weekly_sum = (
        y_daily
        .groupby(np.arange(len(y_daily)) // 7)
        .sum()
    )

    t = np.arange(len(weekly_sum))
    slope, _ = np.polyfit(x=t, y=weekly_sum.values, deg=1)

    mean_level = weekly_sum.mean()
    if mean_level == 0:
        return 1.0

    trend_coef = 1 + (slope * horizont_weeks) / mean_level

    return max(round(trend_coef, 3), min_coef)

def baseline_forecast(
        df: pd.DataFrame,
        forecast_start_date: pd.Timestamp,
        forecast_end_date: pd.Timestamp,
        train_window_days: int
) -> pd.DataFrame:
    """
        Строит baseline-прогноз на период forecast_start_date → forecast_end_date.

        Алгоритм:

        1. Берём историю до даты прогноза (НЕ ВКЛЮЧАЯ !!!)
        2. Ограничиваем её окном train_window_days (к примеру 60 дней окно)
        3. Считаем средние значения по дням недели (Weekly Naive)
        4. Рассчитываем коэффициент недельного тренда (OLS)
        5. Для каждого дня будущего периода прогнозируем:

               forecast = weekly_mean(день_недели) * коэфф_недельного_тренда

        Важно:
        - прогноз строится на все календарные дни месяца
        - праздники НЕ зануляются вручную

        :return: DataFrame [DDATE, FORECAST]
    """

    history_df = (
        df[df['DDATE'] < forecast_start_date]
        .sort_values("DDATE")
        .copy()
    )

    train_df = history_df.tail(train_window_days)

    # среднее по дням недели
    dow_mean = train_df.groupby("DAY_OF_WEEK")["METRIC_VALUE"].mean()

    horizon_days = (forecast_end_date - forecast_start_date).days + 1
    horizon_weeks = int(np.ceil(horizon_days / 7))

    trend_coef = calc_trend_coef_weekly(
        y_daily=train_df['METRIC_VALUE'],
        horizont_weeks=horizon_weeks
    )

    # --- прогноз ---
    forecast_dates = pd.date_range(
        start=forecast_start_date,
        end=forecast_end_date,
        freq="D"
    )

    forecasts = []

    for d in forecast_dates:

        dow = d.dayofweek
        base = dow_mean.get(dow, history_df["METRIC_VALUE"].mean())
        y_pred = base * trend_coef

        forecasts.append(
            {
                "DDATE": d,
                "FORECAST": y_pred
            }
        )

    return pd.DataFrame(forecasts)




