# features/calendar_features.py
import pandas as pd


def add_calendar_features(
        long_df: pd.DataFrame,
        holidays: list
) -> pd.DataFrame:
    """
       Добавляет календарные (временные) признаки в длинный датафрейм.

       Добавляемые признаки:
           - DAY_OF_WEEK      : день недели (0=Пн, 6=Вс)
           - IS_WEEKEND       : признак выходного дня
           - DAY_OF_MONTH     : номер дня в месяце
           - MONTH            : номер месяца
           - WEEK_OF_YEAR     : номер недели в году
           - START_OF_WEEK    : первый рабочий день недели (Пн)
           - END_OF_WEEK      : последний рабочий день недели (Пт)
           - START_OF_MONTH   : первый день месяца
           - END_OF_MONTH     : последний день месяца
           - DAYS_LEFT_IN_MONTH : сколько дней осталось до конца месяца
           - MONTH_PROGRESS   : прогресс месяца (0–1)
           - IS_HOLIDAY       : признак праздничного дня

       Дополнительно:
           - значения METRIC_VALUE на выходных принудительно обнуляются

       :param long_df: длинный датафрейм с фактическими данными
       :param holidays: коллекция дат праздников (datetime.date)
       :return: датафрейм с добавленными календарными признаками
    """

    df = long_df.copy()

    df["DAY_OF_WEEK"] = df["DDATE"].dt.dayofweek
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"]).isin([5, 6]).astype(int)

    df["DAY_OF_MONTH"] = df["DDATE"].dt.day
    df["MONTH"] = df["DDATE"].dt.month
    df["WEEK_OF_YEAR"] = df["DDATE"].dt.isocalendar().week.astype(int)

    df["START_OF_WEEK"] = (df["DAY_OF_WEEK"] == 0).astype(int)
    df["END_OF_WEEK"] = (df["DAY_OF_WEEK"] == 4).astype(int)

    df["START_OF_MONTH"] = (df["DAY_OF_MONTH"] == 1).astype(int)
    df["END_OF_MONTH"] = df["DDATE"].dt.is_month_end.astype(int)

    df["DAYS_LEFT_IN_MONTH"] = (
            df["DDATE"].dt.days_in_month - df["DDATE"].dt.day
    )

    df['MONTH_PROGRESS'] = round(df["DAY_OF_MONTH"] / df["DDATE"].dt.days_in_month, 3)

    df['IS_HOLIDAY'] = df['DDATE'].dt.date.isin(holidays).astype(int)



    # Обнуляем метрики на выходных и на праздники
    # df.loc[df['IS_WEEKEND'] == 1, 'METRIC_VALUE'] = 0
    # df.loc[df['IS_HOLIDAY'] == 1, 'METRIC_VALUE'] = 0

    return df