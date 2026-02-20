# features/lag_features.py
import pandas as pd

def add_lags_means_for_model(
    df: pd.DataFrame
) -> pd.DataFrame:
    '''
    Создаём датафрейм для CATBOOST:
    - лаги (7,14,28)
    - скользящие средние (7,14,28)
    - фильтрация NaN
    - ЗДЕСЬ ВЫХОДНЫЕ ОСТАВЛЯЕМ ЧТОБЫ НЕ НАРУШАТЬ ВРЕМЕННОЙ РЯД
    Всё считается отдельно для каждой связки: FULL_SIGN + METRIC_NAMe
    '''

    df = (df
          .sort_values("DDATE")
          .copy()
          .reset_index(drop=True)
    )

    # ФИЧИ УРОВНЯ (LEVEL FEATURES)
    # Скользящие средние за 7, 14, 28. 3 линии тренда усредненные
    for window in [3, 7, 14, 28]:
        df[f"ROLL_MEAN_{window}"] = (
            df["METRIC_VALUE"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # Добавляем лаги от 1 до 28 дней. Продажи 1 день назад, 7, 14, 28 дней назад.
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f'LAG_{lag}D'] = df['METRIC_VALUE'].shift(lag)

    # Удаляем строки без лагов
    df = df.dropna().reset_index(drop=True)

    return df