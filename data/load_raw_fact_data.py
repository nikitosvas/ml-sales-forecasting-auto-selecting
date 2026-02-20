# data/load_row_fact_data.py
import pandas as pd
from .clickhouse import get_fact_data, get_fact_data_tvoy_doctor

# =============== Преобразуем данные из широково в длинный датафрйм =====================
def load_and_prepare_long_df(
        client,
        subspecies_kp,
        subspecies_td,
        start_date,
        count_hist_dates
):
    '''
        :param client: Клиент клика
        :param subspecies_kp: Словарь с Каналом продаж
        :param subspecies_td: Словарь с Каналом продаж
        :param start_date: Дата от которой строим прогноз и ДО которой берем фактические данные
        :param count_hist_dates: Сколько дней фактических данных брать до текущей даты
        :return: Возвращает длинный датафрейм со всеми каналами, метриками, прзнаками + добавлены EXOG признаки (выходные, праздники, день недели и тд.)
    '''

    all_dfs = []

    for cannel, iris_sign in subspecies_kp:
        df_kp = get_fact_data(
            client=client,
            subspecial=cannel,
            iris_sign=iris_sign,
            start_date=start_date,
            count_hist_fact_dates=count_hist_dates
        )

        all_dfs.append(df_kp)

    for cannel, iris_sign in subspecies_td:
        df_td = get_fact_data_tvoy_doctor(
            client=client,
            subspecial=cannel,
            iris_sign=iris_sign,
            start_date=start_date,
            count_hist_fact_dates=count_hist_dates
        )

        all_dfs.append(df_td)

    wide_df = pd.concat(all_dfs, ignore_index=True)

    # -----------------------------------------
    # 4. Переводим таблицу из ширкой в длинную (одна строка = одна метрика за день).
    # Так каждая метрика будет отдельной строкой, удобнее для анализа
    # -----------------------------------------
    long_df = wide_df.melt(
        id_vars=["DDATE", "SALES_SUBSPECIES", "SIGN_IRIS", "FULL_SIGN"],
        value_vars=[
            "SUM_SNDS",
            "SUM_PROFIT",
            "SUM_PROFIT_NO_KSP",
            "SUM_COST_NONDS_NO_KSP",
            "SUM_COST_SNDS_NO_KSP",
        ],
        var_name="METRIC_NAME",
        value_name="METRIC_VALUE"
    )

    return long_df

def long_to_wide_forecast(
        forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Преобразует long-формат прогноза в широкий вид.
    Добавляет столбцы SUM_NONDS, Type, Fact_type.
    """

    wide_df = forecast_df.pivot_table(
        index=["DDATE", "SALES_SUBSPECIES", "SIGN_IRIS", "FULL_SIGN"],
        columns="METRIC_NAME",
        values="FORECAST"
    ).reset_index()

    # Если были NaN после пивота, заполняем 0
    wide_df = wide_df.fillna(0)

    # Добавляем дополнительные колонки
    wide_df["Type"] = "finFact"
    wide_df["Fact_type"] = "forecast"

    # упорядочим колонки
    cols_order = [
        "DDATE", "SUM_SNDS", "SUM_PROFIT", "SUM_PROFIT_NO_KSP",
        "SUM_COST_NONDS_NO_KSP", "SUM_COST_SNDS_NO_KSP",
        "SALES_SUBSPECIES", "SIGN_IRIS", "FULL_SIGN",
        "Type", "Fact_type"
    ]

    # Некоторые столбцы могут отсутствовать, оставляем только существующие
    cols_order = [c for c in cols_order if c in wide_df.columns]
    wide_df = wide_df[cols_order]

    return wide_df

