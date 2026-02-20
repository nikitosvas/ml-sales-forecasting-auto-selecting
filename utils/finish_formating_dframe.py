# utils/finish_formatting_dframe.py
import pandas as pd

def long_to_wide_forecast(
        long_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Преобразует long-формат прогноза в широкий вид.
    Добавляет столбцы SUM_NONDS, Type, Fact_type.
    """

    forecast_wide_df = long_df.pivot_table(
        index=["DDATE", "FULL_SIGN"],
        columns="METRIC_NAME",
        values="FORECAST"
    ).reset_index()

    # Если были NaN после пивота, заполняем 0
    wide_df = forecast_wide_df.fillna(0)

    # Добавляем дополнительные колонки
    forecast_wide_df["SALES_SUBSPECIES"] = (
        forecast_wide_df["FULL_SIGN"]
        .str.replace(" БЕЗ ИРИС", "")
        .str.replace(" ИРИС", "")
    )
    forecast_wide_df["SIGN_IRIS"] = (
        forecast_wide_df["FULL_SIGN"]
        .apply(lambda x: "ИРИС" if "ИРИС" in x else "БЕЗ ИРИС")
    )
    forecast_wide_df["SUM_NONDS"] = forecast_wide_df["SUM_SNDS"] * 0.9
    forecast_wide_df["Type"] = "finFact"
    forecast_wide_df["Fact_type"] = "forecast"

    # Опционально: упорядочим колонки
    cols_order = [
        "DDATE", "SUM_SNDS", "SUM_PROFIT", "SUM_PROFIT_NO_KSP",
        "SUM_COST_NONDS_NO_KSP", "SUM_COST_SNDS_NO_KSP",
        "SALES_SUBSPECIES", "SIGN_IRIS", "FULL_SIGN",
        "Type", "Fact_type"
    ]

    return forecast_wide_df
