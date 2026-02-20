# data/clickhouse.py
import clickhouse_connect
import pandas as pd
import datetime as dt

''' Подключение к клику и выгрузка датафреймов '''

def create_clickhouse_connect():
    client = clickhouse_connect.get_client(
        host='CLICK',
        username='USER',
        password='PASS',
        port=1111,
        session_id='forecast_models',
        connect_timeout=15,
    )
    return client

# =========================================== Датафрейм с фактом по Реализации в ТД ===================================|
def get_fact_data_tvoy_doctor(
    client,
    subspecial: str,
    iris_sign: str,
    start_date: dt.date,
    count_hist_fact_dates: int
) -> pd.DataFrame:
    '''
        Возвращает Датафрейм с фактическими данными за указанный временной промежуток по каналам Реализация в ТД + БЕЗ ИРИС
    '''

    data = client.query_df(f'''
            WITH
            /* Создаём календарь из последних 60 дней */
            calendar AS (
                SELECT
                     toDate(addDays(toLastDayOfMonth(toDate('{start_date}')), -number)) AS DDATE, -- Минусуем количество дней указанное в numbers
                    '{subspecial}' AS SALES_SUBSPECIES,
                    '{iris_sign}' AS SIGN_IRIS,
                    concat(SALES_SUBSPECIES, ' ', SIGN_IRIS) AS FULL_SIGN
                FROM numbers({count_hist_fact_dates})
            ),
            fact AS (
                SELECT
                    DDATE,
                    SALES_SUBSPECIES,
                    ROUND(SUM(SUM_SNDS), 2) AS SUM_SNDS,
                    ROUND(SUM(SUM_PROFIT), 2) AS SUM_PROFIT,
                    ROUND(SUM(SUM_PROFIT_NO_KSP), 2) AS SUM_PROFIT_NO_KSP,
                    ROUND(SUM(SUM_COST_NONDS_NO_KSP), 2) AS SUM_COST_NONDS_NO_KSP,
                    ROUND(SUM(SUM_COST_SNDS_NO_KSP), 2) AS SUM_COST_SNDS_NO_KSP,
                    '{iris_sign}' AS SIGN_IRIS,
                    concat(SALES_SUBSPECIES, ' ', SIGN_IRIS) AS FULL_SIGN
                FROM 
                    opt_prod.OPT_BASE_FACT_TABLE
                WHERE
                    SALES_SUBSPECIES = '{subspecial}'
                    AND DDATE > toDate('{start_date}') - INTERVAL {count_hist_fact_dates} DAY
                    AND DDATE <= addDays(
                            date_trunc('month', toDate('{start_date}')) + INTERVAL 1 MONTH,
                            -1
                    )
                GROUP BY
                    DDATE,
                    SALES_SUBSPECIES
                ORDER BY
                    DDATE DESC
            )
        /* Джойним календарь и факт, подставляем нули */
        SELECT
            cal.DDATE,
            cal.SALES_SUBSPECIES AS SALES_SUBSPECIES,
            ROUND(coalesce(f.SUM_SNDS, 0)) AS SUM_SNDS,
            ROUND(coalesce(f.SUM_PROFIT, 0)) AS SUM_PROFIT,
            ROUND(coalesce(f.SUM_PROFIT_NO_KSP, 0)) AS SUM_PROFIT_NO_KSP,
            ROUND(coalesce(f.SUM_COST_NONDS_NO_KSP, 0)) AS SUM_COST_NONDS_NO_KSP,
            ROUND(coalesce(f.SUM_COST_SNDS_NO_KSP, 0)) AS SUM_COST_SNDS_NO_KSP,
            cal.SIGN_IRIS AS SIGN_IRIS,
            cal.FULL_SIGN AS FULL_SIGN
        FROM
            calendar cal
        LEFT JOIN
            fact f ON cal.DDATE = f.DDATE
        ORDER BY
            cal.DDATE ASC
    ''')  # noqa: E501

    return data

# ========================= Датафрейм с фактом по Коммерции ЦФО, ЮФО, Дистрибьюции ====================================|
def get_fact_data(
    client,
    subspecial: str,
    iris_sign: str,
    start_date: dt.date,
    count_hist_fact_dates: int,
) -> pd.DataFrame:
    '''
        Возвращает Датафрейм с фактическими данными за указанный временной промежуток по КП РЕГИОН А/ Б + признак И/ БЕЗ И
    '''

    data = client.query_df(f'''
       /* Нам нужно отобрать последние 40 записей кроме первой */
             WITH
            /* Создаём календарь из последних 60 дней */
            calendar AS (
                SELECT
                     toDate(addDays(toLastDayOfMonth(toDate('{start_date}')), -number)) AS DDATE, -- Минусуем количество дней указанное в numbers
                    '{subspecial}' as SALES_SUBSPECIES,
                    '{iris_sign}' AS SIGN_IRIS,
                    concat(SALES_SUBSPECIES, ' ', SIGN_IRIS) as FULL_SIGN
                FROM numbers({count_hist_fact_dates})
            ),
            /* Берём факт, как раньше, но только по тем датам */
            fact AS (
                SELECT
                    DDATE,
                    SALES_SUBSPECIES,
                    ROUND(SUM(SUM_SNDS), 2) AS SUM_SNDS,
                    ROUND(SUM(SUM_PROFIT), 2) AS SUM_PROFIT,
                    ROUND(SUM(SUM_PROFIT_NO_KSP), 2) AS SUM_PROFIT_NO_KSP,
                    ROUND(SUM(SUM_COST_NONDS_NO_KSP), 2) AS SUM_COST_NONDS_NO_KSP,
                    ROUND(SUM(SUM_COST_SNDS_NO_KSP), 2) AS SUM_COST_SNDS_NO_KSP,
                    SIGN_IRIS,
                    FULL_SIGN
                FROM vitrina.FACT_KP_IRIS_NOIRIS
                WHERE
                    SALES_SUBSPECIES = '{subspecial}'
                    AND
                     SIGN_IRIS = '{iris_sign}'
                    AND DDATE > toDate('{start_date}') - INTERVAL {count_hist_fact_dates + 1} DAY
                    AND DDATE <= addDays(
                            date_trunc('month', toDate('{start_date}')) + INTERVAL 1 MONTH,
                            -1
                    )
                GROUP BY 
                    DDATE, 
                    SALES_SUBSPECIES, 
                    SIGN_IRIS,
                    FULL_SIGN
            )
        /* Соединяем календарь и факт: если нет данных — ставим 0 */
        SELECT
            cal.DDATE,
            cal.SALES_SUBSPECIES AS SALES_SUBSPECIES,
            ROUND(coalesce(f.SUM_SNDS, 0)) AS SUM_SNDS,
            ROUND(coalesce(f.SUM_PROFIT, 0)) AS SUM_PROFIT,
            ROUND(coalesce(f.SUM_PROFIT_NO_KSP, 0)) AS SUM_PROFIT_NO_KSP,
            ROUND(coalesce(f.SUM_COST_NONDS_NO_KSP, 0)) AS SUM_COST_NONDS_NO_KSP,
            ROUND(coalesce(f.SUM_COST_SNDS_NO_KSP, 0)) AS SUM_COST_SNDS_NO_KSP,
            cal.SIGN_IRIS,
            cal.FULL_SIGN
        FROM
            calendar cal
        LEFT JOIN
            fact f ON cal.DDATE = f.DDATE
        ORDER BY
            cal.DDATE ASC
    ''')  # noqa: E501

    return data