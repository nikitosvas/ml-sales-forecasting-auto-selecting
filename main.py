import itertools
import pandas as pd
from config.setting import (
    MODELS_TO_RUN,
    TRAIN_WINDOWS,
    BACKTEST_DATES,
    START_FORECAST_DATE,
    MAX_HISTORY_DAYS,
    KP_DISTR_PAIRS,
    TD_PAIRS,
    METRICS,
    POLICY_FILE,
    EXCEL_REPORT_FILE
)

from forecast.policy_month_forecast import run_policy_current_month_forecast

from data.clickhouse import create_clickhouse_connect

from data.load_raw_fact_data import load_and_prepare_long_df, long_to_wide_forecast

from data.calendar_days import HOLIDAYS

from features.calendar_features import add_calendar_features

from utils.printers import print_month_metrics

from utils.pandas_setting import setup_pandas_display

from evaluation.backtests_models_few_periods import run_monthly_backtests
from evaluation.summary_report_metrics import export_report_excel_n_dump_policy

from plots_tables.policy_plots_backtests import plot_policy_backtests

def main():
    """
        Главная функция.

        Последовательность действий:
            1. Загружаются фактические данные из ClickHouse:
               - по всем каналам продаж (FULL_SIGN)
               - по всем финансовым метрикам (METRIC_NAME)
            2. Данные переводятся в длинный формат:
               одна строка = одна метрика в один день.
            3. Добавляются календарные признаки (features):
               - день недели
               - выходной/будний день
               - праздник
               - день месяца, неделя года и т.д.
            4. Далее перебираются все комбинации: канал × метрика
            5. Для каждой связки строятся два прогноза:
               A) Baseline (Weekly Naive + OLS тренд)
               B) ML Forecast (CatBoost + лаги + rolling)
            6. Прогнозы сравниваются с фактом за месяц
                и рассчитываются итоговые месячные метрики качества.

            Важно:
                - прогноз строится начиная с даты START_FORECAST_DATE
                - праздники и выходные НЕ зануляются
                - модель должна учитывать их через признак IS_HOLIDAY
    """


    # ✅ накопители для итогового отчета
    all_results = []

    # --------------------------------------------------
    # 1. Настройка вывода в консоль данных датафрейма (форматы и отображение)
    # --------------------------------------------------
    setup_pandas_display()

    # --------------------------------------------------
    # 2. Загрузка факта из клика по каждой связке канал + метрика
    # И подготовка одного общего длинного датафрейма
    # --------------------------------------------------
    long_df = load_and_prepare_long_df(
        client=create_clickhouse_connect(),
        subspecies_kp=KP_DISTR_PAIRS,
        subspecies_td=TD_PAIRS,
        start_date=START_FORECAST_DATE,
        count_hist_dates=MAX_HISTORY_DAYS
    )

    # --------------------------------------------------------------------------------
    # 3. Добавляем календарные фичи
    # День недели, выходные, праздники, старт/ конец недели, номер дня месяца, начало и тд.
    # --------------------------------------------------------------------------------
    df_w_features = add_calendar_features(
        long_df,
        holidays=HOLIDAYS
    )

    # --------------------------------------------------
    # 4. Перебор каналов и метрик
    # --------------------------------------------------
    full_signs = (
        df_w_features["FULL_SIGN"]
        .dropna()
        .unique()
        .tolist()
    )

    # =========================================================
    # ✅ главный цикл: канал × метрика × окно
    # =========================================================

    for train_window in TRAIN_WINDOWS:
        print(f"\n===============================")
        print(f"✅ TRAIN WINDOW = {train_window} days")
        print(f"===============================\n")

        for full_sign, metric in itertools.product(full_signs, METRICS):

            print(f"\n📌 Тестируем: {full_sign} | {metric}")

            # ---- рабочий df по связке ----
            work_df = df_w_features[
                (df_w_features["FULL_SIGN"] == full_sign) &
                (df_w_features["METRIC_NAME"] == metric)
                ].copy()

            # ---- прогоняем backtests ----
            results_df = run_monthly_backtests(
                df=work_df,
                full_sign=full_sign,
                metric_name=metric,
                forecast_backtest_dates=BACKTEST_DATES,
                train_window_days=train_window,
                models_to_run=MODELS_TO_RUN
            )

            results_df["TRAIN_WINDOW_DAYS"] = train_window
            results_df["FULL_SIGN"] = full_sign
            results_df["METRIC_NAME"] = metric

            all_results.append(results_df)

    # =========================================================
    # ✅ Финальный отчёт
    # =========================================================
    final_report = pd.concat(all_results, ignore_index=True)

    print("\n✅ Итоговый отчёт:")
    print(final_report)

    # =========================================================
    # ✅ Экспорт Excel + графики
    # =========================================================
    export_report_excel_n_dump_policy(
        report_df=final_report,
        models_list=MODELS_TO_RUN,
        file_policy=POLICY_FILE,
        filename_for_report=EXCEL_REPORT_FILE
    )

    run_policy_current_month_forecast(
        df=df_w_features,
        policy_file=POLICY_FILE,
        forecast_start_date=START_FORECAST_DATE
    )

    plot_policy_backtests(
        df=df_w_features,
        policy_df=pd.read_json(POLICY_FILE),
        backtest_dates=BACKTEST_DATES
    )

if __name__ == '__main__':
    main()




