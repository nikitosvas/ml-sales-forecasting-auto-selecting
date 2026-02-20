# evaluation/summary_report_metrics.py

"""
Модуль генерации аналитического Excel-отчёта.

Используется после расчёта backtests,
чтобы быстро оценить:

- где CatBoost выигрывает
- где Baseline сильнее
- какие каналы сложные
- какие метрики нестабильны
- худшие ошибки модели
"""
import json
import datetime as dt
import pandas as pd

def save_policy_json(
        policy_df: pd.DataFrame,
        filename="policy_model.json"
):

    date_run = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    policy_df['RUN_DATE'] = date_run

    new_policy_records = policy_df.to_dict(orient="records")

    # =========================
    # Читаем старый policy
    # =========================
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()

        if not content:
            old_records = []
        else:
            old_records = json.loads(content)

    all_records = old_records + new_policy_records

    # Дозаписываем новые данные в старый json
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=4)

    print(f"Файл policy дополнен, история сохранена в: {filename}")


def build_summary_tables(
        report_df: pd.DataFrame,
        models_check_wmape: list[str]
) -> dict:
    """
    Формирует набор агрегированных таблиц для отчёта.

    Возвращает словарь:
        sheet_name -> dataframe
    """

    df = report_df.copy()
    tables = {}

    # =========================================================
    # Ошибки каких моделей берем в отчет (добавляем в название модели название метрики WMAPE )
    # =========================================================
    wmape_cols = [f"{model}_WMAPE" for model in models_check_wmape]

    # =========================================================
    # ✅ 1) Средний WMAPE по моделям (общий)
    # =========================================================
    summary_models = df[wmape_cols].mean().reset_index()
    summary_models.columns = ["MODEL", "WMAPE_MEAN"]

    tables["SUMMARY_MODELS"] = summary_models.sort_values("WMAPE_MEAN")

    # =========================================================
    # ✅ 2) Средний WMAPE по моделям + окнам обучения
    # =========================================================
    summary_by_window = (
        df
        .groupby("TRAIN_WINDOW_DAYS")[wmape_cols]
        .mean()
        .reset_index()
    )

    tables["SUMMARY_BY_WINDOW"] = summary_by_window

    # =================================================================================
    # ✅ 3) Нужно определить лучшее ОКНО + КАНАЛ + МЕТРИКА и записать эти данные в json
    # Чтобы далее прогнозировать на новый месяц с учетом нескольких моделей
    # ==================================================================================
    melted = df.melt(
        id_vars=["FULL_SIGN", "METRIC_NAME", "TRAIN_WINDOW_DAYS"],
        value_vars=wmape_cols,
        var_name="MODEL",
        value_name="WMAPE"
    )

    # усредняем по всем датам бэктеста
    avg_scores = (
        melted
        .groupby(["FULL_SIGN", "METRIC_NAME", "TRAIN_WINDOW_DAYS", "MODEL"])
        .agg(WMAPE_MEAN=("WMAPE", "mean"))
        .reset_index()
    )

    best_policy = (
        avg_scores
        .sort_values("WMAPE_MEAN")
        .groupby(["FULL_SIGN", "METRIC_NAME"])
        .head(1)
        .reset_index(drop=True)
    )

    best_policy = best_policy.rename(columns={
        "TRAIN_WINDOW_DAYS": "BEST_WINDOW",
        "MODEL": "BEST_MODEL",
        "WMAPE_MEAN": "BEST_MEAN_WMAPE"
    })

    tables["BEST_MODEL_POLICY"] = best_policy

    print(best_policy)

    # =========================================================
    # ✅ 4) Победитель (WINNER) по каждой строке
    # =========================================================
    df["WINNER"] = df[wmape_cols].idxmin(axis=1)

    tables["WIN_RATE"] = (
        df["WINNER"]
        .value_counts(normalize=True)
        .reset_index()
        .rename(columns={"index": "MODEL", "WINNER": "WIN_RATE"})
    )

    # =========================================================
    # ✅ 6) Средний WMAPE по каналам
    # =========================================================
    tables["BY_CHANNEL"] = (
        df
        .groupby("FULL_SIGN")[wmape_cols]
        .mean()
        .reset_index()
    )

    # =========================================================
    # ✅ 7) Средний WMAPE по метрикам
    # =========================================================
    tables["BY_METRIC"] = (
        df
        .groupby("METRIC_NAME")[wmape_cols]
        .mean()
        .reset_index()
    )

    return tables


# ============================================================
# Экспорт Excel
# ============================================================
def export_report_excel_n_dump_policy(
    report_df: pd.DataFrame,
    models_list: list[str],
    file_policy,
    filename_for_report
):
    """
    Экспортирует полный аналитический Excel-отчёт.

    Листы:
        - RAW_RESULTS
        - SUMMARY_MODELS
        - BY_CHANNEL
        - BY_METRIC
        - WIN_RATE
        - WORST_CASES
        - BEST_CASES
        - DAILY_CHART (главный график)

    Дополнительно:
        Вставляет график Fact vs Baseline vs CatBoost.
    """

    print("\n📌 Формируем полный Excel отчёт...")

    # --- таблицы ---
    tables = build_summary_tables(
        report_df=report_df,
        models_check_wmape=models_list
    )

    with pd.ExcelWriter(filename_for_report, engine="openpyxl") as writer:

        # ✅ RAW результаты
        report_df.to_excel(writer, sheet_name="RAW_RESULTS", index=False)

        # ✅ агрегаты
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet, index=False)


    print(f"\n✅ Полный Excel отчёт сохранён: {filename_for_report}")

    save_policy_json(
        tables["BEST_MODEL_POLICY"],
        filename=file_policy
    )

