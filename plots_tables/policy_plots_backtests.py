
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from forecast.models_registry import MODEL_REGISTRY
from evaluation.metrics import calc_month_metrics

matplotlib.use("Agg")

def plot_policy_backtests(
    df: pd.DataFrame,
    policy_df: pd.DataFrame,
    backtest_dates: list[pd.Timestamp],
    save_dir="backtests_plots_policy_daily"
):
    """
    Строит информативные daily графики:

    Train fact + Test fact + Forecast best model
    """

    os.makedirs(save_dir, exist_ok=True)

    for _, policy_row in policy_df.iterrows():

        full_sign = policy_row["FULL_SIGN"]
        metric = policy_row["METRIC_NAME"]
        best_window = policy_row["BEST_WINDOW"]
        best_model = policy_row["BEST_MODEL"].replace("_WMAPE", "")

        print(
            f"\n📊 Plot policy → "
            f"{full_sign} | {metric} | {best_model}"
        )

        # --- рабочий df ---
        work_df = df[
            (df["FULL_SIGN"] == full_sign) &
            (df["METRIC_NAME"] == metric)
        ].copy()

        model_func = MODEL_REGISTRY[best_model]

        for start_date in backtest_dates:

            end_date = start_date + pd.offsets.MonthEnd(0)

            # =========================
            # TRAIN
            # =========================
            train_start = start_date - pd.Timedelta(days=best_window)

            train_df = work_df[
                (work_df["DDATE"] >= train_start) &
                (work_df["DDATE"] < start_date)
            ].copy()

            # =========================
            # TEST
            # =========================
            test_df = work_df[
                (work_df["DDATE"] >= start_date) &
                (work_df["DDATE"] <= end_date)
            ].copy()

            # =========================
            # FORECAST
            # =========================
            forecast_df = model_func(
                df=work_df,
                start=start_date,
                end=end_date,
                window=best_window,
                full_sign=full_sign,
                metric_name=metric
            )

            # =========================
            # WMAPE
            # =========================
            metrics = calc_month_metrics(
                fact_df=test_df,
                forecast_df=forecast_df
            )

            wmape = metrics["WMAPE_MONTH"]

            # =========================
            # PLOT
            # =========================
            plt.figure(figsize=(15, 6))

            # Train
            plt.plot(
                train_df["DDATE"],
                train_df["METRIC_VALUE"],
                label=f"Train ({best_window} days)",
                color="gray"
            )

            # Test fact
            plt.plot(
                test_df["DDATE"],
                test_df["METRIC_VALUE"],
                label="Test fact",
                color="black",
                linewidth=2
            )

            # Forecast
            plt.plot(
                forecast_df["DDATE"],
                forecast_df["FORECAST"],
                label=f"Forecast ({best_model})",
                linewidth=2
            )

            plt.axvline(x=start_date, linestyle="--")

            plt.title(
                f"{full_sign} | {metric}\n"
                f"Model: {best_model} | "
                f"Window: {best_window} | "
                f"Start: {start_date.date()}\n"
                f"WMAPE = {wmape:.2%}"
            )

            plt.legend()
            plt.grid(True)

            filename = (
                f"{full_sign}_{metric}_{start_date.date()}.png"
                .replace(" ", "_")
            )

            plt.savefig(
                os.path.join(save_dir, filename),
                bbox_inches="tight"
            )

            plt.close()

    print(f"\n✅ Policy daily plots saved → {save_dir}")


