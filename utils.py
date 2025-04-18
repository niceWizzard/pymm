
import numpy as np
from pandas import DataFrame, Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_percentage_error as MAPE, mean_absolute_error as MAE


def arima_grid_search(df: DataFrame, d: int, log: bool = True) -> tuple[ARIMA, DataFrame]:
    """
    Assumes the time series is stationary.
    Returns the best ARIMA model (fitted) and a DataFrame of model metrics.
    """
    if not adf_test(df):
        raise ValueError(f"Time series {df.index.name} is not stationary!")

    possible_ar, possible_ma = get_significant_pacf(df), get_significant_acf(df)
    if log:
        print(possible_ar, possible_ma)
    model_list = []

    if log:
        total_combos = len(possible_ar) * len(possible_ma)
        print(f"Trying {total_combos} combinations...")

    for ar in possible_ar:
        for ma in possible_ma:
            try:
                model = ARIMA(df, order=(ar, d, ma)).fit()
                model_list.append((f"ARIMA({ar}, {d}, {ma})", ar, d, ma, model.aic, model.bic))
            except Exception as e:
                if log:
                    print(f"Failed ARIMA({ar}, {d}, {ma}): {e}")

    if not model_list:
        raise RuntimeError("No ARIMA models converged successfully.")

    model_list_df = DataFrame(
        model_list, columns=["text", "ar", "d", "ma", "aic", "bic"]
    ).sort_values("bic")

    if log:
        print(model_list_df)

    best_model_row = model_list_df.iloc[0]
    best_model = ARIMA(df, order=(best_model_row["ar"], best_model_row["d"], best_model_row["ma"])).fit()
    return best_model, best_model_row

        


def get_significant_pacf(series: Series, log: bool = False) -> list[int]:
    significant_lags: list[int] = []
    lag_results, lags_confidence = pacf(series, alpha=0.05)
    for i, (lag_result, (lower, upper)) in enumerate(zip(lag_results, lags_confidence)):
        if i == 0:
            continue
        if 0 < lower or 0 > upper:
            if log:
                print(f"Lag {i} is significant with value: {lag_result}")
            significant_lags.append(i)
    return significant_lags

def get_significant_acf(series: Series, log: bool = False) -> list[int]:
    significant_lags: list[int] = []
    lag_results, lags_confidence = acf(series, alpha=0.05)
    for i, (lag_result, (lower, upper)) in enumerate(zip(lag_results, lags_confidence)):
        if i == 0:
            continue
        if 0 < lower or 0 > upper:
            if log:
                print(f"Lag {i} is significant with value: {lag_result}")
            significant_lags.append(i)
    return significant_lags


def plot_autocorrelations(series : Series) -> None:
    _ = plot_acf(series)
    _ = plot_pacf(series)


def adf_test(series : Series, log: bool=True) -> bool:
    p = adfuller(series)[1]
    is_stationary = p < 0.05
    if log:
        print(f"The {series.index.name} is {"Stationary"if is_stationary else "Not Stationary"} (p-value: {p:.4f})")
    return is_stationary

def split(df : DataFrame) -> tuple[DataFrame, DataFrame]:
    n = int(len(df) - 12)
    train = df.iloc[0:n]
    test = df.iloc[n:len(df)]
    print(f"Total: {len(df)}, Train: {len(train)}, Test: {len(test)}")
    return train, test


def get_metrics(true_values, forecasted) -> DataFrame:
    metrics_result = []
    for key, metric_func in {
        "MAE": MAE,
        "MAPE": MAPE,
        "MSE": MSE,
    }.items():
        metrics_result.append((key, metric_func(true_values, forecasted)))

    return DataFrame(metrics_result, columns=["Metric", "Value"])
