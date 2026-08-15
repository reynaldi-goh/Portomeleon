"""Unit tests for the data layer — Iteration 1."""
import pandas as pd
from src.data_layer import add_moving_average, add_rsi, add_macd, normalize_features, load_price_data


def test_add_moving_average_creates_column():
    df = pd.DataFrame({"Close": list(range(100, 120))})
    result = add_moving_average(df, window=10, col_name="MA10")
    assert "MA10" in result.columns


def test_add_moving_average_values_reasonable():
    # a moving average should stay within the min/max of the underlying prices
    df = pd.DataFrame({"Close": [100, 102, 101, 105, 103, 108, 107, 110, 109, 112, 111, 115]})
    result = add_moving_average(df, window=5, col_name="MA5")
    valid_ma = result["MA5"].dropna()
    assert (valid_ma >= df["Close"].min()).all()
    assert (valid_ma <= df["Close"].max()).all()


def test_add_rsi_range():
    df = pd.DataFrame({"Close": [100, 102, 101, 105, 103, 108, 107, 110, 109, 112, 111, 115, 114, 117, 116]})
    result = add_rsi(df)
    valid_rsi = result["RSI"].dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_add_macd_creates_columns():
    df = pd.DataFrame({"Close": list(range(100, 140))})
    result = add_macd(df)
    assert "MACD" in result.columns
    assert "MACD_Signal" in result.columns
    assert not result["MACD"].isna().all()


def test_normalize_features_range():
    df = pd.DataFrame({"RSI": [10, 20, 50, 80, 90]})
    result = normalize_features(df, columns=["RSI"])
    valid = result["RSI_norm"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_load_price_data_no_nan():
    # full pipeline: after dropna(), no NaNs should remain
    df = load_price_data(ticker="AAPL", period="2y")
    assert not df.isna().any().any()


def test_load_price_data_expected_columns():
    df = load_price_data(ticker="AAPL", period="2y")
    expected = {"Close", "MA10", "RSI", "MACD", "MACD_Signal",
                "Close_norm", "MA10_norm", "RSI_norm", "MACD_norm", "MACD_Signal_norm"}
    assert expected.issubset(set(df.columns))