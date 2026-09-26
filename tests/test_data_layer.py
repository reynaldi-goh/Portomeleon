"""Unit tests for the data layer — Iteration 1."""
import pandas as pd
from src.data_layer import (
    add_moving_average,
    add_rsi,
    add_macd,
    add_percent_features,
    normalize_features,
    fetch_and_engineer,
    align_dates,
    FEATURE_COLUMNS,
)


def test_add_moving_average_creates_column():
    # confirm add_moving_average adds the requested column
    df = pd.DataFrame({"Close": list(range(100, 120))})
    result = add_moving_average(df, window=10, col_name="MA10")
    assert "MA10" in result.columns


def test_add_moving_average_values_reasonable():
    # check a moving average stays within the min/max of the underlying prices
    df = pd.DataFrame({"Close": [100, 102, 101, 105, 103, 108, 107, 110, 109, 112, 111, 115]})
    result = add_moving_average(df, window=5, col_name="MA5")
    valid_ma = result["MA5"].dropna()
    assert (valid_ma >= df["Close"].min()).all()
    assert (valid_ma <= df["Close"].max()).all()


def test_add_rsi_range():
    # verify RSI always falls within its defined 0-100 range
    df = pd.DataFrame({"Close": [100, 102, 101, 105, 103, 108, 107, 110, 109, 112, 111, 115, 114, 117, 116]})
    result = add_rsi(df)
    valid_rsi = result["RSI"].dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_add_macd_creates_columns():
    # check add_macd produces both the MACD line and its signal line
    df = pd.DataFrame({"Close": list(range(100, 140))})
    result = add_macd(df)
    assert "MACD" in result.columns
    assert "MACD_Signal" in result.columns
    assert not result["MACD"].isna().all()


def test_add_percent_features_creates_columns():
    # confirm the dollar-based indicators get turned into percentage features
    df = pd.DataFrame({"Close": [100, 102, 104, 103, 107, 110, 108, 112, 115, 114]})
    df = add_moving_average(df, window=3, col_name="MA10")
    df = add_macd(df)
    result = add_percent_features(df)
    for col in ["Ret_1d", "Close_vs_MA10", "MACD_pct", "MACD_Signal_pct"]:
        assert col in result.columns


def test_normalize_features_default_reference():
    # check normalize_features z-scores a column against its own mean/std
    df = pd.DataFrame({"RSI": [10, 20, 50, 80, 90]})
    result = normalize_features(df, columns=["RSI"])
    valid = result["RSI_norm"].dropna()
    assert abs(valid.mean()) < 1e-8          # a self-referenced z-score centers on zero
    assert (valid >= -5).all() and (valid <= 5).all()  # values stay within the clip bounds


def test_normalize_features_uses_external_reference():
    # confirm passing ref_df scales against another dataframe's stats, not its own
    df = pd.DataFrame({"RSI": [10, 20, 50, 80, 90]})
    ref_df = pd.DataFrame({"RSI": [0, 0, 0, 0, 100]})  # a very different mean/std
    result = normalize_features(df.copy(), columns=["RSI"], ref_df=ref_df)
    own_result = normalize_features(df.copy(), columns=["RSI"])
    assert not result["RSI_norm"].equals(own_result["RSI_norm"])


def test_normalize_features_clips_outliers():
    # verify extreme values get capped at +-5 rather than growing unbounded
    df = pd.DataFrame({"RSI": [10, 10, 10, 10, 10000]})
    result = normalize_features(df, columns=["RSI"])
    assert result["RSI_norm"].max() <= 5
    assert result["RSI_norm"].min() >= -5


def test_fetch_and_engineer_no_nan():
    # confirm the full feature pipeline drops every warm-up NaN row
    df = fetch_and_engineer(ticker="AAPL")
    assert not df.isna().any().any()


def test_fetch_and_engineer_expected_columns():
    # check every raw indicator and percentage feature is present
    df = fetch_and_engineer(ticker="AAPL")
    expected = {"Close", "MA10", "RSI", "MACD", "MACD_Signal"} | set(FEATURE_COLUMNS)
    assert expected.issubset(set(df.columns))


def test_align_dates_keeps_only_shared_dates():
    # verify align_dates trims every ticker down to their common trading dates
    idx_a = pd.date_range("2026-01-01", periods=5, freq="D")
    idx_b = pd.date_range("2026-01-03", periods=5, freq="D")  # only 3 dates overlap with idx_a
    data = {
        "A": pd.DataFrame({"Close": range(5)}, index=idx_a),
        "B": pd.DataFrame({"Close": range(5)}, index=idx_b),
    }
    result = align_dates(data)
    assert len(result["A"]) == len(result["B"]) == 3


def test_align_dates_raises_when_no_overlap():
    # check align_dates fails loudly when tickers share no trading dates at all
    idx_a = pd.date_range("2026-01-01", periods=2, freq="D")
    idx_b = pd.date_range("2027-01-01", periods=2, freq="D")
    data = {
        "A": pd.DataFrame({"Close": [1, 2]}, index=idx_a),
        "B": pd.DataFrame({"Close": [1, 2]}, index=idx_b),
    }
    try:
        align_dates(data)
        assert False, "expected a ValueError for non-overlapping dates"
    except ValueError:
        pass