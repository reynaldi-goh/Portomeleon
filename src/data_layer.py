"""Data layer: load stock prices and create features — Iteration 1."""
import os
import yfinance as yf
import pandas as pd

# the 5 percentage-style features the model sees. They mean the same thing for a
# $40 stock and a $400 stock, so what the model learns carries over to unseen stocks.
FEATURE_COLUMNS = ["Ret_1d", "Close_vs_MA10", "RSI", "MACD_pct", "MACD_Signal_pct"]

# fixed date window, so every run (and anyone reading the repo) sees the same data.
# yfinance treats the end date as exclusive, so the last day included is Fri 2026-09-18.
START_DATE = "2024-09-19"
END_DATE = "2026-09-19"

# downloaded prices are saved in <project root>/data, so each ticker is downloaded only once
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def add_rsi(df, period=14):
    # compute RSI from average gains and losses over a rolling period
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    # compute MACD line and signal line from fast/slow EMAs
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_moving_average(df, window=10, col_name="MA10"):
    # compute a simple moving average over closing price
    df[col_name] = df["Close"].rolling(window=window).mean()
    return df


def add_percent_features(df):
    # turn dollar-based values into percentages, so they mean the same for any stock price
    df["Ret_1d"] = df["Close"].pct_change()               # how much the price moved today
    df["Close_vs_MA10"] = df["Close"] / df["MA10"] - 1    # how far above/below its 10-day average
    df["MACD_pct"] = df["MACD"] / df["Close"]             # MACD as a share of the price
    df["MACD_Signal_pct"] = df["MACD_Signal"] / df["Close"]
    return df


def normalize_features(df, columns, ref_df=None):
    """Put columns on a common scale: (value - mean) / std, clipped to +-5.

    ref_df: if given, use ITS mean/std instead of df's own. Used so the
    test-basket tickers are scaled using the TRAIN-basket's statistics,
    the same scale the model learned on, not their own.
    """
    reference = ref_df if ref_df is not None else df
    for col in columns:
        col_mean = reference[col].mean()
        col_std = reference[col].std()
        df[col + "_norm"] = ((df[col] - col_mean) / col_std).clip(-5, 5)
    return df


def load_prices(ticker, start=START_DATE, end=END_DATE):
    """Return daily closing prices for one ticker between start and end.

    The first time, prices are downloaded and saved to
    data/<ticker>_<start>_<end>.csv. After that the saved file is used, so
    results never change with the day you run. Delete the file to force a
    fresh download.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data downloaded for {ticker} ({start} to {end}).")
    df = df[["Close"]].copy()

    # keep plain dates (no timezone), so the saved file reads back cleanly
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "Date"
    df.to_csv(path)
    return df


def fetch_and_engineer(ticker, start=START_DATE, end=END_DATE):
    """Load price history and add engineered features for one ticker.

    Returns a raw (unnormalized) dataframe indexed by trading date.
    Normalization happens separately, since it depends on which reference
    statistics to use.
    """
    df = load_prices(ticker, start, end)

    # add indicators, then turn them into percentage-style features
    df = add_moving_average(df, window=10, col_name="MA10")
    df = add_rsi(df)
    df = add_macd(df)
    df = add_percent_features(df)

    # drop warm-up rows with missing indicator values, but KEEP the dates as the
    # index, so align_dates can line tickers up by real date
    df = df.dropna()
    return df


def load_training_basket(tickers, start=START_DATE, end=END_DATE):
    """Load and normalize the FULL history for each training ticker.

    Normalization reference is built by pooling all training tickers'
    data together, so the scale reflects the whole basket rather than
    any single stock's own range.
    """
    raw = {t: fetch_and_engineer(t, start, end) for t in tickers}

    # build one shared reference dataframe from all training tickers combined
    reference = pd.concat(raw.values(), ignore_index=True)

    normalized = {
        t: normalize_features(df.copy(), columns=FEATURE_COLUMNS, ref_df=reference)
        for t, df in raw.items()
    }
    return normalized, reference


def load_test_basket(tickers, train_reference, start=START_DATE, end=END_DATE):
    """Load and normalize the FULL history for each test ticker, using the
    TRAINING basket's reference statistics — never the test tickers' own —
    so scaling stays consistent with what the model was trained on.
    """
    raw = {t: fetch_and_engineer(t, start, end) for t in tickers}
    normalized = {
        t: normalize_features(df.copy(), columns=FEATURE_COLUMNS, ref_df=train_reference)
        for t, df in raw.items()
    }
    return normalized


def align_dates(data_dict):
    """Keep only the dates every ticker has, so every stock lines up day-for-day.

    Dates are the index (kept by fetch_and_engineer), so this matches by real
    date, not by row number. Prints a note if any rows had to be dropped.
    """
    tickers = list(data_dict.keys())
    common_dates = data_dict[tickers[0]].index
    for t in tickers[1:]:
        common_dates = common_dates.intersection(data_dict[t].index)
    common_dates = common_dates.sort_values()

    if len(common_dates) == 0:
        raise ValueError("No trading dates are shared by all tickers.")

    for t in tickers:
        dropped = len(data_dict[t]) - len(common_dates)
        if dropped > 0:
            print(f"align_dates: {t} had {dropped} date(s) not shared by every ticker, dropped")

    return {t: data_dict[t].loc[common_dates].reset_index(drop=True) for t in tickers}