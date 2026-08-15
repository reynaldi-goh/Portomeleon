"""Data layer: load stock prices and create features — Iteration 1."""
import yfinance as yf
import pandas as pd


def add_rsi(df, period=14):
    """Adds a Relative Strength Index column to the dataframe."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """Adds MACD line and signal line columns to the dataframe."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_moving_average(df, window=10, col_name="MA10"):
    """Adds a simple moving average column to the dataframe."""
    df[col_name] = df["Close"].rolling(window=window).mean()
    return df


def normalize_features(df, columns):
    """Min-max normalizes the given columns to a 0-1 range."""
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        df[col + "_norm"] = (df[col] - col_min) / (col_max - col_min)
    return df


# load stock prices and create features
def load_price_data(ticker="AAPL", period="2y"):
    # download price history
    df = yf.Ticker(ticker).history(period=period)
    # keep only closing prices
    df = df[["Close"]].copy()

    # add indicators
    df = add_moving_average(df, window=10, col_name="MA10")
    df = add_rsi(df)
    df = add_macd(df)

    # remove rows with missing values (RSI/MACD/MA warm-up periods)
    df = df.dropna()

    # normalize the engineered features
    df = normalize_features(df, columns=["Close", "MA10", "RSI", "MACD", "MACD_Signal"])

    return df.reset_index(drop=True)

df = load_price_data()
print(df)