"""Data layer: load stock prices and create features — Iteration 1."""
import yfinance as yf
import pandas as pd

FEATURE_COLUMNS = ["Close", "MA10", "RSI", "MACD", "MACD_Signal"]


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


def normalize_features(df, columns, ref_df=None):
    """Min-max normalize columns to a 0-1 range.

    ref_df: if given, use ITS min/max instead of df's own. Used so the
    test-basket tickers are scaled using the TRAIN-basket's statistics —
    since train and test are now different tickers entirely, test must
    still be normalized consistently with what the model learned on,
    not with its own, unrelated price range.
    """
    reference = ref_df if ref_df is not None else df
    for col in columns:
        col_min = reference[col].min()
        col_max = reference[col].max()
        df[col + "_norm"] = (df[col] - col_min) / (col_max - col_min)
    return df


def fetch_and_engineer(ticker, period="2y"):
    """Fetch price history and add engineered features for one ticker.

    Returns a raw (unnormalized) dataframe. Normalization happens
    separately, since it depends on which reference statistics to use.
    """
    # download price history and keep only closing price
    df = yf.Ticker(ticker).history(period=period)
    df = df[["Close"]].copy()

    # add indicators
    df = add_moving_average(df, window=10, col_name="MA10")
    df = add_rsi(df)
    df = add_macd(df)

    # drop warm-up rows with missing indicator values
    df = df.dropna().reset_index(drop=True)
    return df


def load_training_basket(tickers, period="2y"):
    """Fetch and normalize the FULL history for each training ticker.

    Normalization reference is built by pooling all training tickers'
    data together, so the scale reflects the whole basket rather than
    any single stock's own range.
    """
    raw = {t: fetch_and_engineer(t, period) for t in tickers}

    # build one shared reference dataframe from all training tickers combined
    reference = pd.concat(raw.values(), ignore_index=True)

    normalized = {
        t: normalize_features(df.copy(), columns=FEATURE_COLUMNS, ref_df=reference)
        for t, df in raw.items()
    }
    return normalized, reference


def load_test_basket(tickers, train_reference, period="2y"):
    """Fetch and normalize the FULL history for each test ticker, using the
    TRAINING basket's reference statistics — never the test tickers' own —
    so scaling stays consistent with what the model was trained on.
    """
    raw = {t: fetch_and_engineer(t, period) for t in tickers}
    normalized = {
        t: normalize_features(df.copy(), columns=FEATURE_COLUMNS, ref_df=train_reference)
        for t, df in raw.items()
    }
    return normalized


def align_dates(data_dict):
    """Intersect trading dates across all tickers in a dict so every stock
    lines up day-for-day."""
    tickers = list(data_dict.keys())
    common_dates = data_dict[tickers[0]].index
    for t in tickers[1:]:
        common_dates = common_dates.intersection(data_dict[t].index)
    return {t: data_dict[t].loc[common_dates].reset_index(drop=True) for t in tickers}