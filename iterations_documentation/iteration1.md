# Iteration 1: Data Layer

## Goal

The goal of this iteration was to build the data layer described in Section 3.3.1 of the project design. This layer is responsible for turning raw historical price data into a clean, feature engineered dataset that the trading environment and DQN agent can later use. Specifically, this iteration covers fetching price data for multiple tickers, calculating technical indicators, converting them into scale-invariant percentage features, and normalizing the result using training-basket statistics so a train/test split can be enforced without leakage. A formal look-ahead bias check is still planned as a short follow-up before Iteration 2 begins.

## What Was Built

The data layer is implemented in `src/data_layer.py` and is made up of the following functions:

- `load_prices(ticker, start, end)`: downloads historical closing prices for one ticker over a fixed date window using the yfinance API, and caches the result to a CSV under `data/` so repeated runs never re-download or drift day to day.
- `add_moving_average(df, window, col_name)`: adds a simple moving average column based on closing price. Currently used to add a 10 day moving average (MA10).
- `add_rsi(df, period)`: adds a Relative Strength Index (RSI) column, calculated using a 14 day period.
- `add_macd(df, fast, slow, signal)`: adds MACD and MACD Signal columns, using the standard 12, 26, 9 day parameters.
- `add_percent_features(df)`: converts the dollar-denominated indicators into percentage-style features (`Ret_1d`, `Close_vs_MA10`, `MACD_pct`, `MACD_Signal_pct`) so a feature means the same thing regardless of a stock's price level.
- `fetch_and_engineer(ticker, start, end)`: the main per-ticker entry point. Runs `load_prices` then adds MA10, RSI, MACD, and the percentage features in sequence, then drops warm-up rows with missing values.
- `normalize_features(df, columns, ref_df=None)`: z-score normalizes a list of columns, `(value - mean) / std`, clipped to +-5, using either the dataframe's own statistics or an external reference dataframe's statistics.
- `load_training_basket(tickers, start, end)`: runs `fetch_and_engineer` across every training ticker, pools their rows into one shared reference table, and normalizes each ticker against that pooled reference. Returns the normalized data plus the reference table itself.
- `load_test_basket(tickers, train_reference, start, end)`: runs `fetch_and_engineer` across the test tickers, but normalizes them against the training basket's reference statistics rather than their own, so the model sees test data on the same scale it was trained on.
- `align_dates(data_dict)`: trims every ticker in a dictionary down to the trading dates they all share, so multi-stock data lines up day-for-day by real calendar date rather than by row position.

The overall order, indicators, then percentage features, then drop missing rows, then normalize, is unchanged from the original design and for the same reason: RSI and MACD need a warm-up period before they're valid, so all indicators are computed before any rows are dropped, and normalization runs last against a clean, complete table.

The pipeline was tested manually on multiple tickers across both the training and test baskets and produced sensible, consistently-scaled results in each case.

## Design Decisions

Choice of indicators: RSI and MACD were chosen because they are the same inputs used in Sarkar (2023), one of the papers reviewed in the literature review, and because together they capture two different kinds of information. RSI measures momentum strength over a recent window, while MACD captures the relationship between a faster and a slower trend. Standard periods were used for both (RSI period of 14, MACD of 12, 26, 9).

Choice of moving average window: MA10 was kept as a short window moving average, since the bot makes daily buy, hold, or sell decisions rather than long term trend following decisions. A short window allows the agent to react to recent momentum instead of lagging behind a slower average.

Percentage features instead of raw indicators: The original pipeline normalized raw dollar-based indicators (closing price, MA10) directly. That approach meant the useful signal, how a price relates to its own average, was entangled with the stock's absolute price level. The pipeline now converts indicators into percentage-style features first (e.g. `Close_vs_MA10` instead of raw `Close` and `MA10`), so a 2% pullback looks identical whether it happens on a $40 stock or a $400 stock. This was necessary groundwork for the multi-ticker basket approach below.

Normalization against training-basket statistics, not each dataframe's own: Rather than min-max scaling each ticker (or each split) against its own minimum and maximum, features are now z-scored against a reference table, and that reference is deliberately built only from the training basket. Test tickers are scaled using the training basket's mean and standard deviation, never their own. This directly closes the look-ahead gap noted in the previous iteration, where a value early in the dataset was scaled using information from later rows: the test set can no longer leak its own statistics into its own scaling. Extreme values are clipped to +-5 standard deviations so a single outlier can't dominate the feature.

Caching and a fixed date window: `load_prices` now pins an explicit start and end date and caches each ticker's download to disk. This was added so that results are reproducible run to run. Earlier, a relative `period` argument meant the exact date window (and therefore the exact features) shifted depending on the day the pipeline was run.

## Testing

Unit tests were written in `tests/test_data_layer.py` using pytest, covering the same two categories from the project's testing plan, functional testing and data validation testing, expanded to cover the new multi-ticker and reference-based normalization behavior.

Functional tests check that each feature engineering function produces values that exist and fall within expected bounds:
- RSI values stay between 0 and 100
- MACD and MACD Signal columns are created and are not entirely empty
- The moving average never exceeds the minimum or maximum of the underlying closing prices
- Percentage features (`Ret_1d`, `Close_vs_MA10`, `MACD_pct`, `MACD_Signal_pct`) are created
- Normalized columns center on zero and stay within the +-5 clip bounds when scaled against their own statistics
- Normalizing against an external reference dataframe produces different results than self-normalizing, confirming `ref_df` is actually used
- Extreme outlier values are capped at +-5 rather than growing unbounded

Data validation tests check the output of the full pipeline:
- No missing values remain after `fetch_and_engineer()` runs
- All expected raw and percentage-feature columns are present in the final dataframe
- `align_dates()` correctly trims multiple tickers down to only their shared trading dates
- `align_dates()` raises a clear error when tickers share no trading dates at all

All 11 tests currently pass.