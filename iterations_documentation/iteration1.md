(need to recheck later)

# Iteration 1: Data Layer

## Goal

The goal of this iteration was to build the data layer described in Section 3.3.1 of the project design. This layer is responsible for turning raw historical price data into a clean, feature engineered dataset that the trading environment and DQN agent can later use. Specifically, this iteration covers fetching price data, calculating technical indicators, and normalizing the resulting features. A train/test split and a formal look-ahead bias check are planned as a short follow up before Iteration 2 begins.

## What Was Built

The data layer is implemented in `src/data_layer.py` and is made up of the following functions:

- `load_price_data(ticker, period)`: downloads historical OHLCV data using the yfinance API and keeps only the closing price. This is the main entry point that calls the other functions below in sequence.
- `add_moving_average(df, window, col_name)`: adds a simple moving average column based on closing price. Currently used to add a 10 day moving average (MA10).
- `add_rsi(df, period)`: adds a Relative Strength Index (RSI) column, calculated using a 14 day period.
- `add_macd(df, fast, slow, signal)`: adds MACD and MACD Signal columns, using the standard 12, 26, 9 day parameters.
- `normalize_features(df, columns)`: applies min-max normalization to a given list of columns so that all features fall within a 0 to 1 range.

`load_price_data()` runs these steps in order: fetch data, add MA10, add RSI, add MACD, drop rows with missing values, then normalize. This order matters because RSI and MACD both need a warm up period before they produce valid values, so all indicators are added before any rows are dropped. Normalization is done last so that the missing values from the warm up period do not affect the min and max values used for scaling.

The pipeline was tested manually in a notebook on two different tickers (AAPL and a second stock) and produced sensible results in both cases, which suggests the function generalizes beyond a single hardcoded ticker.

## Design Decisions

**Choice of indicators.** RSI and MACD were chosen because they are the same inputs used in Sarkar (2023), one of the papers reviewed in the literature review, and because together they capture two different kinds of information: RSI measures momentum strength over a recent window, while MACD captures the relationship between a faster and a slower trend. Standard periods were used for both (RSI period of 14, MACD of 12, 26, 9) since these are the conventional defaults in technical analysis and were not specifically overridden by the source literature.

**Choice of moving average window.** MA10 was kept as a short window moving average, since the bot makes daily buy, hold, or sell decisions rather than long term trend following decisions. A short window allows the agent to react to recent momentum instead of lagging behind a slower average. Since RSI and MACD already provide momentum and trend crossover signals, a longer moving average (such as MA50) may be added later to give the agent both a short term and a medium term view of price direction, rather than three indicators that measure similar information.

**Normalization at the data layer.** Normalization was applied here, rather than only inside the trading environment, so that features stay on a consistent scale before they are eventually combined across multiple stocks in a future iteration. This also avoids the earlier scale mismatch issue noted in the prototype, where cash and portfolio value (in the thousands) dominated price based features (in the hundreds) inside the environment's observation vector.

## Testing

Unit tests were written in `tests/test_data_layer.py` using pytest, covering two categories from the project's testing plan: functional testing and data validation testing.

Functional tests check that each feature engineering function produces values that exist and fall within expected bounds:
- RSI values stay between 0 and 100
- MACD and MACD Signal columns are created and are not entirely empty
- Normalized columns stay between 0 and 1
- The moving average never exceeds the minimum or maximum of the underlying closing prices

Data validation tests check the output of the full pipeline:
- No missing values remain after `load_price_data()` runs
- All expected columns are present in the final dataframe

All 7 tests currently pass.

```
tests\test_data_layer.py .......                          [100%]
======================== 7 passed in 1.41s ========================
```

## Limitations and Next Steps

A few gaps remain from this iteration and are carried forward:

- **Look-ahead bias test not yet written.** Although the pipeline is designed to avoid look-ahead bias (each indicator only uses past and present rows), this has not yet been confirmed with an automated test. This is planned as the next task.
- **No train/test split yet.** The current pipeline returns a single dataframe covering the full requested period. A proper split is needed so the agent can later be evaluated on unseen data, as described in the project design.
- **Min-max normalization has a mild look-ahead property.** Because it uses the minimum and maximum of the entire dataset to scale each column, a value early in the dataset is technically scaled using information from later rows. This is a common simplification at this scope, but it is worth noting rather than treating as fully leak free.
- **MACD's early rows are less accurate.** The exponential moving averages used in MACD do not produce explicit missing values, but they are less accurate for roughly the first 26 rows while they converge. These rows are not currently dropped separately from the other indicators' missing values.
- **Only single ticker fetching has been tested.** The function currently accepts one ticker at a time. Support for looping over a list of tickers is planned for Iteration 2, alongside the trading environment changes needed to actually use multiple stocks.