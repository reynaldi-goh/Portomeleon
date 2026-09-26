# Iteration 2: Trading Environment

## Goal

The goal of this iteration was to extend the trading environment from the single stock prototype into a multi stock environment, as required by the full project design in Chapter 3. The prototype's `TradingEnv` could only hold and trade one stock at a time. This iteration reworks the environment so it can track a portfolio of several stocks at once, so the agent's per stock confidence can be used to decide how much of the available cash goes to each stock on a given day, and so the reward signal reflects risk-adjusted performance rather than raw portfolio change.

## What Was Built

The environment is implemented in `src/trading_env.py` and now looks like this:

- `price_data` is a dictionary of dataframes, one per ticker, instead of a single dataframe. All tickers are expected to be aligned to the same trading dates, using `align_dates()` from Iteration 1.
- `shares_held` and `effective_action` are dictionaries keyed by ticker, instead of single values.
- `_get_obs()` returns a dictionary of observation vectors, one per ticker. Each vector has 7 features: the five percentage-style normalized features from Iteration 1 (`Ret_1d_norm`, `Close_vs_MA10_norm`, `RSI_norm`, `MACD_pct_norm`, `MACD_Signal_pct_norm`), plus two portfolio context features, the stock's current share of total portfolio value (`allocation_pct`) and the current share of the portfolio sitting in cash (`cash_pct`).
- `step()` takes a dictionary of actions, one (action, size fraction) pair per ticker, instead of a single action. It separates the day's decisions into buys, sells, and holds, executes sells first, then pools and allocates cash across the buy signals based on relative confidence, then advances the day once for the whole portfolio.
- A confident sell (size fraction at or above 0.5) is treated as a full exit rather than a partial one, so a sell that would otherwise round down to zero shares is never silently swallowed and misread as a hold.
- Trade timing is now explicit: a step evaluates day k's close, executes trades at day k's close, then advances to day k+1 and revalues the portfolio at day k+1's close. The reward reflects what the position chosen on day k actually earned overnight.
- `_compute_reward()` replaces the raw portfolio percentage change with a rolling Sharpe-style reward (mean return over standard deviation, over a 20 day window, floored and clipped to avoid instability), penalized for drawdown from the episode's peak value, and penalized for sitting in cash.
- `step()` also returns a per-stock reward in `info["stock_rewards"]`, the actual close-to-close return earned by each stock still held after that day's trades, separate from the shared portfolio-level scalar reward.

The two portfolio context features were added so the agent has some information about its current position, not just the price movement of the stock in front of it. Without this, the agent evaluating one stock would have no way of knowing it is already heavily invested in that stock, or that it is nearly out of cash.

## Design Decisions

Discrete per-stock actions over a single joint allocation decision: two designs were considered for how the agent should interact with multiple stocks. The first was a single joint decision across the whole portfolio at once, for example choosing to put 15 percent in stock A and 20 percent in stock B simultaneously. This is closer to how continuous portfolio allocation problems are usually solved, but it does not fit a DQN, which works with a fixed set of discrete actions rather than a continuous range of splits, and supporting it would mean changing to a different algorithm entirely. The second design, the one used here, keeps the same discrete buy, hold, sell action space from the prototype, and evaluates each stock separately with the same trained network, once per stock per day.

Evaluating every stock before committing cash: the environment looks at every stock's decision for the day together before committing any cash, instead of processing one stock fully before moving to the next. This avoids a problem where the first stock evaluated could use up most of the available cash purely because of the order it happened to be checked in, rather than because it was actually the better opportunity.

Pooled buy allocation weighted by confidence: sell decisions are applied per stock independently, since selling does not involve competing for a shared resource. Buy decisions are pooled together first, then the available cash is split across all the stocks that received a buy signal, weighted by how confident the agent was in each one, so a stock the agent is more confident about receives a larger share of that day's cash than one it is less confident about.

Separating trade execution day from revaluation day: trades execute at day k's close, and the portfolio is only revalued at day k+1's close. This keeps the price used to decide and execute a trade separate from the price used to measure its outcome, so the reward for a given day's decision reflects a real overnight price move rather than reusing the same price for both the trade and its evaluation.

Risk-adjusted reward instead of raw portfolio change: the reward function was changed from raw portfolio percentage change to a rolling Sharpe-style reward, penalized for drawdown from the episode's peak. This gives the agent a built-in reason to avoid risky, high variance trades even when they are profitable on average, rather than only rewarding raw return.

Idle penalty for sitting in cash: an opportunity-cost penalty proportional to the cash fraction of the portfolio was added to the reward. This was added after testing in Iteration 3 showed that without it, the agent learned to simply buy and hold cash instead of trading, since sitting entirely in cash keeps the portfolio value flat and therefore avoids the drawdown penalty for free. The idle penalty closes that loophole by making it costly to sit out of the market purely to dodge drawdown, rather than because holding cash was actually the better decision.

## Testing

Unit tests were written in `tests/test_trading_env.py` using pytest, based on small hand built fake price data rather than real yfinance data, so that specific scenarios could be controlled directly.

Some of these tests are genuine unit tests, since they check one method's output in isolation:

- `test_reset_returns_obs_per_ticker` checks that `reset()` returns one correctly shaped observation per ticker
- `test_hold_does_not_change_cash_or_shares` checks that an all hold day leaves cash and shares completely unchanged
- `test_sell_with_no_shares_counts_as_hold` checks that trying to sell a stock with zero shares held is safely treated as a hold, rather than causing an error or a negative share count
- `test_state_resets_correctly_between_episodes` checks that cash, shares held, and effective actions all return to their initial values on `reset()`, rather than carrying over from the previous episode

A few of the other tests are closer to integration tests than pure unit tests, even though they are testing the same environment class. `test_portfolio_value_consistency`, `test_buy_allocation_weighted_by_confidence`, and `test_episode_ends_at_last_step` all depend on more than one part of the environment working correctly together, such as state tracking across multiple steps and the cash allocation math. The per-stock reward tests, `test_stock_rewards_present_in_info`, `test_stock_reward_reflects_price_move_when_held`, and `test_stock_rewards_differ_per_stock`, fall into the same category, since they depend on both the trade execution logic and the next-day price lookup being correct together. These tests are still useful, but they are better described as checking the environment's overall behaviour rather than a single isolated unit. This distinction matters because the project's testing plan lists functional and integration testing as two separate categories, and it would be inaccurate to call every test here a pure unit test.

All 10 tests currently pass.