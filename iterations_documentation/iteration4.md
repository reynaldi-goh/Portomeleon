# Iteration 4: Integration Testing and Baseline

## Goal

The goal of this iteration was to get the three previously separate modules, the data layer, the trading environment, and the agent, running together as one full pipeline for the first time, establish a buy-and-hold baseline to judge the results against, and confirm the trained model is in a state the advisor layer (Iteration 5) can safely consume. No new modules were introduced. This iteration is about wiring `main.py` around the existing ones: a hyperparameter search over the training basket, scoring on disjoint test baskets, a day-by-day simulation on a user-chosen portfolio, and saving and reloading a trained model so it doesn't have to be retrained every run.

## What Was Built

`main.py` implements the pipeline in three stages, matching the project design:

- Stage 1, train: `train()` fits the DQN on the full training basket for a given `(alpha, n_episodes)` combo, blending each stock's own return with the shared portfolio-level reward from Iteration 2 according to `alpha`.
- Stage 2, test: `calibrate_scaler()` runs the finished model once over the full training period to freeze its `MarginScaler` against the confidence spread it actually learned, then the frozen model is scored on 3 disjoint test baskets of unseen tickers. `run_hyperparameter_search()` repeats stages 1 and 2 once per combo in `HYPERPARAM_GRID`, and keeps whichever combo scores best.
- Stage 3, simulate: `simulate_user_portfolio()` runs the best-scoring, frozen model day by day over a user-chosen subset of the training basket, using `run_stepper()`, the same stepping function stage 2 uses for test scoring.
- `compute_buy_and_hold_value()` implements the baseline: an equal cash split across the given tickers, bought once on day 0 and never traded again. This is used two ways, printed alongside the portfolio value on every simulated day so the comparison is visible in the terminal output, and inside `score_test_run()`, where a combo's `excess_return` is its final portfolio value against this same benchmark.
- `save_model()` and `load_model()` persist and restore the network's weights, the frozen `MarginScaler`, and the winning hyperparameters together, so a later run can skip stages 1 and 2 entirely and go straight to stage 3.

Tests were added in `tests/test_main.py`, covering the boundaries between modules that hadn't been exercised together before.

## Design Decisions

Integration testing at pipeline scope: this iteration checks the data layer, environment, and agent working together for the first time, at a larger scope than the per-module tests in Iterations 1 through 3. Combining all three modules at once rather than incrementally is sometimes called big-bang integration testing.

Buy-and-hold as the baseline: a model that beats a raw price increase isn't necessarily good if the market as a whole was rising, and a model that loses money isn't necessarily bad if it lost less than the market did. Buy-and-hold isolates the specific question this project needs answered, is actively trading with this policy better than doing nothing, by holding everything else (which stocks, how much cash, how long) constant between the two.

Synthetic data for most tests, one opt-in real-data check: `test_main.py` follows the same approach as the earlier iterations, small hand built price series for fast, deterministic tests that don't depend on the network. One test is marked separately and left opt-in, confirming that `load_training_basket()`'s real output actually satisfies what `TradingEnv` expects, since that specific boundary, real data layer output feeding the environment, had not been exercised together anywhere before this iteration.

Persisting the trained model separately from training it: `save_model()` and `load_model()` exist so stages 1 and 2, the expensive part, only need to run once. This matters beyond convenience for this iteration: the advisor layer in Iteration 5 needs to load a finished, frozen model without re-running a multi-combo hyperparameter search every time, so this needed to be in place before that work could start.

## Testing

Integration tests were written in `tests/test_main.py` using pytest, focused on the connections between modules rather than any one module's internal correctness, which is already covered by Iterations 1 through 3's own test files.

- `test_select_portfolio_actions_output_consumed_by_env_step` confirms the agent's action dictionary is exactly the shape `TradingEnv.step()` expects.
- `test_train_runs_end_to_end_and_updates_weights` confirms a short training run actually changes the network's weights, not just that it runs without error.
- `test_train_respects_alpha_blend_without_crashing` confirms both ends of the alpha blend, per-stock-only and portfolio-only, run cleanly.
- `test_calibrate_scaler_returns_frozen_scaler_with_samples` confirms `calibrate_scaler()` returns a genuinely frozen `MarginScaler` that has collected real samples.
- `test_compute_buy_and_hold_value_matches_manual_calculation` and `test_compute_buy_and_hold_value_at_day_zero_equals_initial_cash` confirm the baseline's math against hand calculated expected values, including the day-0 edge case.
- `test_score_test_run_excess_return_matches_manual_calculation` confirms the excess-return metric used to rank hyperparameter combos is computed correctly against the baseline.
- `test_simulate_user_portfolio_rejects_empty_selection`, `test_simulate_user_portfolio_rejects_too_many_tickers`, and `test_simulate_user_portfolio_rejects_unknown_ticker` confirm stage 3's input validation.
- `test_simulate_user_portfolio_runs_full_pipeline` confirms stage 3 runs end to end on a subset of the training basket and returns sane output.
- `test_run_hyperparameter_search_rejects_overlapping_baskets` confirms the training and test baskets are checked for overlap before any data is loaded.
- `test_save_and_load_model_roundtrip` confirms weights, the frozen scaler's state, and the winning hyperparameters all survive a real save and reload.
- `test_training_basket_feeds_trading_env_real_data`, marked separately as an opt-in network test, confirms `load_training_basket()`'s real output genuinely satisfies `TradingEnv` on a small real ticker pair over a short date range.

All 14 tests pass.