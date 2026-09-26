# Iteration 6: Streamlit App

## Goal

The goal of this iteration was to give the project a working front end, so a user can pick their own stocks, watch the trained DQN manage a portfolio day by day, and read the advisor's explanations, without touching the terminal or the underlying code. The UI needed to stay a thin layer over everything already built: the trading environment, the agent, and the advisor from earlier iterations. None of the simulation logic itself belongs in the UI, so it can be tested without Streamlit or torch getting in the way.

## What Was Built

The app is split across two files.

- `app.py` is the Streamlit entry point. It defines three pages, switched through `st.session_state.page` with no sidebar: a **dashboard** showing the last saved run, a **setup** page for picking stocks and starting capital, and a **simulation** page for stepping the bot forward and reading its explanations. `load_resources()` loads the trained model, the scaler, and the training basket's reference statistics once per app start and wraps them behind a `policy(obs_dict)` function, so the rest of the app never touches the model directly. Small formatting helpers (`money()`, `drawdown_text()`, `sharpe_text()`, `fmt_date()`, `holdings_table()`, `history_chart()`) keep the page functions themselves focused on layout.
- `src/simulation.py` holds everything about a running simulation. `clean_ticker()` and `load_user_stocks()` turn whatever the user typed into aligned, normalised price data for any ticker, not just the eight the model trained on. `SimulationSession` wraps the trading environment and advances it one day at a time, recording portfolio value, the buy-and-hold benchmark, and the day's `build_day_facts()` output for the advisor. `metrics_from_history()` computes Sharpe, drawdown, and cumulative return from that history. `save_last_simulation()` / `load_last_simulation()` persist the most recent run to a small JSON file, so the dashboard has something to show even after the app restarts.

## Design Decisions

**UI stays thin, simulation logic doesn't**: every page function in `app.py` only draws widgets and handles clicks; anything about how a simulation actually behaves, stepping, recording history, computing metrics, lives in `SimulationSession`. This is the same separation Iteration 5 used for facts versus wording, and it means `simulation.py` can be unit tested with a stub environment and a stub policy, with no Streamlit process and no trained model required.

**The policy is a function, not a model**: `SimulationSession` takes a `policy(obs_dict)` callable and an optional `q_net` (used only so the advisor's fact-builder can read Q-values). It never imports torch or calls `select_portfolio_actions` itself. `app.py` is the only place that wires the real model into a policy function, so tests can hand `SimulationSession` a stub policy and skip loading a model entirely.

**Any stock, not just the training basket**: the setup page lets the user pick 4 to 10 tickers from a common list or type any other symbol, rather than restricting them to the eight stocks the model trained on. `load_user_stocks()` fetches and normalises whatever is picked against the training basket's own scaling statistics, so an unfamiliar stock is still scaled the way the model expects. A caption on the setup page warns that results on very different stocks (penny stocks, crypto-like swings) are less reliable, since the model has never seen that kind of behaviour.

**Day facts are built once, in the same place as Iteration 5**: each `step()` call hands the day's raw state, actions, and portfolio numbers to the existing `build_day_facts()` from the advisor layer, rather than recomputing any of that in the UI. This keeps "facts are computed once, in one place" intact rather than letting the UI grow its own second copy of that logic.

**Explanations are cached per decision day**: `SimulationSession.explain()` only asks the advisor once for a given day and reuses the text afterward, so re-running the Streamlit script (which happens on every widget interaction) never re-sends the same day to Groq. Without an `Advisor` attached, it falls back to the no-LLM `template_explanation()` from Iteration 5.

**Persistence is plain JSON, not a pickled object**: `save_last_simulation()` writes a plain dictionary (`SimulationSession.snapshot()`) to disk instead of pickling the session itself, so the saved file stays human-readable and doesn't depend on the class's internal structure. `load_last_simulation()` returns `None` on a missing or corrupt file rather than raising, so the dashboard can fall back to its empty state.

**Deferred features are named, not silently dropped**: the specific-stock page and the company-details page from the original wireframes are left out of this minimal version on purpose, noted in `app.py`'s own module docstring, rather than half-built.

## Testing

Unit tests were written in `tests/test_simulation.py` using pytest, covering `simulation.py`'s own logic in isolation from the trading environment and advisor internals (both already covered by their own test suites from earlier iterations). A fake `TradingEnv` and a fake `build_day_facts()` are patched in for every test, so nothing here depends on torch, a trained model, or a network call.

- `test_clean_ticker_uppercases_and_strips` and `test_clean_ticker_rejects_bad_input` confirm ticker text is cleaned up correctly and invalid input is rejected.
- `test_load_user_stocks_rejects_too_few` confirms the minimum stock count is enforced.
- `test_load_user_stocks_reports_missing_tickers` confirms a ticker whose data couldn't be fetched is named in the error.
- `test_load_user_stocks_keeps_only_shared_dates` confirms only dates common to every picked stock are kept, with a note for each ticker that lost days.
- `test_buy_and_hold_value_equal_split` confirms the equal-split benchmark is valued correctly.
- `test_metrics_from_history_first_day_has_zero_change` and `test_metrics_from_history_computes_return_and_drawdown` confirm the headline metrics, including the day-one edge case and a later multi-day drawdown.
- `test_session_records_initial_day_on_construction` and `test_session_step_advances_and_records` confirm a session records its starting state on construction and updates correctly after a step.
- `test_explain_uses_advisor_when_present` confirms the advisor is asked exactly once per decision day and the cached text is reused on a second call.
- `test_holdings_shape_matches_tickers` confirms the holdings table covers every picked ticker.
- `test_save_and_load_last_simulation_round_trip` and `test_load_last_simulation_missing_file_returns_none` confirm a saved run can be loaded back correctly, and a missing file is handled gracefully.

All 14 tests currently pass.