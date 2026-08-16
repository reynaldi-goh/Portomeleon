"""Unit tests for the multi-stock trading environment — Iteration 2."""
import numpy as np
import pandas as pd
from src.trading_env import TradingEnv


def make_fake_stock(prices):
    n = len(prices)
    return pd.DataFrame({
        "Close": prices,
        "Close_norm": np.linspace(0, 1, n),
        "MA10_norm": np.linspace(0, 1, n),
        "RSI_norm": np.linspace(0, 1, n),
        "MACD_norm": np.linspace(0, 1, n),
        "MACD_Signal_norm": np.linspace(0, 1, n),
    })


def make_env():
    price_data = {
        "AAPL": make_fake_stock([100, 102, 105, 103, 106]),
        "TSLA": make_fake_stock([200, 198, 205, 210, 208]),
    }
    return TradingEnv(price_data, initial_cash=10000)


def test_reset_returns_obs_per_ticker():
    env = make_env()
    obs, info = env.reset()
    assert set(obs.keys()) == {"AAPL", "TSLA"}
    assert obs["AAPL"].shape == (7,)


def test_portfolio_value_consistency():
    env = make_env()
    env.reset()
    actions = {"AAPL": (1, 0.5), "TSLA": (1, 0.3)}
    obs, reward, terminated, truncated, info = env.step(actions)

    expected_value = info["cash"] + sum(
        info["shares_held"][t] * env._get_price(t, step=env.current_step - 1)
        for t in env.tickers
    )
    assert np.isclose(info["portfolio_value"], expected_value)


def test_buy_allocation_weighted_by_confidence():
    env = make_env()
    env.reset()
    # AAPL has higher confidence than TSLA, so should get a larger cash share
    actions = {"AAPL": (1, 0.6), "TSLA": (1, 0.2)}
    obs, reward, terminated, truncated, info = env.step(actions)

    aapl_spent = info["shares_held"]["AAPL"] * env._get_price("AAPL", step=0)
    tsla_spent = info["shares_held"]["TSLA"] * env._get_price("TSLA", step=0)
    assert aapl_spent > tsla_spent


def test_hold_does_not_change_cash_or_shares():
    env = make_env()
    env.reset()
    actions = {"AAPL": (0, 0.0), "TSLA": (0, 0.0)}
    obs, reward, terminated, truncated, info = env.step(actions)

    assert info["cash"] == env.initial_cash
    assert info["shares_held"]["AAPL"] == 0
    assert info["shares_held"]["TSLA"] == 0


def test_sell_with_no_shares_counts_as_hold():
    env = make_env()
    env.reset()
    actions = {"AAPL": (2, 0.5), "TSLA": (0, 0.0)}  # sell AAPL with 0 shares held
    obs, reward, terminated, truncated, info = env.step(actions)

    assert info["effective_action"]["AAPL"] == 0
    assert info["shares_held"]["AAPL"] == 0


def test_episode_ends_at_last_step():
    env = make_env()
    env.reset()
    truncated = False
    steps_taken = 0
    while not truncated:
        actions = {t: (0, 0.0) for t in env.tickers}
        obs, reward, terminated, truncated, info = env.step(actions)
        steps_taken += 1
    assert steps_taken == env.n_steps - 1