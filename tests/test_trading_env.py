"""Unit tests for the multi-stock trading environment: Iteration 2."""
import numpy as np
import pandas as pd
from src.trading_env import TradingEnv


def make_fake_stock(prices):
    # build a fake price dataframe with the normalized columns the env expects
    n = len(prices)
    return pd.DataFrame({
        "Close": prices,
        "Ret_1d_norm": np.linspace(0, 1, n),
        "Close_vs_MA10_norm": np.linspace(0, 1, n),
        "RSI_norm": np.linspace(0, 1, n),
        "MACD_pct_norm": np.linspace(0, 1, n),
        "MACD_Signal_pct_norm": np.linspace(0, 1, n),
    })


def make_env():
    # set up a two-stock environment with distinct price paths
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
    # check portfolio_value matches cash plus holdings priced at the NEW day's close,
    # since current_step has already advanced by the time step() returns
    env = make_env()
    env.reset()
    actions = {"AAPL": (1, 0.5), "TSLA": (1, 0.3)}
    obs, reward, terminated, truncated, info = env.step(actions)

    expected_value = info["cash"] + sum(
        info["shares_held"][t] * env._get_price(t, step=env.current_step)
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


# ---------- Per-stock reward (info["stock_rewards"]) ----------

def test_stock_rewards_present_in_info():
    env = make_env()
    env.reset()
    actions = {"AAPL": (1, 0.5), "TSLA": (1, 0.3)}
    obs, reward, terminated, truncated, info = env.step(actions)

    assert "stock_rewards" in info
    assert set(info["stock_rewards"].keys()) == {"AAPL", "TSLA"}


def test_stock_reward_reflects_price_move_when_held():
    # a stock bought today is still "held after today's trades", so its reward
    # is the actual close-to-close price move, not zero just because it's a new position
    env = make_env()
    env.reset()
    actions = {"AAPL": (1, 0.5), "TSLA": (0, 0.0)}
    obs, reward, terminated, truncated, info = env.step(actions)

    aapl_price_today = env._get_price("AAPL", step=0)
    aapl_price_next = env._get_price("AAPL", step=1)
    expected_aapl_reward = (aapl_price_next - aapl_price_today) / aapl_price_today

    assert np.isclose(info["stock_rewards"]["AAPL"], expected_aapl_reward)
    assert info["stock_rewards"]["TSLA"] == 0.0  # never bought, so not held


def test_stock_rewards_differ_per_stock():
    # build a scenario where two stocks move by different amounts after both
    # already hold a position, so their rewards should genuinely differ
    env = make_env()
    env.reset()

    # day 1: buy into both, establishing a position
    env.step({"AAPL": (1, 0.9), "TSLA": (1, 0.9)})

    # day 2: hold both, but the fake prices move by different amounts for each
    obs, reward, terminated, truncated, info = env.step(
        {"AAPL": (0, 0.0), "TSLA": (0, 0.0)}
    )

    aapl_reward = info["stock_rewards"]["AAPL"]
    tsla_reward = info["stock_rewards"]["TSLA"]
    assert aapl_reward != tsla_reward


def test_state_resets_correctly_between_episodes():
    # after a step changes cash and holdings, reset() should return to the
    # initial episode state rather than carrying anything over
    env = make_env()
    env.reset()
    env.step({"AAPL": (1, 0.5), "TSLA": (0, 0.0)})

    obs, info = env.reset()
    assert env.cash == env.initial_cash
    assert env.shares_held == {t: 0 for t in env.tickers}
    assert env.effective_action == {t: 0 for t in env.tickers}