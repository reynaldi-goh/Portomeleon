"""Integration tests for the full pipeline: Iteration 4.

These tests exercise the data layer, trading environment, and agent
together through main.py's orchestration functions, rather than any one
module in isolation. All price data here is small and hand built, so
these run without a network call. The one exception, marked below, is
opt-in and touches real data to confirm load_training_basket's output
actually feeds TradingEnv correctly.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.trading_env import TradingEnv
from src.agent import build_agent, select_portfolio_actions, MarginScaler
from main import (
    train,
    calibrate_scaler,
    compute_buy_and_hold_value,
    score_test_run,
    run_hyperparameter_search,
    save_model,
    load_model,
    simulate_user_portfolio,
)


def make_fake_stock(n_days, start_price, seed):
    # build a small synthetic price series with the normalized columns TradingEnv expects
    rng = np.random.default_rng(seed)
    prices = start_price + np.cumsum(rng.normal(0, 1, n_days))
    return pd.DataFrame({
        "Close": prices,
        "Ret_1d_norm": rng.normal(0, 1, n_days),
        "Close_vs_MA10_norm": rng.normal(0, 1, n_days),
        "RSI_norm": rng.normal(0, 1, n_days),
        "MACD_pct_norm": rng.normal(0, 1, n_days),
        "MACD_Signal_pct_norm": rng.normal(0, 1, n_days),
    })


def make_fake_basket(tickers, n_days=40):
    # build a small multi-ticker basket, already "aligned" since every series shares length
    return {t: make_fake_stock(n_days, start_price=100 + 10 * i, seed=i)
            for i, t in enumerate(tickers)}


def make_fresh_agent():
    return build_agent(obs_size=7, n_actions=3, seed=0)


# ---------- env + agent integration ----------

def test_select_portfolio_actions_output_consumed_by_env_step():
    # confirm the agent's action dict is exactly the shape the environment expects
    price_data = make_fake_basket(["AAPL", "TSLA"])
    env = TradingEnv(price_data, initial_cash=10000)
    q_net, target_net, optimizer, loss_fn = make_fresh_agent()

    obs_dict, info = env.reset()
    actions = select_portfolio_actions(obs_dict, q_net, epsilon=0.0, n_actions=3)
    obs_dict, reward, terminated, truncated, info = env.step(actions)

    assert set(actions.keys()) == set(env.tickers)
    assert "stock_rewards" in info


# ---------- train() integration ----------

def test_train_runs_end_to_end_and_updates_weights():
    # confirm a short training run actually changes the network's weights
    price_data = make_fake_basket(["AAPL", "TSLA"], n_days=40)
    env = TradingEnv(price_data, initial_cash=10000)
    q_net, target_net, optimizer, loss_fn = make_fresh_agent()

    initial_weights = [p.clone() for p in q_net.parameters()]
    train(env, q_net, target_net, optimizer, loss_fn, n_actions=3,
          n_episodes=2, batch_size=8, verbose=False)

    changed = any(
        not torch.allclose(before, after)
        for before, after in zip(initial_weights, q_net.parameters())
    )
    assert changed


def test_train_respects_alpha_blend_without_crashing():
    # alpha=0.0 (portfolio-only) and alpha=1.0 (per-stock-only) are both valid blends
    price_data = make_fake_basket(["AAPL", "TSLA"], n_days=30)
    for alpha in (0.0, 1.0):
        env = TradingEnv(price_data, initial_cash=10000)
        q_net, target_net, optimizer, loss_fn = make_fresh_agent()
        train(env, q_net, target_net, optimizer, loss_fn, n_actions=3,
              n_episodes=1, alpha=alpha, batch_size=8, verbose=False)


# ---------- calibrate_scaler() integration ----------

def test_calibrate_scaler_returns_frozen_scaler_with_samples():
    price_data = make_fake_basket(["AAPL", "TSLA"], n_days=40)
    env = TradingEnv(price_data, initial_cash=10000)
    q_net, target_net, optimizer, loss_fn = make_fresh_agent()

    scaler = calibrate_scaler(env, q_net, n_actions=3)

    assert isinstance(scaler, MarginScaler)
    assert scaler.frozen
    assert scaler.count > 0


# ---------- buy-and-hold baseline ----------

def test_compute_buy_and_hold_value_matches_manual_calculation():
    price_data = {
        "A": pd.DataFrame({"Close": [100, 110, 120]}),
        "B": pd.DataFrame({"Close": [50, 45, 60]}),
    }
    value = compute_buy_and_hold_value(price_data, ["A", "B"], initial_cash=1000, step=2)

    # equal cash split on day 0, held to day 2, no trading
    expected = (500 / 100) * 120 + (500 / 50) * 60
    assert np.isclose(value, expected)


def test_compute_buy_and_hold_value_at_day_zero_equals_initial_cash():
    price_data = {"A": pd.DataFrame({"Close": [100, 110, 120]})}
    value = compute_buy_and_hold_value(price_data, ["A"], initial_cash=1000, step=0)
    assert np.isclose(value, 1000)


def test_score_test_run_excess_return_matches_manual_calculation():
    price_data = make_fake_basket(["AAPL", "TSLA"], n_days=30)
    env = TradingEnv(price_data, initial_cash=10000)
    env.reset()
    env.current_step = len(price_data["AAPL"]) - 1  # pretend the run just finished
    info = {"portfolio_value": 11000.0}

    result = score_test_run(info, env, price_data)

    benchmark = compute_buy_and_hold_value(price_data, env.tickers, env.initial_cash, env.current_step)
    expected_excess = (11000.0 - benchmark) / benchmark
    assert np.isclose(result["excess_return"], expected_excess)


# ---------- simulate_user_portfolio() validation ----------

def test_simulate_user_portfolio_rejects_empty_selection():
    with pytest.raises(ValueError):
        simulate_user_portfolio([], None, 3, {})


def test_simulate_user_portfolio_rejects_too_many_tickers():
    train_data = make_fake_basket(["A", "B", "C", "D", "E", "F"])
    too_many = ["A", "B", "C", "D", "E", "F"]
    with pytest.raises(ValueError):
        simulate_user_portfolio(too_many, None, 3, train_data)


def test_simulate_user_portfolio_rejects_unknown_ticker():
    train_data = make_fake_basket(["A", "B"])
    with pytest.raises(ValueError):
        simulate_user_portfolio(["A", "ZZZZ"], None, 3, train_data)


def test_simulate_user_portfolio_runs_full_pipeline():
    # confirm stage 3 runs end to end on a subset of the training basket
    train_data = make_fake_basket(["AAPL", "TSLA", "MSFT"], n_days=30)
    q_net, target_net, optimizer, loss_fn = make_fresh_agent()

    info = simulate_user_portfolio(
        ["AAPL", "TSLA"], q_net, n_actions=3, train_data=train_data,
        initial_cash=10000, interactive=False,
    )

    assert set(info["shares_held"].keys()) == {"AAPL", "TSLA"}
    assert info["portfolio_value"] > 0


# ---------- run_hyperparameter_search() validation ----------

def test_run_hyperparameter_search_rejects_overlapping_baskets():
    # this check happens before any data is fetched, so it needs no network
    with pytest.raises(ValueError):
        run_hyperparameter_search(
            training_tickers=["AAPL", "MSFT"],
            test_baskets={"A": ["AAPL", "GOOG"]},
        )


# ---------- save_model / load_model roundtrip ----------

def test_save_and_load_model_roundtrip(tmp_path):
    q_net, target_net, optimizer, loss_fn = make_fresh_agent()
    scaler = MarginScaler()
    for margin in [0.01, 0.02, 0.03]:
        scaler.update(margin)
    scaler.freeze()

    path = tmp_path / "model.pt"
    best_combo = {"alpha": 0.5, "n_episodes": 30}
    save_model(str(path), q_net, scaler, obs_size=7, n_actions=3, best_combo=best_combo)

    loaded_q_net, n_actions, loaded_scaler, loaded_combo = load_model(str(path))

    assert n_actions == 3
    assert loaded_combo == best_combo
    assert loaded_scaler.count == scaler.count
    assert loaded_scaler.frozen
    assert not loaded_q_net.training  # eval() was called on load

    for p_original, p_loaded in zip(q_net.parameters(), loaded_q_net.parameters()):
        assert torch.allclose(p_original, p_loaded)


# ---------- data layer -> trading env, opt-in real-data check ----------

@pytest.mark.network
def test_training_basket_feeds_trading_env_real_data():
    # confirm load_training_basket's real output shape actually satisfies TradingEnv,
    # using a short date range and two tickers to keep the download small
    from src.data_layer import load_training_basket, align_dates

    raw, _ = load_training_basket(["AAPL", "MSFT"], start="2025-01-01", end="2025-03-01")
    aligned = align_dates(raw)
    env = TradingEnv(aligned, initial_cash=10000)

    obs, info = env.reset()
    assert set(obs.keys()) == {"AAPL", "MSFT"}
    assert obs["AAPL"].shape == (7,)