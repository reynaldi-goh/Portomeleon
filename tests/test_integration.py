"""Integration test: data layer + trading env + agent working together."""
from src.trading_env import TradingEnv
from src.agent import (
    build_agent, select_portfolio_actions, train_step,
    update_target_network, ReplayBuffer,
)


def make_fake_stock_with_real_shape(prices):
    import pandas as pd
    import numpy as np
    n = len(prices)
    prices = np.array(prices, dtype=np.float32)
    norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
    return pd.DataFrame({
        "Close": prices, "Close_norm": norm, "MA10_norm": norm,
        "RSI_norm": norm, "MACD_norm": norm, "MACD_Signal_norm": norm,
    })


def test_full_pipeline_runs_without_error():
    price_data = {
        "AAPL": make_fake_stock_with_real_shape([100, 102, 105, 103, 106, 108, 107, 110, 112, 111]),
        "GOOG": make_fake_stock_with_real_shape([140, 138, 136, 134, 132, 130, 128, 126, 124, 122]),
    }
    env = TradingEnv(price_data, initial_cash=10000)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size, n_actions)
    buffer = ReplayBuffer(capacity=1000)

    obs_dict, info = env.reset()
    done = False
    step_count = 0

    while not done:
        actions = select_portfolio_actions(obs_dict, q_net, epsilon=0.5, n_actions=n_actions)
        next_obs_dict, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        for ticker in env.tickers:
            buffer.push(
                obs_dict[ticker], actions[ticker][0],
                info["stock_rewards"][ticker], next_obs_dict[ticker], done,
            )

        loss = train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size=4)
        step_count += 1
        obs_dict = next_obs_dict

    assert step_count == env.n_steps - 1
    assert info["portfolio_value"] > 0  # never went negative/broke