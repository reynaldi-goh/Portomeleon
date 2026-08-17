"""Terminal-level multi-stock simulation demo."""

from src.data_layer import load_price_data
from src.trading_env import TradingEnv
from src.agent import (
    build_agent,
    select_portfolio_actions,
    train_step,
    update_target_network,
    ReplayBuffer,
)

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def load_multi_stock_data(tickers, period="2y"):
    raw = {ticker: load_price_data(ticker, period) for ticker in tickers}
    common_dates = raw[tickers[0]].index
    for ticker in tickers[1:]:
        common_dates = common_dates.intersection(raw[ticker].index)
    return {ticker: raw[ticker].loc[common_dates].reset_index(drop=True) for ticker in tickers}


def train(env, q_net, target_net, optimizer, loss_fn, n_actions,
          n_episodes=20, batch_size=32, target_update_freq=100):
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.95
    buffer = ReplayBuffer(capacity=10000)
    step_count = 0

    for episode in range(n_episodes):
        obs_dict, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = select_portfolio_actions(obs_dict, q_net, epsilon, n_actions)
            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            for ticker in env.tickers:
                buffer.push(
                    obs_dict[ticker],
                    actions[ticker][0],
                    info["stock_rewards"][ticker],
                    next_obs_dict[ticker],
                    done,
                )

            train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size)
            step_count += 1
            if step_count % target_update_freq == 0:
                update_target_network(q_net, target_net)

            obs_dict = next_obs_dict
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode+1}/{n_episodes} | reward={total_reward:.4f} | "
              f"epsilon={epsilon:.3f} | portfolio=${info['portfolio_value']:.2f}")


def run_stepper(env, q_net, n_actions):
    obs_dict, info = env.reset()
    done = False

    while not done:
        print(f"\nDay {env.current_step} | Cash: ${info['cash']:.2f} | "
              f"Shares: {info['shares_held']} | Portfolio: ${info['portfolio_value']:.2f}")

        actions = select_portfolio_actions(obs_dict, q_net, epsilon=0.0, n_actions=n_actions)
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        for ticker in env.tickers:
            wanted = ACTION_NAMES[actions[ticker][0]]
            did = ACTION_NAMES[info["effective_action"][ticker]]
            size_note = f" (size={actions[ticker][1]:.1%})" if did != "HOLD" else ""
            stock_reward = info["stock_rewards"][ticker]
            print(f"  {ticker}: wanted {wanted} | did {did}{size_note} | stock reward: {stock_reward:+.2%}")

        input("Press Enter for next day...")

    print(f"\nFinal portfolio value: ${info['portfolio_value']:.2f}")


if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "GOOG"]
    price_data = load_multi_stock_data(tickers, period="2y")

    env = TradingEnv(price_data, initial_cash=10000)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net, target_net, optimizer, loss_fn = build_agent(obs_size, n_actions)

    train(env, q_net, target_net, optimizer, loss_fn, n_actions)
    run_stepper(env, q_net, n_actions)