"""Terminal-level multi-stock simulation demo.

Three stages, matching the Portomeleon app design:

1. Train, fit the DQN on the FULL 2-year history of the training basket
   (8 stocks, spread across sectors), for each (alpha, n_episodes) combo.
2. Test, freeze each trained model, score it on the FULL 2-year history
   of a DISJOINT test basket (4 stocks the model never saw in training).
   This tests generalization to unseen tickers, not just unseen dates.
3. Simulation, the BEST scoring model, run day by day across a user-chosen
   portfolio drawn from the TRAINING basket (Option A: the app's actual
   product constraint). This is the deployment/demo view.
"""

import numpy as np
from src.data_layer import load_training_basket, load_test_basket, align_dates
from src.trading_env import TradingEnv
from src.agent import (
    build_agent,
    select_portfolio_actions,
    train_step,
    update_target_network,
    ReplayBuffer,
    get_margin_stats,
    reset_margin_log,
)

# fix a seed so every hyperparameter combo starts from the same random state,
# making the grid an actual controlled comparison instead of luck
RANDOM_SEED = 42

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# training basket: spread across sector and volatility, so the learned
# policy reflects general patterns rather than one sector's quirks
TRAINING_TICKERS = ["AAPL", "NVDA", "JPM", "JNJ", "XOM", "PG", "DIS", "KO"]

# test basket: disjoint from training, one ticker per sector already
# represented in training, so this tests generalization to unseen
# companies within familiar sectors, not a totally unfair extrapolation
TEST_TICKERS = ["MSFT", "BAC", "PFE", "CVX"]

MAX_USER_PORTFOLIO_SIZE = 5

DEFAULT_EPSILON_DECAY = 0.9

# hyperparameter combinations to search over for stage 1+2
HYPERPARAM_GRID = [
    {"alpha": 0.2, "n_episodes": 30},
    {"alpha": 0.5, "n_episodes": 30},
    {"alpha": 0.8, "n_episodes": 30},
    {"alpha": 0.2, "n_episodes": 50},
    {"alpha": 0.5, "n_episodes": 50},
    {"alpha": 0.8, "n_episodes": 50},
]


def train(env, q_net, target_net, optimizer, loss_fn, n_actions,
          n_episodes=20, batch_size=32, target_update_freq=100, alpha=0.5,
          epsilon_decay=DEFAULT_EPSILON_DECAY, verbose=True):
    """alpha controls the blend between per-stock and portfolio-level reward.

    alpha=1.0 means the agent only ever learns from each stock's own return.
    alpha=0.0 means the agent only ever learns from the shaped portfolio-level reward.
    alpha=0.5 means both count equally.
    """
    epsilon, epsilon_min = 1.0, 0.05
    buffer = ReplayBuffer(capacity=10000)
    step_count = 0

    # clear the margin log so stats reported after training reflect this run only
    reset_margin_log()

    for episode in range(n_episodes):
        obs_dict, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # log margins only once exploitation is likely (epsilon low), so
            # the stats reflect the learned policy's actual confidence spread
            log_margin = epsilon < 0.5
            actions = select_portfolio_actions(obs_dict, q_net, epsilon, n_actions,
                                                log_margin=log_margin)
            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # blend per-stock reward with the shared portfolio-level reward
            for ticker in env.tickers:
                blended_reward = (
                    alpha * info["stock_rewards"][ticker] + (1 - alpha) * reward
                )
                buffer.push(
                    obs_dict[ticker],
                    actions[ticker][0],
                    blended_reward,
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
        if verbose:
            print(f"  Episode {episode+1}/{n_episodes} | reward={total_reward:.4f} | "
                  f"epsilon={epsilon:.3f} | portfolio=${info['portfolio_value']:.2f}")

    # report whether margins are saturating the confidence scale (scale=0.02)
    stats = get_margin_stats()
    if stats:
        print(f"  [margin stats] mean={stats['mean']:.4f} | min={stats['min']:.4f} | "
              f"max={stats['max']:.4f} | saturated (>=0.02): {stats['pct_saturated']:.1%} "
              f"of {stats['count']} samples")


def compute_buy_and_hold_value(price_data, tickers, initial_cash, step):
    # value an equal-split, buy-on-day-0, never-trade portfolio for comparison
    cash_per_ticker = initial_cash / len(tickers)
    value = 0.0
    for ticker in tickers:
        day0_price = price_data[ticker].iloc[0]["Close"]
        shares = cash_per_ticker / day0_price
        current_price = price_data[ticker].iloc[step]["Close"]
        value += shares * current_price
    return value


def run_stepper(env, q_net, n_actions, price_data, interactive=True, label="Run", verbose=True):
    """Steps the trained (frozen, epsilon=0) model through price_data day by
    day. Used for both test scoring (unseen tickers) and simulation
    (training-basket tickers, full period)."""
    obs_dict, info = env.reset()
    done = False

    if verbose:
        print(f"\n--- {label}: {len(price_data[env.tickers[0]])} days, tickers={env.tickers} ---")

    while not done:
        cumulative_return = (info["portfolio_value"] - env.initial_cash) / env.initial_cash
        drawdown = (env.peak_value - info["portfolio_value"]) / env.peak_value if env.peak_value > 0 else 0.0

        window = env.return_history[-env.sharpe_window:]
        if len(window) >= env.sharpe_min_history:
            rolling_sharpe = np.mean(window) / (np.std(window) + 1e-6)
            sharpe_note = f"{rolling_sharpe:+.3f}"
        else:
            sharpe_note = "n/a (warming up)"

        benchmark_value = compute_buy_and_hold_value(
            price_data, env.tickers, env.initial_cash, env.current_step
        )

        if verbose:
            print(f"\nDay {env.current_step} | Portfolio: ${info['portfolio_value']:.2f} "
                  f"({cumulative_return:+.1%} since start) | Buy-and-hold benchmark: ${benchmark_value:.2f}")
            print(f"  Cash: ${info['cash']:.2f} | Shares: {info['shares_held']}")
            print(f"  Drawdown from peak: {drawdown:.1%} | Rolling Sharpe (last {env.sharpe_window}d): {sharpe_note}")

        actions = select_portfolio_actions(obs_dict, q_net, epsilon=0.0, n_actions=n_actions)
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        if verbose:
            for ticker in env.tickers:
                wanted = ACTION_NAMES[actions[ticker][0]]
                did = ACTION_NAMES[info["effective_action"][ticker]]
                size_note = f" (size={actions[ticker][1]:.1%})" if did != "HOLD" else ""
                stock_reward = info["stock_rewards"][ticker]
                print(f"  {ticker}: wanted {wanted} | did {did}{size_note} | stock reward: {stock_reward:+.2%}")

        if interactive:
            input("Press Enter for next day...")

    final_benchmark = compute_buy_and_hold_value(
        price_data, env.tickers, env.initial_cash, env.current_step
    )
    if verbose:
        print(f"\n[{label}] Final portfolio value: ${info['portfolio_value']:.2f} "
              f"(buy-and-hold would have given: ${final_benchmark:.2f})")

    return info


def score_test_run(info, env, price_data):
    """Combine excess return over buy-and-hold with risk-adjusted (Sharpe-style)
    performance into a single comparable score across hyperparameter combos."""
    # compute excess return relative to the buy-and-hold benchmark
    benchmark = compute_buy_and_hold_value(
        price_data, env.tickers, env.initial_cash, env.current_step
    )
    excess_return = (info["portfolio_value"] - benchmark) / benchmark

    # compute risk-adjusted return across the whole test run
    returns = env.return_history
    avg_sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0.0

    return excess_return + avg_sharpe


def run_hyperparameter_search(training_tickers=TRAINING_TICKERS, test_tickers=TEST_TICKERS,
                               period="2y", initial_cash=10000, param_grid=HYPERPARAM_GRID):
    """Stages 1 and 2, repeated once per (alpha, n_episodes) combo.

    Train uses the full 2y history of training_tickers.
    Test uses the full 2y history of test_tickers, a disjoint basket the
    model never saw, so this measures generalization to unseen stocks.
    """
    # fetch and normalize training basket, using its own pooled statistics
    train_data_raw, train_reference = load_training_basket(training_tickers, period)
    train_data = align_dates(train_data_raw)

    # fetch and normalize test basket, using the training basket's reference
    test_data_raw = load_test_basket(test_tickers, train_reference, period)
    test_data = align_dates(test_data_raw)

    obs_size = 7
    n_actions = 3

    results = []
    best_score = -np.inf
    best_q_net = None
    best_combo = None

    for combo in param_grid:
        print(f"\n=== Trying hyperparameters: alpha={combo['alpha']}, "
              f"n_episodes={combo['n_episodes']} ===")

        # build a fresh agent from a fixed seed, so every combo starts identically
        train_env = TradingEnv(train_data, initial_cash=initial_cash)
        q_net, target_net, optimizer, loss_fn = build_agent(obs_size, n_actions, seed=RANDOM_SEED)

        # print the full per-episode training log for this combo
        train(train_env, q_net, target_net, optimizer, loss_fn, n_actions,
              n_episodes=combo["n_episodes"], alpha=combo["alpha"],
              epsilon_decay=DEFAULT_EPSILON_DECAY, verbose=True)

        # score the frozen model on the disjoint, unseen test basket
        test_env = TradingEnv(test_data, initial_cash=initial_cash)
        info = run_stepper(test_env, q_net, n_actions, test_data,
                            interactive=False, label="Test (unseen tickers)", verbose=False)
        score = score_test_run(info, test_env, test_data)

        print(f"  >> combo result: final portfolio=${info['portfolio_value']:.2f} | score={score:.4f}")
        results.append({**combo, "score": score, "final_portfolio_value": info["portfolio_value"]})

        # keep this model if it beats the current best
        if score > best_score:
            best_score = score
            best_q_net = q_net
            best_combo = combo

    print(f"\n=== Best hyperparameters: alpha={best_combo['alpha']}, "
          f"n_episodes={best_combo['n_episodes']} (score={best_score:.4f}) ===")

    # return train_data too, since simulation draws user portfolios from training tickers
    return best_q_net, n_actions, train_data, results


def simulate_user_portfolio(chosen_tickers, q_net, n_actions, train_data,
                             initial_cash=10000, interactive=True):
    """Stage 3: the best-scoring, frozen model manages a user-chosen portfolio
    drawn from the training basket, run across its full period. This is the
    deployment/demo view.

    Option A: chosen_tickers must be a subset of the training basket, and
    at most MAX_USER_PORTFOLIO_SIZE of them.
    """
    if len(chosen_tickers) == 0:
        raise ValueError("Choose at least 1 stock.")
    if len(chosen_tickers) > MAX_USER_PORTFOLIO_SIZE:
        raise ValueError(f"Choose at most {MAX_USER_PORTFOLIO_SIZE} stocks "
                          f"(got {len(chosen_tickers)}).")

    unavailable = [t for t in chosen_tickers if t not in train_data]
    if unavailable:
        raise ValueError(
            f"{unavailable} not in the trained basket {list(train_data.keys())}. "
            "Stage 3 currently only supports tickers the model was trained on."
        )

    user_data = {t: train_data[t] for t in chosen_tickers}
    user_env = TradingEnv(user_data, initial_cash=initial_cash)

    return run_stepper(user_env, q_net, n_actions, user_data,
                        interactive=interactive, label="Simulation (full period)")


if __name__ == "__main__":
    # stage 1 + 2: search alpha and n_episodes, score on the disjoint test basket
    q_net, n_actions, train_data, results = run_hyperparameter_search()

    # stage 3: demo a user picking a smaller portfolio from the training basket
    user_choice = ["AAPL", "JPM", "XOM"]  # stand-in for real user input
    simulate_user_portfolio(user_choice, q_net, n_actions, train_data)