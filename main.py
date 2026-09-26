"""Terminal-level multi-stock simulation demo.

1. Train, fit the DQN on the FULL 2-year history of the training basket (8 stocks, spread across sectors), for each (alpha, n_episodes) combo.
2. Test, freeze each trained model, score it on 3 DISJOINT test basket (4 stocks each, none seen in training) over the FULL 2-year history.
   This tests generalization to unseen tickers, not just unseen dates.
   A combo's score is the AVERAGE excess return over buy-and-hold across the 3 test baskets, so one lucky (or unlucky) basket can't decide the winner.
3. Simulation, the BEST scoring model, run day by day across a user-chosen portfolio drawn from the TRAINING basket (Option A: the app's actual
   product constraint). This is the deployment/demo view.

Model persistence: stage 1+2 (hyperparameter search + training) is the expensive part. Once a best model is found, its weights and frozen
MarginScaler are saved to disk (see save_model/load_model below), so later runs can skip straight to stage 3 instead of retraining from scratch.
"""

import os
import pickle

from dotenv import load_dotenv
load_dotenv()  # read .env in the current directory and set its vars into
                # os.environ, e.g. GROQ_API_KEY -- so `advisor.py`'s
                # os.environ.get("GROQ_API_KEY") picks it up without you
                # needing to `export` it in the shell every session

import numpy as np
import torch

from src.data_layer import load_training_basket, load_test_basket, align_dates
from src.trading_env import TradingEnv
from src.agent import (
    build_agent,
    select_portfolio_actions,
    train_step,
    update_target_network,
    ReplayBuffer,
    MarginScaler,
    get_margin_stats,
    reset_margin_log,
)
from src.advisor import Advisor, build_day_facts

# fix a seed for better reproducibility
# making the grid an actual controlled comparison instead of luck
RANDOM_SEED = 42

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# training basket: spread across sector and volatility, so the learned
# policy reflects general patterns rather than one sector's quirks
TRAINING_TICKERS = ["AAPL", "NVDA", "JPM", "JNJ", "XOM", "PG", "DIS", "KO"]

# test baskets: 3 different combinations of stocks the model never saw in
# training. Each basket has one stock per broad sector already represented in
# training (tech, financials, healthcare, energy/staples), so this tests
# generalization to unseen companies in familiar sectors.
# The baskets don't overlap each other, so each one is an independent test.
TEST_BASKETS = {
    "A": ["MSFT", "BAC", "PFE", "CVX"],
    "B": ["AVGO", "GS",  "ABBV", "SHEL"],
    "C": ["AMZN", "MS", "ABT", "WMT"],
}

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

# where the trained model (weights + scaler) is cached between runs
MODEL_PATH = "trained_model.pt"


def train(env, q_net, target_net, optimizer, loss_fn, n_actions,
          n_episodes=20, batch_size=32, target_update_freq=100, alpha=0.5,
          epsilon_decay=DEFAULT_EPSILON_DECAY, verbose=True, scaler=None):
    """alpha controls the blend between per-stock and portfolio-level reward.

    scaler: a MarginScaler that learns the bot's usual confidence spread while
    training. The caller freezes it afterwards and keeps it with the model.

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
                                                log_margin=log_margin, scaler=scaler)
            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # blend per-stock reward with the shared portfolio-level reward
            for ticker in env.tickers:
                blended_reward = (
                    alpha * info["stock_rewards"][ticker] + (1 - alpha) * reward
                )
                buffer.push(
                    obs_dict[ticker],
                    info["effective_action"][ticker],
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

    # report the raw margin spread and the trade sizes it produced
    stats = get_margin_stats()
    if stats:
        print(f"  [margin stats] mean={stats['mean']:.4f} | min={stats['min']:.4f} | "
              f"max={stats['max']:.4f} | margins >=0.02 (old cap, info only): {stats['pct_saturated']:.1%}")
        print(f"  [size stats] mean trade size={stats['mean_size']:.1%} | "
              f"trades sized >=80%: {stats['pct_near_max_size']:.1%} of {stats['count']} samples")


def calibrate_scaler(env, q_net, n_actions, capacity=20000):
    """Run the finished model once over the whole training period and remember
    every confidence gap it produced, so trade sizes are ranked against the
    model's behavior across ALL of training, not just the last stretch.

    Returns a frozen MarginScaler. Keep it with the model.
    (While it builds up, the scaler also sizes the trades in this pass. That is
    fine, we only want the gaps the model produces.)
    """
    scaler = MarginScaler(capacity=capacity)
    obs_dict, info = env.reset()
    done = False
    while not done:
        actions = select_portfolio_actions(obs_dict, q_net, epsilon=0.0,
                                            n_actions=n_actions, scaler=scaler)
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
    scaler.freeze()
    return scaler

# this is the baseline model
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


def run_stepper(env, q_net, n_actions, price_data, interactive=True, label="Run", verbose=True,
                scaler=None, advisor=None):
    """Steps the trained (frozen, epsilon=0) model through price_data day by
    day. Used for both test scoring (unseen tickers) and simulation
    (training-basket tickers, full period).

    scaler: the model's frozen MarginScaler, so trade sizing matches training.
    advisor: an Advisor. If given (and interactive), typing 'why' after a day
    explains that day's decisions in plain language."""
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

        # capture what the bot saw before it acts, so the advisor can explain it afterwards
        explain = advisor is not None and interactive
        if explain:
            decision_day = env.current_step
            states_seen = obs_dict
            raw_rows = {t: price_data[t].iloc[decision_day] for t in env.tickers}
            shares_before = dict(info["shares_held"])
            portfolio_note = {
                "value": round(info["portfolio_value"], 2),
                "return_since_start_pct": round(cumulative_return * 100, 2),
                "buy_and_hold_value": round(benchmark_value, 2),
                "drawdown_pct": round(drawdown * 100, 2),
            }

        actions = select_portfolio_actions(obs_dict, q_net, epsilon=0.0, n_actions=n_actions,
                                            scaler=scaler)
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        day_facts = None
        if explain:
            day_facts = build_day_facts(decision_day, env.tickers, states_seen, q_net, actions,
                                        info, raw_rows, shares_before, portfolio_note)

        if verbose:
            for ticker in env.tickers:
                wanted = ACTION_NAMES[actions[ticker][0]]
                did = ACTION_NAMES[info["effective_action"][ticker]]
                size_note = f" (size={actions[ticker][1]:.1%})" if did != "HOLD" else ""
                stock_reward = info["stock_rewards"][ticker]
                print(f"  {ticker}: wanted {wanted} | did {did}{size_note} | stock reward: {stock_reward:+.2%}")

        if interactive:
            prompt = "Press Enter for next day"
            prompt += ", or type 'why' for the advisor's explanation: " if day_facts else "..."
            while True:
                choice = input(prompt).strip().lower()
                if choice == "why" and day_facts is not None:
                    print("\n" + advisor.explain_day(day_facts) + "\n")
                    continue
                break

    final_benchmark = compute_buy_and_hold_value(
        price_data, env.tickers, env.initial_cash, env.current_step
    )
    if verbose:
        print(f"\n[{label}] Final portfolio value: ${info['portfolio_value']:.2f} "
              f"(buy-and-hold would have given: ${final_benchmark:.2f})")

    return info


def score_test_run(info, env, price_data):
    """Score ONE test basket run against its own buy-and-hold benchmark.

    Returns a dict. "excess_return" is the number the hyperparameter search averages across baskets: how much better (or worse) the model did than
    simply buying equal amounts on day 0 and never trading.
    """
    benchmark = compute_buy_and_hold_value(
        price_data, env.tickers, env.initial_cash, env.current_step
    )
    excess_return = (info["portfolio_value"] - benchmark) / benchmark

    # risk-adjusted return across the run, reported for information only
    returns = env.return_history
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0.0

    return {
        "excess_return": excess_return,
        "sharpe": sharpe,
        "final_value": info["portfolio_value"],
        "benchmark_value": benchmark,
    }


def run_hyperparameter_search(training_tickers=TRAINING_TICKERS, test_baskets=TEST_BASKETS,
                               initial_cash=10000, param_grid=HYPERPARAM_GRID):
    """Stages 1 and 2, repeated once per (alpha, n_episodes) combo.

    Train uses the full pinned history of training_tickers (dates set in data_layer).
    Test uses the same window for EACH basket in test_baskets, all disjoint from training, so this measures generalization to unseen stocks. 
    A combo's score is the average excess return over buy-and-hold across all test baskets.
    """
    # check every test stock is one the model never saw in training
    for name, tickers in test_baskets.items():
        overlap = [t for t in tickers if t in training_tickers]
        if overlap:
            raise ValueError(f"Test basket {name} contains training tickers: {overlap}")

    # load and normalize training basket, using its own pooled statistics
    train_data_raw, train_reference = load_training_basket(training_tickers)
    train_data = align_dates(train_data_raw)

    # load and normalize each test basket, using the training basket's reference
    # (each basket is date-aligned on its own, since baskets are tested separately)
    test_data = {}
    for name, tickers in test_baskets.items():
        raw = load_test_basket(tickers, train_reference)
        test_data[name] = align_dates(raw)

    obs_size = 7
    n_actions = 3

    results = []
    best_score = -np.inf
    best_q_net = None
    best_scaler = None
    best_combo = None

    for combo in param_grid:
        print(f"\n=== Trying hyperparameters: alpha={combo['alpha']}, "
              f"n_episodes={combo['n_episodes']} ===")

        # build a fresh agent from a fixed seed, so every combo starts identically
        train_env = TradingEnv(train_data, initial_cash=initial_cash)
        q_net, target_net, optimizer, loss_fn = build_agent(obs_size, n_actions, seed=RANDOM_SEED)

        # the scaler learns the bot's usual confidence spread during training
        scaler = MarginScaler()

        # print the full per-episode training log for this combo
        train(train_env, q_net, target_net, optimizer, loss_fn, n_actions,
              n_episodes=combo["n_episodes"], alpha=combo["alpha"],
              epsilon_decay=DEFAULT_EPSILON_DECAY, verbose=True, scaler=scaler)

        # replace the training-time scaler with one calibrated on a full pass of the
        # finished model over the training period, then it stays frozen with the model
        scaler = calibrate_scaler(TradingEnv(train_data, initial_cash=initial_cash),
                                  q_net, n_actions)
        print(f"  [scaler] calibrated on {scaler.count} confidence gaps from one full pass")

        # score the frozen model on each disjoint, unseen test basket
        # (the scaler is frozen, so running several baskets in a row can't change it)
        basket_scores = {}
        for name, data in test_data.items():
            test_env = TradingEnv(data, initial_cash=initial_cash)
            info = run_stepper(test_env, q_net, n_actions, data,
                                interactive=False, label=f"Test basket {name}", verbose=False,
                                scaler=scaler)
            basket_scores[name] = score_test_run(info, test_env, data)

            r = basket_scores[name]
            print(f"  >> basket {name} {test_env.tickers}: final=${r['final_value']:.2f} | "
                  f"buy-and-hold=${r['benchmark_value']:.2f} | "
                  f"excess={r['excess_return']:+.2%} | sharpe={r['sharpe']:+.3f}")

        # combo score = average excess return over buy-and-hold across the test baskets
        score = float(np.mean([r["excess_return"] for r in basket_scores.values()]))
        avg_final = float(np.mean([r["final_value"] for r in basket_scores.values()]))

        print(f"  >> combo result: avg excess return={score:+.2%} "
              f"across {len(basket_scores)} baskets | avg final portfolio=${avg_final:.2f}")
        results.append({
            **combo,
            "score": score,
            "basket_excess": {n: r["excess_return"] for n, r in basket_scores.items()},
            "avg_final_portfolio_value": avg_final,
        })

        # keep this model if it beats the current best
        if score > best_score:
            best_score = score
            best_q_net = q_net
            best_scaler = scaler
            best_combo = combo

    print(f"\n=== Best hyperparameters: alpha={best_combo['alpha']}, "
          f"n_episodes={best_combo['n_episodes']} (avg excess return={best_score:+.2%}) ===")

    # return train_data too, since simulation draws user portfolios from training tickers
    # best_scaler is part of the model: simulation must size trades with it
    return best_q_net, n_actions, train_data, results, best_scaler


def save_model(path, q_net, scaler, obs_size, n_actions, best_combo):
    """Saves everything needed to redeploy the trained model without
    retraining: the network's weights, its frozen MarginScaler, and the
    winning hyperparams (kept for the record, not required to reload).
    """
    torch.save({
        "q_net_state_dict": q_net.state_dict(),
        "obs_size": obs_size,
        "n_actions": n_actions,
        "scaler_bytes": pickle.dumps(scaler),  # MarginScaler isn't a torch module, so pickle it
        "best_combo": best_combo,
    }, path)
    print(f"Saved model to {path}")


def load_model(path, seed=RANDOM_SEED):
    """Rebuilds the network architecture, then loads the trained weights and
    scaler back onto it. Returns (q_net, n_actions, scaler, best_combo).
    """
    checkpoint = torch.load(path, weights_only=False)

    q_net, _target_net, _optimizer, _loss_fn = build_agent(
        checkpoint["obs_size"], checkpoint["n_actions"], seed=seed
    )
    q_net.load_state_dict(checkpoint["q_net_state_dict"])
    q_net.eval()  # frozen for inference: no more learning or exploration

    scaler = pickle.loads(checkpoint["scaler_bytes"])
    return q_net, checkpoint["n_actions"], scaler, checkpoint["best_combo"]


def simulate_user_portfolio(chosen_tickers, q_net, n_actions, train_data,
                             initial_cash=10000, interactive=True, scaler=None, advisor=None):
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
                        interactive=interactive, label="Simulation (full period)",
                        scaler=scaler, advisor=advisor)


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        # a saved model exists: skip stage 1+2 entirely and go straight to stage 3
        print(f"Loading saved model from {MODEL_PATH} (skipping search + training)...")
        q_net, n_actions, scaler, best_combo = load_model(MODEL_PATH)

        # still need train_data for stage 3, this is just loading/normalizing
        # price data, not training, so it stays cheap
        train_data_raw, _ = load_training_basket(TRAINING_TICKERS)
        train_data = align_dates(train_data_raw)
        print(f"Loaded model trained with alpha={best_combo['alpha']}, "
              f"n_episodes={best_combo['n_episodes']}")
    else:
        # no saved model yet: run the full stage 1+2 search, then cache the winner
        q_net, n_actions, train_data, results, scaler = run_hyperparameter_search()
        best_combo = max(results, key=lambda r: r["score"])
        save_model(MODEL_PATH, q_net, scaler, obs_size=7, n_actions=n_actions,
                   best_combo=best_combo)

    # stage 3: demo a user picking a smaller portfolio from the training basket
    user_choice = ["JPM", "JNJ", "XOM", "NVDA"]  # stand-in for real user input
    # advisor layer: Groq if GROQ_API_KEY is set, built-in templates otherwise
    advisor = Advisor()
    print(f"\nAdvisor: {advisor.describe()}")
    simulate_user_portfolio(user_choice, q_net, n_actions, train_data, scaler=scaler,
                            advisor=advisor)