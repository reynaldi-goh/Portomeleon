"""Multi-stock trading environment — Iteration 2."""

import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    """Gymnasium trading environment supporting multiple stocks.

    Each trading day, the agent evaluates every stock independently and
    produces an action plus a confidence-based size fraction for each.
    Sells execute independently per stock. Buys are pooled and the
    available cash is split across stocks proportionally to their
    relative confidence (size fraction).

    Timing of one step (day k):
      1. The agent sees day k's close and decides.
      2. Trades execute at day k's close.
      3. The clock moves to day k+1 and the portfolio is revalued at
         day k+1's close.
      4. Reward = what the position chosen on day k earned between
         close k and close k+1.
    So portfolio_value always matches the price of self.current_step.
    """

    def __init__(self, price_data, initial_cash=10000):
        super().__init__()
        # price_data: dict of {ticker: dataframe}, all aligned to the same trading dates
        self.price_data = price_data
        self.tickers = list(price_data.keys())
        self.initial_cash = initial_cash

        # 3 actions per stock: hold, buy, sell
        self.action_space = gym.spaces.Discrete(3)

        # observation per stock: 5 percentage-style features + 2 portfolio-context features
        # (daily return, price vs 10-day average, RSI, MACD %, MACD signal %, allocation_pct, cash_pct)
        self.n_features = 7
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        # assumes all tickers share the same number of trading days
        self.n_steps = len(price_data[self.tickers[0]])

        # reward shaping parameters
        self.sharpe_window = 20          # rolling window size for sharpe-style reward
        self.sharpe_min_history = 5      # minimum days before sharpe kicks in
        self.drawdown_penalty_weight = 0.5  # how harshly drawdown is punished
        self.idle_penalty_weight = 0.01     # opportunity cost for sitting in cash

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.initial_cash
        self.effective_action = {ticker: 0 for ticker in self.tickers}

        # reward-shaping state, reset each episode
        self.return_history = []
        self.peak_value = self.initial_cash

        return self._get_obs(), self._get_info()

    def _get_price(self, ticker, step=None):
        step = self.current_step if step is None else step
        return self.price_data[ticker].iloc[step]["Close"]

    def _get_obs(self):
        # build one observation vector per ticker
        obs = {}
        for ticker in self.tickers:
            row = self.price_data[ticker].iloc[self.current_step]
            stock_value = self.shares_held[ticker] * self._get_price(ticker)

            obs[ticker] = np.array([
                row["Ret_1d_norm"],
                row["Close_vs_MA10_norm"],
                row["RSI_norm"],
                row["MACD_pct_norm"],
                row["MACD_Signal_pct_norm"],
                stock_value / self.portfolio_value if self.portfolio_value > 0 else 0.0,  # allocation_pct
                self.cash / self.portfolio_value if self.portfolio_value > 0 else 0.0,     # cash_pct
            ], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "effective_action": dict(self.effective_action),
            "cash": self.cash,
            "shares_held": dict(self.shares_held),
        }

    def _compute_reward(self, prev_value):
        """Rolling Sharpe-style reward, penalized for drawdown from the episode's peak,
        with a small opportunity-cost penalty for sitting in cash — otherwise the agent
        can dodge the drawdown penalty for free by simply not being in the market.

        prev_value: portfolio value right after today's trades (at today's close).
        self.portfolio_value: value at the next day's close. The difference is
        the return earned by the position chosen today.
        """
        raw_return = (self.portfolio_value - prev_value) / prev_value
        self.return_history.append(raw_return)

        # rolling sharpe component: mean return / std return, over a sliding window
        window = self.return_history[-self.sharpe_window:]
        if len(window) < self.sharpe_min_history:
            # not enough history yet, fall back to raw return
            sharpe_component = raw_return
        else:
            mean_return = np.mean(window)
            std_return = max(np.std(window), 0.005)  # floor at realistic daily vol, not near-zero
            sharpe_component = np.clip(mean_return / std_return, -3.0, 3.0)  # cap outliers

        # drawdown penalty: how far below the episode's peak we currently are
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # opportunity cost: being in cash isn't free just because it dodges drawdown
        cash_fraction = self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0
        idle_penalty = self.idle_penalty_weight * cash_fraction

        return sharpe_component - self.drawdown_penalty_weight * drawdown - idle_penalty

    def step(self, actions):
        # today's close: the price the agent just observed, and the price it trades at
        today_prices = {t: self._get_price(t) for t in self.tickers}

        buy_tickers = [t for t, (a, _) in actions.items() if a == 1]
        sell_tickers = [t for t, (a, _) in actions.items() if a == 2]
        hold_tickers = [t for t, (a, _) in actions.items() if a == 0]

        # mark every hold-action ticker as HOLD
        for ticker in hold_tickers:
            self.effective_action[ticker] = 0

        # full-exit threshold: below whole-share granularity, a partial sell and
        # a full sell are functionally the same, so treat a confident sell as
        # "get me fully out" rather than silently rounding to zero shares
        FULL_EXIT_THRESHOLD = 0.5

        for ticker in sell_tickers:
            _, size_fraction = actions[ticker]
            price = today_prices[ticker]
            held = self.shares_held[ticker]

            # exit fully when the agent is confident, so a rounding-to-zero
            # sell never gets silently swallowed and misread as a HOLD
            if size_fraction >= FULL_EXIT_THRESHOLD:
                shares_to_sell = held
            else:
                shares_to_sell = int(held * size_fraction)

            if shares_to_sell > 0:
                self.cash += shares_to_sell * price
                self.shares_held[ticker] -= shares_to_sell
                self.effective_action[ticker] = 2
            else:
                self.effective_action[ticker] = 0

        if buy_tickers:
            size_fractions = {t: actions[t][1] for t in buy_tickers}
            total_size = sum(size_fractions.values())
            cash_pool = min(total_size, 1.0) * self.cash

            for ticker in buy_tickers:
                weight = size_fractions[ticker] / total_size if total_size > 0 else 1 / len(buy_tickers)
                cash_for_stock = cash_pool * weight
                price = today_prices[ticker]
                # round buys down only; never spend more than the allocated cash
                shares_to_buy = int(cash_for_stock // price)
                if shares_to_buy > 0:
                    self.cash -= shares_to_buy * price
                    self.shares_held[ticker] += shares_to_buy
                    self.effective_action[ticker] = 1
                else:
                    self.effective_action[ticker] = 0

        # portfolio value right after today's trades, still at today's prices
        # (a trade swaps cash for shares at the same price, so nothing is gained or lost yet)
        value_after_trades = self.cash + sum(
            self.shares_held[t] * today_prices[t] for t in self.tickers
        )

        # move to the next day and revalue everything at the NEW day's close
        self.current_step += 1
        next_prices = {t: self._get_price(t) for t in self.tickers}
        self.portfolio_value = self.cash + sum(
            self.shares_held[t] * next_prices[t] for t in self.tickers
        )

        # per-stock reward: the price move from today's close to the next close,
        # only for stocks actually held after today's trades
        stock_rewards = {}
        for t in self.tickers:
            if self.shares_held[t] > 0:
                stock_rewards[t] = (next_prices[t] - today_prices[t]) / today_prices[t]
            else:
                stock_rewards[t] = 0.0

        # compute the shared portfolio-level reward (rolling sharpe minus drawdown minus idle cost)
        reward = self._compute_reward(value_after_trades)

        terminated = False
        truncated = self.current_step >= self.n_steps - 1

        info = self._get_info()
        info["stock_rewards"] = stock_rewards

        return self._get_obs(), reward, terminated, truncated, info