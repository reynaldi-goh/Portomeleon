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
    """

    def __init__(self, price_data, initial_cash=10000):
        super().__init__()
        # price_data: dict of {ticker: dataframe}, all aligned to the same trading dates
        self.price_data = price_data
        self.tickers = list(price_data.keys())
        self.initial_cash = initial_cash

        # 3 actions per stock: hold, buy, sell
        self.action_space = gym.spaces.Discrete(3)

        # observation per stock: 5 normalized features + 2 portfolio-context features
        # (close_norm, MA10_norm, RSI_norm, MACD_norm, MACD_Signal_norm, allocation_pct, cash_pct)
        self.n_features = 7
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        # assumes all tickers share the same number of trading days
        self.n_steps = len(price_data[self.tickers[0]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        # shares held per ticker, starts at 0 for everything
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.initial_cash
        # what actually happened per ticker (may differ from what the agent wanted)
        self.effective_action = {ticker: 0 for ticker in self.tickers}

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
                row["Close_norm"],
                row["MA10_norm"],
                row["RSI_norm"],
                row["MACD_norm"],
                row["MACD_Signal_norm"],
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

    def step(self, actions):
        """
        actions: dict of {ticker: (action, size_fraction)}
            action: 0 = hold, 1 = buy, 2 = sell
            size_fraction: confidence-based fraction already computed by the agent
                           (e.g. via margin_to_size), between 0 and 1
        """
        prev_value = self.portfolio_value
        # lock in today's prices before anything moves the day forward
        today_prices = {t: self._get_price(t) for t in self.tickers}

        buy_tickers = [t for t, (a, _) in actions.items() if a == 1]
        sell_tickers = [t for t, (a, _) in actions.items() if a == 2]
        hold_tickers = [t for t, (a, _) in actions.items() if a == 0]

        for ticker in hold_tickers:
            self.effective_action[ticker] = 0

        # sells execute independently, same logic as the single-stock version
        for ticker in sell_tickers:
            _, size_fraction = actions[ticker]
            price = today_prices[ticker]
            shares_to_sell = int(self.shares_held[ticker] * size_fraction)
            if shares_to_sell > 0:
                self.cash += shares_to_sell * price
                self.shares_held[ticker] -= shares_to_sell
                self.effective_action[ticker] = 2
            else:
                self.effective_action[ticker] = 0  # less than 1 share, counts as hold

        # buys are pooled and split proportionally by relative confidence
        if buy_tickers:
            size_fractions = {t: actions[t][1] for t in buy_tickers}
            total_size = sum(size_fractions.values())

            # cash pool: combined confidence across all buy signals, capped at available cash
            cash_pool = min(total_size, 1.0) * self.cash

            for ticker in buy_tickers:
                weight = size_fractions[ticker] / total_size if total_size > 0 else 1 / len(buy_tickers)
                cash_for_stock = cash_pool * weight
                price = today_prices[ticker]
                shares_to_buy = int(cash_for_stock // price)
                if shares_to_buy > 0:
                    self.cash -= shares_to_buy * price
                    self.shares_held[ticker] += shares_to_buy
                    self.effective_action[ticker] = 1
                else:
                    self.effective_action[ticker] = 0  # not enough cash for even 1 share

        # advance one day for the whole portfolio
        self.current_step += 1

        # portfolio value uses today's prices, since that's when the trades happened
        self.portfolio_value = self.cash + sum(
            self.shares_held[t] * today_prices[t] for t in self.tickers
        )

        reward = (self.portfolio_value - prev_value) / prev_value

        terminated = False
        truncated = self.current_step >= self.n_steps - 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()