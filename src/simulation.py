"""Simulation session for the UI — Iteration 7.

app.py should stay thin: it draws things and handles clicks. Everything about a
running simulation lives here, so it can be tested without Streamlit or torch:

- clean_ticker / load_user_stocks: turn what the user typed into aligned,
  normalised price data (works for ANY ticker, not just the training basket)
- SimulationSession: one run, advanced day by day, with the advisor attached
- metrics_from_history: Sharpe, drawdown and friends from the value history
- save_last_simulation / load_last_simulation: so the dashboard survives restarts

Timing reminder (same as trading_env): each step, the bot decides at the close
of day k, trades at that close, then the portfolio is revalued at the close of
day k+1. So after a step the screen shows day k+1, and the "latest decisions"
belong to day k.
"""

import json
import os
import re
from datetime import datetime

import numpy as np

from src.advisor import build_day_facts, template_explanation
from src.data_layer import FEATURE_COLUMNS, fetch_and_engineer, normalize_features
from src.trading_env import TradingEnv

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}  # matches main.py's convention

MIN_STOCKS = 4
MAX_STOCKS = 10
MIN_TRADING_DAYS = 60          # fewer shared days than this is too short to simulate
LAST_SIM_PATH = "last_simulation.json"

# a ticker becomes part of a cache file name, so only plain ticker characters are allowed
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


# ---------------------------------------------------------------- loading stocks

def clean_ticker(text):
    """'  aapl ' -> 'AAPL'. Raises ValueError if it can't be a ticker symbol."""
    t = text.strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"'{text.strip()}' doesn't look like a stock ticker "
                         "(letters and numbers only, for example AAPL or BRK-B).")
    return t


def load_user_stocks(tickers, train_reference):
    """Load, engineer and normalise any tickers the user picked.

    Prices are scaled with the TRAINING basket's statistics (train_reference),
    the same scale the model learned on, exactly like the test baskets.

    Returns (data, dates, notes):
      data:  {ticker: dataframe}, aligned so row i is the same date for every stock
      dates: those shared dates, kept here because the aligned frames use plain row numbers
      notes: plain-language warnings for the UI
    Raises ValueError with a friendly message for anything the user can fix.
    """
    tickers = list(dict.fromkeys(tickers))  # drop duplicates, keep order
    if len(tickers) < MIN_STOCKS:
        raise ValueError(f"Pick at least {MIN_STOCKS} stocks (you have {len(tickers)}).")
    if len(tickers) > MAX_STOCKS:
        raise ValueError(f"Pick at most {MAX_STOCKS} stocks (you have {len(tickers)}).")

    raw, missing = {}, []
    for t in tickers:
        try:
            raw[t] = fetch_and_engineer(t)
        except Exception:
            missing.append(t)
    if missing:
        raise ValueError(f"Couldn't get price data for: {', '.join(missing)}. "
                         "Check the spelling, or try again if you are offline.")

    normalized = {
        t: normalize_features(df.copy(), columns=FEATURE_COLUMNS, ref_df=train_reference)
        for t, df in raw.items()
    }

    # keep only the dates every stock has, matched by real date
    common = normalized[tickers[0]].index
    for t in tickers[1:]:
        common = common.intersection(normalized[t].index)
    common = common.sort_values()

    if len(common) < MIN_TRADING_DAYS:
        shortest = min(raw, key=lambda t: len(raw[t]))
        raise ValueError(
            f"Only {len(common)} trading days are shared by all these stocks "
            f"(need at least {MIN_TRADING_DAYS}). {shortest} has the shortest history "
            f"({len(raw[shortest])} days), so try removing it.")

    notes = []
    for t in tickers:
        dropped = len(normalized[t]) - len(common)
        if dropped > 0:
            notes.append(f"{t}: {dropped} day(s) left out because not every stock traded on them.")

    data = {t: normalized[t].loc[common].reset_index(drop=True) for t in tickers}
    return data, list(common), notes


# ---------------------------------------------------------------------- metrics

def buy_and_hold_value(data, tickers, initial_cash, step):
    """Equal split on day 0, never trade again. Same formula as main.py."""
    cash_per_ticker = initial_cash / len(tickers)
    total = 0.0
    for t in tickers:
        shares = cash_per_ticker / data[t].iloc[0]["Close"]
        total += shares * data[t].iloc[step]["Close"]
    return float(total)


def metrics_from_history(values, benchmark_values, initial_cash):
    """Summary numbers from the daily portfolio values (and the buy-and-hold ones).

    Sharpe is annualised (x sqrt(252)) with a risk-free rate of 0, and is None
    until there are at least 5 daily returns to base it on.
    max_drawdown is a positive fraction: 0.076 means "down 7.6% from the peak".
    """
    values = np.asarray(values, dtype=float)
    bench = np.asarray(benchmark_values, dtype=float)

    total = float(values[-1])
    day_change = float(values[-1] / values[-2] - 1) if len(values) > 1 else 0.0

    peaks = np.maximum.accumulate(values)
    max_drawdown = float(((peaks - values) / peaks).max())

    returns = np.diff(values) / values[:-1]
    sharpe = None
    if len(returns) >= 5:
        std = returns.std()
        sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.0

    return {
        "total_value": total,
        "cumulative_return": total / initial_cash - 1,
        "day_change": day_change,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "buy_and_hold_value": float(bench[-1]),
        "vs_buy_and_hold": total / float(bench[-1]) - 1,
    }


def _iso(d):
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


# ---------------------------------------------------------------------- session

class SimulationSession:
    """One simulation run, advanced a day at a time.

    policy:       function(obs_dict) -> {ticker: (action, size_fraction)}. In the app
                  this wraps the frozen model, so this class never touches torch.
    q_net:        only passed through to the advisor's fact builder.
    advisor:      an Advisor, or None to use the built-in template wording.
    q_values_fn:  optional override for how facts read the network (tests use this).
    """

    def __init__(self, tickers, data, dates, policy, q_net=None, advisor=None,
                 q_values_fn=None, initial_cash=10000):
        self.tickers = list(tickers)
        self.data = data
        self.dates = list(dates)
        self.policy = policy
        self.q_net = q_net
        self.advisor = advisor
        self.q_values_fn = q_values_fn
        self.initial_cash = initial_cash

        self.env = TradingEnv(data, initial_cash=initial_cash)
        self.obs, self.info = self.env.reset()
        self.done = False

        self.day_facts = None       # facts about the most recent step's decisions
        self.decision_date = None   # the date those decisions were made at
        self.explanations = {}      # {decision day: text}, so a page rerun never re-asks the LLM
        self.history = []           # one row per day shown, for the chart and the metrics
        self._record()

    # -- state --------------------------------------------------------------

    @property
    def current_date(self):
        return self.dates[self.env.current_step]

    @property
    def days_simulated(self):
        return len(self.history) - 1

    @property
    def total_days(self):
        return len(self.dates) - 1

    def _record(self, effective_action=None):
        step = self.env.current_step
        # effective_action is info["effective_action"] from the env.step() that led to this
        # day (raw 0/1/2 ints); None for the initial day 0 row, before any step has run.
        if effective_action is None:
            actions = {t: "HOLD" for t in self.tickers}
        else:
            actions = {t: ACTION_NAMES[effective_action[t]] for t in self.tickers}
        self.history.append({
            "date": self.dates[step],
            "portfolio": float(self.info["portfolio_value"]),
            "buy_and_hold": buy_and_hold_value(self.data, self.tickers, self.initial_cash, step),
            "actions": actions,
        })

    # -- moving forward -----------------------------------------------------

    def step(self):
        """Advance one trading day. Does nothing once the data has run out."""
        if self.done:
            return
        env = self.env
        day = env.current_step
        value = self.info["portfolio_value"]

        # remember what the bot saw before it acts, so the advisor can explain it afterwards
        states_seen = self.obs
        raw_rows = {t: self.data[t].iloc[day] for t in self.tickers}
        shares_before = dict(self.info["shares_held"])
        portfolio_note = {
            "value": round(value, 2),
            "return_since_start_pct": round((value / self.initial_cash - 1) * 100, 2),
            "buy_and_hold_value": round(self.history[-1]["buy_and_hold"], 2),
            "drawdown_pct": round((env.peak_value - value) / env.peak_value * 100, 2)
                            if env.peak_value > 0 else 0.0,
        }

        actions = self.policy(self.obs)
        self.obs, _reward, terminated, truncated, self.info = env.step(actions)
        self.done = terminated or truncated

        self.decision_date = self.dates[day]
        self.day_facts = build_day_facts(day, self.tickers, states_seen, self.q_net, actions,
                                         self.info, raw_rows, shares_before, portfolio_note,
                                         q_values_fn=self.q_values_fn)
        self._record(effective_action=self.info.get("effective_action"))

    def step_many(self, n):
        for _ in range(n):
            if self.done:
                break
            self.step()

    # -- the advisor --------------------------------------------------------

    def cached_explanation(self):
        """The explanation for the latest decisions if it was already asked for, else None."""
        if self.day_facts is None:
            return None
        return self.explanations.get(self.day_facts["day"])

    def explain(self):
        """Explain the latest decisions. Asks the advisor once per day, then reuses the text."""
        if self.day_facts is None:
            return None
        day = self.day_facts["day"]
        if day not in self.explanations:
            if self.advisor is None:
                self.explanations[day] = template_explanation(self.day_facts)
            else:
                self.explanations[day] = self.advisor.explain_day(self.day_facts)
        return self.explanations[day]

    # -- numbers for the screen ---------------------------------------------

    def metrics(self):
        m = metrics_from_history([h["portfolio"] for h in self.history],
                                 [h["buy_and_hold"] for h in self.history],
                                 self.initial_cash)
        m["wallet"] = float(self.info["cash"])
        return m

    def holdings(self):
        """One row per stock: price and move today, shares, value, and the bot's latest action."""
        step = self.env.current_step
        latest = {s["ticker"]: s for s in self.day_facts["stocks"]} if self.day_facts else {}
        rows = []
        for t in self.tickers:
            row = self.data[t].iloc[step]
            shares = int(self.info["shares_held"][t])
            price = float(row["Close"])
            if t in latest:
                s = latest[t]
                did = s["did"] if s["did"] == "HOLD" else f"{s['did']} {abs(s['shares_change'])}"
            else:
                did = "-"
            rows.append({"Stock": t, "Price": price, "Day change": float(row["Ret_1d"]),
                         "Shares": shares, "Value": shares * price,
                         "Last action": did, "Action taken": did})
        return rows

    def chart_history(self):
        return {
            "dates": [_iso(h["date"]) for h in self.history],
            "portfolio": [h["portfolio"] for h in self.history],
            "buy_and_hold": [h["buy_and_hold"] for h in self.history],
            "actions": [h["actions"] for h in self.history],
        }

    def snapshot(self):
        """Everything the dashboard needs to show this run later."""
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "tickers": self.tickers,
            "initial_cash": self.initial_cash,
            "start_date": _iso(self.history[0]["date"]),
            "end_date": _iso(self.history[-1]["date"]),
            "days": self.days_simulated,
            "metrics": self.metrics(),
            "history": self.chart_history(),
            "holdings": self.holdings(),
        }


# ------------------------------------------------------------------ persistence

def save_last_simulation(session, path=LAST_SIM_PATH):
    with open(path, "w") as f:
        json.dump(session.snapshot(), f, indent=1, default=float)
    return path


def load_last_simulation(path=LAST_SIM_PATH):
    """The last saved run, or None if there isn't one (or the file is unreadable)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None