# test simulation.py logic in isolation from streamlit/torch
# run: pytest test_simulation.py -v

import json

import pandas as pd
import pytest

import src.simulation as sim


def make_price_df(closes):
    closes = pd.Series(closes, dtype=float)
    ret = closes.pct_change().fillna(0.0)
    return pd.DataFrame({"Close": closes, "Ret_1d": ret})


@pytest.fixture
def two_ticker_data():
    return {
        "AAA": make_price_df([100, 101, 99, 102, 103]),
        "BBB": make_price_df([50, 49, 51, 52, 53]),
    }


@pytest.fixture
def dates():
    return pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )


# stand in for src.trading_env.TradingEnv
class FakeEnv:
    def __init__(self, data, initial_cash):
        self.data = data
        self.initial_cash = initial_cash
        self.n_days = len(next(iter(data.values())))
        self.current_step = 0
        self.peak_value = initial_cash

    def reset(self):
        self.current_step = 0
        info = {
            "portfolio_value": self.initial_cash,
            "cash": self.initial_cash,
            "shares_held": {t: 0 for t in self.data},
        }
        return {"step": 0}, info

    def step(self, actions):
        self.current_step += 1
        value = self.initial_cash + self.current_step * 10.0
        self.peak_value = max(self.peak_value, value)
        terminated = self.current_step >= self.n_days - 1
        info = {
            "portfolio_value": value,
            "cash": value * 0.5,
            "shares_held": {t: self.current_step for t in self.data},
            "effective_action": {t: 1 for t in self.data},
        }
        return {"step": self.current_step}, 0.0, terminated, False, info


def fake_policy(obs_dict):
    return {"AAA": (0, 0.0), "BBB": (0, 0.0)}


# stand in for src.advisor.build_day_facts, so tests don't need real state vectors
def fake_build_day_facts(day, tickers, states, q_net, actions, info_after, raw_rows,
                         shares_before, portfolio, q_values_fn=None):
    return {
        "day": day,
        "stocks": [{"ticker": t, "did": "HOLD", "shares_change": 0} for t in tickers],
    }


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setattr(sim, "TradingEnv", FakeEnv)
    monkeypatch.setattr(sim, "build_day_facts", fake_build_day_facts)


@pytest.fixture
def session(two_ticker_data, dates):
    return sim.SimulationSession(
        list(two_ticker_data.keys()), two_ticker_data, dates,
        policy=fake_policy, initial_cash=1000,
    )


def test_clean_ticker_uppercases_and_strips():
    assert sim.clean_ticker("  aapl ") == "AAPL"


def test_clean_ticker_rejects_bad_input():
    with pytest.raises(ValueError):
        sim.clean_ticker("$$$")


def test_load_user_stocks_rejects_too_few():
    with pytest.raises(ValueError, match="at least"):
        sim.load_user_stocks(["AAA", "BBB"], train_reference=None)


def test_load_user_stocks_reports_missing_tickers(monkeypatch):
    def fake_fetch(ticker):
        if ticker == "BAD":
            raise RuntimeError("no data")
        return make_price_df([1, 2, 3, 4] * 20)

    monkeypatch.setattr(sim, "fetch_and_engineer", fake_fetch)
    monkeypatch.setattr(sim, "normalize_features", lambda df, columns, ref_df: df)

    with pytest.raises(ValueError, match="BAD"):
        sim.load_user_stocks(["AAA", "BBB", "CCC", "BAD"], train_reference=None)


def test_load_user_stocks_keeps_only_shared_dates(monkeypatch):
    idx_a = pd.date_range("2024-01-01", periods=80, freq="D")
    idx_b = idx_a[5:]

    frames = {
        "AAA": pd.DataFrame({"Close": range(80)}, index=idx_a),
        "BBB": pd.DataFrame({"Close": range(75)}, index=idx_b),
        "CCC": pd.DataFrame({"Close": range(80)}, index=idx_a),
        "DDD": pd.DataFrame({"Close": range(80)}, index=idx_a),
    }
    monkeypatch.setattr(sim, "fetch_and_engineer", lambda t: frames[t])
    monkeypatch.setattr(sim, "normalize_features", lambda df, columns, ref_df: df)

    data, common_dates, notes = sim.load_user_stocks(["AAA", "BBB", "CCC", "DDD"], None)

    assert len(common_dates) == 75
    assert all(len(df) == 75 for df in data.values())
    assert any("AAA" in n for n in notes)


def test_buy_and_hold_value_equal_split():
    data = {"AAA": make_price_df([100, 200]), "BBB": make_price_df([50, 50])}
    value = sim.buy_and_hold_value(data, ["AAA", "BBB"], initial_cash=1000, step=1)
    assert value == pytest.approx(1500.0)


def test_metrics_from_history_first_day_has_zero_change():
    m = sim.metrics_from_history([1000.0], [1000.0], initial_cash=1000)
    assert m["day_change"] == 0.0
    assert m["sharpe"] is None
    assert m["max_drawdown"] == 0.0


def test_metrics_from_history_computes_return_and_drawdown():
    values = [1000, 1100, 900, 950]
    bench = [1000, 1000, 1000, 1000]
    m = sim.metrics_from_history(values, bench, initial_cash=1000)
    assert m["total_value"] == 950
    assert m["cumulative_return"] == pytest.approx(-0.05)
    assert m["max_drawdown"] == pytest.approx((1100 - 900) / 1100)


def test_session_records_initial_day_on_construction(session):
    assert session.days_simulated == 0
    assert session.history[0]["portfolio"] == 1000
    assert session.history[0]["actions"] == {"AAA": "HOLD", "BBB": "HOLD"}


def test_session_step_advances_and_records(session):
    session.step()
    assert session.days_simulated == 1
    assert session.history[-1]["portfolio"] == pytest.approx(1010.0)
    assert session.decision_date == session.dates[0]


def test_explain_uses_advisor_when_present(two_ticker_data, dates):
    calls = []

    class StubAdvisor:
        def explain_day(self, day_facts):
            calls.append(day_facts)
            return "advisor says hold"

        def describe(self):
            return "stub"

    s = sim.SimulationSession(
        list(two_ticker_data.keys()), two_ticker_data, dates,
        policy=fake_policy, advisor=StubAdvisor(), initial_cash=1000,
    )
    s.step()
    assert s.cached_explanation() is None
    assert s.explain() == "advisor says hold"
    assert s.explain() == "advisor says hold"
    assert len(calls) == 1  # cached, not re-asked


def test_holdings_shape_matches_tickers(session):
    session.step()
    rows = session.holdings()
    assert {r["Stock"] for r in rows} == {"AAA", "BBB"}


def test_save_and_load_last_simulation_round_trip(tmp_path, session):
    session.step_many(2)
    path = tmp_path / "last_simulation.json"
    sim.save_last_simulation(session, path=str(path))

    loaded = sim.load_last_simulation(path=str(path))
    assert loaded["tickers"] == session.tickers
    assert loaded["days"] == session.days_simulated
    with open(path) as f:
        json.load(f)


def test_load_last_simulation_missing_file_returns_none(tmp_path):
    missing = tmp_path / "nope.json"
    assert sim.load_last_simulation(path=str(missing)) is None