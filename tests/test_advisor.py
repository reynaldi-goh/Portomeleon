"""Tests for the advisor layer (Iteration 4). No network, no torch needed."""

from types import SimpleNamespace

from src.advisor import (
    Advisor, build_day_facts, signal_notes, template_explanation, SYSTEM_PROMPT,
)


def make_row(rsi=50.0, gap=0.0, macd=0.01, signal=0.0, ret=0.0):
    return {"RSI": rsi, "Close_vs_MA10": gap, "MACD_pct": macd,
            "MACD_Signal_pct": signal, "Ret_1d": ret}


def fake_q(q_net, state):
    return [0.10, 0.50, -0.20]  # HOLD, BUY, SELL


def make_facts():
    tickers = ["AAA", "BBB", "CCC"]
    state = [0, 0, 0, 0, 0, 0.25, 0.40]  # index 5 = allocation, 6 = cash share
    states = {t: state for t in tickers}
    actions = {"AAA": (1, 0.40), "BBB": (0, 0.0), "CCC": (2, 0.90)}
    info_after = {
        "effective_action": {"AAA": 1, "BBB": 0, "CCC": 0},   # CCC wanted SELL but did nothing
        "shares_held": {"AAA": 12, "BBB": 3, "CCC": 0},
    }
    raw_rows = {t: make_row() for t in tickers}
    shares_before = {"AAA": 0, "BBB": 3, "CCC": 0}
    portfolio = {"value": 10000.0, "return_since_start_pct": 0.0,
                 "buy_and_hold_value": 10000.0, "drawdown_pct": 0.0}
    return build_day_facts(5, tickers, states, None, actions, info_after,
                           raw_rows, shares_before, portfolio, q_values_fn=fake_q)


class FakeClient:
    """Mimics client.chat.completions.create(...) without any network."""

    def __init__(self, text=None, error=None):
        self.text, self.error, self.last_kwargs = text, error, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_signal_notes_rsi_zones():
    assert "overbought" in signal_notes(make_row(rsi=75))[0]
    assert "oversold" in signal_notes(make_row(rsi=25))[0]
    assert "neutral" in signal_notes(make_row(rsi=50))[0]


def test_signal_notes_price_vs_average_and_macd():
    notes = signal_notes(make_row(gap=-0.031, macd=-0.01, signal=0.0))
    assert "3.1% below its 10-day average" in notes[1]
    assert "downward momentum" in notes[2]


def test_facts_record_wanted_did_and_share_change():
    facts = make_facts()
    aaa, bbb, ccc = facts["stocks"]
    assert (aaa["wanted"], aaa["did"], aaa["shares_change"]) == ("BUY", "BUY", 12)
    assert (bbb["wanted"], bbb["did"], bbb["shares_change"]) == ("HOLD", "HOLD", 0)
    assert (ccc["wanted"], ccc["did"]) == ("SELL", "HOLD")
    assert aaa["size_pct"] == 40 and bbb["size_pct"] == 0
    assert aaa["preference_scores"] == {"HOLD": 0.1, "BUY": 0.5, "SELL": -0.2}
    assert facts["portfolio_before_trades"]["cash_pct"] == 40.0


def test_missed_sell_explains_why():
    ccc = make_facts()["stocks"][2]
    assert "no shares" in ccc["note"]


def test_missed_buy_explains_why():
    facts = make_facts()
    facts["stocks"][0]["did"] = "HOLD"
    from src.advisor import _mismatch_note
    assert "whole share" in _mismatch_note("BUY", "HOLD", 0)


def test_template_mentions_every_ticker():
    text = template_explanation(make_facts())
    for t in ("AAA", "BBB", "CCC"):
        assert t in text
    assert "bought 12 shares" in text
    assert "wanted to sell but did not" in text


def test_advisor_without_key_uses_template(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    advisor = Advisor()
    assert advisor.client is None
    assert "built-in templates" in advisor.describe()
    assert advisor.explain_day(make_facts()) == template_explanation(make_facts())


def test_advisor_sends_facts_and_returns_llm_text():
    client = FakeClient(text="  AAA was bought. All good.  ")
    advisor = Advisor(client=client, model="test-model")
    out = advisor.explain_day(make_facts())
    assert out == "AAA was bought. All good."
    assert client.last_kwargs["model"] == "test-model"
    assert client.last_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
    assert '"ticker": "AAA"' in client.last_kwargs["messages"][1]["content"]


def test_advisor_falls_back_when_api_fails():
    advisor = Advisor(client=FakeClient(error=RuntimeError("rate limited")))
    out = advisor.explain_day(make_facts())
    assert "AAA" in out                       # template text is still there
    assert "Groq unavailable" in out and "rate limited" in out


def test_advisor_falls_back_on_empty_reply():
    advisor = Advisor(client=FakeClient(text="   "))
    assert "Groq unavailable" in advisor.explain_day(make_facts())