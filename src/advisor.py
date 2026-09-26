"""Advisor layer: turns the bot's daily decisions into plain-language explanations: Iteration 5.

The DQN can't explain itself, so the facts here (what the bot saw, preferred,
chose, and what happened) are computed in plain Python. An LLM (Groq) only
phrases them for a beginner, never invents anything. If Groq is unavailable or
its reply is incomplete, a built-in template is used instead.
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()  # load here too (not just main.py), so GROQ_API_KEY is found
                # regardless of which entry point imports this module first

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

DEFAULT_MODEL = "openai/gpt-oss-120b"

# gpt-oss spends part of its token budget on internal reasoning before
# replying; "medium" often left too little room for the actual reply, so
# default to "low" and lean on 120B's size for quality instead
DEFAULT_REASONING_EFFORT = "low"

# trade-size thresholds (percent of cash/shares committed) mapped to a
# plain-language confidence label
CONFIDENCE_HIGH_THRESHOLD = 60
CONFIDENCE_MODERATE_THRESHOLD = 25

# a BUY at or above this size_pct is flagged as a concentrated bet
RISK_FLAG_THRESHOLD = 60

# wording for confidence_label, kept in sync with what SYSTEM_PROMPT asks the LLM to convey
CONFIDENCE_PHRASES = {
    "low": "it leaned into this trade only a little",
    "moderate": "it put a fair amount of weight behind this trade",
    "high": "it leaned strongly into this trade",
}

SYSTEM_PROMPT = """You explain the daily decisions of a stock-trading bot to a complete beginner with very little financial knowledge. Write like a knowledgeable friend talking them through today's trades, in your own words , not like you're filling in a fixed-format report. Two different days should not read like the same paragraph with different numbers swapped in: vary your structure, sentence order, and phrasing each time.

You're given a JSON with, per stock: what the bot wanted to do vs what actually happened, plain-language signal facts, a confidence_label for any trade, a risk_flag, the model's internal preference_scores, and the day's portfolio value vs a buy-and-hold benchmark. Use all of it as you see fit , there's no required sentence order, no required grouping of HOLD stocks, no fixed paragraph count. If several stocks genuinely held for the same reason, it's natural to group them; if one HOLD stock's facts are actually more interesting or different from the others, feel free to give it its own moment. Structure it however reads best for this particular day.

Only these are hard rules , everything else is your call:
1. Never invent a fact, number, news item, or comparison that isn't in the JSON. This includes technical mechanics not present in the facts , e.g. don't say "signal line" or "crossed below" even though you know what MACD is; the signal facts are already translated to plain language on purpose, so stick to what's actually there rather than adding real trading vocabulary from your own knowledge.
2. Never state or imply that a signal caused, made, led, or resulted in a decision. The bot is a neural network choosing from learned preferences, not a rule reacting to a trigger.
   Bad: "These signals made the bot decide to buy NVDA." / "The high RSI caused the bot to sell."
   Good: "The bot saw NVDA's price well above its recent average, and it chose to buy."
3. preference_scores (the model's internal Q-values) are NOT on a fixed or comparable scale , never print them, never describe their magnitude ("strongly preferred", "by a wide margin"), and never treat them as a percentage or probability. They exist so you know a real preference drove the choice, not so you can narrate how big it was.
4. For every BUY/SELL, convey the confidence_label in your own natural wording, without ever printing the words "low", "moderate", "high", or "confidence_label" themselves. This is not optional , a trade with no confidence conveyed is an incomplete explanation.
5. If a stock's signals point in different directions (e.g. one reads as caution, another as strength), say so, without implying either one caused the choice.
6. If, and only if, a stock's risk_flag is true, note in passing that committing this much to one stock is a concentrated bet with more risk than spreading across several stocks. If no stock has risk_flag true, don't bring up risk or concentration at all.
7. Use the "note" field to explain any stock the bot wanted to trade but couldn't.
8. Use portfolio_before_trades.benchmark_comparison (already computed) for how the portfolio compares to buy-and-hold , don't compute or guess this yourself from the dollar values.
9. No financial advice, no telling the reader what to do. Do not just describe, explain why the bot makes certain decision.
10. Plain text, no markdown or bullets. Roughly 150-350 words , long enough to actually explain things, short enough that it isn't a chore to read."""

def q_values_for(q_net, state):
    """The network's preference score for [HOLD, BUY, SELL] in this state."""
    import torch  # imported here so the rest of this file can be tested without torch

    with torch.no_grad():
        q = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    return [float(v) for v in q]


def signal_notes(row):
    """Turn one stock-day's raw indicators into short, outcome-first plain phrases
    (e.g. "momentum was fading" rather than "MACD crossed below its signal line")."""
    notes = []

    rsi = float(row["RSI"])
    if rsi >= 70:
        notes.append("this stock's price has climbed quickly over the past couple of weeks, "
                      "a pattern sometimes called overbought (a reading called RSI)")
    elif rsi <= 30:
        notes.append("this stock's price has dropped quickly over the past couple of weeks, "
                      "a pattern sometimes called oversold (a reading called RSI)")
    else:
        notes.append("this stock's recent price swings have been fairly ordinary, "
                      "not stretched in either direction (a reading called RSI)")

    gap = float(row["Close_vs_MA10"])
    side = "above" if gap >= 0 else "below"
    notes.append(f"price {abs(gap):.1%} {side} its 10-day average")

    if float(row["MACD_pct"]) > float(row["MACD_Signal_pct"]):
        notes.append("this stock's short-term momentum was picking up (a signal called MACD)")
    else:
        notes.append("this stock's short-term momentum was fading (a signal called MACD)")

    notes.append(f"that day's price move {float(row['Ret_1d']):+.1%}")
    return notes


def _mismatch_note(wanted, did, shares_before):
    """Why the bot's wish did not become a trade (only when wanted != did)."""
    if wanted == did:
        return ""
    if wanted == "BUY":
        return "its share of the cash was too small to buy even one whole share"
    if wanted == "SELL":
        if shares_before == 0:
            return "it held no shares of this stock, so there was nothing to sell"
        return "the planned partial sale rounded down to zero whole shares"
    return ""


def confidence_label(size_pct):
    """Map a calibrated trade size onto a plain confidence bucket, computed here
    in plain Python since raw Q-values have no fixed scale for an LLM to reason from.
    Returns None for HOLD (size_pct == 0)."""
    if size_pct <= 0:
        return None
    if size_pct >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if size_pct >= CONFIDENCE_MODERATE_THRESHOLD:
        return "moderate"
    return "low"


def build_day_facts(day, tickers, states, q_net, actions, info_after, raw_rows,
                    shares_before, portfolio, q_values_fn=None):
    """Collect everything about one day's decisions into one plain dict.

    states:        {ticker: state vector the bot saw BEFORE deciding}
    actions:       {ticker: (action, size_fraction)} the bot chose
    info_after:    env info after the step (effective_action, shares_held)
    raw_rows:      {ticker: that day's raw indicator row (unnormalised)}
    shares_before: {ticker: shares held before today's trades}
    portfolio:     dict of portfolio-level numbers from the caller
    """
    q_fn = q_values_fn or q_values_for
    stocks = []
    for t in tickers:
        state = states[t]
        action, size = actions[t]
        wanted = ACTION_NAMES[action]
        did = ACTION_NAMES[info_after["effective_action"][t]]
        q = q_fn(q_net, state)
        size_pct = round(size * 100) if wanted != "HOLD" else 0
        stocks.append({
            "ticker": t,
            "wanted": wanted,
            "did": did,
            "size_pct": size_pct,
            "confidence_label": confidence_label(size_pct),
            "risk_flag": did == "BUY" and size_pct >= RISK_FLAG_THRESHOLD,
            "preference_scores": {"HOLD": round(q[0], 4), "BUY": round(q[1], 4), "SELL": round(q[2], 4)},
            "signals": signal_notes(raw_rows[t]),
            "shares_change": info_after["shares_held"][t] - shares_before[t],
            "holding_pct_of_portfolio": round(float(state[5]) * 100, 1),
            "note": _mismatch_note(wanted, did, shares_before[t]),
        })

    portfolio = dict(portfolio)
    portfolio["cash_pct"] = round(float(states[tickers[0]][6]) * 100, 1)
    portfolio["benchmark_comparison"] = _benchmark_comparison(
        portfolio.get("value"), portfolio.get("buy_and_hold_value")
    )
    return {"day": day, "portfolio_before_trades": portfolio, "stocks": stocks}


def _benchmark_comparison(value, benchmark_value):
    """Compare portfolio value to buy-and-hold here, so the LLM never has to
    eyeball two dollar figures. Within 0.5% counts as "about the same as"
    rather than false precision on a difference too small to be meaningful."""
    if not value or not benchmark_value:
        return "about the same as"
    diff_pct = (value - benchmark_value) / benchmark_value
    if abs(diff_pct) < 0.005:
        return "about the same as"
    return "ahead of" if diff_pct > 0 else "behind"


def template_explanation(day_facts):
    """Built-in explanation with no LLM: same facts, fixed wording. Also the
    fallback whenever the LLM's reply drifts from house style, so it follows
    every rule SYSTEM_PROMPT asks the LLM to follow."""
    hold_stocks = [s["ticker"] for s in day_facts["stocks"] if s["did"] == "HOLD" and s["wanted"] == "HOLD"]
    para1 = []
    if hold_stocks:
        para1.append(f"{', '.join(hold_stocks)} showed no strong signal either way, so the bot held them unchanged.")

    risk_note = ""
    for s in day_facts["stocks"]:
        if s["did"] == "HOLD" and s["wanted"] == "HOLD":
            continue
        n = abs(s["shares_change"])
        if s["did"] == "BUY":
            action_text = f"bought {n} share{'s' if n != 1 else ''}"
        elif s["did"] == "SELL":
            action_text = f"sold {n} share{'s' if n != 1 else ''}"
        else:
            action_text = f"wanted to {s['wanted'].lower()} but did not, because {s['note']}"
        conf_phrase = CONFIDENCE_PHRASES.get(s.get("confidence_label"))
        conf_clause = f" - {conf_phrase}" if conf_phrase and s["did"] in ("BUY", "SELL") else ""
        para1.append(f"For {s['ticker']}, it {action_text}{conf_clause}. The bot saw: {'; '.join(s['signals'])}.")
        if s["risk_flag"]:
            risk_note = (f" This commits a large share of the portfolio to {s['ticker']}, "
                         "a more concentrated bet that carries more risk than spreading money across several stocks.")

    p = day_facts["portfolio_before_trades"]
    para2 = (f"The portfolio is currently worth ${p.get('value', 0):,.2f}, "
             f"a {p.get('return_since_start_pct', 0):+.1f}% return since the start, "
             f"which is {p.get('benchmark_comparison', 'about the same as')} "
             f"a simple buy-and-hold approach worth ${p.get('buy_and_hold_value', 0):,.2f}."
             f"{risk_note}")

    return " ".join(para1) + "\n\n" + para2


def _make_groq_client():
    """Return (client, reason). client is None when Groq cannot be used."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None, "no GROQ_API_KEY set"
    try:
        from groq import Groq
    except ImportError:
        return None, "groq package not installed (pip install groq)"
    return Groq(api_key=key), None


class Advisor:
    """Explains a day's decisions. Uses Groq when it can, the template otherwise.

    client: optional, any object with client.chat.completions.create(...). Only
    passed in by tests, so no real API call is needed.
    """

    def __init__(self, use_llm=True, model=None, reasoning_effort=None, client=None):
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "GROQ_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        )
        self.client = client
        self.reason = None
        if client is None:
            if use_llm:
                self.client, self.reason = _make_groq_client()
            else:
                self.reason = "LLM turned off"

    def describe(self):
        if self.client is not None:
            return f"Groq ({self.model}, reasoning_effort={self.reasoning_effort})"
        return f"built-in templates ({self.reason})"

    def explain_day(self, day_facts):
        if self.client is None:
            return template_explanation(day_facts)

        try:
            reply = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        "Explain today's decisions to a beginner.\n\nFACTS (JSON):\n"
                        + json.dumps(day_facts, indent=1, default=float)},
                ],
                temperature=0.7,  # 0.3 made replies template-identical; this adds variation
                max_tokens=2400,  # generous: 120B + reasoning can burn much of the budget before replying
                reasoning_effort=self.reasoning_effort,
                timeout=20,
            )
            choice = reply.choices[0]
            text = (choice.message.content or "").strip()
            finish_reason = choice.finish_reason
            if not text or finish_reason != "stop":
                # "stop" means it finished normally; anything else (usually "length")
                # means the reply was cut off mid-thought, so treat it as incomplete
                usage = getattr(reply, "usage", None)
                raise ValueError(f"incomplete reply (finish_reason={finish_reason}, usage={usage})")
            return text
        except Exception as e:
            # never let an API problem stop the demo: fall back and say why
            return (template_explanation(day_facts)
                    + f"\n(Groq unavailable: {type(e).__name__}: {str(e)[:300]}. "
                      "Showing the built-in explanation.)")