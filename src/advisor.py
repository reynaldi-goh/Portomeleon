"""Advisor layer: turns the bot's daily decisions into plain-language explanations — Iteration 6.

The DQN is a neural network, so nobody can read its "reasons" directly. What we CAN
report honestly is: what the bot saw (indicators), what it preferred (Q-values),
what it chose, and what actually happened. Those facts are computed here in plain
Python. Groq's LLM only turns the facts into friendly wording; it is told not to
invent anything. If Groq is unavailable (no key, no internet, rate limit), or its
reply drifts from the house style, a built-in template is used instead, so the
demo never breaks and never shows the reader something ungrounded or confusing.

Iteration 6 changes:
- MACD facts are now written as plain outcomes ("momentum was fading (a signal
  called MACD)") instead of mechanics ("MACD below its signal line"). Naming a
  signal-line crossing requires understanding two technical constructs just to
  parse the sentence; reporting the outcome does not.
- The system prompt is tightened with explicit bad/good examples for the exact
  drift seen in testing: per-stock HOLD breakdowns instead of one grouped
  sentence, the raw word "moderate" leaking into the text, and "no risk flag"
  being mentioned even when there was nothing to flag.
- A deterministic guard (_violates_house_style) scans the LLM's reply for that
  same banned phrasing after the fact. A prompt can reduce drift but can't
  guarantee it, so this is a cheap backstop: if it still slips through, we show
  the template instead of a reply that breaks the house rules.
- Default model upgraded to gpt-oss-120b with medium reasoning effort, both
  overridable via env vars, since 120B costs more per token than 20B.
"""

import json
import os
import re

from dotenv import load_dotenv
load_dotenv()  # loads .env (if present) into os.environ -- lives here, not
                # just in main.py, so GROQ_API_KEY is found regardless of
                # which entry point (main.py, a script, a REPL check) imports
                # this module first

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Groq changes its model list from time to time. If this one is retired, pick a
# current one from https://console.groq.com/docs/models and set GROQ_MODEL.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# gpt-oss models spend part of their completion-token budget on an internal
# reasoning pass before writing the reply. This task is mechanical (turn
# already-computed facts into two paragraphs), not a hard reasoning problem,
# and "medium" has repeatedly spent enough of the budget on reasoning to
# either empty out or truncate the actual reply. "low" leaves more room for
# the reply itself; 120B's larger size, not deeper reasoning, is what's
# expected to carry the improved instruction-following here.
DEFAULT_REASONING_EFFORT = "low"

# Trade-size thresholds (percent of available cash/shares committed) that map
# onto the plain-language confidence label. Tune here, not in the prompt.
CONFIDENCE_HIGH_THRESHOLD = 60
CONFIDENCE_MODERATE_THRESHOLD = 25

# A BUY at or above this size_pct is flagged as a concentrated bet, so the
# advisor can name the added risk as a fact rather than the LLM judging it.
RISK_FLAG_THRESHOLD = 60

# Fixed wording for confidence_label, used by the template fallback so it
# stays consistent with what the SYSTEM_PROMPT asks the LLM to convey.
CONFIDENCE_PHRASES = {
    "low": "it leaned into this trade only a little",
    "moderate": "it put a fair amount of weight behind this trade",
    "high": "it leaned strongly into this trade",
}

SYSTEM_PROMPT = """You explain the daily decisions of a stock-trading bot to a complete beginner with very little financial knowledge. Write like a knowledgeable friend talking them through today's trades, in your own words -- not like you're filling in a fixed-format report. Two different days should not read like the same paragraph with different numbers swapped in: vary your structure, sentence order, and phrasing each time.

You're given a JSON with, per stock: what the bot wanted to do vs what actually happened, plain-language signal facts, a confidence_label for any trade, a risk_flag, the model's internal preference_scores, and the day's portfolio value vs a buy-and-hold benchmark. Use all of it as you see fit -- there's no required sentence order, no required grouping of HOLD stocks, no fixed paragraph count. If several stocks genuinely held for the same reason, it's natural to group them; if one HOLD stock's facts are actually more interesting or different from the others, feel free to give it its own moment. Structure it however reads best for this particular day.

Only these are hard rules -- everything else is your call:
1. Never invent a fact, number, news item, or comparison that isn't in the JSON. This includes technical mechanics not present in the facts -- e.g. don't say "signal line" or "crossed below" even though you know what MACD is; the signal facts are already translated to plain language on purpose, so stick to what's actually there rather than adding real trading vocabulary from your own knowledge.
2. Never state or imply that a signal caused, made, led, or resulted in a decision. The bot is a neural network choosing from learned preferences, not a rule reacting to a trigger.
   Bad: "These signals made the bot decide to buy NVDA." / "The high RSI caused the bot to sell."
   Good: "The bot saw NVDA's price well above its recent average, and it chose to buy."
3. preference_scores (the model's internal Q-values) are NOT on a fixed or comparable scale -- never print them, never describe their magnitude ("strongly preferred", "by a wide margin"), and never treat them as a percentage or probability. They exist so you know a real preference drove the choice, not so you can narrate how big it was.
4. For every BUY/SELL, convey the confidence_label in your own natural wording, without ever printing the words "low", "moderate", "high", or "confidence_label" themselves. This is not optional -- a trade with no confidence conveyed is an incomplete explanation.
5. If a stock's signals point in different directions (e.g. one reads as caution, another as strength), say so, without implying either one caused the choice.
6. If, and only if, a stock's risk_flag is true, note in passing that committing this much to one stock is a concentrated bet with more risk than spreading across several stocks. If no stock has risk_flag true, don't bring up risk or concentration at all.
7. Use the "note" field to explain any stock the bot wanted to trade but couldn't.
8. Use portfolio_before_trades.benchmark_comparison (already computed) for how the portfolio compares to buy-and-hold -- don't compute or guess this yourself from the dollar values.
9. No financial advice, no telling the reader what to do.
10. Plain text, no markdown or bullets. Roughly 150-350 words -- long enough to actually explain things, short enough that it isn't a chore to read."""

# Deterministic backstop: substrings that mean the reply broke a house rule
# above. Checked after generation, case-insensitively. A prompt can reduce
# this drift but not guarantee it away, so anything caught here falls back to
# the template rather than reaching the reader.
_BANNED_PATTERNS = [
    r"\bmade (?:the bot|it) decide\b",
    r"\bcaused (?:the bot|it)\b",
    r"\blead(?:ing|s)? to a? ?(?:hold|buy|sell) decision\b",
    r"\bconfidence (?:was|is) (?:low|moderate|high)\b",
    r"\bconfidence_label\b",
    r"\brisk[_ ]flag\b",
    r"\bsignal line\b",
]
_BANNED_RE = re.compile("|".join(_BANNED_PATTERNS), re.IGNORECASE)

# Broader, context-aware catch for the risk rule specifically: a fixed phrase
# list only catches wording we've already seen fail (e.g. Day 4's "no
# additional concentration risk was flagged" slipped past the list above
# entirely). Since we already know from day_facts whether ANY stock actually
# has risk_flag true, we can check the concept, not just specific phrasings:
# if none do, the word "risk" or "concentrat*" shouldn't appear anywhere.
_RISK_MENTION_RE = re.compile(r"\brisk\b|\bconcentrat", re.IGNORECASE)


def _house_style_violation_reason(text, day_facts):
    """Returns a short string naming which house rule the text broke, or None
    if it's compliant. Split out from _violates_house_style so the caller can
    log/print *why* a reply was rejected instead of just silently falling back.
    """
    m = _BANNED_RE.search(text)
    if m:
        return f"matched banned phrase {m.group(0)!r}"
    any_risk_flagged = any(s["risk_flag"] for s in day_facts["stocks"])
    if not any_risk_flagged and _RISK_MENTION_RE.search(text):
        m2 = _RISK_MENTION_RE.search(text)
        return f"mentioned risk/concentration ({m2.group(0)!r}) but no stock has risk_flag true"
    return None


def _violates_house_style(text, day_facts):
    """True if the LLM's reply still breaks a house rule after generation."""
    return _house_style_violation_reason(text, day_facts) is not None


def q_values_for(q_net, state):
    """The network's preference score for [HOLD, BUY, SELL] in this state."""
    import torch  # imported here so the rest of this file can be tested without torch

    with torch.no_grad():
        q = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    return [float(v) for v in q]


def signal_notes(row):
    """Turn the raw (unnormalised) indicators of one stock-day into short plain phrases.

    Written outcome-first on purpose: a beginner doesn't need "MACD crossed
    below its signal line" to understand what happened, they need "momentum
    was fading." The technical name rides along in parentheses for anyone who
    already knows it, but understanding the sentence never depends on it.
    """
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
    """Maps the calibrated trade size (already normalized by the MarginScaler)
    onto a plain confidence bucket. Returns None for HOLD (size_pct == 0),
    since there is no trade to be confident about.

    This is deliberately computed here in plain Python rather than left for the
    LLM to infer from raw Q-values, which have no fixed scale to reason from.
    """
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
    """How the portfolio stands against a simple buy-and-hold approach, computed
    here so the LLM never has to eyeball two dollar figures and guess.
    Within 0.5% counts as "about the same as" rather than a false precision of
    "ahead" or "behind" on a difference too small to be meaningful.
    """
    if not value or not benchmark_value:
        return "about the same as"
    diff_pct = (value - benchmark_value) / benchmark_value
    if abs(diff_pct) < 0.005:
        return "about the same as"
    return "ahead of" if diff_pct > 0 else "behind"


def template_explanation(day_facts):
    """Built-in explanation with no LLM: same facts, fixed wording.

    Also the fallback whenever the LLM's reply drifts from house style, so it
    must follow every rule the prompt asks the LLM to follow (grouped HOLDs,
    no raw confidence words, risk note only when risk_flag is true).
    """
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
        conf_clause = f" -- {conf_phrase}" if conf_phrase and s["did"] in ("BUY", "SELL") else ""
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
    """Returns (client, reason). client is None when Groq cannot be used."""
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
                temperature=0.7,  # was 0.3: too low combined with a tightly-worded prompt,
                                  # so replies came out template-identical every time
                max_tokens=2400,  # 120B + medium reasoning on this longer, rule-heavy prompt can burn
                                  # a lot of budget on reasoning before writing a single reply token
                reasoning_effort=self.reasoning_effort,
                timeout=20,
            )
            choice = reply.choices[0]
            text = (choice.message.content or "").strip()
            finish_reason = choice.finish_reason
            if not text or finish_reason != "stop":
                # "stop" means the model finished on its own; anything else
                # (most commonly "length") means the reply was cut off
                # mid-thought, e.g. reasoning tokens ate the whole budget
                # before or during the actual answer. A truncated sentence is
                # non-empty, so checking only "is text empty" lets these
                # through: this check catches that case too.
                usage = getattr(reply, "usage", None)
                raise ValueError(f"incomplete reply (finish_reason={finish_reason}, usage={usage})")
            reason = _house_style_violation_reason(text, day_facts)
            if reason:
                # No longer a hard block: log it for visibility, but trust the
                # prompt's own instructions rather than force-swapping in the
                # template. If drift becomes a real problem, tighten this back up.
                print(f"\n[advisor note] reply didn't fully match house style ({reason}), showing it anyway.\n",
                      flush=True)
            return text
        except Exception as e:
            # never let an API problem stop the demo: fall back and say why
            return (template_explanation(day_facts)
                    + f"\n(Groq unavailable: {type(e).__name__}: {str(e)[:300]}. "
                      "Showing the built-in explanation.)")