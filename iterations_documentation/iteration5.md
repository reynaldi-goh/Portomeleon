# Iteration 5: Advisor Layer

## Goal

The goal of this iteration was to build the advisor layer described in the project design, the piece that turns the DQN's daily trading decisions into plain-language explanations a beginner can actually read. A neural network's internal preferences aren't something that can be read out as a "reason," so this iteration needed a way to explain decisions honestly without pretending the model reasons in words it never actually used.

## What Was Built

The advisor layer is implemented in `src/advisor.py`.

- `signal_notes()` turns one stock-day's raw indicators into short, outcome-first phrases, "momentum was fading" rather than "MACD crossed below its signal line", so understanding the sentence never depends on already knowing what RSI or MACD mean.
- `confidence_label()` maps a calibrated trade size onto a plain low, moderate, or high bucket.
- `_mismatch_note()` explains, in plain language, why a wanted trade didn't happen, for example a planned sell that rounded down to zero whole shares.
- `build_day_facts()` collects everything about one day, per-stock signals, what was wanted versus what actually happened, confidence, a risk flag, the model's raw Q-values, and portfolio-level numbers, into one plain dictionary. This is the only place any of these facts are computed.
- `_benchmark_comparison()` compares the portfolio's value to the buy-and-hold baseline from Iteration 4, so the wording ("ahead of", "behind", "about the same as") is decided here rather than left for an LLM to eyeball two dollar figures.
- `template_explanation()` turns a day's facts into a fixed-wording explanation with no LLM involved at all.
- The `Advisor` class ties it together. `explain_day()` sends the day's facts to Groq with a system prompt that lists the house rules (never invent a fact, never imply a signal caused a decision, never print raw Q-values, always convey confidence in words, mention risk only when actually flagged), and falls back to `template_explanation()` whenever Groq is unavailable (no key, package missing, or the call raises) or its reply is incomplete (empty, or cut off before finishing, both checked against the response's `finish_reason`).

## Design Decisions

Facts computed in plain Python, wording left to the LLM: everything the explanation could possibly state as fact, the indicators, what was chosen, whether it executed, the benchmark comparison, is computed deterministically in `build_day_facts()` before Groq is ever called. Groq's only job is to phrase already-true facts for a beginner, and the system prompt explicitly forbids inventing anything not in that dictionary. This keeps a hallucinated number or an invented reason from ever reaching the reader, regardless of how the LLM phrases things.

Outcome-first signal language instead of technical terms: `signal_notes()` deliberately describes what an indicator means before naming it, "price has climbed quickly, a pattern sometimes called overbought (a reading called RSI)", rather than leading with the indicator's name. A beginner reading the explanation shouldn't need to already know what RSI or MACD are to follow along.

Confidence computed here, not inferred by the LLM from Q-values: the model's raw Q-values have no fixed or comparable scale, the same margin can mean very different things depending on the model, so asking an LLM to eyeball them and describe "how confident" the bot was would be guessing. `confidence_label()` instead maps the already-calibrated trade size (from the `MarginScaler` in Iteration 3) onto a fixed low, moderate, or high bucket, and the system prompt forbids the LLM from printing or reasoning about the raw Q-values directly.

Fallback only for genuine failure, not for wording drift: `explain_day()` falls back to the template when Groq can't be reached, errors, or returns an incomplete reply. An earlier version of this file also scanned a completed reply afterward for phrasing that broke a house rule and force-replaced it with the template if so. That check was removed, the system prompt's own rules were reliable enough in practice, and the extra layer added complexity for a case that wasn't actually protecting against real failures the way the completeness check does.

Testing against a fake client instead of the real API: `Advisor` accepts any object exposing `client.chat.completions.create(...)`, the same shape Groq's SDK exposes, so tests can pass in a stand-in that returns a fixed reply, an error, or a specific `finish_reason` without ever making a network call. This keeps the test suite fast, deterministic, and free to run without a real API key.

## Testing

Unit tests were written in `tests/test_advisor.py` using pytest, covering the fact-building logic and the advisor's fallback behavior. None of them touch torch or the network.

- `test_signal_notes_rsi_zones` and `test_signal_notes_price_vs_average_and_macd` confirm the plain-language phrasing for overbought, oversold, and ordinary RSI zones, and for both directions of the price-vs-average and MACD comparisons.
- `test_facts_record_wanted_did_and_share_change` confirms `build_day_facts()` correctly records what was wanted versus what happened, the resulting share change, trade size, raw preference scores, and the portfolio's cash percentage.
- `test_missed_sell_explains_why` and `test_missed_buy_explains_why` confirm `_mismatch_note()` gives the right plain-language reason for each way a wanted trade can fail to execute.
- `test_template_mentions_every_ticker` confirms the no-LLM template covers every ticker, including one that wanted to sell but couldn't.
- `test_advisor_without_key_uses_template` confirms `Advisor` falls back to the template automatically when no Groq API key is set.
- `test_advisor_sends_facts_and_returns_llm_text` confirms the day's facts are actually sent to the client as JSON, and a clean reply is returned as-is.
- `test_advisor_falls_back_when_api_fails`, `test_advisor_falls_back_on_empty_reply`, and `test_advisor_falls_back_on_truncated_reply` confirm the template fallback fires correctly for an API error, an empty reply, and a reply cut off before finishing.

All 11 tests currently pass.