# Iteration 6 - run the streamlit ui
# streamlit run app.py

import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from main import MODEL_PATH, TRAINING_TICKERS, load_model
from src.advisor import Advisor
from src.agent import select_portfolio_actions
from src.data_layer import END_DATE, START_DATE, load_training_basket
from src.simulation import (
    MAX_STOCKS, MIN_STOCKS, SimulationSession, clean_ticker,
    load_last_simulation, load_user_stocks, save_last_simulation,
)

COMMON_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "AVGO", "LLY",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "ABBV",
    "MRK", "CVX", "KO", "PEP", "ADBE", "CRM", "BAC", "WMT", "DIS", "NFLX",
]
STEP_BUTTON_DAYS = 5

st.set_page_config(page_title="Portomeleon", layout="centered")


@st.cache_resource(show_spinner="Loading the trained model...")
def load_resources():
    # load model once per app start
    q_net, n_actions, scaler, _best_combo = load_model(MODEL_PATH)
    # load reference basket for scaling
    _, train_reference = load_training_basket(TRAINING_TICKERS)

    def policy(obs_dict):
        return select_portfolio_actions(obs_dict, q_net, epsilon=0.0,
                                        n_actions=n_actions, scaler=scaler)

    return {"q_net": q_net, "policy": policy, "reference": train_reference, "advisor": Advisor()}


# switch page
def go(page):
    st.session_state.page = page
    st.session_state.setup_msg = None
    st.rerun()


# format a dollar amount
def money(x):
    return f"${x:,.2f}"


# format a drawdown percentage
def drawdown_text(x):
    return "0.0%" if x < 0.0005 else f"-{x:.1%}"


# format a sharpe ratio
def sharpe_text(x):
    return "n/a (too few days)" if x is None else f"{x:.2f}"


# format a date
def fmt_date(d):
    return d.strftime("%d %b %Y") if hasattr(d, "strftime") else str(d)


# format holdings rows for display
def holdings_table(rows):
    return pd.DataFrame([{
        "Stock": r["Stock"],
        "Price": money(r["Price"]),
        "Day change": f"{r['Day change']:+.1%}",
        "Shares": r["Shares"],
        "Value": money(r["Value"]),
        "Last action": r.get("Last action", r.get("Action taken", "-")),
        "Action taken": r.get("Action taken", r.get("Last action", "-")),
    } for r in rows])


# build a chart-ready frame from history
def history_frame(history):
    actions = history.get("actions", [{}] * len(history["dates"]))
    return pd.DataFrame({
        "Date": pd.to_datetime(history["dates"]),
        "Bot portfolio": history["portfolio"],
        "Buy and hold": history["buy_and_hold"],
        # flag buy/sell days
        "Buy day": [any(a == "BUY" for a in day.values()) for day in actions],
        "Sell day": [any(a == "SELL" for a in day.values()) for day in actions],
    })


# draw the portfolio-vs-benchmark chart
def history_chart(df):
    base = alt.Chart(df).encode(x=alt.X("Date:T", title=None))
    lines = base.transform_fold(
        ["Bot portfolio", "Buy and hold"], as_=["Series", "Value"]
    ).mark_line().encode(
        y=alt.Y("Value:Q", title="Value ($)"),
        color=alt.Color("Series:N", title=None),
    )
    buys = base.transform_filter("datum['Buy day']").mark_point(
        shape="triangle-up", size=100, color="green", filled=True,
    ).encode(y=alt.Y("Bot portfolio:Q"), tooltip=[alt.Tooltip("Date:T", title="Buy day")])
    sells = base.transform_filter("datum['Sell day']").mark_point(
        shape="triangle-down", size=100, color="red", filled=True,
    ).encode(y=alt.Y("Bot portfolio:Q"), tooltip=[alt.Tooltip("Date:T", title="Sell day")])
    return (lines + buys + sells).properties(height=320).interactive()


# escape dollar signs before rendering text
def show_text(text):
    st.write(text.replace("$", "\\$"))


# add tickers typed or clicked by the user
def add_tickers(text):
    picked = st.session_state.picked
    for part in [p for p in re.split(r"[,\s]+", text.strip()) if p]:
        try:
            t = clean_ticker(part)
        except ValueError as e:
            st.session_state.setup_msg = str(e)
            return
        if t in picked:
            continue
        if len(picked) >= MAX_STOCKS:
            st.session_state.setup_msg = f"You can pick at most {MAX_STOCKS} stocks."
            return
        picked.append(t)
    st.session_state.setup_msg = None


# handle the add-ticker button
def add_typed():
    add_tickers(st.session_state.typed_tickers)


# handle the dropdown selection
def add_from_dropdown():
    add_tickers(" ".join(st.session_state.common_tickers_select))


# remove a picked ticker
def remove_ticker(t):
    if t in st.session_state.picked:
        st.session_state.picked.remove(t)
    st.session_state.setup_msg = None

# render the new-simulation setup page
def page_setup(res):
    if st.button("← Back"):
        go("dashboard")
    st.header("New simulation")
    st.caption(f"The bot will trade {START_DATE} to {END_DATE} (fixed for now). "
               f"Pick {MIN_STOCKS} to {MAX_STOCKS} stocks.")

    if st.session_state.setup_msg:
        st.warning(st.session_state.setup_msg)

    st.multiselect("Add from common tickers", COMMON_TICKERS,
                   key="common_tickers_select", on_change=add_from_dropdown,
                   placeholder="Choose one or more...")

    with st.form("add_stock", clear_on_submit=True):
        st.text_input("Or type any other ticker", key="typed_tickers",
                      placeholder="e.g. RIVN, BRK-B")
        st.form_submit_button("Add", on_click=add_typed)

    picked = st.session_state.picked
    st.subheader(f"Holdings ({len(picked)})")
    if not picked:
        st.caption(f"No stocks yet. Add at least {MIN_STOCKS}.")
    for t in picked:
        c1, c2 = st.columns([4, 1])
        seen = "the bot trained on this stock" if t in TRAINING_TICKERS else "new to the bot"
        c1.write(f"**{t}**  ({seen})")
        c2.button("Remove", key=f"rm_{t}", on_click=remove_ticker, args=(t,))

    capital = st.number_input("Starting capital ($)", min_value=1000, max_value=10_000_000,
                              value=10000, step=1000)

    st.caption("The bot was trained on 8 large US stocks, so results on very different "
               "stocks (penny stocks, crypto-like swings) are less reliable.")

    if st.button("Start simulation", type="primary", disabled=len(picked) < MIN_STOCKS):
        try:
            with st.spinner("Preparing price data (the first time for a stock it is downloaded)..."):
                data, dates, notes = load_user_stocks(picked, res["reference"])
        except ValueError as e:
            st.error(str(e))
            return
        st.session_state.sim = SimulationSession(
            picked, data, dates, policy=res["policy"], q_net=res["q_net"],
            advisor=res["advisor"], initial_cash=capital)
        st.session_state.sim_notes = notes
        go("simulation")


# render the running-simulation page
def page_simulation():
    sim = st.session_state.sim
    if sim is None:
        go("dashboard")

    st.header("Simulation")
    st.subheader(fmt_date(sim.current_date))
    st.caption(f"Day {sim.days_simulated} of {sim.total_days}. Each step, the bot decides at "
               "one day's close and the portfolio is revalued at the next day's close.")
    for note in st.session_state.sim_notes:
        st.caption(note)

    st.altair_chart(history_chart(history_frame(sim.chart_history())), width="stretch")
    st.caption("Green triangle = bot bought that day. Red triangle = bot sold that day.")
    st.dataframe(holdings_table(sim.holdings()), hide_index=True)

    m = sim.metrics()
    row1 = st.columns(3)
    row1[0].metric("Wallet", money(m["wallet"]))
    row1[1].metric("Total value", money(m["total_value"]))
    row1[2].metric("Performance (today)", f"{m['day_change']:+.2%}")
    row2 = st.columns(3)
    row2[0].metric("Sharpe ratio", sharpe_text(m["sharpe"]))
    row2[1].metric("Cumulative return", f"{m['cumulative_return']:+.1%}")
    row2[2].metric("Max drawdown", drawdown_text(m["max_drawdown"]))
    side = "ahead of" if m["vs_buy_and_hold"] >= 0 else "behind"
    st.caption(f"Buy-and-hold would be worth {money(m['buy_and_hold_value'])}. "
               f"The bot is {abs(m['vs_buy_and_hold']):.1%} {side} it.")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Step 1 day", type="primary", disabled=sim.done):
        sim.step()
        st.rerun()
    if b2.button(f"Step {STEP_BUTTON_DAYS} days", disabled=sim.done):
        sim.step_many(STEP_BUTTON_DAYS)
        st.rerun()
    if b3.button("Run to end", disabled=sim.done):
        with st.spinner("Running the rest of the simulation..."):
            sim.step_many(sim.total_days)
        st.rerun()
    if b4.button("End simulation"):
        if sim.days_simulated > 0:
            save_last_simulation(sim)
        st.session_state.sim = None
        go("dashboard")
    if sim.done:
        st.info("Reached the last day of data. Click 'End simulation' to save this run.")

    if sim.day_facts is not None:
        st.subheader("AI advisor")
        st.caption(f"Explaining the bot's decisions at the close of {fmt_date(sim.decision_date)}.")
        text = sim.cached_explanation()
        if text is None:
            if st.button("Explain these decisions"):
                with st.spinner("Asking the advisor..."):
                    sim.explain()
                st.rerun()
        else:
            show_text(text)
        if sim.advisor is not None:
            st.caption(f"Advisor: {sim.advisor.describe()}")


# render the dashboard page
def page_dashboard():
    top = st.columns([3, 1])
    top[0].header("Dashboard")
    if top[1].button("New simulation", type="primary"):
        go("setup")

    last = load_last_simulation()
    if last is None:
        st.info("No simulation yet. Click 'New simulation' to let the bot manage a portfolio.")
        return

    m = last["metrics"]
    st.subheader("Last simulation")
    st.caption(f"{', '.join(last['tickers'])} | {last['start_date']} to {last['end_date']} "
               f"({last['days']} trading days)")
    left, right = st.columns([3, 1])
    left.altair_chart(history_chart(history_frame(last["history"])), width="stretch")
    right.metric("Total value", money(m["total_value"]))
    right.metric("Performance", f"{m['cumulative_return']:+.1%}")

    st.subheader("AI metrics")
    a, b, c = st.columns(3)
    a.metric("Sharpe ratio", sharpe_text(m["sharpe"]))
    b.metric("Max drawdown", drawdown_text(m["max_drawdown"]))
    c.metric("vs buy-and-hold", f"{m['vs_buy_and_hold']:+.1%}")

    st.subheader("Holdings at the end")
    st.dataframe(holdings_table(last["holdings"]), hide_index=True)


def main():
    st.title("Portomeleon")

    # init session state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
        st.session_state.sim = None
        st.session_state.sim_notes = []
        st.session_state.picked = []
        st.session_state.setup_msg = None

    if not os.path.exists(MODEL_PATH):
        st.error(f"No trained model found at '{MODEL_PATH}'. Run `python main.py` once to "
                 "train and save one, then start the app from the project root folder.")
        st.stop()
    res = load_resources()

    # route to the current page
    page = st.session_state.page
    if page == "setup":
        page_setup(res)
    elif page == "simulation":
        page_simulation()
    else:
        page_dashboard()

    st.divider()
    st.caption("Disclaimer: Educational project only")

main()