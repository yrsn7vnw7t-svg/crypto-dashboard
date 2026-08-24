import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Dashboard")
st.caption("Marktdata via Bitvavo • Geen echte orders • Alleen analyse en paper trading")

COINS = {
    "Bitcoin": "BTC-EUR",
    "Ethereum": "ETH-EUR",
    "Solana": "SOL-EUR",
    "XRP": "XRP-EUR",
    "Cardano": "ADA-EUR"
}


def get_candles(market, interval="1h", limit=200):
    url = f"https://api.bitvavo.com/v2/{market}/candles"
    params = {
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def analyze_coin(df):
    df = df.copy()

    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["close"])

    latest = df.iloc[-1]

    current_price = latest["close"]
    previous_24h = df.iloc[-25]["close"] if len(df) >= 25 else df.iloc[0]["close"]

    change_24h = ((current_price - previous_24h) / previous_24h) * 100

    score = 50
    reasons = []

    if latest["MA20"] > latest["MA50"]:
        score += 15
        reasons.append("korte trend ligt boven lange trend")
    else:
        score -= 15
        reasons.append("korte trend ligt onder lange trend")

    if latest["close"] > latest["MA20"]:
        score += 10
        reasons.append("koers ligt boven het 20-uurs gemiddelde")
    else:
        score -= 10
        reasons.append("koers ligt onder het 20-uurs gemiddelde")

    rsi = latest["RSI"]

    if pd.notna(rsi):
        if rsi < 30:
            score += 15
            reasons.append("RSI wijst op mogelijke oversold situatie")
        elif rsi > 70:
            score -= 15
            reasons.append("RSI wijst op mogelijke overbought situatie")
        elif 45 <= rsi <= 60:
            score += 5
            reasons.append("RSI is relatief gezond")
        else:
            reasons.append("RSI is neutraal")

    recent_volume = df["volume"].tail(5).mean()
    average_volume = df["volume"].tail(50).mean()

    if recent_volume > average_volume * 1.25:
        score += 10
        reasons.append("recent handelsvolume ligt duidelijk hoger")
    elif recent_volume < average_volume * 0.75:
        score -= 5
        reasons.append("recent handelsvolume is relatief laag")

    score = int(max(0, min(100, score)))

    if score >= 70:
        signal = "BUY WATCH"
    elif score <= 35:
        signal = "SELL WATCH"
    else:
        signal = "WAIT"

    return {
        "price": current_price,
        "change_24h": change_24h,
        "rsi": rsi,
        "score": score,
        "signal": signal,
        "reasons": reasons,
        "df": df
    }


results = {}

for name, market in COINS.items():
    try:
        candles = get_candles(market)
        results[name] = analyze_coin(candles)
    except Exception as e:
        results[name] = {
            "error": str(e)
        }


st.subheader("Marktoverzicht")

cols = st.columns(len(COINS))

for col, (name, market) in zip(cols, COINS.items()):
    result = results[name]

    with col:
        if "error" in result:
            st.error(f"{name}\nData niet beschikbaar")
        else:
            st.metric(
                label=name,
                value=f"€ {result['price']:,.4f}",
                delta=f"{result['change_24h']:.2f}%"
            )

            st.write(f"Score: **{result['score']}/100**")
            st.write(f"Signaal: **{result['signal']}**")


st.divider()

st.subheader("Scanner")

scanner_rows = []

for name, result in results.items():
    if "error" not in result:
        scanner_rows.append({
            "Coin": name,
            "Prijs": result["price"],
            "24u %": result["change_24h"],
            "RSI": result["rsi"],
            "Score": result["score"],
            "Signaal": result["signal"]
        })

scanner_df = pd.DataFrame(scanner_rows)

st.dataframe(
    scanner_df,
    use_container_width=True,
    hide_index=True
)


st.divider()

st.subheader("Coin analyse")

selected_coin = st.selectbox(
    "Kies een coin",
    list(COINS.keys())
)

selected_result = results[selected_coin]

if "error" in selected_result:
    st.error("Kon geen data ophalen voor deze coin.")
else:
    df = selected_result["df"]

    left, right = st.columns([2, 1])

    with left:
        chart_df = df.set_index("timestamp")[["close", "MA20", "MA50"]]

        st.line_chart(
            chart_df,
            use_container_width=True
        )

    with right:
        st.metric(
            "Huidige prijs",
            f"€ {selected_result['price']:,.4f}"
        )

        st.metric(
            "24 uur",
            f"{selected_result['change_24h']:.2f}%"
        )

        st.metric(
            "RSI",
            f"{selected_result['rsi']:.1f}"
            if pd.notna(selected_result["rsi"])
            else "n.v.t."
        )

        st.metric(
            "Signaalscore",
            f"{selected_result['score']}/100"
        )

        st.write(f"### {selected_result['signal']}")


    st.write("### Waarom dit signaal?")

    for reason in selected_result["reasons"]:
        st.write(f"• {reason}")


st.divider()

st.subheader("Paper trading")

if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = 1000.0

if "paper_positions" not in st.session_state:
    st.session_state.paper_positions = {}

if "paper_history" not in st.session_state:
    st.session_state.paper_history = []

paper_col1, paper_col2 = st.columns(2)

with paper_col1:
    st.metric(
        "Virtueel cash",
        f"€ {st.session_state.paper_cash:,.2f}"
    )

with paper_col2:
    total_position_value = 0

    for coin, amount in st.session_state.paper_positions.items():
        if coin in results and "error" not in results[coin]:
            total_position_value += amount * results[coin]["price"]

    total_portfolio = st.session_state.paper_cash + total_position_value

    st.metric(
        "Totale paper portfolio",
        f"€ {total_portfolio:,.2f}",
        delta=f"{total_portfolio - 1000:.2f} EUR"
    )


paper_coin = st.selectbox(
    "Coin voor paper trade",
    list(COINS.keys()),
    key="paper_coin"
)

paper_amount = st.number_input(
    "Bedrag in euro",
    min_value=10.0,
    max_value=1000.0,
    value=100.0,
    step=10.0
)

buy_col, sell_col = st.columns(2)

with buy_col:
    if st.button("Virtueel kopen", use_container_width=True):

        price = results[paper_coin]["price"]

        if paper_amount <= st.session_state.paper_cash:
            units = paper_amount / price

            current_units = st.session_state.paper_positions.get(
                paper_coin,
                0
            )

            st.session_state.paper_positions[paper_coin] = (
                current_units + units
            )

            st.session_state.paper_cash -= paper_amount

            st.session_state.paper_history.append({
                "Tijd": datetime.now(),
                "Actie": "BUY",
                "Coin": paper_coin,
                "Bedrag": paper_amount,
                "Prijs": price
            })

            st.success("Virtuele aankoop uitgevoerd.")
        else:
            st.warning("Niet genoeg virtueel cash.")


with sell_col:
    if st.button("Alles virtueel verkopen", use_container_width=True):

        units = st.session_state.paper_positions.get(
            paper_coin,
            0
        )

        if units > 0:
            price = results[paper_coin]["price"]

            value = units * price

            st.session_state.paper_cash += value
            st.session_state.paper_positions[paper_coin] = 0

            st.session_state.paper_history.append({
                "Tijd": datetime.now(),
                "Actie": "SELL",
                "Coin": paper_coin,
                "Bedrag": value,
                "Prijs": price
            })

            st.success("Virtuele positie verkocht.")
        else:
            st.warning("Je hebt geen virtuele positie in deze coin.")


if st.session_state.paper_positions:

    st.write("### Virtuele posities")

    position_rows = []

    for coin, units in st.session_state.paper_positions.items():

        if units > 0 and coin in results:

            current_price = results[coin]["price"]

            position_rows.append({
                "Coin": coin,
                "Aantal": units,
                "Huidige prijs": current_price,
                "Waarde": units * current_price
            })

    if position_rows:
        st.dataframe(
            pd.DataFrame(position_rows),
            use_container_width=True,
            hide_index=True
        )


if st.session_state.paper_history:

    st.write("### Trade history")

    st.dataframe(
        pd.DataFrame(st.session_state.paper_history),
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "Deze signalen zijn experimenteel en geen financieel advies. "
    "We gebruiken eerst paper trading om te testen of de strategie werkt."
)
