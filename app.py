import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Bitvavo Market Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Bitvavo Market Scanner")
st.caption(
    "Alle actieve EUR-assets • Live Bitvavo-data • "
    "Technische ranking • Geen automatische orders"
)

BASE_URL = "https://api.bitvavo.com/v2"


# =========================================================
# DATA
# =========================================================

@st.cache_data(ttl=900)
def get_markets():

    response = requests.get(
        f"{BASE_URL}/markets",
        timeout=15
    )

    response.raise_for_status()

    markets = response.json()

    eur_markets = []

    for market in markets:

        if (
            market.get("quote") == "EUR"
            and market.get("status") == "trading"
        ):
            eur_markets.append({
                "market": market["market"],
                "asset": market["base"]
            })

    return eur_markets


@st.cache_data(ttl=300)
def get_ticker_24h():

    response = requests.get(
        f"{BASE_URL}/ticker/24h",
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    return {
        item["market"]: item
        for item in data
    }


@st.cache_data(ttl=300)
def get_candles(
    market,
    interval="1h",
    limit=250
):

    response = requests.get(
        f"{BASE_URL}/{market}/candles",
        params={
            "interval": interval,
            "limit": limit
        },
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        return pd.DataFrame()

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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    return (
        100
        - (
            100
            / (1 + rs)
        )
    )


def add_indicators(df):

    df = df.copy()

    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    df["RSI"] = calculate_rsi(
        df["close"]
    )

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["VOL_AVG20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    return df


# =========================================================
# SCORE
# =========================================================

def classify_score(score):

    if score >= 75:
        return "🟢 Sterk interessant"

    if score >= 65:
        return "🟢 Interessant"

    if score >= 45:
        return "🟡 Afwachten"

    if score >= 35:
        return "🟠 Voorzichtig"

    return "🔴 Ongunstig"


def analyze_timeframe(df):

    if len(df) < 50:
        return None

    df = add_indicators(df)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    score = 50
    reasons = []

    # ------------------------------------------------
    # TREND
    # ------------------------------------------------

    if (
        latest["close"]
        > latest["EMA20"]
        > latest["EMA50"]
    ):

        score += 15
        reasons.append(
            "Koers en korte EMA's wijzen omhoog"
        )

    elif (
        latest["close"]
        < latest["EMA20"]
        < latest["EMA50"]
    ):

        score -= 15
        reasons.append(
            "Koers en korte EMA's wijzen omlaag"
        )

    else:

        reasons.append(
            "Korte trend is gemengd"
        )

    # ------------------------------------------------
    # LONGER TREND
    # ------------------------------------------------

    if len(df) >= 200:

        if (
            latest["close"]
            > latest["EMA200"]
        ):

            score += 8
            reasons.append(
                "Koers boven EMA200"
            )

        else:

            score -= 8
            reasons.append(
                "Koers onder EMA200"
            )

    # ------------------------------------------------
    # RSI
    # ------------------------------------------------

    rsi = latest["RSI"]

    if pd.notna(rsi):

        if rsi < 25:

            score += 3
            reasons.append(
                "RSI zeer laag / oversold"
            )

        elif 25 <= rsi < 40:

            score += 7
            reasons.append(
                "RSI laat herstelruimte zien"
            )

        elif 40 <= rsi <= 60:

            score += 5
            reasons.append(
                "RSI neutraal en gezond"
            )

        elif 60 < rsi <= 70:

            score += 3
            reasons.append(
                "Positief momentum"
            )

        elif rsi > 70:

            score -= 8
            reasons.append(
                "RSI mogelijk overbought"
            )

    # ------------------------------------------------
    # MACD
    # ------------------------------------------------

    if (
        latest["MACD"]
        > latest["MACD_SIGNAL"]
        and
        previous["MACD"]
        <= previous["MACD_SIGNAL"]
    ):

        score += 12
        reasons.append(
            "Bullish MACD crossover"
        )

    elif (
        latest["MACD"]
        < latest["MACD_SIGNAL"]
        and
        previous["MACD"]
        >= previous["MACD_SIGNAL"]
    ):

        score -= 12
        reasons.append(
            "Bearish MACD crossover"
        )

    elif (
        latest["MACD"]
        > latest["MACD_SIGNAL"]
    ):

        score += 5
        reasons.append(
            "MACD positief"
        )

    else:

        score -= 5
        reasons.append(
            "MACD negatief"
        )

    # ------------------------------------------------
    # VOLUME
    # ------------------------------------------------

    if (
        pd.notna(
            latest["VOL_AVG20"]
        )
        and
        latest["VOL_AVG20"] > 0
    ):

        volume_ratio = (
            latest["volume"]
            / latest["VOL_AVG20"]
        )

        if volume_ratio >= 1.5:

            score += 8
            reasons.append(
                "Sterk bovengemiddeld volume"
            )

        elif volume_ratio >= 1.15:

            score += 4
            reasons.append(
                "Volume boven gemiddeld"
            )

        elif volume_ratio < 0.60:

            score -= 3
            reasons.append(
                "Laag handelsvolume"
            )

    score = int(
        max(
            0,
            min(
                100,
                round(score)
            )
        )
    )

    return {
        "score": score,
        "rating": classify_score(score),
        "rsi": rsi,
        "reasons": reasons,
        "df": df
    }


# =========================================================
# MARKET SCANNER
# =========================================================

markets = get_markets()
ticker_data = get_ticker_24h()

st.write(
    f"**{len(markets)} actieve EUR-assets gevonden op Bitvavo**"
)

progress = st.progress(0)

scanner_rows = []

total = len(markets)

for index, item in enumerate(markets):

    market = item["market"]
    asset = item["asset"]

    try:

        candles = get_candles(
            market,
            "1h",
            250
        )

        analysis = analyze_timeframe(
            candles
        )

        ticker = ticker_data.get(
            market,
            {}
        )

        if analysis is None:
            continue

        last_price = float(
            ticker.get(
                "last",
                candles.iloc[-1]["close"]
            )
        )

        open_24h = float(
            ticker.get(
                "open",
                last_price
            )
        )

        if open_24h > 0:

            change_24h = (
                (
                    last_price
                    - open_24h
                )
                / open_24h
                * 100
            )

        else:

            change_24h = 0

        volume_eur = float(
            ticker.get(
                "volumeQuote",
                0
            )
        )

        scanner_rows.append({
            "Asset": asset,
            "Market": market,
            "Koers": last_price,
            "24u %": round(
                change_24h,
                2
            ),
            "24u volume €": round(
                volume_eur,
                0
            ),
            "RSI": round(
                analysis["rsi"],
                1
            )
            if pd.notna(
                analysis["rsi"]
            )
            else None,
            "Score": analysis["score"],
            "Beoordeling":
                analysis["rating"]
        })

    except Exception:
        pass

    progress.progress(
        min(
            (index + 1) / total,
            1.0
        )
    )

progress.empty()


scanner_df = pd.DataFrame(
    scanner_rows
)

scanner_df = scanner_df.sort_values(
    [
        "Score",
        "24u volume €"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(
    drop=True
)

scanner_df.insert(
    0,
    "#",
    range(
        1,
        len(scanner_df) + 1
    )
)


# =========================================================
# TOP OPPORTUNITIES
# =========================================================

st.divider()

st.subheader(
    "🏆 Hoogste scores"
)

top = scanner_df.head(5)

top_cols = st.columns(5)

for col, (_, row) in zip(
    top_cols,
    top.iterrows()
):

    with col:

        st.metric(
            row["Asset"],
            f"{row['Score']}/100",
            f"{row['24u %']:+.2f}%"
        )

        st.write(
            row["Beoordeling"]
        )


# =========================================================
# FILTERS
# =========================================================

st.divider()

st.subheader(
    "🔎 Alle Bitvavo-assets"
)

search = st.text_input(
    "Zoek asset",
    placeholder="Bijvoorbeeld BTC, SOL, LINK..."
)

minimum_volume = st.number_input(
    "Minimaal 24u handelsvolume in €",
    min_value=0,
    value=0,
    step=10000
)

filtered_df = scanner_df.copy()

if search:

    filtered_df = filtered_df[
        filtered_df[
            "Asset"
        ].str.contains(
            search.upper(),
            case=False,
            na=False
        )
    ]

if minimum_volume > 0:

    filtered_df = filtered_df[
        filtered_df[
            "24u volume €"
        ]
        >= minimum_volume
    ]


st.dataframe(
    filtered_df[
        [
            "#",
            "Asset",
            "Koers",
            "24u %",
            "24u volume €",
            "RSI",
            "Score",
            "Beoordeling"
        ]
    ],
    hide_index=True,
    use_container_width=True,
    height=600
)


# =========================================================
# DETAILED COIN ANALYSIS
# =========================================================

st.divider()

st.subheader(
    "📊 Uitgebreide analyse"
)

available_assets = (
    scanner_df["Asset"]
    .tolist()
)

selected_asset = st.selectbox(
    "Selecteer een asset",
    available_assets
)

selected_row = scanner_df[
    scanner_df["Asset"]
    == selected_asset
].iloc[0]

selected_market = (
    selected_row["Market"]
)

timeframes = {
    "1 uur": "1h",
    "4 uur": "4h",
    "1 dag": "1d"
}

detailed_results = {}

for label, interval in (
    timeframes.items()
):

    try:

        df = get_candles(
            selected_market,
            interval,
            250
        )

        detailed_results[
            label
        ] = analyze_timeframe(
            df
        )

    except Exception:

        detailed_results[
            label
        ] = None


valid_results = {
    key: value
    for key, value
    in detailed_results.items()
    if value is not None
}

if valid_results:

    detailed_score = round(
        sum(
            result["score"]
            for result
            in valid_results.values()
        )
        / len(
            valid_results
        )
    )

    detailed_rating = (
        classify_score(
            detailed_score
        )
    )

    positive_timeframes = sum(
        result["score"] >= 65
        for result
        in valid_results.values()
    )

    negative_timeframes = sum(
        result["score"] < 45
        for result
        in valid_results.values()
    )

    if (
        positive_timeframes
        == len(valid_results)
        or
        negative_timeframes
        == len(valid_results)
    ):

        confidence = "HOOG"

    elif (
        positive_timeframes >= 2
        or
        negative_timeframes >= 2
    ):

        confidence = "GEMIDDELD"

    else:

        confidence = "LAAG"


    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Asset",
        selected_asset
    )

    c2.metric(
        "Multi-timeframe score",
        f"{detailed_score}/100"
    )

    c3.metric(
        "Beoordeling",
        detailed_rating
    )

    c4.metric(
        "Confidence",
        confidence
    )


    # ---------------------------------------------
    # TIMEFRAME TABLE
    # ---------------------------------------------

    timeframe_rows = []

    for label, result in (
        valid_results.items()
    ):

        timeframe_rows.append({
            "Timeframe": label,
            "Score": result["score"],
            "RSI": round(
                result["rsi"],
                1
            )
            if pd.notna(
                result["rsi"]
            )
            else None,
            "Beoordeling":
                result["rating"]
        })

    st.write(
        "### Timeframes"
    )

    st.dataframe(
        pd.DataFrame(
            timeframe_rows
        ),
        hide_index=True,
        use_container_width=True
    )


    # ---------------------------------------------
    # CHART
    # ---------------------------------------------

    if (
        detailed_results[
            "1 uur"
        ]
        is not None
    ):

        chart_df = (
            detailed_results[
                "1 uur"
            ]["df"]
            .set_index(
                "timestamp"
            )
        )

        st.write(
            "### Koers & EMA-trend"
        )

        st.line_chart(
            chart_df[
                [
                    "close",
                    "EMA20",
                    "EMA50"
                ]
            ],
            use_container_width=True
        )


    # ---------------------------------------------
    # WHY
    # ---------------------------------------------

    st.write(
        "### Waarom deze beoordeling?"
    )

    tabs = st.tabs(
        list(
            timeframes.keys()
        )
    )

    for tab, label in zip(
        tabs,
        timeframes.keys()
    ):

        with tab:

            result = (
                detailed_results[
                    label
                ]
            )

            if result is None:

                st.write(
                    "Onvoldoende data."
                )

            else:

                st.write(
                    f"**{result['rating']} "
                    f"— {result['score']}/100**"
                )

                for reason in (
                    result["reasons"]
                ):

                    st.write(
                        f"• {reason}"
                    )


st.divider()

st.caption(
)
