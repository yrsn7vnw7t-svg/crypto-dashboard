import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alert_scanner import (
    scan_market,
    get_candles,
    analyze_timeframe,
    get_recent_news,
    classify_news,
    build_forecast,
    get_market_context,
    classify_score,
)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Crypto Market Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Market Scanner")

st.caption(
    "Bitvavo market scanner • Timing • Nieuwscontext • 72h trend forecast"
)


# =========================================================
# HELPERS
# =========================================================

def fmt_pct(value):
    if value is None or pd.isna(value):
        return "n.v.t."

    return f"{value:+.1f}%"


def fmt_price(value):
    if value is None:
        return "n.v.t."

    if value >= 1000:
        return f"€ {value:,.2f}"

    if value >= 1:
        return f"€ {value:,.4f}"

    return f"€ {value:,.8f}"


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_market_scan():
    return scan_market()


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_market_context():
    return get_market_context()


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_news(asset):
    articles = get_recent_news(
        asset,
        max_records=5
    )

    classification = classify_news(
        articles
    )

    return articles, classification


# =========================================================
# LOAD MARKET
# =========================================================

with st.spinner(
    "Bitvavo-markt wordt geanalyseerd..."
):
    rows = load_market_scan()


if not rows:
    st.error(
        "Er konden geen Bitvavo-assets worden geanalyseerd."
    )
    st.stop()


scanner_df = pd.DataFrame(rows)

market_context = load_market_context()


# =========================================================
# OVERVIEW
# =========================================================

total_assets = len(scanner_df)

interesting_assets = len(
    scanner_df[
        scanner_df["score"] >= 65
    ]
)

strong_assets = len(
    scanner_df[
        scanner_df["score"] >= 75
    ]
)

early_assets = len(
    scanner_df[
        scanner_df["phase"]
        == "🟢 Vroeg momentum"
    ]
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Assets gescand",
    total_assets
)

c2.metric(
    "Interessant",
    interesting_assets
)

c3.metric(
    "Sterk interessant",
    strong_assets
)

c4.metric(
    "Vroeg momentum",
    early_assets
)


st.info(
    f"Brede marktcontext: {market_context['regime']}"
)


# =========================================================
# TOP 5
# =========================================================

st.divider()

st.subheader(
    "🏆 Top 5 op dit moment"
)


top5 = scanner_df.head(5)

top_cols = st.columns(5)


for col, (_, row) in zip(
    top_cols,
    top5.iterrows()
):

    articles, news = load_news(
        row["asset"]
    )

    forecast = build_forecast(
        row.to_dict(),
        news,
        market_context
    )

    with col:

        st.metric(
            row["asset"],
            f"{row['score']}/100",
            fmt_pct(
                row["change_1d"]
            )
        )

        st.write(
            row["rating"]
        )

        st.caption(
            row["phase"]
        )

        st.write(
            f"🔮 {forecast['bias']}"
        )

        st.caption(
            f"{forecast['confidence']}% confidence"
        )


# =========================================================
# COMPLETE TABLE
# =========================================================

st.divider()

st.subheader(
    "🔎 Alle Bitvavo-assets"
)


f1, f2, f3 = st.columns(3)


with f1:

    search = st.text_input(
        "Zoek asset",
        placeholder="BTC, ONG, SOL..."
    )


with f2:

    min_volume = st.number_input(
        "Minimaal 24u-volume (€)",
        min_value=0,
        value=0,
        step=25000
    )


with f3:

    timing_filter = st.selectbox(
        "Timing",
        [
            "Alles",
            "🟢 Vroeg momentum",
            "🟡 Lopend momentum",
            "🟡 Neutraal",
            "🟠 Mogelijk laat"
        ]
    )


filtered = scanner_df.copy()


if search:

    filtered = filtered[
        filtered["asset"]
        .str.contains(
            search.upper(),
            case=False,
            na=False
        )
    ]


if min_volume > 0:

    filtered = filtered[
        filtered["volume_eur"]
        >= min_volume
    ]


if timing_filter != "Alles":

    filtered = filtered[
        filtered["phase"]
        == timing_filter
    ]


display_df = filtered[
    [
        "asset",
        "price",
        "change_1d",
        "change_3d",
        "change_7d",
        "volume_eur",
        "liquidity",
        "technical_score",
        "late_penalty",
        "score",
        "phase",
        "rating",
    ]
].copy()


display_df.columns = [
    "Asset",
    "Koers",
    "1d %",
    "3d %",
    "7d %",
    "24u volume €",
    "Liquiditeit",
    "Technische score",
    "Late-entry aftrek",
    "Score",
    "Timing",
    "Beoordeling",
]


display_df.insert(
    0,
    "#",
    range(
        1,
        len(display_df) + 1
    )
)


st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True,
    height=650
)


# =========================================================
# COIN SELECTION
# =========================================================

st.divider()

st.subheader(
    "📊 Uitgebreide coin-analyse"
)


available_assets = (
    scanner_df["asset"]
    .tolist()
)


selected_asset = st.selectbox(
    "Selecteer een asset",
    available_assets
)


selected = scanner_df[
    scanner_df["asset"]
    == selected_asset
].iloc[0]


selected_dict = selected.to_dict()


# =========================================================
# NEWS
# =========================================================

articles, news = load_news(
    selected_asset
)


# =========================================================
# FORECAST
# =========================================================

forecast = build_forecast(
    selected_dict,
    news,
    market_context
)


# =========================================================
# SUMMARY
# =========================================================

m1, m2, m3, m4, m5 = (
    st.columns(5)
)


m1.metric(
    "Koers",
    fmt_price(
        selected["price"]
    )
)

m2.metric(
    "Score",
    f"{selected['score']}/100"
)

m3.metric(
    "1 dag",
    fmt_pct(
        selected["change_1d"]
    )
)

m4.metric(
    "3 dagen",
    fmt_pct(
        selected["change_3d"]
    )
)

m5.metric(
    "7 dagen",
    fmt_pct(
        selected["change_7d"]
    )
)


st.write(
    f"## {selected['rating']}"
)

st.write(
    f"**Timing:** {selected['phase']}"
)

st.write(
    f"**Liquiditeit:** {selected['liquidity']}"
)


# =========================================================
# 72H FORECAST CARD
# =========================================================

st.divider()

st.subheader(
    "🔮 72h Trend Forecast"
)


f1, f2, f3, f4 = st.columns(4)


f1.metric(
    "Forecast",
    forecast["bias"]
)

f2.metric(
    "Confidence",
    f"{forecast['confidence']}%"
)

f3.metric(
    "Base scenario",
    f"{forecast['center_pct']:+.1f}%"
)

f4.metric(
    "Scenario-range",
    (
        f"{forecast['low_pct']:+.1f}% "
        f"tot "
        f"{forecast['high_pct']:+.1f}%"
    )
)


st.write(
    forecast["scenario"]
)


st.caption(
    f"Marktcontext: {forecast['market_regime']}"
)

st.caption(
    f"Nieuwscontext: {forecast['news_label']}"
)


# =========================================================
# FORECAST PRICE LEVELS
# =========================================================

current_price = selected["price"]

base_target = (
    current_price
    * (
        1
        + forecast["center_pct"]
        / 100
    )
)

low_target = (
    current_price
    * (
        1
        + forecast["low_pct"]
        / 100
    )
)

high_target = (
    current_price
    * (
        1
        + forecast["high_pct"]
        / 100
    )
)


p1, p2, p3 = st.columns(3)


p1.metric(
    "Bear scenario",
    fmt_price(
        low_target
    )
)

p2.metric(
    "Base scenario",
    fmt_price(
        base_target
    )
)

p3.metric(
    "Bull scenario",
    fmt_price(
        high_target
    )
)


# =========================================================
# FORECAST CHART
# =========================================================

try:

    historical_df = get_candles(
        selected["market"],
        "1h",
        168
    )

    historical_df = (
        historical_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    recent_history = (
        historical_df
        .tail(72)
        .copy()
    )

    last_time = (
        recent_history[
            "timestamp"
        ].iloc[-1]
    )

    forecast_hours = np.array(
        [
            0,
            24,
            48,
            72
        ]
    )


    center_end = (
        forecast[
            "center_pct"
        ]
        / 100
    )

    low_end = (
        forecast[
            "low_pct"
        ]
        / 100
    )

    high_end = (
        forecast[
            "high_pct"
        ]
        / 100
    )


    # Geleidelijke overgang naar het 72h scenario
    progress = (
        forecast_hours
        / 72
    )


    center_prices = (
        current_price
        * (
            1
            + center_end
            * progress
        )
    )


    low_prices = (
        current_price
        * (
            1
            + low_end
            * progress
        )
    )


    high_prices = (
        current_price
        * (
            1
            + high_end
            * progress
        )
    )


    forecast_times = [
        last_time
        + pd.Timedelta(
            hours=int(hour)
        )
        for hour
        in forecast_hours
    ]


    st.write(
        "### Forecast-grafiek"
    )


    fig, ax = plt.subplots(
        figsize=(11, 5)
    )


    ax.plot(
        recent_history[
            "timestamp"
        ],
        recent_history[
            "close"
        ],
        label="Historische koers"
    )


    ax.plot(
        forecast_times,
        center_prices,
        linestyle="--",
        label="72h forecast"
    )


    ax.fill_between(
        forecast_times,
        low_prices,
        high_prices,
        alpha=0.18,
        label="Scenario-band"
    )


    ax.axvline(
        last_time,
        linestyle=":",
        alpha=0.7
    )


    ax.set_xlabel(
        "Tijd"
    )

    ax.set_ylabel(
        "Koers (€)"
    )

    ax.legend()


    st.pyplot(
        fig,
        use_container_width=True
    )


except Exception as e:

    st.warning(
        "De forecast-grafiek kon op dit moment "
        "niet worden opgebouwd."
    )


# =========================================================
# LATE ENTRY
# =========================================================

if selected[
    "late_penalty"
] >= 20:

    st.error(
        "⚠️ Verhoogd late-entry risico. "
        "De coin is volgens de scanner al "
        "aanzienlijk opgelopen."
    )


elif selected[
    "late_penalty"
] >= 10:

    st.warning(
        "⚠️ Een deel van de beweging heeft "
        "mogelijk al plaatsgevonden."
    )


elif selected[
    "phase"
] == "🟢 Vroeg momentum":

    st.success(
        "🟢 Volgens de huidige regels bevindt "
        "de beweging zich nog relatief vroeg."
    )


if selected[
    "risk_reasons"
]:

    st.write(
        "**Timing-risico's:**"
    )

    for reason in selected[
        "risk_reasons"
    ]:

        st.write(
            f"• {reason}"
        )


# =========================================================
# SCORE BREAKDOWN
# =========================================================

st.write(
    "### 🧮 Score-opbouw"
)


score_df = pd.DataFrame(
    [
        {
            "Onderdeel":
                "Technische score",
            "Waarde":
                selected[
                    "technical_score"
                ],
        },
        {
            "Onderdeel":
                "Liquiditeit",
            "Waarde":
                selected[
                    "liquidity_score"
                ],
        },
        {
            "Onderdeel":
                "Late-entry aftrek",
            "Waarde":
                -selected[
                    "late_penalty"
                ],
        },
        {
            "Onderdeel":
                "Eindscore",
            "Waarde":
                selected[
                    "score"
                ],
        },
    ]
)


st.dataframe(
    score_df,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# MULTI TIMEFRAME
# =========================================================

st.write(
    "### ⏱️ Multi-timeframe"
)


timeframes = {
    "1 uur": "1h",
    "4 uur": "4h",
    "1 dag": "1d",
}


timeframe_results = {}


for label, interval in (
    timeframes.items()
):

    try:

        df = get_candles(
            selected[
                "market"
            ],
            interval,
            250
        )

        timeframe_results[
            label
        ] = analyze_timeframe(
            df
        )

    except Exception:

        timeframe_results[
            label
        ] = None


timeframe_rows = []


for label, result in (
    timeframe_results.items()
):

    if result is None:

        timeframe_rows.append({
            "Timeframe":
                label,
            "Score":
                None,
            "RSI":
                None,
            "Beoordeling":
                "Onvoldoende data"
        })

    else:

        timeframe_rows.append({
            "Timeframe":
                label,
            "Score":
                result[
                    "technical_score"
                ],
            "RSI":
                round(
                    result[
                        "rsi"
                    ],
                    1
                )
                if result[
                    "rsi"
                ]
                is not None
                else None,
            "Beoordeling":
                classify_score(
                    result[
                        "technical_score"
                    ]
                )
        })


st.dataframe(
    pd.DataFrame(
        timeframe_rows
    ),
    hide_index=True,
    use_container_width=True
)


# =========================================================
# TECHNICAL REASONS
# =========================================================

st.write(
    "### 🔬 Technische signalen"
)


tabs = st.tabs(
    [
        "1 uur",
        "4 uur",
        "1 dag"
    ]
)


for tab, label in zip(
    tabs,
    [
        "1 uur",
        "4 uur",
        "1 dag"
    ]
):

    with tab:

        result = (
            timeframe_results[
                label
            ]
        )

        if result is None:

            st.write(
                "Onvoldoende historische data."
            )

        else:

            st.write(
                f"**Score: "
                f"{result['technical_score']}/100**"
            )

            if (
                result["rsi"]
                is not None
            ):

                st.write(
                    f"**RSI:** "
                    f"{result['rsi']:.1f}"
                )

            st.write(
                f"**Volume versus normaal:** "
                f"{result['volume_ratio']:.2f}×"
            )

            for reason in result[
                "reasons"
            ]:

                st.write(
                    f"• {reason}"
                )


# =========================================================
# NEWS
# =========================================================

st.divider()

st.subheader(
    f"📰 Recent nieuws — {selected_asset}"
)


st.write(
    f"### {news['label']}"
)

st.write(
    news["context"]
)


if articles:

    for article in articles:

        title = article[
            "title"
        ]

        domain = article[
            "domain"
        ]

        url = article[
            "url"
        ]

        if url:

            st.markdown(
                f"**[{title}]({url})**  \n"
                f"*Bron: {domain}*"
            )

        else:

            st.write(
                f"**{title}**"
            )

            st.caption(
                f"Bron: {domain}"
            )


else:

    st.info(
        "Geen duidelijk recent nieuws gevonden."
    )
