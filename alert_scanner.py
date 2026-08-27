import os
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests


# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://api.bitvavo.com/v2"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

FORCE_DIGEST = (
    os.getenv("FORCE_DIGEST", "false").lower() == "true"
)

STATE_FILE = "alert_state.json"

AMSTERDAM = ZoneInfo("Europe/Amsterdam")

TOP_N = 5

# Algemene drempels
INTERESTING_THRESHOLD = 65
STRONG_THRESHOLD = 75

# Voor tussentijdse "vroeg momentum"-alerts
EARLY_ALERT_THRESHOLD = 70
EARLY_ALERT_COOLDOWN_HOURS = 12

# Vaste samenvatting
DIGEST_HOURS = {8, 12, 16, 20}

# Bekende projectnamen om nieuwszoekopdrachten beter te maken
NEWS_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "XRP",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "LINK": "Chainlink",
    "DOT": "Polkadot",
    "AVAX": "Avalanche",
    "ATOM": "Cosmos",
    "LTC": "Litecoin",
    "BCH": "Bitcoin Cash",
    "UNI": "Uniswap",
    "AAVE": "Aave",
    "ARB": "Arbitrum",
    "OP": "Optimism",
    "NEAR": "NEAR Protocol",
    "SUI": "Sui",
    "APT": "Aptos",
    "INJ": "Injective",
    "TON": "Toncoin",
    "TRX": "TRON",
    "HBAR": "Hedera",
    "VET": "VeChain",
    "ALGO": "Algorand",
    "FIL": "Filecoin",
    "ICP": "Internet Computer",
    "ETC": "Ethereum Classic",
    "XLM": "Stellar",
    "SHIB": "Shiba Inu",
    "PEPE": "Pepe",
    "ZBCN": "Zebec Network",
    "MOVR": "Moonriver",
    "BLZ": "Bluzelle",
    "ONG": "Ontology Gas",
    "ONT": "Ontology",
}


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "Telegram secrets ontbreken."
        )

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=30,
    )

    response.raise_for_status()


# =========================================================
# STATE / MEMORY
# =========================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "assets": {},
            "last_digest": None,
            "last_event_alerts": {},
        }

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

        # Migratie van onze oude state-structuur
        if "assets" not in state:
            state = {
                "assets": state,
                "last_digest": None,
                "last_event_alerts": {},
            }

        state.setdefault("assets", {})
        state.setdefault("last_digest", None)
        state.setdefault("last_event_alerts", {})

        return state

    except Exception:
        return {
            "assets": {},
            "last_digest": None,
            "last_event_alerts": {},
        }


def save_state(state):
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# BITVAVO DATA
# =========================================================

def get_markets():
    response = requests.get(
        f"{BASE_URL}/markets",
        timeout=30,
    )

    response.raise_for_status()

    markets = response.json()

    return [
        {
            "market": market["market"],
            "asset": market["base"],
        }
        for market in markets
        if (
            market.get("quote") == "EUR"
            and market.get("status") == "trading"
        )
    ]


def get_ticker_24h():
    response = requests.get(
        f"{BASE_URL}/ticker/24h",
        timeout=30,
    )

    response.raise_for_status()

    return {
        item["market"]: item
        for item in response.json()
    }


def get_candles(
    market,
    interval="1h",
    limit=250,
):
    response = requests.get(
        f"{BASE_URL}/{market}/candles",
        params={
            "interval": interval,
            "limit": limit,
        },
        timeout=30,
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
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return (
        df
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    df["RSI"] = calculate_rsi(
        df["close"]
    )

    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["VOL_AVG20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    return df


# =========================================================
# HISTORICAL PRICE CHANGE
# =========================================================

def price_change_since(df, hours):
    if df.empty:
        return None

    latest = df.iloc[-1]

    target_time = (
        latest["timestamp"]
        - pd.Timedelta(hours=hours)
    )

    older = df[
        df["timestamp"] <= target_time
    ]

    if older.empty:
        return None

    old_price = older.iloc[-1]["close"]
    new_price = latest["close"]

    if old_price <= 0:
        return None

    return (
        (new_price - old_price)
        / old_price
        * 100
    )


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def analyze_timeframe(df):
    if len(df) < 50:
        return None

    df = add_indicators(df)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    score = 50
    reasons = []

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if (
        latest["close"] > latest["EMA20"]
        and latest["EMA20"] > latest["EMA50"]
    ):
        score += 15
        reasons.append(
            "EMA-trend is positief"
        )

    elif (
        latest["close"] < latest["EMA20"]
        and latest["EMA20"] < latest["EMA50"]
    ):
        score -= 15
        reasons.append(
            "EMA-trend is negatief"
        )

    else:
        reasons.append(
            "EMA-trend is gemengd"
        )

    # -----------------------------------------------------
    # EMA200
    # -----------------------------------------------------

    if len(df) >= 200:
        if latest["close"] > latest["EMA200"]:
            score += 8
            reasons.append(
                "Koers boven EMA200"
            )
        else:
            score -= 8
            reasons.append(
                "Koers onder EMA200"
            )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi = latest["RSI"]

    if pd.notna(rsi):
        if rsi < 25:
            score += 3
            reasons.append(
                "RSI zeer laag"
            )

        elif 25 <= rsi < 40:
            score += 7
            reasons.append(
                "RSI toont herstelruimte"
            )

        elif 40 <= rsi <= 60:
            score += 5
            reasons.append(
                "RSI gezond/neutraal"
            )

        elif 60 < rsi <= 70:
            score += 3
            reasons.append(
                "RSI toont positief momentum"
            )

        elif rsi > 70:
            score -= 8
            reasons.append(
                "RSI is mogelijk overbought"
            )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        latest["MACD"] > latest["MACD_SIGNAL"]
        and previous["MACD"]
        <= previous["MACD_SIGNAL"]
    ):
        score += 12
        reasons.append(
            "Nieuwe bullish MACD-crossover"
        )

    elif (
        latest["MACD"] < latest["MACD_SIGNAL"]
        and previous["MACD"]
        >= previous["MACD_SIGNAL"]
    ):
        score -= 12
        reasons.append(
            "Nieuwe bearish MACD-crossover"
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

    # -----------------------------------------------------
    # VOLUME MOMENTUM
    # -----------------------------------------------------

    volume_ratio = 1.0

    if (
        pd.notna(latest["VOL_AVG20"])
        and latest["VOL_AVG20"] > 0
    ):
        volume_ratio = (
            latest["volume"]
            / latest["VOL_AVG20"]
        )

        if volume_ratio >= 2.0:
            score += 8
            reasons.append(
                "Handelsvolume versnelt sterk"
            )

        elif volume_ratio >= 1.25:
            score += 4
            reasons.append(
                "Handelsvolume bovengemiddeld"
            )

        elif volume_ratio < 0.60:
            score -= 3
            reasons.append(
                "Handelsvolume relatief laag"
            )

    score = int(
        max(
            0,
            min(
                100,
                round(score),
            ),
        )
    )

    return {
        "technical_score": score,
        "rsi": (
            float(rsi)
            if pd.notna(rsi)
            else None
        ),
        "volume_ratio": float(volume_ratio),
        "reasons": reasons,
        "df": df,
    }


# =========================================================
# LIQUIDITY
# =========================================================

def liquidity_info(volume_eur):
    if volume_eur >= 10_000_000:
        return 100, "🟢 Zeer hoog"

    if volume_eur >= 2_000_000:
        return 90, "🟢 Hoog"

    if volume_eur >= 500_000:
        return 75, "🟡 Goed"

    if volume_eur >= 100_000:
        return 55, "🟠 Beperkt"

    if volume_eur >= 25_000:
        return 35, "🟠 Laag"

    return 15, "🔴 Zeer laag"


# =========================================================
# MOMENTUM / ENTRY TIMING
# =========================================================

def timing_analysis(
    change_1d,
    change_3d,
    change_7d,
    rsi,
    volume_ratio,
):
    c1 = change_1d or 0
    c3 = change_3d or 0
    c7 = change_7d or 0
    rsi_value = rsi or 50

    penalty = 0
    risk_reasons = []

    # Zeer harde korte pump
    if c1 >= 20:
        penalty += 18
        risk_reasons.append(
            "koers al ≥20% gestegen in 24u"
        )

    elif c1 >= 12:
        penalty += 10
        risk_reasons.append(
            "forse stijging in de laatste 24u"
        )

    # Meerdere dagen al hard omhoog
    if c3 >= 35:
        penalty += 18
        risk_reasons.append(
            "koers al ≥35% gestegen in 3 dagen"
        )

    elif c3 >= 22:
        penalty += 10
        risk_reasons.append(
            "koers al sterk opgelopen in 3 dagen"
        )

    # Weekmove
    if c7 >= 60:
        penalty += 12
        risk_reasons.append(
            "zeer sterke stijging over 7 dagen"
        )

    elif c7 >= 40:
        penalty += 7
        risk_reasons.append(
            "sterke stijging over 7 dagen"
        )

    # Overbought
    if rsi_value >= 78:
        penalty += 10
        risk_reasons.append(
            "RSI zeer hoog"
        )

    elif rsi_value >= 72:
        penalty += 5
        risk_reasons.append(
            "RSI aan hoge kant"
        )

    # Extreme volume spike na stijging
    if (
        volume_ratio >= 3.0
        and c1 >= 10
    ):
        penalty += 8
        risk_reasons.append(
            "extreme volumepiek na sterke stijging"
        )

    penalty = min(
        penalty,
        35,
    )

    # Timing label
    if penalty >= 20:
        phase = "🟠 Mogelijk laat"

    elif penalty >= 10:
        phase = "🟡 Lopend momentum"

    elif (
        c1 >= 0
        and c1 <= 10
        and c3 <= 18
        and rsi_value <= 70
    ):
        phase = "🟢 Vroeg momentum"

    else:
        phase = "🟡 Neutraal"

    return {
        "phase": phase,
        "late_penalty": penalty,
        "risk_reasons": risk_reasons,
    }


# =========================================================
# CLASSIFICATION
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


# =========================================================
# NEWS
# =========================================================

POSITIVE_NEWS_WORDS = [
    "partnership",
    "partners",
    "integration",
    "integrates",
    "launch",
    "launches",
    "launched",
    "listing",
    "listed",
    "upgrade",
    "mainnet",
    "approval",
    "approved",
    "buyback",
    "expansion",
    "adoption",
    "funding",
    "collaboration",
    "release",
    "rollout",
    "acquisition",
    "acquires",
    "staking",
]

NEGATIVE_NEWS_WORDS = [
    "hack",
    "hacked",
    "exploit",
    "attack",
    "delist",
    "delisting",
    "lawsuit",
    "investigation",
    "outage",
    "shutdown",
    "bankruptcy",
    "vulnerability",
    "warning",
    "fraud",
]

MOMENTUM_NEWS_WORDS = [
    "rally",
    "surge",
    "surges",
    "soars",
    "jumps",
    "pump",
    "price prediction",
    "price rises",
    "gains",
    "whale",
]


def news_search_name(asset):
    return NEWS_NAMES.get(
        asset,
        asset,
    )


def get_recent_news(asset, max_records=5):
    project = news_search_name(asset)

    if project.upper() == asset.upper():
        query = (
            f'"{asset}" '
            f'(crypto OR cryptocurrency OR blockchain OR token)'
        )
    else:
        query = (
            f'("{project}" OR "{asset}") '
            f'(crypto OR cryptocurrency OR blockchain OR token)'
        )

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "timespan": "3d",
        "sort": "HybridRel",
    }

    try:
        response = requests.get(
            GDELT_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        articles = data.get(
            "articles",
            [],
        )

        cleaned = []

        for article in articles:
            title = (
                article.get("title")
                or ""
            ).strip()

            url = (
                article.get("url")
                or ""
            ).strip()

            if not title:
                continue

            domain = (
                article.get("domain")
                or urlparse(url).netloc
                or "Onbekende bron"
            )

            cleaned.append({
                "title": title,
                "url": url,
                "domain": domain,
                "seendate": article.get(
                    "seendate",
                    "",
                ),
            })

        return cleaned[:max_records]

    except Exception:
        return []


def classify_news(articles):
    if not articles:
        return {
            "label": "⚪ Geen duidelijke katalysator",
            "context": (
                "Geen relevant recent nieuws gevonden "
                "in de gebruikte nieuwsbron."
            ),
            "positive": 0,
            "negative": 0,
            "momentum": 0,
        }

    positive = 0
    negative = 0
    momentum = 0

    for article in articles:
        title = article["title"].lower()

        positive += sum(
            word in title
            for word in POSITIVE_NEWS_WORDS
        )

        negative += sum(
            word in title
            for word in NEGATIVE_NEWS_WORDS
        )

        momentum += sum(
            word in title
            for word in MOMENTUM_NEWS_WORDS
        )

    if negative > positive:
        label = "🔴 Negatieve/risicovolle context"
        context = (
            "Recente headlines bevatten vooral "
            "negatieve of risicovolle ontwikkelingen."
        )

    elif positive > negative and positive > 0:
        label = "🟢 Positieve katalysator"
        context = (
            "Recente headlines bevatten mogelijk "
            "fundamenteel positieve ontwikkelingen."
        )

    elif momentum > 0:
        label = "🟡 Vooral momentum-nieuws"
        context = (
            "De recente berichtgeving gaat vooral "
            "over koersbeweging/momentum en minder "
            "over een duidelijke fundamentele katalysator."
        )

    else:
        label = "⚪ Neutrale nieuwscontext"
        context = (
            "Er is recente berichtgeving, maar geen "
            "duidelijke positieve of negatieve katalysator."
        )

    return {
        "label": label,
        "context": context,
        "positive": positive,
        "negative": negative,
        "momentum": momentum,
    }


# =========================================================
# MARKET SCAN
# =========================================================

def scan_market():
    markets = get_markets()
    ticker_data = get_ticker_24h()

    rows = []

    for item in markets:
        market = item["market"]
        asset = item["asset"]

        try:
            candles = get_candles(
                market,
                "1h",
                250,
            )

            analysis = analyze_timeframe(
                candles
            )

            if analysis is None:
                continue

            ticker = ticker_data.get(
                market,
                {},
            )

            latest_price = float(
                ticker.get(
                    "last",
                    candles.iloc[-1]["close"],
                )
            )

            volume_eur = float(
                ticker.get(
                    "volumeQuote",
                    0,
                )
            )

            liquidity_score, liquidity_label = (
                liquidity_info(
                    volume_eur
                )
            )

            change_1d = price_change_since(
                candles,
                24,
            )

            change_3d = price_change_since(
                candles,
                72,
            )

            change_7d = price_change_since(
                candles,
                168,
            )

            timing = timing_analysis(
                change_1d,
                change_3d,
                change_7d,
                analysis["rsi"],
                analysis["volume_ratio"],
            )

            # Eerst techniek + liquiditeit
            base_score = (
                analysis["technical_score"]
                * 0.80
                + liquidity_score
                * 0.20
            )

            # Daarna aftrek als we waarschijnlijk
            # achter de beweging aanlopen.
            final_score = round(
                base_score
                - timing["late_penalty"]
            )

            final_score = int(
                max(
                    0,
                    min(
                        100,
                        final_score,
                    ),
                )
            )

            rows.append({
                "asset": asset,
                "market": market,
                "price": latest_price,
                "change_1d": (
                    round(change_1d, 2)
                    if change_1d is not None
                    else None
                ),
                "change_3d": (
                    round(change_3d, 2)
                    if change_3d is not None
                    else None
                ),
                "change_7d": (
                    round(change_7d, 2)
                    if change_7d is not None
                    else None
                ),
                "volume_eur": round(
                    volume_eur,
                    0,
                ),
                "liquidity": liquidity_label,
                "technical_score":
                    analysis["technical_score"],
                "liquidity_score":
                    liquidity_score,
                "late_penalty":
                    timing["late_penalty"],
                "score": final_score,
                "rating":
                    classify_score(final_score),
                "phase":
                    timing["phase"],
                "rsi":
                    analysis["rsi"],
                "volume_ratio":
                    analysis["volume_ratio"],
                "reasons":
                    analysis["reasons"],
                "risk_reasons":
                    timing["risk_reasons"],
            })

        except Exception:
            continue

    rows.sort(
        key=lambda x: (
            x["score"],
            x["technical_score"],
            x["volume_eur"],
        ),
        reverse=True,
    )

    return rows


# =========================================================
# FORMATTING
# =========================================================

def fmt_pct(value):
    if value is None:
        return "n.v.t."

    return f"{value:+.1f}%"


def build_coin_message(
    coin,
    news_articles,
):
    news = classify_news(
        news_articles
    )

    lines = [
        (
            f"{coin['rating']} "
            f"{coin['asset']} — "
            f"{coin['score']}/100"
        ),
        (
            f"1d {fmt_pct(coin['change_1d'])} | "
            f"3d {fmt_pct(coin['change_3d'])} | "
            f"7d {fmt_pct(coin['change_7d'])}"
        ),
        (
            f"Timing: {coin['phase']} | "
            f"Liquiditeit: {coin['liquidity']}"
        ),
    ]

    # Kort technisch beeld
    technical_reasons = coin[
        "reasons"
    ][:3]

    if technical_reasons:
        lines.append(
            "Technisch: "
            + ", ".join(
                technical_reasons
            )
        )

    lines.append(
        f"Nieuws: {news['label']}"
    )

    lines.append(
        f"Context: {news['context']}"
    )

    if coin["risk_reasons"]:
        lines.append(
            "⚠️ Timing-risico: "
            + "; ".join(
                coin["risk_reasons"][:2]
            )
        )

    if news_articles:
        lines.append(
            "Gevonden:"
        )

        for article in news_articles[:2]:
            title = article["title"]

            if len(title) > 105:
                title = (
                    title[:102]
                    + "..."
                )

            lines.append(
                f"• {title} "
                f"({article['domain']})"
            )

    return "\n".join(lines)


# =========================================================
# DIGEST LOGIC
# =========================================================

def digest_key(now):
    return now.strftime(
        "%Y-%m-%d-%H"
    )


def should_send_digest(
    now,
    state,
):
    if FORCE_DIGEST:
        return True

    if now.hour not in DIGEST_HOURS:
        return False

    current_key = digest_key(now)

    return (
        state.get("last_digest")
        != current_key
    )


def send_top_digest(
    rows,
    state,
    now,
):
    top = rows[:TOP_N]

    if not top:
        return

    lines = [
        "📊 CRYPTO UPDATE",
        now.strftime(
            "%d-%m-%Y %H:%M"
        ),
        "",
        "Top 5 op dit moment:",
        "",
    ]

    for index, coin in enumerate(
        top,
        start=1,
    ):
        # Nieuws alleen voor top 5 ophalen
        articles = get_recent_news(
            coin["asset"],
            max_records=4,
        )

        coin_text = build_coin_message(
            coin,
            articles,
        )

        lines.append(
            f"#{index}"
        )

        lines.append(
            coin_text
        )

        lines.append("")

    lines.append(
        "Een hoge score betekent een gunstige "
        "technische setup volgens deze scanner, "
        "niet dat verdere stijging zeker is."
    )

    message = "\n".join(lines)

    # Telegram heeft een limiet;
    # deze versie houdt de tekst bewust compact.
    if len(message) > 4000:
        message = message[:3950] + "\n..."

    send_telegram(
        message
    )

    state["last_digest"] = (
        digest_key(now)
    )


# =========================================================
# EARLY OPPORTUNITY ALERTS
# =========================================================

def parse_iso(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value
        )
    except Exception:
        return None


def can_send_event_alert(
    asset,
    state,
    now,
):
    previous_time = parse_iso(
        state.get(
            "last_event_alerts",
            {},
        ).get(asset)
    )

    if previous_time is None:
        return True

    if previous_time.tzinfo is None:
        previous_time = (
            previous_time.replace(
                tzinfo=AMSTERDAM
            )
        )

    return (
        now - previous_time
        >= timedelta(
            hours=EARLY_ALERT_COOLDOWN_HOURS
        )
    )


def detect_early_opportunities(
    rows,
    previous_assets,
):
    alerts = []

    for coin in rows:
        if (
            coin["score"]
            < EARLY_ALERT_THRESHOLD
        ):
            continue

        # We willen juist GEEN late pump-alert
        if coin["late_penalty"] >= 10:
            continue

        if (
            coin["phase"]
            != "🟢 Vroeg momentum"
        ):
            continue

        # Redelijke liquiditeit vereist
        if coin["liquidity_score"] < 55:
            continue

        previous = previous_assets.get(
            coin["asset"],
            {},
        )

        previous_score = int(
            previous.get(
                "score",
                0,
            )
        )

        # Nieuwe doorbraak naar interessant
        crossed_threshold = (
            previous_score
            < INTERESTING_THRESHOLD
            and coin["score"]
            >= EARLY_ALERT_THRESHOLD
        )

        # Of sterke verbetering
        score_jump = (
            coin["score"]
            - previous_score
            >= 10
        )

        # Volume moet minimaal gezond zijn
        volume_confirmed = (
            coin["volume_ratio"]
            >= 1.15
        )

        if (
            (crossed_threshold or score_jump)
            and volume_confirmed
        ):
            alerts.append(
                coin
            )

    return alerts


def send_early_alerts(
    alerts,
    state,
    now,
):
    sent = 0

    for coin in alerts:
        if sent >= 3:
            break

        if not can_send_event_alert(
            coin["asset"],
            state,
            now,
        ):
            continue

        articles = get_recent_news(
            coin["asset"],
            max_records=4,
        )

        news = classify_news(
            articles
        )

        # Als het nieuws expliciet negatief is,
        # geen enthousiaste early opportunity-alert.
        if news["negative"] > news["positive"]:
            continue

        lines = [
            "🚨 VROEG SIGNAAL",
            "",
            build_coin_message(
                coin,
                articles,
            ),
            "",
            (
                "Dit signaal verschijnt omdat de "
                "technische score vroeg verbetert "
                "zonder dat de koers volgens onze "
                "late-entry-regels al extreem is opgelopen."
            ),
        ]

        message = "\n".join(
            lines
        )

        if len(message) > 4000:
            message = (
                message[:3950]
                + "\n..."
            )

        send_telegram(
            message
        )

        state.setdefault(
            "last_event_alerts",
            {},
        )

        state[
            "last_event_alerts"
        ][coin["asset"]] = (
            now.isoformat()
        )

        sent += 1


# =========================================================
# SAVE CURRENT MARKET STATE
# =========================================================

def build_asset_state(rows):
    state = {}

    for coin in rows:
        state[
            coin["asset"]
        ] = {
            "score":
                coin["score"],
            "technical_score":
                coin["technical_score"],
            "change_1d":
                coin["change_1d"],
            "change_3d":
                coin["change_3d"],
            "change_7d":
                coin["change_7d"],
            "phase":
                coin["phase"],
            "price":
                coin["price"],
        }

    return state


# =========================================================
# MAIN
# =========================================================

def main():
    now = datetime.now(
        AMSTERDAM
    )

    state = load_state()

    previous_assets = state.get(
        "assets",
        {},
    )

    rows = scan_market()

    if not rows:
        raise RuntimeError(
            "Geen Bitvavo-assets konden worden geanalyseerd."
        )

    # ---------------------------------------------
    # VASTE TOP-5 UPDATE
    # ---------------------------------------------

    if should_send_digest(
        now,
        state,
    ):
        send_top_digest(
            rows,
            state,
            now,
        )

    # ---------------------------------------------
    # TUSSENTIJDSE VROEGE SIGNALEN
    # ---------------------------------------------

    early_alerts = (
        detect_early_opportunities(
            rows,
            previous_assets,
        )
    )

    send_early_alerts(
        early_alerts,
        state,
        now,
    )

    # ---------------------------------------------
    # NIEUWE SCORES OPSLAAN
    # ---------------------------------------------

    state["assets"] = (
        build_asset_state(
            rows
        )
    )

    save_state(
        state
    )


if __name__ == "__main__":
    main()
