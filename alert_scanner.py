import os
import json
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

# FAST_MODE wordt later door GitHub Actions gebruikt voor de lichte
# 15-minuten scanner. De normale dashboard/full scan blijft hetzelfde.
FAST_MODE = (
    os.getenv("FAST_MODE", "false").lower() == "true"
)

STATE_FILE = "alert_state.json"

AMSTERDAM = ZoneInfo("Europe/Amsterdam")

TOP_N = 5

INTERESTING_THRESHOLD = 65
EARLY_ALERT_THRESHOLD = 70
EARLY_ALERT_COOLDOWN_HOURS = 12

# Fast-mover instellingen
FAST_ALERT_COOLDOWN_MINUTES = 60
FAST_MIN_LIQUIDITY_EUR = 25_000

DIGEST_HOURS = {8, 12, 16, 20}


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
        raise RuntimeError("Telegram secrets ontbreken.")

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
# STATE
# =========================================================

def default_state():
    return {
        "assets": {},
        "last_digest": None,
        "last_event_alerts": {},
        "last_fast_alerts": {},
        "known_markets": [],
        "forecast_history": [],
    }


def load_state():
    if not os.path.exists(STATE_FILE):
        return default_state()

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

        if "assets" not in state:
            state = {
                "assets": state,
                "last_digest": None,
                "last_event_alerts": {},
                "last_fast_alerts": {},
                "known_markets": [],
                "forecast_history": [],
            }

        state.setdefault("assets", {})
        state.setdefault("last_digest", None)
        state.setdefault("last_event_alerts", {})
        state.setdefault("last_fast_alerts", {})
        state.setdefault("known_markets", [])
        state.setdefault("forecast_history", [])

        return state

    except Exception:
        return default_state()


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
# BITVAVO
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

    # Volatiliteit
    df["RETURN"] = (
        df["close"]
        .pct_change()
    )

    df["VOLATILITY_24H"] = (
        df["RETURN"]
        .rolling(24)
        .std()
    )

    return df


# =========================================================
# PRICE CHANGE
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
# FAST MOMENTUM HELPERS
# =========================================================

def candle_change(df, candles_back):
    """
    Percentage change from N completed 15m candles back to the latest candle.
    candles_back=1  -> roughly 15 minutes
    candles_back=4  -> roughly 1 hour
    candles_back=8  -> roughly 2 hours
    """
    if df.empty or len(df) <= candles_back:
        return None

    old_price = float(df.iloc[-(candles_back + 1)]["close"])
    new_price = float(df.iloc[-1]["close"])

    if old_price <= 0:
        return None

    return (new_price - old_price) / old_price * 100


def fast_volume_ratio(df):
    """
    Vergelijkt het volume van de laatste 15m candle met het gemiddelde
    van de 8 candles daarvoor (ongeveer 2 uur).
    """
    if df.empty or len(df) < 10:
        return 1.0

    previous = df.iloc[-9:-1]["volume"]
    average = float(previous.mean()) if not previous.empty else 0.0
    latest = float(df.iloc[-1]["volume"])

    if average <= 0:
        return 1.0

    return latest / average


def classify_fast_momentum(change_15m, change_1h, change_2h, volume_ratio):
    """
    Geeft een event-type terug voor plotselinge korte-termijn bewegingen.

    EARLY:
      beweging begint nu en wordt door volume bevestigd.

    FAST:
      duidelijke sterke versnelling, maar nog niet per se extreem laat.

    LATE:
      beweging is al zo groot dat FOMO/terugvalrisico duidelijk hoger is.
    """
    c15 = change_15m or 0
    c1 = change_1h or 0
    c2 = change_2h or 0
    vr = volume_ratio or 1.0

    # Eerst de late/pump-regels, zodat +30% in 1u nooit "vroeg" kan heten.
    if c1 >= 15 or c2 >= 25 or c15 >= 10:
        return {
            "type": "late",
            "label": "⚠️ EXTREME MOVER / LATE ENTRY",
            "priority": 3,
        }

    if c1 >= 7 or c2 >= 10 or c15 >= 4:
        return {
            "type": "fast",
            "label": "🔥 FAST MOVER",
            "priority": 2,
        }

    early = (
        (
            c15 >= 1.5
            and c1 >= 2.0
        )
        or c15 >= 2.5
        or (
            c1 >= 3.5
            and c2 < 8
        )
    )

    if early and vr >= 1.35 and c2 < 12:
        return {
            "type": "early",
            "label": "🚀 EARLY MOMENTUM",
            "priority": 1,
        }

    return None


def fast_alert_key(asset, event_type):
    return f"{asset}:{event_type}"


def can_send_fast_alert(asset, event_type, state, now):
    key = fast_alert_key(asset, event_type)

    previous_time = parse_iso(
        state.get("last_fast_alerts", {}).get(key)
    )

    if previous_time is None:
        return True

    if previous_time.tzinfo is None:
        previous_time = previous_time.replace(tzinfo=AMSTERDAM)

    return (
        now - previous_time
        >= timedelta(minutes=FAST_ALERT_COOLDOWN_MINUTES)
    )


def detect_new_markets(markets, state):
    """
    Detecteert nieuwe Bitvavo EUR-markten.
    Bij de allereerste run wordt alleen een baseline opgeslagen, zodat
    niet alle bestaande markten als 'nieuw' worden gemeld.
    """
    current = sorted(
        item["market"]
        for item in markets
    )

    previous = state.get("known_markets", [])

    if not previous:
        state["known_markets"] = current
        return []

    previous_set = set(previous)

    new_items = [
        item
        for item in markets
        if item["market"] not in previous_set
    ]

    state["known_markets"] = current
    return new_items


def send_new_listing_alerts(new_markets, state, now):
    for item in new_markets[:5]:
        asset = item["asset"]
        market = item["market"]

        message = "\n".join([
            "🆕 NIEUWE BITVAVO LISTING",
            "",
            f"{asset} ({market}) is nieuw actief op Bitvavo.",
            "",
            "Ik volg vanaf nu de eerste 15m / 1u / 2u beweging.",
            "⚠️ Nieuwe listings kunnen extreem volatiel zijn.",
        ])

        send_telegram(message)


def scan_fast_movers(state, now):
    """
    Lichte scanner voor GitHub Actions.
    Haalt voor iedere actieve EUR-market alleen een klein blok 15m candles op.
    Hiermee kunnen we echte korte-termijn bewegingen detecteren zonder de
    volledige EMA/RSI/MACD scan te draaien.
    """
    markets = get_markets()
    ticker_data = get_ticker_24h()

    new_markets = detect_new_markets(markets, state)
    if new_markets:
        send_new_listing_alerts(
            new_markets,
            state,
            now,
        )

    alerts = []

    for item in markets:
        market = item["market"]
        asset = item["asset"]

        try:
            ticker = ticker_data.get(market, {})
            volume_eur = float(
                ticker.get("volumeQuote", 0) or 0
            )

            # Hele dunne markten negeren voor momentum-alerts.
            if volume_eur < FAST_MIN_LIQUIDITY_EUR:
                continue

            candles = get_candles(
                market,
                "15m",
                16,
            )

            if len(candles) < 10:
                continue

            c15 = candle_change(candles, 1)
            c1 = candle_change(candles, 4)
            c2 = candle_change(candles, 8)
            vr = fast_volume_ratio(candles)

            event = classify_fast_momentum(
                c15,
                c1,
                c2,
                vr,
            )

            if event is None:
                continue

            latest_price = float(
                ticker.get(
                    "last",
                    candles.iloc[-1]["close"],
                )
            )

            alerts.append({
                "asset": asset,
                "market": market,
                "price": latest_price,
                "volume_eur": volume_eur,
                "change_15m": round(c15, 2) if c15 is not None else None,
                "change_1h": round(c1, 2) if c1 is not None else None,
                "change_2h": round(c2, 2) if c2 is not None else None,
                "fast_volume_ratio": round(vr, 2),
                "event_type": event["type"],
                "event_label": event["label"],
                "priority": event["priority"],
            })

        except Exception:
            continue

    # Eerst late/extreme movers, daarna fast, daarna early;
    # binnen elk type grootste 1u-beweging eerst.
    alerts.sort(
        key=lambda x: (
            x["priority"],
            x["change_1h"] or 0,
            x["change_15m"] or 0,
        ),
        reverse=True,
    )

    sent = 0

    for coin in alerts:
        if sent >= 5:
            break

        if not can_send_fast_alert(
            coin["asset"],
            coin["event_type"],
            state,
            now,
        ):
            continue

        if coin["event_type"] == "early":
            explanation = (
                "🟢 De beweging lijkt recent te starten en volume bevestigt."
            )
        elif coin["event_type"] == "fast":
            explanation = (
                "🟠 Sterke versnelling. Instaprisico loopt snel op."
            )
        else:
            explanation = (
                "🔴 De beweging is al extreem groot. Hoog late-entry/FOMO-risico."
            )

        lines = [
            coin["event_label"],
            "",
            f"{coin['asset']} ({coin['market']})",
            f"15m {fmt_pct(coin['change_15m'])}",
            f"1u  {fmt_pct(coin['change_1h'])}",
            f"2u  {fmt_pct(coin['change_2h'])}",
            f"Volume laatste 15m: {coin['fast_volume_ratio']:.1f}× normaal",
            f"24u volume: €{coin['volume_eur']:,.0f}",
            "",
            explanation,
        ]

        send_telegram(
            "\n".join(lines)
        )

        state.setdefault(
            "last_fast_alerts",
            {},
        )[fast_alert_key(
            coin["asset"],
            coin["event_type"],
        )] = now.isoformat()

        sent += 1

    return alerts


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

    # EMA trend
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

    # EMA200
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

    # RSI
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
                "RSI mogelijk overbought"
            )

    # MACD
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

    # Volume
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

    volatility = (
        float(latest["VOLATILITY_24H"])
        if pd.notna(
            latest["VOLATILITY_24H"]
        )
        else 0.0
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
        "volatility": volatility,
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
# ENTRY TIMING
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
                "Geen relevant recent nieuws gevonden."
            ),
            "positive": 0,
            "negative": 0,
            "momentum": 0,
            "score_adjustment": 0,
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
        adjustment = -8

    elif positive > negative and positive > 0:
        label = "🟢 Positieve katalysator"
        context = (
            "Recente headlines bevatten mogelijk "
            "fundamenteel positieve ontwikkelingen."
        )
        adjustment = 6

    elif momentum > 0:
        label = "🟡 Vooral momentum-nieuws"
        context = (
            "Nieuws gaat vooral over koersbeweging "
            "en minder over een fundamentele katalysator."
        )
        adjustment = 0

    else:
        label = "⚪ Neutrale nieuwscontext"
        context = (
            "Er is recente berichtgeving, maar geen "
            "duidelijke positieve of negatieve katalysator."
        )
        adjustment = 0

    return {
        "label": label,
        "context": context,
        "positive": positive,
        "negative": negative,
        "momentum": momentum,
        "score_adjustment": adjustment,
    }


# =========================================================
# MARKET CONTEXT
# =========================================================

def get_market_context():
    try:
        btc_df = get_candles(
            "BTC-EUR",
            "1h",
            200,
        )

        btc_analysis = analyze_timeframe(
            btc_df
        )

        btc_1d = price_change_since(
            btc_df,
            24,
        )

        btc_3d = price_change_since(
            btc_df,
            72,
        )

        if btc_analysis is None:
            raise RuntimeError()

        score = btc_analysis[
            "technical_score"
        ]

        if (
            score >= 65
            and (btc_1d or 0) >= 0
        ):
            regime = "🟢 Positieve cryptomarkt"
            adjustment = 5

        elif (
            score < 45
            and (btc_1d or 0) < 0
        ):
            regime = "🔴 Zwakke cryptomarkt"
            adjustment = -7

        else:
            regime = "🟡 Gemengde cryptomarkt"
            adjustment = 0

        return {
            "regime": regime,
            "adjustment": adjustment,
            "btc_score": score,
            "btc_1d": btc_1d,
            "btc_3d": btc_3d,
        }

    except Exception:
        return {
            "regime": "⚪ Marktcontext onbekend",
            "adjustment": 0,
            "btc_score": None,
            "btc_1d": None,
            "btc_3d": None,
        }


# =========================================================
# 72H FORECAST
# =========================================================

def build_forecast(
    coin,
    news=None,
    market_context=None,
):
    if news is None:
        news = {
            "score_adjustment": 0,
            "label": "⚪ Geen nieuwscontext",
        }

    if market_context is None:
        market_context = {
            "adjustment": 0,
            "regime": "⚪ Marktcontext onbekend",
        }

    technical = coin[
        "technical_score"
    ]

    score = 50

    # Technische richting
    score += (
        technical - 50
    ) * 0.55

    # Momentum
    c1 = coin["change_1d"] or 0
    c3 = coin["change_3d"] or 0

    if 0 < c1 <= 8:
        score += 5

    elif 8 < c1 <= 15:
        score += 2

    elif c1 >= 20:
        score -= 8

    if 0 < c3 <= 18:
        score += 5

    elif c3 >= 35:
        score -= 10

    # Volume bevestiging
    volume_ratio = coin[
        "volume_ratio"
    ]

    if volume_ratio >= 1.5:
        score += 5

    elif volume_ratio < 0.6:
        score -= 4

    # Late-entry risico
    score -= (
        coin["late_penalty"]
        * 0.65
    )

    # Nieuws
    score += news.get(
        "score_adjustment",
        0,
    )

    # Brede cryptomarkt
    score += market_context.get(
        "adjustment",
        0,
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

    # Direction
    if score >= 65:
        bias = "🟢 Bullish"
        direction = 1

    elif score <= 40:
        bias = "🔴 Bearish"
        direction = -1

    else:
        bias = "🟡 Sideways / onzeker"
        direction = 0

    # Confidence
    distance = abs(
        score - 50
    )

    confidence = int(
        min(
            85,
            45 + distance * 1.25,
        )
    )

    if (
        coin["liquidity_score"] < 55
    ):
        confidence -= 10

    if (
        coin["late_penalty"] >= 20
    ):
        confidence -= 8

    confidence = int(
        max(
            35,
            min(
                85,
                confidence,
            ),
        )
    )

    # Volatiliteit gebruiken voor range
    volatility = coin.get(
        "volatility",
        0.02,
    )

    if not volatility or volatility <= 0:
        volatility = 0.02

    # Geschatte 72u volatiliteitsband
    expected_move = (
        volatility
        * np.sqrt(72)
        * 100
    )

    expected_move = float(
        max(
            2.0,
            min(
                30.0,
                expected_move,
            ),
        )
    )

    # Richtingscomponent
    directional_strength = (
        abs(score - 50)
        / 50
    )

    center_move = (
        expected_move
        * directional_strength
        * direction
        * 0.65
    )

    if direction == 0:
        center_move = 0

    band = (
        expected_move
        * 0.65
    )

    low = center_move - band
    high = center_move + band

    # Voorkom absurde ranges
    low = max(
        low,
        -35,
    )

    high = min(
        high,
        35,
    )

    if bias == "🟢 Bullish":
        scenario = (
            "Opwaartse voortzetting heeft op dit moment "
            "meer technische ondersteuning dan een daling."
        )

    elif bias == "🔴 Bearish":
        scenario = (
            "Neerwaarts risico is momenteel groter dan "
            "de kans op directe trendvoortzetting omhoog."
        )

    else:
        scenario = (
            "Er is onvoldoende overtuiging voor een "
            "duidelijke richting in de komende 72 uur."
        )

    return {
        "forecast_score": score,
        "bias": bias,
        "confidence": confidence,
        "low_pct": round(
            low,
            1,
        ),
        "high_pct": round(
            high,
            1,
        ),
        "center_pct": round(
            center_move,
            1,
        ),
        "scenario": scenario,
        "market_regime":
            market_context.get(
                "regime",
                "Onbekend",
            ),
        "news_label":
            news.get(
                "label",
                "Geen nieuwscontext",
            ),
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

            base_score = (
                analysis["technical_score"]
                * 0.80
                + liquidity_score
                * 0.20
            )

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
                "liquidity":
                    liquidity_label,
                "liquidity_score":
                    liquidity_score,
                "technical_score":
                    analysis[
                        "technical_score"
                    ],
                "late_penalty":
                    timing[
                        "late_penalty"
                    ],
                "score":
                    final_score,
                "rating":
                    classify_score(
                        final_score
                    ),
                "phase":
                    timing["phase"],
                "rsi":
                    analysis["rsi"],
                "volume_ratio":
                    analysis[
                        "volume_ratio"
                    ],
                "volatility":
                    analysis[
                        "volatility"
                    ],
                "reasons":
                    analysis[
                        "reasons"
                    ],
                "risk_reasons":
                    timing[
                        "risk_reasons"
                    ],
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


# =========================================================
# DIGEST
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

    return (
        state.get("last_digest")
        != digest_key(now)
    )


def build_digest_coin(
    coin,
    market_context,
):
    articles = get_recent_news(
        coin["asset"],
        max_records=4,
    )

    news = classify_news(
        articles
    )

    forecast = build_forecast(
        coin,
        news,
        market_context,
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
            f"Timing: {coin['phase']}"
        ),
        (
            f"🔮 72h: {forecast['bias']} "
            f"({forecast['confidence']}%)"
        ),
        (
            f"Scenario-range: "
            f"{forecast['low_pct']:+.1f}% "
            f"tot {forecast['high_pct']:+.1f}%"
        ),
        (
            f"Nieuws: {news['label']}"
        ),
    ]

    if articles:
        title = articles[0]["title"]

        if len(title) > 120:
            title = (
                title[:117]
                + "..."
            )

        lines.append(
            f"• {title}"
        )

    if coin["risk_reasons"]:
        lines.append(
            "⚠️ "
            + "; ".join(
                coin[
                    "risk_reasons"
                ][:2]
            )
        )

    return "\n".join(lines)


def send_top_digest(
    rows,
    state,
    now,
):
    top = rows[:TOP_N]

    if not top:
        return

    market_context = (
        get_market_context()
    )

    lines = [
        "📊 CRYPTO UPDATE",
        now.strftime(
            "%d-%m-%Y %H:%M"
        ),
        "",
        (
            f"Markt: "
            f"{market_context['regime']}"
        ),
        "",
        "Top 5:",
        "",
    ]

    for index, coin in enumerate(
        top,
        start=1,
    ):
        lines.append(
            f"#{index}"
        )

        lines.append(
            build_digest_coin(
                coin,
                market_context,
            )
        )

        lines.append("")

    message = "\n".join(lines)

    if len(message) > 4000:
        message = (
            message[:3950]
            + "\n..."
        )

    send_telegram(
        message
    )

    state["last_digest"] = (
        digest_key(now)
    )


# =========================================================
# EARLY ALERTS
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
            hours=
            EARLY_ALERT_COOLDOWN_HOURS
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

        if (
            coin["late_penalty"]
            >= 10
        ):
            continue

        if (
            coin["phase"]
            != "🟢 Vroeg momentum"
        ):
            continue

        if (
            coin["liquidity_score"]
            < 55
        ):
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

        crossed = (
            previous_score
            < INTERESTING_THRESHOLD
            and coin["score"]
            >= EARLY_ALERT_THRESHOLD
        )

        jumped = (
            coin["score"]
            - previous_score
            >= 10
        )

        volume_confirmed = (
            coin["volume_ratio"]
            >= 1.15
        )

        if (
            (crossed or jumped)
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
    market_context = (
        get_market_context()
    )

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

        if (
            news["negative"]
            > news["positive"]
        ):
            continue

        forecast = build_forecast(
            coin,
            news,
            market_context,
        )

        lines = [
            "🚨 VROEG SIGNAAL",
            "",
            (
                f"{coin['asset']} — "
                f"{coin['score']}/100"
            ),
            (
                f"Timing: "
                f"{coin['phase']}"
            ),
            (
                f"1d {fmt_pct(coin['change_1d'])} | "
                f"3d {fmt_pct(coin['change_3d'])}"
            ),
            "",
            (
                f"🔮 72h forecast: "
                f"{forecast['bias']}"
            ),
            (
                f"Confidence: "
                f"{forecast['confidence']}%"
            ),
            (
                f"Range: "
                f"{forecast['low_pct']:+.1f}% "
                f"tot "
                f"{forecast['high_pct']:+.1f}%"
            ),
            "",
            (
                f"Nieuws: "
                f"{news['label']}"
            ),
            (
                f"Markt: "
                f"{market_context['regime']}"
            ),
        ]

        if articles:
            lines.append("")
            lines.append(
                "Recent nieuws:"
            )

            for article in articles[:2]:
                lines.append(
                    f"• {article['title']}"
                )

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
# FORECAST HISTORY
# =========================================================

def record_forecasts(
    rows,
    state,
    now,
):
    history = state.setdefault(
        "forecast_history",
        [],
    )

    market_context = (
        get_market_context()
    )

    # Alleen top 5 bewaren om state klein te houden
    for coin in rows[:5]:
        forecast = build_forecast(
            coin,
            news=None,
            market_context=
                market_context,
        )

        history.append({
            "created_at":
                now.isoformat(),
            "target_at":
                (
                    now
                    + timedelta(hours=72)
                ).isoformat(),
            "asset":
                coin["asset"],
            "start_price":
                coin["price"],
            "bias":
                forecast["bias"],
            "confidence":
                forecast["confidence"],
            "low_pct":
                forecast["low_pct"],
            "high_pct":
                forecast["high_pct"],
            "actual_pct":
                None,
        })

    # Max ongeveer 30 dagen historie
    if len(history) > 500:
        state[
            "forecast_history"
        ] = history[-500:]


def evaluate_old_forecasts(
    rows,
    state,
    now,
):
    current_prices = {
        coin["asset"]:
            coin["price"]
        for coin in rows
    }

    history = state.get(
        "forecast_history",
        [],
    )

    for item in history:
        if (
            item.get(
                "actual_pct"
            )
            is not None
        ):
            continue

        try:
            target = (
                datetime
                .fromisoformat(
                    item[
                        "target_at"
                    ]
                )
            )

            if target.tzinfo is None:
                target = target.replace(
                    tzinfo=AMSTERDAM
                )

            if now < target:
                continue

            asset = item[
                "asset"
            ]

            current = (
                current_prices
                .get(asset)
            )

            start = item[
                "start_price"
            ]

            if (
                current is None
                or start <= 0
            ):
                continue

            actual = (
                (current - start)
                / start
                * 100
            )

            item[
                "actual_pct"
            ] = round(
                actual,
                2,
            )

        except Exception:
            continue


# =========================================================
# ASSET STATE
# =========================================================

def build_asset_state(rows):
    result = {}

    for coin in rows:
        result[
            coin["asset"]
        ] = {
            "score":
                coin["score"],
            "technical_score":
                coin[
                    "technical_score"
                ],
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

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    now = datetime.now(
        AMSTERDAM
    )

    state = load_state()

    # ---------------------------------------------------------
    # FAST MODE
    # ---------------------------------------------------------
    # Wordt later door GitHub Actions iedere 15 minuten gestart.
    # Deze route doet alleen listing + 15m/1u/2u momentum checks.
    if FAST_MODE:
        scan_fast_movers(
            state,
            now,
        )

        save_state(
            state
        )

        return

    # ---------------------------------------------------------
    # VOLLEDIGE SCAN
    # ---------------------------------------------------------
    previous_assets = (
        state.get(
            "assets",
            {},
        )
    )

    # Houd ook bij welke Bitvavo-markten bestaan.
    # Dit zorgt dat de fast scanner later nieuwe listings kan herkennen.
    try:
        markets = get_markets()
        detect_new_markets(
            markets,
            state,
        )
    except Exception:
        pass

    rows = scan_market()

    if not rows:
        raise RuntimeError(
            "Geen assets konden worden geanalyseerd."
        )

    # Eerdere forecasts beoordelen
    evaluate_old_forecasts(
        rows,
        state,
        now,
    )

    # Vaste update
    if should_send_digest(
        now,
        state,
    ):
        send_top_digest(
            rows,
            state,
            now,
        )

        record_forecasts(
            rows,
            state,
            now,
        )

    # Bestaande technische vroege signalen blijven actief.
    # Deze zijn trager dan de fast scanner en dienen als extra bevestiging.
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

    # Nieuwe marktstatus
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
