import os
import json
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.bitvavo.com/v2"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = "alert_state.json"

STRONG_THRESHOLD = 75
EXIT_THRESHOLD = 65
BIG_SCORE_INCREASE = 10


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram secrets ontbreken.")

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()


# =========================================================
# STATE
# =========================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception:
        return {}


def save_state(state):
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False
        )


# =========================================================
# BITVAVO DATA
# =========================================================

def get_markets():
    response = requests.get(
        f"{BASE_URL}/markets",
        timeout=20
    )

    response.raise_for_status()

    markets = response.json()

    return [
        {
            "market": market["market"],
            "asset": market["base"]
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
        timeout=30
    )

    response.raise_for_status()

    return {
        item["market"]: item
        for item in response.json()
    }


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
        timeout=20
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
# TECHNICAL SCORE
# =========================================================

def analyze(df):
    if len(df) < 50:
        return None

    df = add_indicators(df)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    score = 50

    # Trend
    if (
        latest["close"] > latest["EMA20"]
        and latest["EMA20"] > latest["EMA50"]
    ):
        score += 15

    elif (
        latest["close"] < latest["EMA20"]
        and latest["EMA20"] < latest["EMA50"]
    ):
        score -= 15

    # EMA200
    if len(df) >= 200:
        if latest["close"] > latest["EMA200"]:
            score += 8
        else:
            score -= 8

    # RSI
    rsi = latest["RSI"]

    if pd.notna(rsi):
        if rsi < 25:
            score += 3

        elif 25 <= rsi < 40:
            score += 7

        elif 40 <= rsi <= 60:
            score += 5

        elif 60 < rsi <= 70:
            score += 3

        elif rsi > 70:
            score -= 8

    # MACD
    if (
        latest["MACD"] > latest["MACD_SIGNAL"]
        and previous["MACD"] <= previous["MACD_SIGNAL"]
    ):
        score += 12

    elif (
        latest["MACD"] < latest["MACD_SIGNAL"]
        and previous["MACD"] >= previous["MACD_SIGNAL"]
    ):
        score -= 12

    elif latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 5

    else:
        score -= 5

    # Volume momentum
    if (
        pd.notna(latest["VOL_AVG20"])
        and latest["VOL_AVG20"] > 0
    ):
        volume_ratio = (
            latest["volume"]
            / latest["VOL_AVG20"]
        )

        if volume_ratio >= 1.5:
            score += 8

        elif volume_ratio >= 1.15:
            score += 4

        elif volume_ratio < 0.60:
            score -= 3

    return int(
        max(
            0,
            min(
                100,
                round(score)
            )
        )
    )


# =========================================================
# LIQUIDITY
# =========================================================

def liquidity_score(volume_eur):
    if volume_eur >= 10_000_000:
        return 100

    if volume_eur >= 2_000_000:
        return 90

    if volume_eur >= 500_000:
        return 75

    if volume_eur >= 100_000:
        return 55

    if volume_eur >= 25_000:
        return 35

    return 15


# =========================================================
# MAIN SCANNER
# =========================================================

def main():
    previous_state = load_state()

    markets = get_markets()
    ticker_data = get_ticker_24h()

    current_state = {}

    new_strong = []
    stronger = []
    lost_signal = []

    for item in markets:
        market = item["market"]
        asset = item["asset"]

        try:
            candles = get_candles(
                market,
                "1h",
                250
            )

            technical_score = analyze(
                candles
            )

            if technical_score is None:
                continue

            ticker = ticker_data.get(
                market,
                {}
            )

            volume_eur = float(
                ticker.get(
                    "volumeQuote",
                    0
                )
            )

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
                    (last_price - open_24h)
                    / open_24h
                    * 100
                )
            else:
                change_24h = 0

            liq_score = liquidity_score(
                volume_eur
            )

            final_score = int(
                round(
                    technical_score * 0.80
                    + liq_score * 0.20
                )
            )

            current_state[asset] = {
                "score": final_score,
                "technical_score": technical_score,
                "price": last_price,
                "change_24h": round(
                    change_24h,
                    2
                )
            }

            previous = previous_state.get(
                asset
            )

            # Eerste keer dat een asset >= 75 komt
            if previous is not None:
                previous_score = int(
                    previous.get(
                        "score",
                        0
                    )
                )

                if (
                    final_score >= STRONG_THRESHOLD
                    and previous_score < STRONG_THRESHOLD
                ):
                    new_strong.append({
                        "asset": asset,
                        "score": final_score,
                        "previous": previous_score,
                        "change": change_24h
                    })

                # Al sterk, maar ineens nóg veel sterker
                elif (
                    final_score >= STRONG_THRESHOLD
                    and previous_score >= STRONG_THRESHOLD
                    and (
                        final_score
                        - previous_score
                    ) >= BIG_SCORE_INCREASE
                ):
                    stronger.append({
                        "asset": asset,
                        "score": final_score,
                        "previous": previous_score,
                        "change": change_24h
                    })

                # Was sterk, maar verliest overtuiging
                elif (
                    previous_score >= STRONG_THRESHOLD
                    and final_score < EXIT_THRESHOLD
                ):
                    lost_signal.append({
                        "asset": asset,
                        "score": final_score,
                        "previous": previous_score,
                        "change": change_24h
                    })

        except Exception:
            continue

    # -----------------------------------------------------
    # EERSTE RUN
    # -----------------------------------------------------

    if not previous_state:
        strong_now = []

        for asset, data in current_state.items():
            if data["score"] >= STRONG_THRESHOLD:
                strong_now.append({
                    "asset": asset,
                    "score": data["score"],
                    "change": data["change_24h"]
                })

        strong_now = sorted(
            strong_now,
            key=lambda x: x["score"],
            reverse=True
        )

        if strong_now:
            lines = [
                "📊 Crypto Scanner gestart",
                "",
                "Huidige sterk interessante assets:"
            ]

            for item in strong_now[:10]:
                lines.append(
                    f"🟢 {item['asset']} — "
                    f"{item['score']}/100 "
                    f"({item['change']:+.2f}% 24u)"
                )

            if len(strong_now) > 10:
                lines.append(
                    f"\n+ {len(strong_now) - 10} andere"
                )

            send_telegram(
                "\n".join(lines)
            )

    # -----------------------------------------------------
    # CHANGES
    # -----------------------------------------------------

    else:
        messages = []

        if new_strong:
            new_strong = sorted(
                new_strong,
                key=lambda x: x["score"],
                reverse=True
            )

            messages.append(
                "🟢 NIEUW STERK INTERESSANT"
            )

            for item in new_strong[:10]:
                messages.append(
                    f"{item['asset']} — "
                    f"{item['previous']} → "
                    f"{item['score']}/100 "
                    f"({item['change']:+.2f}% 24u)"
                )

        if stronger:
            if messages:
                messages.append("")

            messages.append(
                "🚀 STERK VERBETERD"
            )

            for item in stronger[:10]:
                messages.append(
                    f"{item['asset']} — "
                    f"{item['previous']} → "
                    f"{item['score']}/100"
                )

        if lost_signal:
            if messages:
                messages.append("")

            messages.append(
                "🔴 STERK SIGNAAL VERLOREN"
            )

            for item in lost_signal[:10]:
                messages.append(
                    f"{item['asset']} — "
                    f"{item['previous']} → "
                    f"{item['score']}/100"
                )

        if messages:
            send_telegram(
                "\n".join(messages)
            )

    save_state(
        current_state
    )


if __name__ == "__main__":
    main()
