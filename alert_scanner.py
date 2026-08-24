import os
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.bitvavo.com/v2"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message):
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=15
    )


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

    rs = avg_gain / avg_loss.replace(0, np.nan)

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

    return df


def analyze(df):
    if len(df) < 50:
        return None

    df = add_indicators(df)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    score = 50

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

    rsi = latest["RSI"]

    if pd.notna(rsi):
        if 25 <= rsi < 40:
            score += 7

        elif 40 <= rsi <= 60:
            score += 5

        elif 60 < rsi <= 70:
            score += 3

        elif rsi > 70:
            score -= 8

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

    return max(
        0,
        min(
            100,
            round(score)
        )
    )


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


def main():
    markets = get_markets()
    ticker_data = get_ticker_24h()

    opportunities = []

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

            liq_score = liquidity_score(
                volume_eur
            )

            final_score = round(
                technical_score * 0.8
                + liq_score * 0.2
            )

            if final_score >= 75:
                opportunities.append({
                    "asset": asset,
                    "score": final_score
                })

        except Exception:
            continue

    opportunities = sorted(
        opportunities,
        key=lambda x: x["score"],
        reverse=True
    )

    if opportunities:
        top = opportunities[:5]

        lines = [
            "🚨 Crypto Scanner",
            "",
            "Sterk interessante assets:"
        ]

        for item in top:
            lines.append(
                f"🟢 {item['asset']} — {item['score']}/100"
            )

        send_telegram(
            "\n".join(lines)
        )


if __name__ == "__main__":
    main()
