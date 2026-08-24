import streamlit as st
import requests

st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Mijn Crypto Dashboard")

response = requests.get(
    "https://api.bitvavo.com/v2/ticker/price",
    params={"market": "BTC-EUR"}
)

data = response.json()
btc_price = float(data["price"])

st.metric(
    label="Bitcoin",
    value=f"€ {btc_price:,.2f}"
)
