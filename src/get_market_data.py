"""
Download world gold (GC=F futures) and USD/THB from Yahoo Finance.
Yahoo only serves 1-minute bars for the last ~30 days and hourly bars for ~2 years,
so re-running this later downloads a different window than the one in the README.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from config import DATA

TICKERS = {"gold_usd": "GC=F", "usd_thb": "THB=X"}


def pull_minute(ticker, days=29):
    end = datetime.now(timezone.utc)
    s, frames = end - timedelta(days=days), []
    while s < end:  # Yahoo allows at most 7 days of 1-minute data per request
        e = min(s + timedelta(days=7), end)
        h = yf.Ticker(ticker).history(start=s, end=e, interval="1m")
        if not h.empty:
            frames.append(h["Close"])
        s = e
    out = pd.concat(frames)
    return out[~out.index.duplicated()]


def pull_hourly(ticker):
    return yf.Ticker(ticker).history(period="730d", interval="1h")["Close"]


if __name__ == "__main__":
    for name, pull in [("market_1m.csv", pull_minute), ("market_1h.csv", pull_hourly)]:
        cols = {}
        for col, t in TICKERS.items():
            s = pull(t)
            s.index = s.index.tz_convert("Asia/Bangkok").tz_localize(None)
            cols[col] = s
        df = pd.DataFrame(cols).sort_index()
        df.index.name = "time"
        df.to_csv(DATA / name)
        print(f"{name}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
