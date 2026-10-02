"""
Download Thai Gold Traders Association re-price history into a CSV.

Usage:
    pip install requests pandas
    python scrape_thai_gold.py                      # from 2020-01-01 to today
    python scrape_thai_gold.py --start 2023-01-01   # custom start date

Output: thai_gold_reprices.csv — one row per re-price, with the committee's
inputs (gold spot, USD/THB) and a computed international fair value/premium.
"""
import argparse
import time
from datetime import date

import pandas as pd
import requests
from config import DATA

URL = "https://www.goldtraders.or.th/api/GoldPricesDaily/pricechanges"
HEADERS = {
    "accept": "*/*",
    "referer": "https://www.goldtraders.or.th/updatepricelist",
    "user-agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"),
}
# If you get HTTP 403 errors, paste the cf_clearance value from your browser's
# cURL here (it expires after a while, so grab a fresh one if needed).
COOKIES = {}  # e.g. {"cf_clearance": "..."}

BAHT_WEIGHT_G, TROY_OZ_G, PURITY = 15.244, 31.1034768, 0.965
COLUMNS = {
    "asTime": "time",
    "seq": "seq",
    "bL_BuyPrice": "bar_buy",
    "bL_SellPrice": "bar_sell",
    "oM965_SellPrice": "ornament_sell",
    "goldSpot": "gold_spot_usd",
    "bahtPerUSD": "usd_thb",
    "priceChangeFromPrevRow": "change",
    "modifiedBy": "operator",
}


def month_ranges(start, end):
    for m in pd.date_range(start.replace(day=1), end, freq="MS"):
        m_end = min((m + pd.offsets.MonthEnd(0)).date(), end)
        yield max(m.date(), start), m_end


def fetch(start, end, session, retries=3):
    params = {"StartDate": start.isoformat(), "EndDate": end.isoformat()}
    for attempt in range(retries):
        r = session.get(URL, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"  HTTP {r.status_code}, retrying in 10s...")
        time.sleep(10)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default=str(DATA / "thai_gold_reprices.csv"))
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)

    rows = []
    for s, e in month_ranges(date.fromisoformat(args.start), date.fromisoformat(args.end)):
        data = fetch(s, e, session)
        print(f"{s} to {e}: {len(data)} re-prices")
        rows.extend(data)
        time.sleep(3)  # be polite to the server

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data returned.")
        return
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"])  # Bangkok local time (UTC+7)
    df = df.drop_duplicates(subset=["time", "seq"]).sort_values("time")

    df["bar_mid"] = (df["bar_buy"] + df["bar_sell"]) / 2
    df["bar_spread"] = df["bar_sell"] - df["bar_buy"]
    df["intl_fair_value"] = (df["gold_spot_usd"] * df["usd_thb"]
                             * BAHT_WEIGHT_G / TROY_OZ_G * PURITY)
    df["premium_bp"] = 1e4 * (df["bar_mid"] / df["intl_fair_value"] - 1)

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} re-prices to {args.out}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Median bar spread: {df['bar_spread'].median():.0f} baht")
    print(f"Median premium: {df['premium_bp'].median():.1f} bp")


if __name__ == "__main__":
    main()
