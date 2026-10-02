import numpy as np, pandas as pd
from config import DATA
from kalman import PremiumKalman, BAHT_WEIGHT_G, TROY_OZ_G, PURITY

CONV = BAHT_WEIGHT_G / TROY_OZ_G * PURITY

def load(market_file, bar=None):
    g = pd.read_csv(DATA / "thai_gold_clean.csv", parse_dates=["time"])
    m = pd.read_csv(DATA / market_file, parse_dates=["time"]).set_index("time").sort_index().ffill().dropna()
    # yfinance labels bars by START time with the CLOSE price: shift labels to bar end
    # so we never use a price before it existed (removes look-ahead bias)
    if bar is None:
        bar = "1min" if "1m" in market_file else "1h"
    m.index = m.index + pd.Timedelta(bar)
    m["fv"] = m["gold_usd"] * m["usd_thb"] * CONV
    g = g[(g.time >= m.index.min()) & (g.time <= m.index.max())].reset_index(drop=True)
    return g, m

def market_premium(g, m):
    """Kalman premium of posted bar mid over MARKET fair value (futures basis absorbed)."""
    fv_ev = m["fv"].reindex(g.time, method="ffill").to_numpy()
    z = np.log(g.bar_mid.to_numpy() / fv_ev)
    dt = np.maximum(g.time.diff().dt.total_seconds().fillna(3600).to_numpy() / 3600, 1/60)
    kf = PremiumKalman().fit(z, dt)
    p_hat, _ = kf.filter(z, dt)
    g = g.copy(); g["p_hat"] = p_hat
    return g, kf

def grid(g, m, freq):
    """Trading-hours grid; at each point: posted price (from last re-price strictly before),
    shadow price = market fv * exp(premium known from last re-price), gap, event flag."""
    days = g.time.dt.normalize().unique()
    idx = pd.DatetimeIndex(np.concatenate([
        pd.date_range(d + pd.Timedelta("09:00:00"), d + pd.Timedelta("17:30:00"), freq=freq)
        for d in days]))
    G = pd.DataFrame(index=idx)
    G["fv"] = m["fv"].reindex(idx, method="ffill")
    ev = g.set_index("time")
    # last re-price strictly before each grid point
    pos = np.searchsorted(g.time.values, idx.values, side="left") - 1
    ok = pos >= 0
    G = G[ok]; pos = pos[ok]
    G["posted_mid"] = g.bar_mid.values[pos]
    G["posted_buy"] = g.bar_buy.values[pos]
    G["posted_sell"] = g.bar_sell.values[pos]
    G["p_known"] = g.p_hat.values[pos]
    G["same_day"] = g.time.dt.normalize().values[pos] == G.index.normalize().values
    G["shadow"] = G["fv"] * np.exp(G["p_known"])
    G["gap"] = G["shadow"] - G["posted_mid"]
    # did a re-price happen during (t, t+freq]?
    nxt = np.searchsorted(g.time.values, (G.index + pd.Timedelta(freq)).values, side="right")
    cur = np.searchsorted(g.time.values, G.index.values, side="right")
    G["n_events"] = nxt - cur
    return G.dropna()
