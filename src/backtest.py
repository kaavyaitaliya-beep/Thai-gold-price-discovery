import numpy as np, pandas as pd

def backtest(G, threshold):
    """Long-only (Thai retail can't short bars). Enter: gap >= threshold -> buy at posted sell.
    Exit: first point the posted mid reaches the shadow price seen at entry (the catch-up),
    selling at posted buy; otherwise sell at posted buy at the day's last grid point."""
    trades = []
    t, n = 0, len(G)
    gap, shadow = G["gap"].values, G["shadow"].values
    mid, buy, sell = G["posted_mid"].values, G["posted_buy"].values, G["posted_sell"].values
    day = G.index.normalize().values
    while t < n:
        if gap[t] >= threshold:
            entry, target, d = sell[t], shadow[t] - 25, day[t]
            u = t + 1
            while u < n and day[u] == d and mid[u] < target:
                u += 1
            if u >= n or day[u] != d:
                u -= 1  # end of day: exit at close
            trades.append(dict(entry_time=G.index[t], exit_time=G.index[u],
                               pnl=buy[u] - entry, spread=sell[t] - buy[t]))
            t = u + 1
        else:
            t += 1
    return pd.DataFrame(trades)

def summarize(tr):
    if tr.empty: return dict(trades=0)
    return dict(trades=len(tr), win_rate=round(float((tr.pnl > 0).mean()), 3),
                avg_pnl=round(float(tr.pnl.mean()), 1), total_pnl=round(float(tr.pnl.sum()), 0),
                avg_hold_min=round(float((tr.exit_time - tr.entry_time).dt.total_seconds().mean()) / 60, 1))
