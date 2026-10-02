"""Final results: lag, re-pricing model (gap vs Hawkes), direction, backtest."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from final_common import load, market_premium, grid
from hawkes import fit, fit_bounded
from backtest import backtest, summarize
from config import FIGURES

# 1. How closely do the committee's inputs track the live market?
g, m = load("market_1m.csv"); g, kf = market_premium(g, m)
corr = {}
for k in range(0, 11):
    mk = m["gold_usd"].reindex(g.time - pd.Timedelta(minutes=k), method="ffill").values
    corr[k] = np.corrcoef(np.diff(g.gold_spot_usd.values), np.diff(mk))[0, 1]
best_lag = max(corr, key=corr.get)

# 2. Staleness of the posted price, minute by minute
G = grid(g, m, "1min"); G = G[G.same_day]
a = G.gap.abs()

# 3. Direction: does the sign of the gap predict the next re-price?
pos = np.searchsorted(G.index.values, g.time.values, side="left") - 1
ok = pos >= 0
chg, gp = g.change.values[ok], G.gap.values[pos[ok]]
nz = chg != 0
direction_acc = (np.sign(chg[nz]) == np.sign(gp[nz])).mean()

# 4. When: gap-driven vs self-exciting (Hawkes)
y = G.n_events.values.astype(float); x = a.values / 50
day = G.index.normalize().factorize()[0]
m_gap = fit(y, x, day, use_hawkes=False)
m_hawkes = fit(y, x, day, use_gap=False)
m_both = fit_bounded(y, x, day)

# 5. Backtests (minute: Sep 2026; hourly: May 2024 - Sep 2026)
bt_min = {th: summarize(backtest(G, th)) for th in [50, 100, 150, 200]}
gh, mh = load("market_1h.csv"); gh, _ = market_premium(gh, mh)
H = grid(gh, mh, "1h"); H = H[H.same_day]
bt_h = {sp: {th: summarize(backtest(H[mask], th)) for th in [100, 200, 300, 400]}
        for sp, mask in [(100, H.index < "2026-02-02"), (200, H.index >= "2026-02-02")]}

if __name__ == "__main__":
    print(f"Sep 2026 re-prices: {len(g)} | minutes analysed: {len(G)}")
    print(f"Committee inputs vs market: best lag {best_lag} min (corr {corr[best_lag]:.3f}; "
          f"0 min {corr[0]:.3f}, 10 min {corr[10]:.3f})")
    print(f"Posted-price gap: median {a.median():.0f} baht | >=50: {(a>=50).mean():.1%} "
          f"| >=100: {(a>=100).mean():.1%} | >=200: {(a>=200).mean():.2%} | max {a.max():.0f}")
    print(f"Direction accuracy from gap sign: {direction_acc:.1%} of {nz.sum()} re-prices")
    for name, r in [("gap only", m_gap), ("Hawkes only", m_hawkes), ("gap + Hawkes", m_both)]:
        print(f"{name:13s} AIC {r['aic']:.0f}" + (f" | branching {r['branching']:.2f}, half-life "
              f"{r['half_life_min']:.0f} min" if "branching" in r else "") +
              (f" | per-minute prob per 50 baht of gap {r['a']:.3f}" if "a" in r else ""))
    print("Backtest, minute data (spread 200):"); [print(" ", k, v) for k, v in bt_min.items()]
    for sp, d in bt_h.items():
        print(f"Backtest, hourly data (spread {sp}):"); [print(" ", k, v) for k, v in d.items()]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(a.clip(upper=300), bins=60, color="C0")
    for v, lab in [(50, "1 step"), (200, "spread")]:
        ax[0].axvline(v, ls="--", c="k" if v == 200 else "gray"); ax[0].text(v + 4, ax[0].get_ylim()[1]*.9, lab)
    ax[0].set_xlabel("|shadow − posted| (baht)"); ax[0].set_ylabel("minutes")
    ax[0].set_title("How stale is the posted price? (Sep 2026)")
    bins = np.arange(0, 201, 25); b = pd.cut(a, bins)
    emp = G.groupby(b, observed=True)["n_events"].apply(lambda s: (s > 0).mean())
    ctr = bins[:-1] + 12.5
    ax[1].bar(ctr, emp.values, width=20, alpha=.5, label="observed")
    xs = np.linspace(0, 200, 100)
    ax[1].plot(xs, 1 - np.exp(-(m_gap["mu"] + m_gap["a"] * xs / 50)), c="C1", lw=2, label="gap model")
    ax[1].set_xlabel("|gap| (baht)"); ax[1].set_ylabel("P(re-price next minute)")
    ax[1].set_title("Re-pricing is driven by the gap"); ax[1].legend()
    ths = list(bt_min); ax[2].plot(ths, [bt_min[t]["avg_pnl"] for t in ths], marker="o", label="minute, spread 200")
    for sp, d in bt_h.items():
        t2 = list(d); ax[2].plot(t2, [d[t]["avg_pnl"] for t in t2], marker="s", label=f"hourly, spread {sp}")
    ax[2].axhline(0, c="k", lw=.7); ax[2].set_xlabel("entry threshold (baht)")
    ax[2].set_ylabel("avg P&L per trade (baht)"); ax[2].set_title("Can you profit from the lag?"); ax[2].legend()
    fig.tight_layout(); fig.savefig(FIGURES / "final_results.png", dpi=130)
