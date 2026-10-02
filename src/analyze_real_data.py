"""
Stage 1 on real data: clean the association's records, estimate the domestic
premium with the Kalman filter, and chart spread history and re-price activity.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kalman import PremiumKalman
from config import DATA, FIGURES

df = pd.read_csv(DATA / "thai_gold_reprices.csv", parse_dates=["time"]).dropna(
    subset=["gold_spot_usd", "usd_thb"]).reset_index(drop=True)

# --- Cleaning: flag isolated data-entry errors in the committee's spot/FX inputs.
# A row is an error if its premium jumps >100 bp away from its neighbours'
# median (real premium episodes persist across many rows; typos don't).
nbr_med = df["premium_bp"].rolling(9, center=True, min_periods=3).median()
df["entry_error"] = (df["premium_bp"] - nbr_med).abs() > 100
clean = df[~df["entry_error"]].reset_index(drop=True)

# --- Kalman filter on the log-premium, observed at each re-price.
z = np.log(clean["bar_mid"] / clean["intl_fair_value"]).to_numpy()
dt = clean["time"].diff().dt.total_seconds().fillna(3600).to_numpy() / 3600
dt = np.maximum(dt, 1 / 60)  # at least one minute between observations
kf = PremiumKalman().fit(z, dt)
p_hat, P = kf.filter(z, dt)
clean["premium_kalman_bp"] = 1e4 * p_hat
clean["premium_se_bp"] = 1e4 * np.sqrt(P)

# --- Monthly re-price counts, raw and normalised to the Jan-2020 price level.
m = clean.set_index("time").resample("MS").agg(
    reprices=("bar_mid", "size"), mid=("bar_mid", "mean"))
m["reprices_price_adj"] = m["reprices"] * m["mid"].iloc[0] / m["mid"]

if __name__ == "__main__":
    print(f"rows: {len(df)} | flagged entry errors: {int(df['entry_error'].sum())}")
    print(df.loc[df["entry_error"], ["time", "gold_spot_usd", "usd_thb", "premium_bp"]]
          .to_string(index=False))
    print(f"\nKalman: drift sd per hour {1e4*np.sqrt(kf.q):.2f} bp | "
          f"observation noise sd {1e4*np.sqrt(kf.r):.2f} bp "
          f"(50-baht rounding alone ≈ {1e4*50/np.sqrt(12)/clean['bar_mid'].mean():.2f} bp)")
    yr = clean.groupby(clean["time"].dt.year)["premium_kalman_bp"].agg(["median", "min", "max"])
    print("\nKalman premium by year (bp):\n", yr.round(1).to_string())
    yc = m.groupby(m.index.year)[["reprices", "reprices_price_adj"]].mean()
    print("\nAvg re-prices per month:\n", yc.round(0).to_string())
    spread_changes = clean.loc[clean["bar_spread"] != clean["bar_spread"].shift(),
                               ["time", "bar_spread"]]
    print("\nSpread changes:\n", spread_changes.to_string(index=False))

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].scatter(clean["time"], clean["premium_bp"], s=1, c="lightgray", label="observed")
    ax[0].plot(clean["time"], clean["premium_kalman_bp"], lw=1, label="Kalman estimate")
    ax[0].axhline(0, c="k", lw=.5)
    ax[0].set_ylim(-260, 220); ax[0].set_ylabel("premium (bp)")
    ax[0].set_title("Thai bar gold premium over international fair value")
    ax[0].legend(loc="lower right")
    ax[1].step(clean["time"], clean["bar_spread"], where="post")
    ax[1].set_ylabel("baht"); ax[1].set_title("Bar buy/sell spread")
    ax[2].bar(m.index, m["reprices"], width=25, alpha=.35, label="raw")
    ax[2].plot(m.index, m["reprices_price_adj"], lw=1.5, c="C1",
               label="adjusted to Jan-2020 price level")
    ax[2].set_ylabel("re-prices / month"); ax[2].legend(loc="upper left")
    ax[2].set_title("Re-pricing activity")
    fig.tight_layout()
    fig.savefig(FIGURES / "premium_spread_activity.png", dpi=130)
    clean.to_csv(DATA / "thai_gold_clean.csv", index=False)
