"""
Thai Gold Market Making Model — Stage 1: Fair value via Kalman filter.

Model (log space):
    international fair value  F_t = XAU_USD * USDTHB * (15.244 / 31.1035) * 0.965
    hidden log-premium        p_t = p_{t-1} + w_t,     w_t ~ N(0, q * dt)
    observation at re-price   z_t = log(posted_t) - log(F_t) = p_t + v_t,  v_t ~ N(0, r)

Only association re-price events are used as observations: between updates the
posted price is stale and would bias the premium estimate.
The "shadow price" F_t * exp(p_hat_t), rounded to 50 baht, is what the
association would post if it updated right now.
"""
import numpy as np
from config import FIGURES
from scipy.optimize import minimize

BAHT_WEIGHT_G = 15.244
TROY_OZ_G = 31.1034768
PURITY = 0.965
STEP = 50.0  # association prices move in 50-baht steps


def intl_fair_value(xau_usd, usdthb):
    """International gold price in THB per baht-weight of 96.5% gold."""
    return xau_usd * usdthb * (BAHT_WEIGHT_G / TROY_OZ_G) * PURITY


def round_step(x):
    return np.round(x / STEP) * STEP


class PremiumKalman:
    """Local-level Kalman filter on the log-premium, with irregular time gaps."""

    def __init__(self, q=None, r=None):
        self.q, self.r = q, r

    @staticmethod
    def _filter(z, dt, q, r, p0=None, P0=1e-2):
        n = len(z)
        p_hat, P = np.empty(n), np.empty(n)
        p, var = (z[0] if p0 is None else p0), P0
        nll = 0.0
        for t in range(n):
            var_pred = var + q * dt[t]           # predict: premium drifts
            S = var_pred + r                     # innovation variance
            innov = z[t] - p
            nll += 0.5 * (np.log(2 * np.pi * S) + innov**2 / S)
            K = var_pred / S                     # Kalman gain
            p = p + K * innov                    # update
            var = (1 - K) * var_pred
            p_hat[t], P[t] = p, var
        return p_hat, P, nll

    def fit(self, z, dt):
        """Estimate q (drift variance per hour) and r (obs noise) by MLE."""
        def nll(theta):
            q, r = np.exp(theta)
            return self._filter(z, dt, q, r)[2]
        res = minimize(nll, x0=np.log([1e-7, 1e-7]), method="Nelder-Mead")
        self.q, self.r = np.exp(res.x)
        return self

    def filter(self, z, dt):
        return self._filter(z, dt, self.q, self.r)[:2]


def simulate(hours=24 * 120, seed=0, xau0=3500.0, fx0=33.0,
             vol_xau=0.002, vol_fx=0.0006, q_true=4e-8, reprice_prob=0.25):
    """Synthetic market: continuous intl price, drifting premium, 50-baht steps."""
    rng = np.random.default_rng(seed)
    xau = xau0 * np.exp(np.cumsum(rng.normal(0, vol_xau, hours)))
    fx = fx0 * np.exp(np.cumsum(rng.normal(0, vol_fx, hours)))
    F = intl_fair_value(xau, fx)
    p_true = 0.002 + np.cumsum(rng.normal(0, np.sqrt(q_true), hours))
    shadow = F * np.exp(p_true)

    posted = np.empty(hours)
    posted[0] = round_step(shadow[0])
    events = [0]
    for t in range(1, hours):
        posted[t] = posted[t - 1]
        if abs(shadow[t] - posted[t]) >= STEP and rng.random() < reprice_prob:
            posted[t] = round_step(shadow[t])
            events.append(t)
    return dict(F=F, p_true=p_true, shadow=shadow, posted=posted,
                events=np.array(events), q_true=q_true)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = simulate()
    ev = sim["events"]
    z = np.log(sim["posted"][ev]) - np.log(sim["F"][ev])
    dt = np.diff(ev, prepend=ev[0] - 1).astype(float)

    kf = PremiumKalman().fit(z, dt)
    p_hat, P = kf.filter(z, dt)

    # Rounding to 50 baht is ~uniform noise of +-25 baht -> variance (50^2/12)/price^2
    r_implied = (STEP**2 / 12) / np.mean(sim["posted"][ev]) ** 2
    rmse_bp = 1e4 * np.sqrt(np.mean((p_hat - sim["p_true"][ev]) ** 2))
    rmse_raw_bp = 1e4 * np.sqrt(np.mean((z - sim["p_true"][ev]) ** 2))

    print(f"re-price events: {len(ev)} over {len(sim['F'])} hours")
    print(f"q: true {sim['q_true']:.2e} | estimated {kf.q:.2e}")
    print(f"r: rounding-implied {r_implied:.2e} | estimated {kf.r:.2e}")
    print(f"premium RMSE: raw observations {rmse_raw_bp:.2f} bp -> Kalman {rmse_bp:.2f} bp")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(1e4 * sim["p_true"], lw=1, label="true premium")
    ax[0].scatter(ev, 1e4 * z, s=4, c="gray", alpha=.5, label="observed at re-price")
    ax[0].plot(ev, 1e4 * p_hat, lw=1.2, label="Kalman estimate")
    ax[0].set_ylabel("premium (bp)"); ax[0].legend(loc="upper left")
    w = slice(0, 24 * 10)
    hrs = np.arange(len(sim["F"]))[w]
    ax[1].plot(hrs, sim["shadow"][w], lw=1, label="true shadow price")
    ax[1].step(hrs, sim["posted"][w], where="post", lw=1.2, label="posted (50-baht steps)")
    ax[1].set_ylabel("THB / baht-weight"); ax[1].set_xlabel("hour")
    ax[1].legend(loc="upper left"); ax[1].set_title("First 10 days: stale posted price vs shadow price")
    fig.tight_layout()
    fig.savefig(FIGURES / "kalman_validation_synthetic.png", dpi=130)
