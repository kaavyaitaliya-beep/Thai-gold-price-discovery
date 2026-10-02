import numpy as np
from scipy.optimize import minimize

def excitation(y, day_id, beta):
    """E_t = sum over earlier same-day events of exp(-beta * minutes since event)."""
    E = np.zeros(len(y)); d = np.exp(-beta)
    for t in range(1, len(y)):
        E[t] = 0.0 if day_id[t] != day_id[t-1] else d * (E[t-1] + y[t-1])
    return E

def nll(theta, y, gap, day_id, use_gap, use_hawkes):
    mu = np.exp(theta[0]); a = np.exp(theta[1]) if use_gap else 0.0
    alpha = np.exp(theta[2]) if use_hawkes else 0.0
    beta = np.exp(theta[3]) if use_hawkes else 1.0
    lam = mu + a * gap
    if use_hawkes:
        lam = lam + alpha * excitation(y, day_id, beta)
    return -(y * np.log(lam) - lam).sum()

def fit(y, gap, day_id, use_gap=True, use_hawkes=True):
    x0 = np.log([max(y.mean() / 2, 1e-4), 1e-4, 0.05, 0.1])
    res = minimize(nll, x0, args=(y, gap, day_id, use_gap, use_hawkes), method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-4, fatol=1e-3))
    mu, a, alpha, beta = np.exp(res.x)
    k = 1 + use_gap + 2 * use_hawkes
    out = dict(mu=mu, loglik=-res.fun, aic=2 * k + 2 * res.fun)
    if use_gap: out["a"] = a
    if use_hawkes:
        out.update(alpha=alpha, beta=beta, half_life_min=np.log(2) / beta,
                   branching=alpha * np.exp(-beta) / (1 - np.exp(-beta)))
    return out

def fit_bounded(y, gap, day_id):
    """Gap + Hawkes with bounded params (avoids degenerate solutions)."""
    from scipy.optimize import minimize
    def f(p):
        mu, a, alpha, beta = p
        lam = mu + a * gap + alpha * excitation(y, day_id, beta)
        return -(y * np.log(lam) - lam).sum()
    best = None
    for b0 in [0.02, 0.1, 0.5]:
        r = minimize(f, [1e-4, 0.05, 0.01, b0], method="L-BFGS-B",
                     bounds=[(1e-6, 1), (0, 1), (0, 1), (1e-3, 5)])
        if best is None or r.fun < best.fun: best = r
    mu, a, alpha, beta = best.x
    return dict(mu=mu, a=a, alpha=alpha, beta=beta, loglik=-best.fun, aic=8 + 2 * best.fun,
                branching=alpha * np.exp(-beta) / (1 - np.exp(-beta)), half_life_min=np.log(2) / beta)
