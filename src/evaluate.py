"""Evaluation primitives: rank IC, tail AUC, TE/IR, deflated Sharpe, factor alpha, decay fit."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

EULER = 0.5772156649


def rank_ic(signal: pd.Series, fwd: pd.Series) -> float:
    x = pd.concat([signal, fwd], axis=1).dropna()
    if len(x) < 20 or x.iloc[:, 0].nunique() < 2:
        return np.nan
    return stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1]).statistic


def tail_auc(signal: pd.Series, is_tail: pd.Series) -> float:
    x = pd.concat([signal, is_tail.astype(float)], axis=1).dropna()
    pos = x[x.iloc[:, 1] == 1].iloc[:, 0]; neg = x[x.iloc[:, 1] == 0].iloc[:, 0]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return u / (len(pos) * len(neg))


def te_ir(port: pd.Series, bench: pd.Series, periods: int = 12) -> dict:
    a = (port - bench).dropna()
    te = a.std(ddof=1) * np.sqrt(periods)
    ex = a.mean() * periods
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 and a.std() > 0 else np.nan
    return {"excess_ann": ex, "te": te, "ir": ex / te if te > 0 else np.nan, "t": t, "n": len(a)}


def sharpe(r: pd.Series, periods=12) -> float:
    r = r.dropna()
    return r.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std() > 0 else np.nan


def deflated_sharpe(r: pd.Series, n_trials: int, sr_var_across_trials: float | None = None) -> float:
    """Bailey & Lopez de Prado (2014). `r` is per-period (non-annualized) returns.
    Expected max Sharpe under the null across N trials uses the cross-trial variance of
    per-period SRs; if unknown, use 1/T (the variance of an SR estimate under the null)."""
    r = r.dropna(); T = len(r)
    if T < 10 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r); g4 = stats.kurtosis(r, fisher=False)
    v = sr_var_across_trials if sr_var_across_trials else 1.0 / T
    N = max(int(n_trials), 1)
    if N == 1:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(v) * ((1 - EULER) * stats.norm.ppf(1 - 1 / N) + EULER * stats.norm.ppf(1 - 1 / (N * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def factor_alpha(active: pd.Series, factors: pd.DataFrame, periods=12) -> dict:
    import statsmodels.api as sm
    x = pd.concat([active.rename("y"), factors], axis=1).dropna()
    if len(x) < 24:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "n": len(x)}
    m = sm.OLS(x["y"], sm.add_constant(x.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {"alpha_ann": m.params["const"] * periods, "alpha_t": m.tvalues["const"], "n": len(x),
            "betas": m.params.drop("const").round(3).to_dict()}


def fit_decay(lags: np.ndarray, ic: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """IC(L) = IC0 * exp(-L/tau). Log-linear fit on positive ICs is fragile, so use
    nonlinear least squares and bootstrap the (lag, ic) pairs."""
    from scipy.optimize import curve_fit
    f = lambda L, a, tau: a * np.exp(-L / tau)
    ok = np.isfinite(ic)
    lags, ic = np.asarray(lags)[ok], np.asarray(ic)[ok]
    def one(L, y):
        try:
            p, _ = curve_fit(f, L, y, p0=[max(y[0], 1e-3), 10], bounds=([-1, 0.1], [1, 1000]), maxfev=5000)
            return p
        except Exception:
            return [np.nan, np.nan]
    p = one(lags, ic)
    rng = np.random.default_rng(seed); taus = []
    for _ in range(n_boot):
        i = rng.integers(0, len(lags), len(lags))
        if len(set(lags[i])) < 2:
            continue
        taus.append(one(lags[i], ic[i])[1])
    taus = np.array([t for t in taus if np.isfinite(t)])
    hl = lambda t: t * np.log(2)
    return {"ic0": p[0], "tau": p[1], "half_life": hl(p[1]),
            "half_life_ci": (hl(np.percentile(taus, 5)), hl(np.percentile(taus, 95))) if len(taus) > 20 else (np.nan, np.nan)}
