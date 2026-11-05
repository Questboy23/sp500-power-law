"""Clauset-Shalizi-Newman (2009) fit: MLE alpha, x_min by min KS, LR test vs lognormal.
Alpha here is the CCDF exponent (P(X>x) ~ x^-alpha), i.e. the pdf exponent minus 1."""
from __future__ import annotations
import numpy as np, warnings
import powerlaw as _pl


def csn_fit(x, discrete=False) -> dict:
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 50:
        return {"n": len(x), "alpha": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = _pl.Fit(x, discrete=discrete, verbose=False)
        R, p = f.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    xs = np.sort(x)[::-1]
    hill = {}
    for q in (0.10, 0.05):
        k = max(int(len(xs) * q), 5)
        hill[f"hill_alpha_top{int(q*100)}"] = float(1 / np.mean(np.log(xs[:k] / xs[k])))
    return {**hill, "n": len(x), "n_tail": int((x >= f.xmin).sum()), "xmin": float(f.xmin),
            "alpha": float(f.power_law.alpha - 1), "alpha_se": float(f.power_law.sigma),
            "ks": float(f.power_law.D), "lr_pl_vs_ln": float(R), "lr_p": float(p),
            "better_fit": ("power_law" if R > 0 else "lognormal") if p < 0.1 else "indistinguishable"}
