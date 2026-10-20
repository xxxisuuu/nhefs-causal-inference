import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# censoring model occasionally doesnt converge in a bootstrap sample (few events). estimates still fine
warnings.simplefilter("ignore", ConvergenceWarning)

# quadratic terms for continuous vars like in the book
PS_FORMULA = ("qsmk ~ C(sex) + C(race) + age + I(age**2) + C(education) + smokeintensity"
              " + I(smokeintensity**2) + smokeyrs + I(smokeyrs**2) + C(exercise) + C(active)"
              " + wt71 + I(wt71**2)")
OUT_RHS = PS_FORMULA.split("~")[1]


def propensity(d):
    m = smf.logit(PS_FORMULA, data=d).fit(disp=0)
    return m.predict(d), m


def ip_weights(d, stabilized=True, censoring=True):
    ps, _ = propensity(d)
    p_trt = d.qsmk.mean()
    if stabilized:
        w = np.where(d.qsmk == 1, p_trt / ps, (1 - p_trt) / (1 - ps))
    else:
        w = np.where(d.qsmk == 1, 1 / ps, 1 / (1 - ps))
    w = pd.Series(w, index=d.index)

    if censoring:
        # IPCW: P(uncensored | A, L) -- book uses same covariates + qsmk
        cm = smf.logit("cens ~ qsmk + " + OUT_RHS, data=d).fit(disp=0)
        p_unc = 1 - cm.predict(d)
        num = 1 - smf.logit("cens ~ qsmk", data=d).fit(disp=0).predict(d)
        w = w * (num / p_unc if stabilized else 1 / p_unc)
    return w


def est_naive(d):
    u = d[d.cens == 0]
    return u.loc[u.qsmk == 1, "wt82_71"].mean() - u.loc[u.qsmk == 0, "wt82_71"].mean()


def est_regression(d):
    u = d[d.cens == 0]
    m = smf.ols("wt82_71 ~ qsmk + " + OUT_RHS, data=u).fit()
    return m.params["qsmk"]


def est_ipw(d):
    w = ip_weights(d)
    u = d[d.cens == 0].copy()
    u["w"] = w[u.index]
    # weighted MSM, wls is fine for point est (bootstrap does the CI)
    m = smf.wls("wt82_71 ~ qsmk", data=u, weights=u.w).fit()
    return m.params["qsmk"]


def est_gformula(d):
    """standardization / parametric g-formula. outcome model w/ qsmk x smokeintensity interaction"""
    u = d[d.cens == 0]
    m = smf.ols("wt82_71 ~ qsmk + qsmk:smokeintensity + " + OUT_RHS, data=u).fit()
    d1 = d.copy(); d1["qsmk"] = 1
    d0 = d.copy(); d0["qsmk"] = 0
    return m.predict(d1).mean() - m.predict(d0).mean()


def est_aipw(d):
    # doubly robust. ignoring censoring here (only 63) -- noted in README
    u = d[d.cens == 0].copy()
    ps = smf.logit(PS_FORMULA, data=u).fit(disp=0).predict(u)
    ps = ps.clip(0.01, 0.99)
    m = smf.ols("wt82_71 ~ qsmk + qsmk:smokeintensity + " + OUT_RHS, data=u).fit()
    u1 = u.copy(); u1["qsmk"] = 1
    u0 = u.copy(); u0["qsmk"] = 0
    mu1, mu0 = m.predict(u1), m.predict(u0)
    a, y = u.qsmk, u.wt82_71
    psi1 = mu1 + a * (y - mu1) / ps
    psi0 = mu0 + (1 - a) * (y - mu0) / (1 - ps)
    return (psi1 - psi0).mean()


def est_psmatch(d, caliper=0.05):
    """1:1 nearest neighbour on logit PS, with replacement. ATT"""
    u = d[d.cens == 0].copy()
    ps = smf.logit(PS_FORMULA, data=u).fit(disp=0).predict(u)
    lps = np.log(ps / (1 - ps))
    t_idx = np.where(u.qsmk == 1)[0]; c_idx = np.where(u.qsmk == 0)[0]
    lt, lc = lps.values[t_idx], lps.values[c_idx]
    cal = caliper * lps.std() * 4   # ~0.2 sd of logit ps, rough
    # brute force is ok, n is small
    dist = np.abs(lt[:, None] - lc[None, :])
    j = dist.argmin(axis=1)
    ok = dist[np.arange(len(t_idx)), j] < cal
    y = u.wt82_71.values
    return (y[t_idx[ok]] - y[c_idx[j[ok]]]).mean()


ESTIMATORS = {
    "Unadjusted": est_naive,
    "Outcome regression": est_regression,
    "PS matching (ATT)": est_psmatch,
    "IP weighting": est_ipw,
    "G-formula": est_gformula,
    "AIPW (doubly robust)": est_aipw,
}


def bootstrap(d, n_boot=500, seed=11):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ESTIMATORS}
    for b in range(n_boot):
        db = d.sample(len(d), replace=True, random_state=rng.integers(1e9)).reset_index(drop=True)
        for k, f in ESTIMATORS.items():
            try:
                out[k].append(f(db))
            except Exception as e:   # perfect separation etc. rare
                out[k].append(np.nan)
        if b % 100 == 0:
            print(f"  boot {b}/{n_boot}")
    return pd.DataFrame(out)


def smd(x, t, w=None):
    """standardized mean difference, weighted if w given"""
    if w is None:
        w = np.ones(len(x))
    x, t, w = np.asarray(x, float), np.asarray(t), np.asarray(w, float)
    m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1])
    v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)
