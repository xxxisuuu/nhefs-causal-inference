import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import (ConvergenceWarning, PerfectSeparationError,
                                             PerfectSeparationWarning)

# censoring model occasionally doesnt converge in a bootstrap sample (few events). estimates still fine
warnings.simplefilter("ignore", ConvergenceWarning)

# quadratic terms for continuous vars like in the book
PS_FORMULA = ("qsmk ~ C(sex) + C(race) + age + I(age**2) + C(education) + smokeintensity"
              " + I(smokeintensity**2) + smokeyrs + I(smokeyrs**2) + C(exercise) + C(active)"
              " + wt71 + I(wt71**2)")
OUT_RHS = PS_FORMULA.split("~")[1]
OUT_FORMULA = "wt82_71 ~ qsmk + qsmk:smokeintensity + " + OUT_RHS

# one truncation rule for every estimator that divides by the PS (IPW, AIPW).
# observed PS range is ~0.05-0.79, so at 0.01 this is a safety net, not an adjustment
PS_TRIM = 0.01

# errors a bootstrap resample is allowed to hit (separation in the logit models, singular design).
# anything else is a real bug and should crash
BOOT_OK_ERRORS = (PerfectSeparationError, np.linalg.LinAlgError)


def _trim(p, trim=PS_TRIM):
    return p.clip(trim, 1 - trim) if trim else p


def propensity(d, trim=PS_TRIM):
    m = smf.logit(PS_FORMULA, data=d).fit(disp=0)
    return _trim(m.predict(d), trim), m


def censoring_probs(d):
    """P(uncensored | A, L) and the marginal P(uncensored | A) for the stabilized numerator"""
    cm = smf.logit("cens ~ qsmk + " + OUT_RHS, data=d).fit(disp=0)
    p_unc = 1 - cm.predict(d)
    num = 1 - smf.logit("cens ~ qsmk", data=d).fit(disp=0).predict(d)
    return p_unc, num


def ip_weights(d, stabilized=True, censoring=True, trim=PS_TRIM):
    ps, _ = propensity(d, trim)
    p_trt = d.qsmk.mean()
    if stabilized:
        w = np.where(d.qsmk == 1, p_trt / ps, (1 - p_trt) / (1 - ps))
    else:
        w = np.where(d.qsmk == 1, 1 / ps, 1 / (1 - ps))
    w = pd.Series(w, index=d.index)

    if censoring:
        # IPCW: P(uncensored | A, L) -- book uses same covariates + qsmk
        p_unc, num = censoring_probs(d)
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


def _outcome_model(d):
    """outcome model fit on the uncensored, weighted by 1/P(uncensored | A, L) so it targets
    the full baseline population (same censoring handling as IPW)"""
    p_unc, _ = censoring_probs(d)
    u = d[d.cens == 0].copy()
    u["wc"] = 1 / p_unc[u.index]
    return smf.wls(OUT_FORMULA, data=u, weights=u.wc).fit(), p_unc


def est_gformula(d):
    """standardization / parametric g-formula over all 1,629 (censored included).
    outcome model w/ qsmk x smokeintensity interaction, IPCW-weighted"""
    m, _ = _outcome_model(d)
    d1 = d.copy(); d1["qsmk"] = 1
    d0 = d.copy(); d0["qsmk"] = 0
    return m.predict(d1).mean() - m.predict(d0).mean()


def est_aipw(d):
    """doubly robust with censoring: the residual term is weighted by 1 / [P(A|L) P(uncensored|A,L)]
    and averaged over everyone, censored people contribute only through the outcome model"""
    ps, _ = propensity(d)
    m, p_unc = _outcome_model(d)
    d1 = d.copy(); d1["qsmk"] = 1
    d0 = d.copy(); d0["qsmk"] = 0
    mu1, mu0 = m.predict(d1), m.predict(d0)
    a, c = d.qsmk, d.cens
    y = d.wt82_71.fillna(0)   # only used where c == 0
    unc = (1 - c) / p_unc
    psi1 = mu1 + a * unc * (y - mu1) / ps
    psi0 = mu0 + (1 - a) * unc * (y - mu0) / (1 - ps)
    return (psi1 - psi0).mean()


def psmatch(d, caliper_sd=0.2):
    """1:1 nearest neighbour on logit PS, with replacement, caliper = caliper_sd x SD(logit PS).
    complete case, targets the ATT. returns (estimate, info dict)"""
    u = d[d.cens == 0].copy()
    ps = smf.logit(PS_FORMULA, data=u).fit(disp=0).predict(u)
    lps = np.log(ps / (1 - ps))
    t_idx = np.where(u.qsmk == 1)[0]; c_idx = np.where(u.qsmk == 0)[0]
    lt, lc = lps.values[t_idx], lps.values[c_idx]
    cal = caliper_sd * lps.std()
    # brute force is ok, n is small
    dist = np.abs(lt[:, None] - lc[None, :])
    j = dist.argmin(axis=1)
    ok = dist[np.arange(len(t_idx)), j] < cal
    y = u.wt82_71.values
    est = (y[t_idx[ok]] - y[c_idx[j[ok]]]).mean()
    info = dict(n_treated=len(t_idx), n_unmatched=int((~ok).sum()), caliper=cal,
                treated=u.index[t_idx[ok]], controls=u.index[c_idx[j[ok]]])
    return est, info


def est_psmatch(d, caliper_sd=0.2):
    return psmatch(d, caliper_sd)[0]


ESTIMATORS = {
    "Unadjusted": est_naive,
    "Outcome regression": est_regression,
    "PS matching (ATT)": est_psmatch,
    "IP weighting": est_ipw,
    "G-formula": est_gformula,
    "AIPW (doubly robust)": est_aipw,
}

# what each estimator targets. matching drops controls and keeps quitters -> ATT
TARGET = {k: "ATE" for k in ESTIMATORS}
TARGET["PS matching (ATT)"] = "ATT"
TARGET["Unadjusted"] = "crude"
CENSORING = {"Unadjusted": "complete case", "Outcome regression": "complete case",
             "PS matching (ATT)": "complete case", "IP weighting": "IPCW",
             "G-formula": "IPCW", "AIPW (doubly robust)": "IPCW"}


def bootstrap(d, n_boot=500, seed=11):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ESTIMATORS}
    fails = {k: 0 for k in ESTIMATORS}
    with warnings.catch_warnings():
        # statsmodels >= 0.14 warns (not raises) on separation; make it an error so it is counted
        warnings.simplefilter("error", PerfectSeparationWarning)
        for b in range(n_boot):
            db = d.sample(len(d), replace=True, random_state=rng.integers(1e9)).reset_index(drop=True)
            for k, f in ESTIMATORS.items():
                try:
                    out[k].append(f(db))
                except (PerfectSeparationWarning, *BOOT_OK_ERRORS):
                    out[k].append(np.nan)
                    fails[k] += 1
            if b % 100 == 0:
                print(f"  boot {b}/{n_boot}")
    bad = {k: v for k, v in fails.items() if v}
    print("  failed resamples:", bad if bad else "none")
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


# --- sensitivity to unmeasured confounding -------------------------------------------
def _evalue_rr(rr):
    rr = np.where(rr < 1, 1 / rr, rr)
    return rr + np.sqrt(rr * (rr - 1))


def evalue(est, lo, hi, sd_y):
    """E-value for a mean difference (VanderWeele & Ding 2017): standardize by SD(Y),
    approximate RR = exp(0.91 * d). returns (E-value for the estimate, for the CI limit nearest 1).
    the CI E-value is 1 when the CI crosses 0"""
    rr = np.exp(0.91 * np.array([est, lo, hi]) / sd_y)
    e_est = float(_evalue_rr(rr[0]))
    if lo <= 0 <= hi:
        e_ci = 1.0
    else:
        near = rr[1] if est > 0 else rr[2]
        e_ci = float(_evalue_rr(near))
    return e_est, e_ci
