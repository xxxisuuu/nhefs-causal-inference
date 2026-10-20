"""
Does quitting smoking cause weight gain? NHEFS 1971 -> 1982
runs everything + writes figures/*.html (interactive) and *.png

  python src/analysis.py --nboot 500
"""
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from pathlib import Path

import prep
import models

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

C_QUIT = "#E4572E"
C_SMOKE = "#29335C"
C_ACC = "#F3A712"
FONT = dict(family="Helvetica Neue, Arial", size=13, color="#222")
BG = "#FAF9F6"


def save(fig, name, w=1000, h=650):
    fig.write_html(FIG / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(FIG / f"{name}.png", width=w, height=h, scale=2)
    except Exception as e:
        # kaleido needs chrome, set BROWSER_PATH if this fails
        print(f"  (png skipped for {name}: {type(e).__name__})")


def base_layout(fig, title, sub=None):
    t = f"<b>{title}</b>" + (f"<br><span style='font-size:13px;color:#666'>{sub}</span>" if sub else "")
    fig.update_layout(title=dict(text=t, x=0.02, xanchor="left"), font=FONT,
                      paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(t=95, l=70, r=40, b=60))
    return fig


# ----------------------------------------------------------------------------
def fig_ps_overlap(d):
    ps, _ = models.propensity(d)
    w = models.ip_weights(d, censoring=False)
    bins = np.linspace(0, 0.8, 41)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False, horizontal_spacing=0.08,
                        subplot_titles=("Before weighting (counts)", "After IP weighting (weighted counts)"))
    for col, wt in [(1, None), (2, w)]:
        for grp, colr, sign, lab in [(1, C_QUIT, 1, "Quit smoking"), (0, C_SMOKE, -1, "Kept smoking")]:
            m = d.qsmk == grp
            h, e = np.histogram(ps[m], bins=bins, weights=None if wt is None else wt[m])
            mid = (e[:-1] + e[1:]) / 2
            fig.add_bar(x=mid, y=sign * h, marker_color=colr, name=lab, showlegend=(col == 1),
                        width=bins[1] - bins[0], marker_line_width=0, opacity=0.9,
                        hovertemplate=f"{lab}<br>PS %{{x:.2f}}<br>n = %{{customdata:.0f}}<extra></extra>",
                        customdata=np.abs(h), row=1, col=col)
    fig.update_layout(barmode="overlay", bargap=0, legend=dict(orientation="v", y=0.98, x=0.99, xanchor="right"))
    fig.update_xaxes(title_text="Propensity score  P(quit | L)", gridcolor="#eee")
    fig.update_yaxes(zeroline=True, zerolinecolor="#444", gridcolor="#eee", tickformat="d")
    # abs value tick labels -- plotly can't do it directly, set them manually
    for c in (1, 2):
        ys = np.concatenate([t.y for t in fig.data if t.xaxis == ("x" if c == 1 else "x2")])
        top = np.abs(ys).max()
        step = 20 if top < 120 else 50
        tv = np.arange(-np.ceil(top / step) * step, np.ceil(top / step) * step + 1, step)
        fig.update_yaxes(tickvals=tv, ticktext=[str(int(abs(v))) for v in tv], row=1, col=c)
    base_layout(fig, "Quitters and smokers overlap well on the propensity score",
                "Mirrored histograms: quitters up, continuing smokers down. Weighting rebalances the two groups.")
    save(fig, "fig1_ps_overlap", 1150, 560)


def fig_love(d):
    w = models.ip_weights(d, censoring=False)
    # dummies for categoricals
    X = pd.get_dummies(d[prep.CONFOUNDERS], columns=["education", "exercise", "active"], dtype=float)
    nice = {"sex": "Female", "race": "Non-white", "age": "Age", "smokeintensity": "Cigs/day (1971)",
            "smokeyrs": "Years smoking", "wt71": "Weight 1971 (kg)",
            "education_1": "Edu: 8th grade or less", "education_2": "Edu: HS dropout", "education_3": "Edu: HS",
            "education_4": "Edu: college dropout", "education_5": "Edu: college+",
            "exercise_0": "Exercise: much", "exercise_1": "Exercise: moderate", "exercise_2": "Exercise: little/none",
            "active_0": "Active: very", "active_1": "Active: moderate", "active_2": "Active: inactive"}
    rows = []
    for c in X.columns:
        rows.append(dict(var=nice.get(c, c), raw=models.smd(X[c], d.qsmk), wtd=models.smd(X[c], d.qsmk, w)))
    s = pd.DataFrame(rows)
    s["absraw"] = s.raw.abs()
    s = s.sort_values("absraw")

    fig = go.Figure()
    fig.add_vrect(x0=-0.1, x1=0.1, fillcolor="#2a9d8f", opacity=0.08, line_width=0)
    for _, r in s.iterrows():
        fig.add_shape(type="line", x0=r.raw, x1=r.wtd, y0=r["var"], y1=r["var"], line=dict(color="#bbb", width=2))
    fig.add_scatter(x=s.raw, y=s["var"], mode="markers", name="Unweighted",
                    marker=dict(size=11, color="white", line=dict(color=C_QUIT, width=2.5)),
                    hovertemplate="%{y}<br>SMD %{x:.3f}<extra>unweighted</extra>")
    fig.add_scatter(x=s.wtd, y=s["var"], mode="markers", name="IP weighted",
                    marker=dict(size=11, color="#2a9d8f"), hovertemplate="%{y}<br>SMD %{x:.3f}<extra>weighted</extra>")
    fig.add_vline(x=0, line_color="#444", line_width=1)
    fig.add_annotation(x=0.1, y=1.02, yref="paper", text="|SMD| < 0.1", showarrow=False,
                       font=dict(size=11, color="#2a9d8f"), xanchor="left")
    fig.update_xaxes(title="Standardized mean difference (quit - smoke)", gridcolor="#eee", zeroline=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0.55))
    base_layout(fig, "Covariate balance before and after weighting",
                f"Max |SMD| drops from {s.raw.abs().max():.2f} to {s.wtd.abs().max():.2f}")
    save(fig, "fig2_love_plot", 950, 700)
    return s


def fig_estimates(point, boot):
    order = list(models.ESTIMATORS.keys())[::-1]
    fig = go.Figure()
    pal = ["#9AA0A6", "#6C8EAD", "#8E7DBE", C_QUIT, C_ACC, "#2a9d8f"][::-1]
    for i, k in enumerate(order):
        b = boot[k].dropna()
        lo, hi = np.percentile(b, [2.5, 97.5])
        fig.add_violin(x=b, y=[k] * len(b), orientation="h", side="positive", width=1.6, points=False,
                       line_color=pal[i], fillcolor=pal[i], opacity=0.55, name=k, showlegend=False,
                       hoverinfo="skip", meanline_visible=False)
        fig.add_scatter(x=[lo, hi], y=[k, k], mode="lines", line=dict(color="#222", width=2.5), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=[point[k]], y=[k], mode="markers", showlegend=False,
                        marker=dict(size=12, color="white", line=dict(color="#222", width=2.5)),
                        hovertemplate=f"<b>{k}</b><br>{point[k]:.2f} kg (95% CI {lo:.2f} to {hi:.2f})<extra></extra>")
        fig.add_annotation(x=9.6, y=k, text=f"{point[k]:.2f} [{lo:.2f}, {hi:.2f}]", showarrow=False,
                           xanchor="right", font=dict(size=12, color="#333"))
    fig.add_vline(x=0, line_dash="dot", line_color="#888")
    fig.update_xaxes(range=[-0.5, 9.7], title="Effect of quitting on weight change 1971-82 (kg)", gridcolor="#eee")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    base_layout(fig, "Quitting smoking adds about 3.5 kg, once confounding is handled",
                f"Point estimate + bootstrap distribution ({len(boot)} resamples) and percentile 95% CI")
    save(fig, "fig3_estimates", 1050, 620)


def fig_modification(d, nboot=300, seed=7):
    """effect of quitting as a function of baseline smoking intensity (g-formula w/ interaction)"""
    u = d[d.cens == 0]
    f = "wt82_71 ~ qsmk + qsmk:smokeintensity + " + models.OUT_RHS
    grid = np.arange(1, 61)
    m = smf.ols(f, data=u).fit()
    est = m.params["qsmk"] + m.params["qsmk:smokeintensity"] * grid
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(nboot):
        ub = u.sample(len(u), replace=True, random_state=rng.integers(1e9))
        mb = smf.ols(f, data=ub).fit()
        curves.append(mb.params["qsmk"] + mb.params["qsmk:smokeintensity"] * grid)
    curves = np.array(curves)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2], shared_xaxes=True, vertical_spacing=0.04)
    # spaghetti of 40 boot curves, looks nice + shows uncertainty
    for c in curves[:40]:
        fig.add_scatter(x=grid, y=c, mode="lines", line=dict(color=C_QUIT, width=0.6), opacity=0.18,
                        hoverinfo="skip", showlegend=False, row=1, col=1)
    fig.add_scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor="rgba(228,87,46,0.12)",
                    line=dict(width=0), hoverinfo="skip", showlegend=False, row=1, col=1)
    fig.add_scatter(x=grid, y=est, mode="lines", line=dict(color=C_QUIT, width=4), name="Effect",
                    hovertemplate="%{x} cigs/day: %{y:.2f} kg<extra></extra>", showlegend=False, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=1, col=1)
    fig.add_histogram(x=d.smokeintensity, xbins=dict(start=0, end=61, size=2), marker_color=C_SMOKE, opacity=0.7,
                      showlegend=False, hovertemplate="%{x}: %{y} ppl<extra></extra>", row=2, col=1)
    fig.update_yaxes(title_text="Effect (kg)", gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(title_text="n", gridcolor="#eee", row=2, col=1)
    fig.update_xaxes(title_text="Cigarettes per day at baseline (1971)", range=[0, 61], row=2, col=1)
    fig.update_xaxes(gridcolor="#eee")
    sl = m.params["qsmk:smokeintensity"]; p = m.pvalues["qsmk:smokeintensity"]
    base_layout(fig, "Heavier smokers might gain a bit more after quitting, but the evidence is weak",
                f"+{sl:.2f} kg per extra cigarette/day (interaction p = {p:.2f}); band = bootstrap 95% CI. Wide at the tails - few ppl smoke >40/day")
    save(fig, "fig4_effect_by_intensity", 1000, 680)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--figs-only", action="store_true", help="reuse results_bootstrap.csv, skip the slow part")
    args = ap.parse_args()

    d = prep.load()
    print(f"n = {len(d)}, quitters = {d.qsmk.sum()}, censored = {d.cens.sum()}")

    w = models.ip_weights(d)
    print(f"stabilized weights: mean {w.mean():.3f}, range {w.min():.2f}-{w.max():.2f}")   # mean shld be ~1

    point = {k: f(d) for k, f in models.ESTIMATORS.items()}
    print(pd.Series(point).round(2))

    bfile = FIG.parent / "results_bootstrap.csv"
    if args.figs_only and bfile.exists():
        boot = pd.read_csv(bfile)
    else:
        print("bootstrapping... (~5 min for 500)")
        boot = models.bootstrap(d, n_boot=args.nboot)
        boot.to_csv(bfile, index=False)

    res = pd.DataFrame({"estimate": point, "ci_lo": boot.quantile(0.025), "ci_hi": boot.quantile(0.975)})
    res.round(3).to_csv(FIG.parent / "results.csv")
    print(res.round(2))

    fig_ps_overlap(d)
    bal = fig_love(d)
    fig_estimates(point, boot)
    fig_modification(d)
    print("done ->", FIG)
