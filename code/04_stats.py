"""
Step 4 — full statistical analysis for the paper (reviewer-proof).

Reads pilot_scores_full.csv (VLM judges: name,judge,score,artifact,mos,cls,content,...)
and metrics.csv (per-PVS objective baselines: name,mos,psnr,ssim,vmaf,lpips).
Aggregation unit = PVS (matches per-PVS MOS); judge score = mean over crops/frames/reps.

Produces, for each predictor (3 judges + VMAF/PSNR/SSIM/LPIPS), pooled and split by codec
class (traditional/neural):
  * SROCC (primary, rank-based -> no normality assumption), KROCC.
  * PLCC after ITU-T P.1401 / VQEG 5-parameter logistic mapping (+ RMSE).
  * 95% CI for SROCC via content-level BOOTSTRAP (B=2000) and Fisher-z CI for PLCC.
  * per-content SROCC (averaged) to remove the content confound.
NORMALITY: Shapiro-Wilk on MOS and on logistic-fit residuals -> justifies the
  non-parametric primary metric (Spearman) and the use of bootstrap / Wilcoxon.
SIGNIFICANCE of "judge vs VMAF":
  * Williams's test for two DEPENDENT correlations sharing MOS (+ Holm across baselines).
  * Paired t-test AND Wilcoxon signed-rank on per-PVS absolute prediction error.
Plus inter-judge agreement and artifact distribution by codec class.

Usage: python3 04_stats.py --scores pilot_scores_full.csv --metrics metrics.csv
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy.stats import (pearsonr, spearmanr, kendalltau, shapiro, ttest_rel,
                         wilcoxon, t as tdist, norm)
from scipy.optimize import curve_fit

ALPHA = 0.05

# ---- ITU-T P.1401 / VQEG 5-parameter logistic ----
def _logistic(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(np.clip(b2 * (x - b3), -500, 500)))) + b4 * x + b5
def fit_map(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    p0 = [np.max(y) - np.min(y), 1.0 / (np.std(x) + 1e-9), np.mean(x), 0.0, np.mean(y)]
    try:
        popt, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=20000); return _logistic(x, *popt)
    except Exception:
        a, b = np.polyfit(x, y, 1); return a * x + b

def fisher_ci(r, n, alpha=ALPHA):
    if not np.isfinite(r) or n < 4 or abs(r) >= 1: return (np.nan, np.nan)
    z = np.arctanh(r); se = 1.0 / np.sqrt(n - 3); zc = norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))

def per_content_srocc(df, val):
    vals = [spearmanr(g[val], g.mos)[0] for _, g in df.groupby("content")
            if len(g) >= 3 and g[val].nunique() > 1]
    return (float(np.nanmean(vals)), len(vals)) if vals else (np.nan, 0)

def bootstrap_srocc_ci(df, val, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("content")]
    stats = []
    for _ in range(n):
        samp = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if samp[val].nunique() > 1: stats.append(spearmanr(samp[val], samp.mos)[0])
    return tuple(np.percentile(stats, [2.5, 97.5])) if stats else (np.nan, np.nan)

def williams_test(mos, a, b):
    """H0: corr(mos,a) == corr(mos,b), dependent (share mos). Returns (t, df, p)."""
    mos, a, b = (np.asarray(z, float) for z in (mos, a, b))
    n = len(mos)
    r_ja, r_jb, r_ab = pearsonr(mos, a)[0], pearsonr(mos, b)[0], pearsonr(a, b)[0]
    R = (1 - r_ja**2 - r_jb**2 - r_ab**2) + 2 * r_ja * r_jb * r_ab
    num = (r_ja - r_jb) * np.sqrt((n - 1) * (1 + r_ab))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * R + ((r_ja + r_jb) ** 2) / 4 * (1 - r_ab) ** 3)
    if den == 0 or n < 4: return (np.nan, n - 3, np.nan)
    tt = num / den
    return (float(tt), n - 3, float(2 * (1 - tdist.cdf(abs(tt), n - 3))))

def holm(pvals):
    pvals = list(pvals); order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx]); adj[idx] = min(1.0, run)
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="pilot_scores_full.csv")
    ap.add_argument("--metrics", default="metrics.csv")
    ap.add_argument("--out-prefix", default="results_")
    a = ap.parse_args()

    js = pd.read_csv(a.scores)
    met = pd.read_csv(a.metrics)
    meta = js.drop_duplicates("name")[["name", "cls", "content"]]
    # judge per-PVS = mean score over crops/frames/reps
    jagg = (js.groupby(["name", "judge"]).agg(val=("score", "mean"), mos=("mos", "first"),
            cls=("cls", "first"), content=("content", "first")).reset_index())
    # baselines per-PVS (join cls/content)
    mb = met.merge(meta, on="name", how="inner")

    JUDGES = sorted(jagg.judge.unique())
    BASE = [c for c in ["vmaf", "psnr", "ssim", "lpips"] if c in mb.columns]
    SUBSETS = ["all", "traditional", "neural"]

    def predictor_frame(pred, subset):
        if pred in JUDGES:
            d = jagg[jagg.judge == pred][["name", "val", "mos", "cls", "content"]].copy()
        else:
            d = mb[["name", pred, "mos", "cls", "content"]].rename(columns={pred: "val"}).copy()
        if subset != "all": d = d[d.cls == subset]
        return d.dropna(subset=["val", "mos"])

    # ---------- correlation + CIs table ----------
    rows = []
    for pred in JUDGES + BASE:
        for subset in SUBSETS:
            d = predictor_frame(pred, subset)
            if len(d) < 4 or d.val.nunique() < 2:
                rows.append([pred, subset, len(d), *[np.nan] * 9]); continue
            sc = spearmanr(d.val, d.mos)[0]; kc = kendalltau(d.val, d.mos)[0]
            xh = fit_map(d.val, d.mos); plcc = pearsonr(xh, d.mos)[0]
            rmse = float(np.sqrt(np.mean((xh - d.mos) ** 2)))
            slo, shi = bootstrap_srocc_ci(d, "val")
            plo, phi = fisher_ci(plcc, len(d))
            pc, npc = per_content_srocc(d, "val")
            rows.append([pred, subset, len(d), sc, slo, shi, kc, plcc, plo, phi, rmse, pc])
    corr = pd.DataFrame(rows, columns=["predictor", "subset", "n", "SROCC", "SROCC_lo", "SROCC_hi",
                        "KROCC", "PLCC", "PLCC_lo", "PLCC_hi", "RMSE", "SROCC_per_content"])
    corr.to_csv(f"{a.out_prefix}correlation.csv", index=False)

    # ---------- NORMALITY (Shapiro-Wilk) ----------
    nrows = []
    mos_all = mb.mos.values
    W, p = shapiro(mos_all)
    nrows.append(["MOS", "all", len(mos_all), W, p, "normal" if p > ALPHA else "NON-normal"])
    for pred in JUDGES + BASE:
        d = predictor_frame(pred, "all")
        if len(d) < 4: continue
        resid = d.mos.values - fit_map(d.val, d.mos)
        W, p = shapiro(resid)
        nrows.append([f"resid:{pred}", "all", len(d), W, p, "normal" if p > ALPHA else "NON-normal"])
    norm_df = pd.DataFrame(nrows, columns=["variable", "subset", "n", "shapiro_W", "p", "verdict"])
    norm_df.to_csv(f"{a.out_prefix}normality.csv", index=False)

    # ---------- judge vs VMAF: Williams + paired error tests ----------
    cmp_rows = []
    ref = "vmaf" if "vmaf" in BASE else BASE[0]
    for judge in JUDGES:
        for subset in SUBSETS:
            dj = predictor_frame(judge, subset).rename(columns={"val": "jval"})
            db = predictor_frame(ref, subset).rename(columns={"val": "bval"})
            d = dj.merge(db[["name", "bval"]], on="name")
            if len(d) < 5: continue
            # Williams (dependent correlations sharing MOS) across ALL baselines, Holm
            wt = {}
            pvs = []; labs = []
            for base in BASE:
                dbb = predictor_frame(base, subset).rename(columns={"val": "bv"})
                dd = dj.merge(dbb[["name", "bv"]], on="name")
                if len(dd) < 5 or dd.bv.nunique() < 2: continue
                t, dfree, p = williams_test(dd.mos, dd.jval, dd.bv)
                pvs.append(p); labs.append((base, t, p))
            adj = holm([p for _, _, p in labs]) if labs else []
            for (base, t, p), pa in zip(labs, adj):
                wt[base] = (t, p, pa)
            # paired error vs ref: |map(judge)-mos| vs |map(ref)-mos|
            ej = np.abs(fit_map(d.jval, d.mos) - d.mos)
            eb = np.abs(fit_map(d.bval, d.mos) - d.mos)
            tt_p = ttest_rel(ej, eb).pvalue
            try: w_p = wilcoxon(ej, eb).pvalue
            except ValueError: w_p = np.nan
            wv = wt.get(ref, (np.nan, np.nan, np.nan))
            cmp_rows.append([judge, subset, len(d), float(ej.mean()), float(eb.mean()),
                             wv[0], wv[1], wv[2], tt_p, w_p])
    cmp = pd.DataFrame(cmp_rows, columns=["judge", "subset", "n", "MAE_judge", f"MAE_{ref}",
        f"williams_t_vs_{ref}", f"williams_p_vs_{ref}", f"williams_p_holm",
        "paired_ttest_p", "wilcoxon_p"])
    cmp.to_csv(f"{a.out_prefix}judge_vs_{ref}.csv", index=False)

    # ---------- inter-judge agreement ----------
    wide = jagg.pivot_table(index="name", columns="judge", values="val")
    inter = []
    for i in range(len(JUDGES)):
        for j in range(i + 1, len(JUDGES)):
            dd = wide[[JUDGES[i], JUDGES[j]]].dropna()
            if len(dd) >= 3:
                inter.append([JUDGES[i], JUDGES[j], len(dd), spearmanr(dd.iloc[:, 0], dd.iloc[:, 1])[0]])
    pd.DataFrame(inter, columns=["judge_a", "judge_b", "n", "spearman"]).to_csv(
        f"{a.out_prefix}inter_judge.csv", index=False)

    # ---------- artifact distribution by codec class ----------
    # one dominant artifact per PVS (majority over all judges x crops)
    maj = (js.groupby(["name", "cls"]).artifact
             .agg(lambda s: s.mode().iloc[0]).reset_index())
    art = maj.groupby(["cls", "artifact"]).size().unstack(fill_value=0)
    art.to_csv(f"{a.out_prefix}artifacts.csv")

    # ---------- console summary ----------
    pd.set_option("display.width", 160)
    print("\n===== CORRELATION (predictor x MOS) =====")
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n===== NORMALITY (Shapiro-Wilk) =====")
    print(norm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n===== JUDGE vs {ref.upper()} (Williams + paired error) =====")
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n===== INTER-JUDGE (Spearman) =====")
    print(pd.DataFrame(inter, columns=["a", "b", "n", "spearman"]).to_string(index=False))
    print("\n===== ARTIFACT by codec class =====")
    print(art.to_string())
    print(f"\nwrote {a.out_prefix}{{correlation,normality,judge_vs_{ref},inter_judge,artifacts}}.csv",
          file=sys.stderr)

if __name__ == "__main__":
    main()
