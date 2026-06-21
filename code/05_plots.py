"""
Step 5 — publication figures (vector PDF) for the paper.
Reads results_correlation.csv / results_artifacts.csv (from 04_stats.py) plus the raw
pilot_scores_full.csv + metrics.csv for the scatter grid. Writes to ../latex/figs/.

Figures:
  fig_srocc_bars.pdf  : SROCC (|.|) per predictor with 95% bootstrap CI, grouped all/trad/neural.
  fig_scatter.pdf     : score-vs-MOS scatter for {claude,gpt,gemini,VMAF} + P.1401 logistic fit.
  fig_artifacts.pdf   : dominant-artifact distribution by codec class (stacked bars).
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

plt.rcParams.update({"font.size": 11, "axes.spineright": False} if False else {"font.size": 11})
FIGS = "../latex/figs"
os.makedirs(FIGS, exist_ok=True)
# colorblind-friendly (Okabe-Ito)
CB = {"claude": "#E69F00", "gpt": "#56B4E9", "gemini": "#009E73",
      "vmaf": "#0072B2", "ssim": "#D55E00", "psnr": "#CC79A7", "lpips": "#999999"}
PRETTY = {"claude": "Claude", "gpt": "GPT-5.5", "gemini": "Gemini",
          "vmaf": "VMAF", "ssim": "SSIM", "psnr": "PSNR", "lpips": "LPIPS"}

def _logistic(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(np.clip(b2 * (x - b3), -500, 500)))) + b4 * x + b5
def fit_map(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    p0 = [y.max() - y.min(), 1 / (x.std() + 1e-9), x.mean(), 0.0, y.mean()]
    try:
        popt, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=20000)
        xs = np.linspace(x.min(), x.max(), 200); return xs, _logistic(xs, *popt)
    except Exception:
        a, b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 200); return xs, a * xs + b

def fig_bars(corr):
    order = ["vmaf", "gemini", "claude", "gpt", "psnr", "lpips", "ssim"]
    order = [p for p in order if p in set(corr.predictor)]
    subs = ["all", "traditional", "neural"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.26
    for si, sub in enumerate(subs):
        xs, ys, los, his = [], [], [], []
        for pi, p in enumerate(order):
            r = corr[(corr.predictor == p) & (corr.subset == sub)]
            if r.empty: continue
            s = abs(float(r.SROCC.iloc[0]))
            lo, hi = abs(float(r.SROCC_lo.iloc[0])), abs(float(r.SROCC_hi.iloc[0]))
            lo, hi = min(lo, hi), max(lo, hi)
            xs.append(pi + (si - 1) * w); ys.append(s)
            los.append(s - lo); his.append(hi - s)
        ax.bar(xs, ys, width=w, yerr=[los, his], capsize=2.5,
               label=sub.capitalize(), edgecolor="black", linewidth=0.4,
               color=["#0072B2", "#56B4E9", "#9ecae1"][si])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PRETTY[p] for p in order])
    ax.set_ylabel("SROCC vs. MOS  (|·|)")
    ax.set_ylim(0, 1.0); ax.axhline(0.9, ls=":", c="gray", lw=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_srocc_bars.pdf"); plt.close(fig)
    print("wrote fig_srocc_bars.pdf")

def fig_scatter(jagg, mb):
    preds = [("claude", jagg), ("gpt", jagg), ("gemini", jagg), ("vmaf", mb)]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, (p, src) in zip(axs, preds):
        if p in ("claude", "gpt", "gemini"):
            d = src[src.judge == p]; x = d.val.values
        else:
            x = src[p].values
        y = (src[src.judge == p].mos.values if p in ("claude", "gpt", "gemini") else src.mos.values)
        cls = (src[src.judge == p].cls.values if p in ("claude", "gpt", "gemini") else src.cls.values)
        for cc, mk in [("traditional", "o"), ("neural", "^")]:
            m = cls == cc
            ax.scatter(x[m], y[m], s=14, marker=mk, alpha=0.6,
                       color=CB[p] if cc == "traditional" else "#444444",
                       edgecolor="none", label=cc)
        xs, ys = fit_map(x, y); ax.plot(xs, ys, color="black", lw=1.4)
        sr = spearmanr(x, y)[0]
        ax.set_title(f"{PRETTY[p]}  (SROCC={abs(sr):.2f})", fontsize=10)
        ax.set_xlabel("predictor score"); ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("MOS"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_scatter.pdf"); plt.close(fig)
    print("wrote fig_scatter.pdf")

def fig_artifacts(art):
    cats = [c for c in ["texture-blur", "blocking", "ringing", "color-shift", "none"] if c in art.columns]
    colors = {"texture-blur": "#56B4E9", "blocking": "#D55E00", "ringing": "#E69F00",
              "color-shift": "#009E73", "none": "#cccccc"}
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    classes = list(art.index)
    left = np.zeros(len(classes))
    for c in cats:
        vals = art[c].values
        ax.barh(classes, vals, left=left, label=c, color=colors[c], edgecolor="black", linewidth=0.4)
        left += vals
    ax.set_xlabel("number of PVS (dominant artifact per PVS)")
    ax.legend(frameon=False, ncol=len(cats), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_artifacts.pdf"); plt.close(fig)
    print("wrote fig_artifacts.pdf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results_")
    ap.add_argument("--scores", default="pilot_scores_full.csv")
    ap.add_argument("--metrics", default="metrics.csv")
    a = ap.parse_args()
    corr = pd.read_csv(f"{a.prefix}correlation.csv")
    art = pd.read_csv(f"{a.prefix}artifacts.csv", index_col=0)
    js = pd.read_csv(a.scores); met = pd.read_csv(a.metrics)
    meta = js.drop_duplicates("name")[["name", "cls", "content"]]
    jagg = (js.groupby(["name", "judge"]).agg(val=("score", "mean"), mos=("mos", "first"),
            cls=("cls", "first")).reset_index())
    mb = met.merge(meta, on="name", how="inner")
    fig_bars(corr); fig_scatter(jagg, mb); fig_artifacts(art)

if __name__ == "__main__":
    main()
