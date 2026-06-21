"""Qualitative figure (balanced: successes AND a failure, to avoid cherry-picking).
Selection rule (disclosed in caption): brightest legible crop within each target category
(neural-blur, traditional-blocking, and the largest judge-vs-MOS over-rating error).
For each: reference vs reconstruction crop + judges' scores/labels + one justification + a
status tag comparing the judge verdict to MOS. -> ../latex/figs/fig_qualitative.pdf"""
import csv, os, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# (name, family label, status string, ok?)
CASES = [
    ("bigbuckbunny_dcvcfm_640x360_q38", "Neural (DCVC-FM)",  "judge tracks MOS", True),
    ("water_av1_1280x720_q61",          "Traditional (AV1)", "judge tracks MOS", True),
    ("bigbuckbunny_vvc_640x360_q34",    "Traditional (VVC)", "judge OVER-rates (failure)", False),
]
rows = list(csv.DictReader(open("pilot_scores_full.csv")))
PRETTY = {"claude": "Claude", "gpt": "GPT-5.5", "gemini": "Gemini"}

def info(name):
    cand = [r for r in rows if r["name"] == name
            if os.path.exists(f"frames_full/{name}__f{r['frame']}__c{r['crop_id']}__ref.png")]
    r = cand[0]
    base = f"frames_full/{name}__f{r['frame']}__c{r['crop_id']}"
    v = {}
    for x in rows:
        if x["name"] == name and x["judge"] not in v:
            v[x["judge"]] = (float(x["score"]), x["artifact"], x["justification"])
    return base + "__ref.png", base + "__dis.png", float(r["mos"]), v

fig, axs = plt.subplots(len(CASES), 3, figsize=(11, 8.0),
                        gridspec_kw={"width_ratios": [1, 1, 1.6]})
for i, (name, label, status, ok) in enumerate(CASES):
    ref, dis, mos, v = info(name)
    axs[i, 0].imshow(Image.open(ref)); axs[i, 0].set_title("Reference", fontsize=11)
    axs[i, 1].imshow(Image.open(dis)); axs[i, 1].set_title("Reconstruction", fontsize=11)
    for j in (0, 1):
        axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
    axs[i, 0].set_ylabel(f"{label}\nMOS={mos:.2f}/5\n({mos*20:.0f}/100)", fontsize=10)
    ax = axs[i, 2]; ax.axis("off")
    art = max(set(a for _, a, _ in v.values()), key=[a for _, a, _ in v.values()].count)
    just = next(jx for _, a, jx in v.values() if a == art)
    jmean = np.mean([s for s, _, _ in v.values()])
    tag = "OK" if ok else "FAIL"
    lines = [f"[{tag}] {status}",
             f"judge mean score: {jmean:.0f}/100   (MOS {mos*20:.0f}/100)",
             f"dominant artifact: {art}", "scores per judge:"]
    for jn in ["claude", "gpt", "gemini"]:
        if jn in v: lines.append(f"   {PRETTY[jn]}: {v[jn][0]:.0f}  [{v[jn][1]}]")
    lines.append("justification (Gemini):")
    lines += textwrap.wrap("“" + just.rstrip(".") + ".”", width=44)
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
            family="monospace", transform=ax.transAxes,
            color="black" if ok else "#a11")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                 lw=0.8, ec="0.6" if ok else "#a11"))

fig.tight_layout()
fig.savefig("../latex/figs/fig_qualitative.pdf")
print("wrote fig_qualitative.pdf")
