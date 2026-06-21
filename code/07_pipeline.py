"""Pipeline diagram (real figure, replaces the placeholder). Boxes + arrows + real
reference/reconstruction crop thumbnails. -> ../latex/figs/fig_pipeline.pdf"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# a clear, bright crop pair for the thumbnail
EX = "water_av1_1280x720_q61"
rows = list(csv.DictReader(open("pilot_scores_full.csv")))
r = next(x for x in rows if x["name"] == EX
         if os.path.exists(f"frames_full/{EX}__f{x['frame']}__c{x['crop_id']}__ref.png"))
base = f"frames_full/{EX}__f{r['frame']}__c{r['crop_id']}"
ref_img, dis_img = Image.open(base + "__ref.png"), Image.open(base + "__dis.png")

fig, ax = plt.subplots(figsize=(7.4, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

BLUE, GRAY, GREEN, ORANGE = "#0072B2", "#f0f0f0", "#009E73", "#E69F00"
def box(x, y, w, h, text, fc=GRAY, ec="#333", fs=8.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                 fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", color="#111")
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                 lw=1.3, color="#555", shrinkA=2, shrinkB=2))

# Stage 1: source + reconstruction
box(0.5, 12, 17, 11, "PVS\nUHD source +\nreconstruction\n(AV1/VVC/\nDCVC-FM/RT)", fc="#eaf2fb")
arrow(17.5, 17.5, 20.5, 17.5)

# Stage 2: aligned native crops (with real thumbnails, side by side)
box(20.5, 6.5, 23, 21, "", fc="#ffffff", ec="#999")
ax.text(32, 25.2, "aligned native\n512$\\times$512 crops", ha="center", va="center", fontsize=7.6)
for img, xx, lab in [(ref_img, 22.3, "reference"), (dis_img, 32.0, "reconstruction")]:
    inset = ax.inset_axes([xx, 12.5, 9.0, 9.0], transform=ax.transData)
    inset.imshow(img); inset.set_xticks([]); inset.set_yticks([])
    ax.text(xx + 4.5, 10.6, lab, fontsize=6.6, ha="center", va="center")
arrow(43.5, 17.5, 45.5, 17.5)

# Stage 3: VLM judges, two passes
box(45.5, 7, 23, 21, "", fc="#eaf7f1", ec=GREEN)
ax.text(57, 25.6, "Zero-shot VLM judge", ha="center", fontsize=8.2, weight="bold")
ax.text(57, 22.6, "Claude $\\cdot$ GPT-5.5 $\\cdot$ Gemini", ha="center", fontsize=7.4)
box(47.2, 14.5, 19.6, 5.6, "score pass\n(ITU-R BT.500 anchor)", fc="#ffffff", ec="#7bc4a6", fs=7.4)
box(47.2, 8.2, 19.6, 5.6, "artifact pass\n(label + justification)", fc="#ffffff", ec="#7bc4a6", fs=7.4)
arrow(68.5, 17.5, 73, 17.5)

# Stage 4: JSON output
box(73, 11, 17, 13, "JSON\n{score,\nartifact,\njustification}", fc="#fff6e6", ec=ORANGE, fs=7.8)
arrow(81.5, 11, 81.5, 6.5)

# Stage 5: aggregate + validate (bottom, wide)
box(40, 0.3, 45, 5.6,
    "aggregate per-PVS (mean over crops)  $\\rightarrow$  validate vs human MOS\n"
    "& PSNR / SSIM / VMAF / LPIPS  (SROCC, PLCC, Williams, CIs)",
    fc="#eaf2fb", ec=BLUE, fs=7.6)
arrow(73, 3.1, 85.5, 3.1)  # JSON column feeds into the bar (visual continuity)

fig.savefig("../latex/figs/fig_pipeline.pdf", bbox_inches="tight")
print("wrote fig_pipeline.pdf")
