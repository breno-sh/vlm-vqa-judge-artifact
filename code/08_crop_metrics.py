"""Recompute PSNR/SSIM/LPIPS on the SAME 512x512 crops the judges saw (frames_full),
so the objective baselines are frame-level apples-to-apples with the judges. VMAF stays
video-level (it is a video metric) and is carried over from metrics.csv.
Output: metrics_crop.csv  (name,mos,psnr,ssim,vmaf,lpips)  -- psnr/ssim/lpips = crop-level."""
import csv, glob, os, re
from collections import defaultdict
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch, lpips

dev = "cuda" if torch.cuda.is_available() else "cpu"
loss = lpips.LPIPS(net="alex").to(dev).eval()
print(f"device={dev}", flush=True)

def to_t(img):  # HxWx3 uint8 -> 1x3xHxW in [-1,1]
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return t.to(dev)

pat = re.compile(r"^(.*)__f\d+__c\d+__ref\.png$")
per = defaultdict(lambda: {"psnr": [], "ssim": [], "lpips": []})
pairs = sorted(glob.glob("frames_full/*__ref.png"))
with torch.no_grad():
    for i, ref in enumerate(pairs, 1):
        name = pat.match(os.path.basename(ref)).group(1)
        dis = ref.replace("__ref.png", "__dis.png")
        if not os.path.exists(dis):
            continue
        R = np.asarray(Image.open(ref).convert("RGB"))
        D = np.asarray(Image.open(dis).convert("RGB"))
        per[name]["psnr"].append(psnr(R, D, data_range=255))
        per[name]["ssim"].append(ssim(R, D, channel_axis=2, data_range=255))
        per[name]["lpips"].append(float(loss(to_t(Image.fromarray(R)), to_t(Image.fromarray(D))).item()))
        if i % 100 == 0:
            print(f"  {i}/{len(pairs)}", flush=True)

# carry over mos + video-level VMAF from the original metrics.csv
orig = {r["name"]: r for r in csv.DictReader(open("metrics.csv"))}
rows = []
for name, m in per.items():
    if name not in orig:
        continue
    rows.append({"name": name, "mos": orig[name]["mos"], "vmaf": orig[name]["vmaf"],
                 "psnr": np.mean(m["psnr"]), "ssim": np.mean(m["ssim"]),
                 "lpips": np.mean(m["lpips"])})
with open("metrics_crop.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "mos", "psnr", "ssim", "vmaf", "lpips"])
    w.writeheader(); w.writerows(rows)
print(f"wrote metrics_crop.csv ({len(rows)} PVS) -- psnr/ssim/lpips crop-level, vmaf video-level")
