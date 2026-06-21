# Beyond the Score — Zero-Shot VLM Judge for Video-Codec Quality

Reproducibility artifact for the WebMedia 2026 submission *"Beyond the Score: A Zero-Shot
Vision–Language Judge for Interpretable, Artifact-Aware Quality Assessment of Traditional and
Neural Video Codecs."*

> **Anonymized for double-blind review.** Author/institution information is intentionally omitted.

## What this is
Three frontier commercial VLMs (Claude, GPT-5.5, Gemini) are used **zero-shot** as
full-reference perceptual judges of compressed video: given a reference and a reconstructed
crop, each returns a quality score (anchored to the ITU-R BT.500 impairment scale), a
dominant-artifact label, and a one-sentence justification. We validate against human MOS and
compare with PSNR/SSIM/LPIPS (recomputed on the *same* crops) and full-video VMAF.

## Dataset
**AVT-VQDB-UHD-1-NVC** (public; 6 UHD contents × {AV1, VVC, DCVC-FM, DCVC-RT} × quality levels,
216 PVS with per-PVS MOS). Obtain it from the dataset authors' public distribution. Place
`subjective.csv`, `results.json`, `decoded/` and `original/` under `AVT-VQDB-UHD-1-NVC/`.

## Setup
```bash
pip install -r requirements.txt
cp keys.env.example keys.env   # then fill in your own API keys
source keys.env
```
Models are env-overridable: `PAPER51_CLAUDE_MODEL`, `PAPER51_GPT_MODEL`, `PAPER51_GEMINI_MODEL`.
No API keys are stored in this repository; the code reads them from environment variables only.

## Pipeline
```bash
# 1. extract aligned native-resolution crops (parallel; caches the 4K source frame per content)
python code/fast_extract.py --subjective AVT-VQDB-UHD-1-NVC/subjective.csv \
    --pvs-dir AVT-VQDB-UHD-1-NVC/decoded --src-dir AVT-VQDB-UHD-1-NVC/original \
    --out frames_full --n-frames 2 --crops 1 --workers 12
#    (build_manifest.py rebuilds frames_full/manifest.json from the PNGs if extraction is interrupted)

# 2. run the three judges (two-pass: BT.500 score + artifact), resumable
python code/02_run_judges.py --manifest frames_full/manifest.json --out scores.csv \
    --judges claude gpt gemini --two-pass --workers 8

# 3. recompute PSNR/SSIM/LPIPS on the SAME crops (fair, frame-level baselines)
python code/08_crop_metrics.py            # -> metrics_crop.csv  (VMAF carried over, video-level)

# 4. full statistics: SROCC/PLCC/KROCC, bootstrap + Fisher-z CIs, Shapiro-Wilk normality,
#    Williams's test (dependent correlations), paired t / Wilcoxon, inter-judge agreement
python code/04_stats.py --scores scores.csv --metrics metrics_crop.csv

# 5. figures (SROCC bars, scatter vs MOS, artifact distribution, pipeline, qualitative)
python code/05_plots.py --prefix results_ --metrics metrics_crop.csv
python code/07_pipeline.py ; python code/06_qualfig.py
```

## Headline result (all 216 PVS, on matched crops)
Judges SROCC vs MOS: Gemini 0.736, Claude 0.723, GPT-5.5 0.705 — **significantly above**
frame-level PSNR (0.627), LPIPS (0.623), SSIM (0.585) under Williams's test (p<0.01); below only
**full-video** VMAF (0.907). Inter-model Spearman 0.88–0.95. `blocking` labeled exclusively for
traditional codecs. Pre-computed result CSVs are in `results/`.

## Notes
- `pairwise_test.py` reproduces the pilot pairwise-ranking check (did not improve over anchored
  absolute scoring on single crops).
- Commercial model versions drift; the exact returned version string is logged with each verdict.

## License
Code released under the MIT License (see LICENSE). Dataset is the property of its original authors.
