# Zero-Shot VLM Judge for Compressed-Video Quality Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Anonymous Submission](https://img.shields.io/badge/paper-double--blind%20review-orange.svg)]()

> **Anonymous submission — double-blind review.**
> Authors and institution are intentionally omitted. The de-anonymized version will be published on acceptance.

---

## What is this?

Standard video-quality metrics (PSNR, SSIM, VMAF, LPIPS) tell you *how much* a compressed
frame degrades — but not *why*. Blocking, texture blur, ringing: the artifact type changes
the fix, yet every metric collapses these into a single number.

This work asks: **can an off-the-shelf AI assistant, with zero video-quality training, look
at a reference frame and its compressed version and act as a perceptual judge — rating quality
*and* naming the artifact?**

We tested three frontier vision-language models (**Claude Opus 4.8, GPT-5.5, Gemini 3.5 Flash**)
zero-shot on 216 compressed video sequences covering both **traditional codecs** (AV1, VVC)
and **neural codecs** (DCVC-FM, DCVC-RT), validating every output against human MOS.

---

## Key Results

### Correlation with Human MOS (all 216 PVS, matched per-frame inputs)

| Predictor | SROCC | 95% CI | PLCC | SROCC per-content |
|-----------|------:|--------|-----:|------------------:|
| **VMAF** *(full-video)* | **0.907** | [0.877, 0.952] | **0.911** | 0.940 |
| — | — | — | — | — |
| **Gemini 3.5 Flash** *(ours)* | **0.736** | [0.690, 0.830] | 0.732 | 0.793 |
| **Claude Opus 4.8** *(ours)* | **0.723** | [0.641, 0.826] | 0.719 | 0.811 |
| **GPT-5.5** *(ours)* | **0.705** | [0.643, 0.797] | 0.711 | 0.750 |
| — | — | — | — | — |
| PSNR *(frame-crop)* | 0.627 | [0.550, 0.738] | 0.618 | 0.734 |
| LPIPS *(frame-crop)* | 0.623 | [0.530, 0.773] | 0.629 | 0.695 |
| SSIM *(frame-crop)* | 0.585 | [0.461, 0.716] | 0.560 | 0.621 |

> **The three zero-shot AI judges outperform every classical frame-level metric** (Williams's
> test, p < 0.01), with no training on video quality data. They trail only full-video VMAF,
> which integrates the entire sequence — an advantage the single-crop judge does not have.

### Artifact Detection (something no scalar metric can do)

| Codec family | blocking | texture-blur | color-shift |
|---|---:|---:|---:|
| Traditional (AV1, VVC) | **6 PVS** | 101 | 1 |
| Neural (DCVC-FM, DCVC-RT) | **0 PVS** | 108 | 0 |

Every single blocking call landed on a traditional codec. Not one was wrongly attributed
to a neural codec. Three independent AI vendors agreed with inter-model Spearman **0.88–0.95**.

### Cross-Vendor Agreement

| Pair | Spearman |
|------|----------:|
| Claude × Gemini | 0.891 |
| Claude × GPT-5.5 | 0.882 |
| Gemini × GPT-5.5 | **0.946** |

---

## How It Works

```
PVS (video)
    │
    ▼
aligned 512×512 crop pairs (reference + reconstruction)
    │
    ▼
Zero-shot VLM judge  ──── score pass  (ITU-R BT.500 scale, 0–100)
  Claude / GPT / Gemini ── artifact pass (blocking / texture-blur / color-shift / ringing)
    │
    ▼
JSON  →  aggregate per-PVS  →  compare vs. human MOS + PSNR/SSIM/VMAF/LPIPS
```

Each model is queried in **two passes**:
1. **Score pass** — integer 0–100 anchored to the ITU-R BT.500 five-level impairment scale
2. **Artifact pass** — dominant artifact label from a fixed taxonomy + one-sentence justification

---

## Dataset

**AVT-VQDB-UHD-1-NVC** (public):
- 6 UHD source contents × {AV1, VVC, DCVC-FM, DCVC-RT} × quality levels = **216 PVS**
- Per-PVS human MOS collected under ITU-R BT.500 / ITU-T P.910

Obtain from the dataset authors. Place `subjective.csv`, `results.json`, `decoded/`, and
`original/` under `AVT-VQDB-UHD-1-NVC/`.

---

## Setup

```bash
pip install -r requirements.txt
cp keys.env.example keys.env   # fill in your API keys
source keys.env
```

Models are overridable via environment variables:
`PAPER51_CLAUDE_MODEL`, `PAPER51_GPT_MODEL`, `PAPER51_GEMINI_MODEL`

No API keys are stored in this repository.

---

## Reproducing the Results

```bash
# 1. Extract aligned native-resolution 512×512 crops
python code/fast_extract.py \
    --subjective AVT-VQDB-UHD-1-NVC/subjective.csv \
    --pvs-dir    AVT-VQDB-UHD-1-NVC/decoded \
    --src-dir    AVT-VQDB-UHD-1-NVC/original \
    --out        frames_full --n-frames 2 --crops 1 --workers 12

# 2. Run the three VLM judges (resumable; ~216 × 3 API calls)
python code/02_run_judges.py \
    --manifest frames_full/manifest.json \
    --out      scores.csv \
    --judges   claude gpt gemini \
    --two-pass --workers 8

# 3. Recompute PSNR / SSIM / LPIPS on the same crops
python code/08_crop_metrics.py   # → metrics_crop.csv

# 4. Statistics: SROCC/PLCC/KROCC, bootstrap + Fisher-z CIs,
#    Shapiro-Wilk normality, Williams's test, paired t / Wilcoxon
python code/04_stats.py \
    --scores  scores.csv \
    --metrics metrics_crop.csv

# 5. Generate all paper figures
python code/05_plots.py
python code/06_qualfig.py
python code/07_pipeline.py
```

Pre-computed results are in `results/` — no API keys needed to verify the numbers.

---

## Repository Structure

```
├── code/
│   ├── fast_extract.py             # crop extraction (parallelized, 4K-safe)
│   ├── 02_run_judges.py            # VLM querying (two-pass, resumable)
│   ├── 08_crop_metrics.py          # PSNR/SSIM/LPIPS on same crops
│   ├── 04_stats.py                 # full statistical analysis
│   ├── 05_plots.py                 # SROCC bar chart, scatter vs MOS
│   ├── 06_qualfig.py               # qualitative figure
│   └── pairwise_test.py            # pilot pairwise ranking check
├── results/
│   ├── results_correlation.csv     # SROCC/PLCC/KROCC per predictor + subset
│   ├── results_artifacts.csv       # artifact label distribution by codec family
│   ├── results_inter_judge.csv     # cross-vendor Spearman correlations
│   ├── results_normality.csv       # Shapiro-Wilk normality tests
│   ├── results_judge_vs_vmaf.csv   # paired error tests (judge vs VMAF)
│   └── gemini_reproducibility_R3.csv  # test-retest stability (R=3)
├── keys.env.example                # API key template
├── requirements.txt
└── LICENSE                         # MIT
```

---

## Limitations

- **VMAF gap is real**: the judges do not match full-video VMAF (0.91), which sees the entire
  sequence. On per-frame input the comparison is fair; temporal artifacts are not directly observed.
- **Resolution bias**: judges over-rate upscaled low-resolution clips (+0.49 SD at ≤360p) and
  under-rate native 4K (−0.22 SD). The most systematic failure mode identified.
- **Low blocking recall**: blocking is high-precision (0 false positives on neural codecs) but
  low-recall (detected in only 6 of the traditional PVS).
- **Single dataset**: all results come from AVT-VQDB-UHD-1-NVC. Generalization to other content
  and codec types remains open.
- **Prompt sensitivity**: the two-pass anchored design was chosen based on pilot data; a differently
  engineered prompt could shift the absolute numbers.

---

## License

Code released under the [MIT License](LICENSE).
Dataset (AVT-VQDB-UHD-1-NVC) is the property of its original authors — see their distribution terms.
