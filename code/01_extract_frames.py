"""
Step 1 — extract aligned NATIVE-RESOLUTION CROP pairs (ref, reconstructed) from
AVT-VQDB-UHD-1-NVC for VLM judging.

Why crops (not full frames): the VLM APIs downscale large images internally, which
would destroy the fine texture-blur that distinguishes neural codecs. We compare at
the DISPLAY resolution (reconstruction upscaled to the UHD source size, as the dataset
is reproduced) and send small native-pixel crops the API ingests without downscaling.

For each sampled frame we take K crops at deterministic, content-derived positions
(seeded by name+frame so runs are reproducible and we are not cherry-picking regions).

manifest.json fields per crop pair: name, content, codec, cls, res, qp, frame, crop_id,
x, y, size, mos, ci, ref, dis.

Usage:
  python 01_extract_frames.py --subjective subjective.csv \
      --pvs-dir ./pvs --src-dir ./src --out ./frames \
      --n-frames 3 --crops 3 --crop-size 512 [--sample 30]

Requires ffmpeg/ffprobe.
"""
import argparse, csv, json, os, subprocess, sys, hashlib
from PIL import Image

def parse_name(name):
    # <content>_<codec>_<WxH>_q##  ; codec may contain a hyphen (dcvc-fm); content may have '_'
    t = name.split("_")
    qp, res, codec = t[-1], t[-2], t[-3]
    content = "_".join(t[:-3])
    cls = "neural" if codec.lower().startswith("dcvc") else "traditional"
    return content, codec, cls, res, qp

def list_sources(src_dir):
    out = {}
    for f in os.listdir(src_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".mp4", ".mkv", ".y4m", ".yuv", ".hevc", ".mov", ".webm"):
            out[stem] = os.path.join(src_dir, f)
    return out

def match_source(content, sources):
    # AVT-VQDB-UHD-1-NVC: source = "<content>_original_3840x2160_q0.mkv"
    cands = [s for s in sources if s.startswith(content + "_")]
    if not cands:  # fallback: any stem starting with the content
        cands = [s for s in sources if s.startswith(content)]
    return sources[min(cands, key=len)] if cands else None

def find_pvs(pvs_dir, name):
    # AVT-VQDB-UHD-1-NVC reconstructions are named "<name>.decoded.mkv"
    for cand in (name + ".decoded.mkv", name + ".mkv"):
        p = os.path.join(pvs_dir, cand)
        if os.path.exists(p):
            return p
    for f in os.listdir(pvs_dir):  # fallback
        if os.path.splitext(f)[0].replace(".decoded", "") == name:
            return os.path.join(pvs_dir, f)
    return None

def probe(path, key):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                        "-show_entries", f"stream={key}", "-of","csv=p=0", path],
                       capture_output=True, text=True)
    return r.stdout.strip()

def nframes(path):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
                        "-show_entries","stream=nb_read_frames","-of","csv=p=0", path],
                       capture_output=True, text=True)
    try: return int(r.stdout.strip())
    except: return 0

def extract_frame(path, idx, out_png, scale_to=None):
    vf = f"select=eq(n\\,{idx})"
    if scale_to:
        vf += f",scale={scale_to[0]}:{scale_to[1]}:flags=bicubic"
    subprocess.run(["ffmpeg","-y","-v","error","-i",path,"-vf",vf,"-vframes","1",out_png], check=True)

def crop_positions(name, frame, W, H, S, k):
    # deterministic pseudo-random top-left positions seeded by name+frame
    h = int(hashlib.sha256(f"{name}:{frame}".encode()).hexdigest(), 16)
    pos = []
    for i in range(k):
        h = (h * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        x = (h >> 16) % max(1, W - S)
        h = (h * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        y = (h >> 16) % max(1, H - S)
        pos.append((x, y))
    return pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjective", required=True)
    ap.add_argument("--pvs-dir", required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp"); os.makedirs(tmp, exist_ok=True)
    sources = list_sources(a.src_dir)
    rows = list(csv.DictReader(open(a.subjective)))
    if a.sample: rows = rows[: a.sample]
    S = a.crop_size

    manifest = []
    for row in rows:
        name = row["name"]
        content, codec, cls, res, qp = parse_name(name)
        pvs, src = find_pvs(a.pvs_dir, name), match_source(content, sources)
        if not pvs or not src:
            print(f"SKIP {name}: pvs={bool(pvs)} src={bool(src)}", file=sys.stderr); continue
        # display resolution = source (UHD) size; reconstruction is upscaled to it
        try:
            W, H = int(probe(src, "width")), int(probe(src, "height"))
        except ValueError:
            print(f"SKIP {name}: cannot probe source size", file=sys.stderr); continue
        total = nframes(pvs)
        if total <= 0 or W < S or H < S:
            print(f"SKIP {name}: frames={total} {W}x{H} < crop {S}", file=sys.stderr); continue
        idxs = [int(total * (k + 1) / (a.n_frames + 1)) for k in range(a.n_frames)]
        for fi in idxs:
            rf = os.path.join(tmp, f"{name}_f{fi}_ref.png")
            df = os.path.join(tmp, f"{name}_f{fi}_dis.png")
            try:
                extract_frame(src, fi, rf)                 # source at native UHD
                extract_frame(pvs, fi, df, scale_to=(W, H))  # reconstruction upscaled to display res
            except subprocess.CalledProcessError:
                print(f"SKIP {name} f{fi}: ffmpeg", file=sys.stderr); continue
            R, D = Image.open(rf).convert("RGB"), Image.open(df).convert("RGB")
            for ci, (x, y) in enumerate(crop_positions(name, fi, W, H, S, a.crops)):
                rcp = os.path.join(a.out, f"{name}__f{fi}__c{ci}__ref.png")
                dcp = os.path.join(a.out, f"{name}__f{fi}__c{ci}__dis.png")
                R.crop((x, y, x + S, y + S)).save(rcp)
                D.crop((x, y, x + S, y + S)).save(dcp)
                manifest.append({"name": name, "content": content, "codec": codec, "cls": cls,
                                 "res": res, "qp": qp, "frame": fi, "crop_id": ci,
                                 "x": x, "y": y, "size": S,
                                 "mos": float(row["mos"]), "ci": float(row.get("ci", "nan") or "nan"),
                                 "ref": rcp, "dis": dcp})
            os.remove(rf); os.remove(df)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"wrote {len(manifest)} crop pairs -> {a.out}/manifest.json")

if __name__ == "__main__":
    main()
