"""
Fast PARALLEL frame/crop extractor for the full set. Key insight: the SOURCE clips are
4K LOSSLESS (~2GB) and decoding to a frame is the bottleneck — so we decode each source
frame ONCE per content and crop every PVS of that content from the cached full frame
(instead of re-decoding the 2GB source ~36x per content). The distorted (decoded) frames
are cheap and decoded per-PVS, frame-accurate (select=eq), so ref/dis stay aligned.

Produces manifest.json compatible with 02_run_judges.py.

Usage: python3 fast_extract.py --subjective ../AVT-VQDB-UHD-1-NVC/subjective.csv \
   --pvs-dir ../AVT-VQDB-UHD-1-NVC/decoded --src-dir ../AVT-VQDB-UHD-1-NVC/original \
   --out ./frames_full --n-frames 2 --crops 1 --workers 12
"""
import argparse, csv, importlib.util, json, os, subprocess, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

spec = importlib.util.spec_from_file_location("ef", "01_extract_frames.py")
ef = importlib.util.module_from_spec(spec); spec.loader.exec_module(ef)

def total_frames(path):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_packets",
                        "-show_entries","stream=nb_read_packets","-of","csv=p=0", path],
                       capture_output=True, text=True).stdout.strip()
    return int(r) if r.isdigit() else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjective", required=True)
    ap.add_argument("--pvs-dir", required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--crops", type=int, default=1)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp"); os.makedirs(tmp, exist_ok=True)
    sources = ef.list_sources(a.src_dir)
    rows = list(csv.DictReader(open(a.subjective)))
    S = a.crop_size

    groups = defaultdict(list)
    for r in rows:
        groups[ef.parse_name(r["name"])[0]].append(r)

    # ---- phase 1: per content, decode source full frames ONCE (parallel across contents) ----
    plan = {}   # content -> (src, W, H, idxs)
    for content in groups:
        src = ef.match_source(content, sources)
        if not src:
            print(f"SKIP content {content}: no source", file=sys.stderr); continue
        W, H = int(ef.probe(src, "width")), int(ef.probe(src, "height"))
        total = total_frames(src)
        idxs = [min(int(total * (k + 1) / (a.n_frames + 1)), total - 2) for k in range(a.n_frames)]
        plan[content] = (src, W, H, idxs)

    ref_full = {}  # (content, fi) -> PIL.Image (full source frame)
    def dec_ref(content_fi):
        content, fi = content_fi
        src = plan[content][0]
        out = os.path.join(tmp, f"__src_{content}_f{fi}.png")
        ef.extract_frame(src, fi, out)
        return content_fi, Image.open(out).convert("RGB")
    ref_tasks = [(c, fi) for c, (_, _, _, idxs) in plan.items() for fi in idxs]
    print(f"fase 1: decodificando {len(ref_tasks)} frames-fonte (4K lossless)...", flush=True)
    with ThreadPoolExecutor(max_workers=min(a.workers, len(ref_tasks) or 1)) as ex:
        for key, img in ex.map(dec_ref, ref_tasks):
            ref_full[key] = img
    print("fase 1 ok.", flush=True)

    # ---- phase 2: per PVS, decode distorted frame (cheap) and crop both ----
    manifest = []
    lock = threading.Lock()
    done = err = 0
    def work(row):
        name = row["name"]
        content, codec, cls, res, qp = ef.parse_name(name)
        if content not in plan: return ("skip", name, "no plan")
        pvs = ef.find_pvs(a.pvs_dir, name)
        if not pvs: return ("skip", name, "no pvs")
        src, W, H, idxs = plan[content]
        out_rows = []
        for fi in idxs:
            R = ref_full.get((content, fi))
            if R is None: continue
            df = os.path.join(tmp, f"{name}_f{fi}_dis.png")
            try:
                ef.extract_frame(pvs, fi, df, scale_to=(W, H))
                if not os.path.exists(df): continue
                D = Image.open(df).convert("RGB")
            except Exception:
                continue
            for ci, (x, y) in enumerate(ef.crop_positions(name, fi, W, H, S, a.crops)):
                rcp = os.path.join(a.out, f"{name}__f{fi}__c{ci}__ref.png")
                dcp = os.path.join(a.out, f"{name}__f{fi}__c{ci}__dis.png")
                R.crop((x, y, x + S, y + S)).save(rcp)
                D.crop((x, y, x + S, y + S)).save(dcp)
                out_rows.append({"name": name, "content": content, "codec": codec, "cls": cls,
                                 "res": res, "qp": qp, "frame": fi, "crop_id": ci,
                                 "x": x, "y": y, "size": S,
                                 "mos": float(row["mos"]), "ci": float(row.get("ci", "nan") or "nan"),
                                 "ref": rcp, "dis": dcp})
            try: os.remove(df)
            except OSError: pass
        return ("ok", name, out_rows)

    print(f"fase 2: {len(rows)} PVS...", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            status, name, payload = f.result()
            with lock:
                if status == "ok": manifest.extend(payload); done += 1
                else: err += 1; print(f"SKIP {name}: {payload}", file=sys.stderr)
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} PVS (ok={done} skip={err} pares={len(manifest)})", flush=True)

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"done -> {a.out}/manifest.json  ({len(manifest)} pares, {done} PVS)")

if __name__ == "__main__":
    main()
