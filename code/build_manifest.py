"""Reconstruct manifest.json from already-extracted crop PNGs in frames_full/
(the extractor only writes the manifest at the very end; this recovers it from disk).
Pairs <name>__f<fi>__c<ci>__ref.png with __dis.png, joins mos/ci from subjective.csv."""
import csv, glob, importlib.util, json, os, re, sys

spec = importlib.util.spec_from_file_location("ef", "01_extract_frames.py")
ef = importlib.util.module_from_spec(spec); spec.loader.exec_module(ef)

OUT = sys.argv[1] if len(sys.argv) > 1 else "frames_full"
subj = {r["name"]: r for r in csv.DictReader(open("../AVT-VQDB-UHD-1-NVC/subjective.csv"))}

pat = re.compile(r"^(.*)__f(\d+)__c(\d+)__ref\.png$")
manifest = []
for ref in sorted(glob.glob(os.path.join(OUT, "*__ref.png"))):
    m = pat.match(os.path.basename(ref))
    if not m: continue
    name, fi, ci = m.group(1), int(m.group(2)), int(m.group(3))
    dis = ref.replace("__ref.png", "__dis.png")
    if not os.path.exists(dis) or name not in subj: continue
    content, codec, cls, res, qp = ef.parse_name(name)
    r = subj[name]
    manifest.append({"name": name, "content": content, "codec": codec, "cls": cls,
                     "res": res, "qp": qp, "frame": fi, "crop_id": ci,
                     "mos": float(r["mos"]), "ci": float(r.get("ci", "nan") or "nan"),
                     "ref": ref, "dis": dis})

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"))
pvs = {x["name"] for x in manifest}
cls = {c: len({x["name"] for x in manifest if x["cls"] == c}) for c in {x["cls"] for x in manifest}}
print(f"manifest: {len(manifest)} pares | {len(pvs)} PVS | cls(PVS)={cls}")
print("conteúdos:", {c: len({x['name'] for x in manifest if x['content']==c}) for c in {x['content'] for x in manifest}})
