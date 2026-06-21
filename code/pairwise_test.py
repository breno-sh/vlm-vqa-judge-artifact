"""
Quick test of approach B: PAIRWISE ranking (vs absolute scoring).
Literature (Q-Align, Zheng et al. NeurIPS'23) says VLMs order better than they score.

For ONE content we extract ALIGNED crops (same frame + same position across all PVS),
then for sampled PVS pairs (A,B) we show the judge REF + A + B and ask which reconstruction
is more faithful. We measure pairwise concordance with MOS, and compare to VMAF's concordance.

Usage: source ~/.paper51_keys.env; PAPER51_GPT_MODEL=gpt-5.5 python3 pairwise_test.py
"""
import base64, csv, hashlib, itertools, json, os, random, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import judges as J  # reuse _b64, _media_type, model env vars

CONTENT = os.environ.get("PW_CONTENT", "vegetables")
FRAME   = int(os.environ.get("PW_FRAME", "150"))
S = 512
N_PAIRS = int(os.environ.get("PW_PAIRS", "60"))
DEC = "../AVT-VQDB-UHD-1-NVC/decoded"
SRC = f"../AVT-VQDB-UHD-1-NVC/original/{CONTENT}_original_3840x2160_q0.mkv"
OUT = f"./pairwise_{CONTENT}"
os.makedirs(OUT, exist_ok=True)

def pos(name, frame, W=3840, H=2160):
    h = int(hashlib.sha256(f"{name}:{frame}".encode()).hexdigest(), 16)
    h = (h * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    x = (h >> 16) % (W - S)
    h = (h * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    y = (h >> 16) % (H - S)
    return x, y
X, Y = pos(CONTENT, FRAME)  # SAME position for every PVS of this content

def ff(infile, out, scale):
    vf = (f"select=eq(n\\,{FRAME}),scale=3840:2160," if scale else f"select=eq(n\\,{FRAME}),") + f"crop={S}:{S}:{X}:{Y}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", infile, "-vf", vf, "-vframes", "1", out], check=True)

# --- load PVS + MOS + VMAF for this content ---
met = [r for r in csv.DictReader(open("metrics.csv")) if r["name"].startswith(CONTENT + "_")]
have = set(f[:-len(".decoded.mkv")] for f in os.listdir(DEC) if f.endswith(".decoded.mkv"))
met = [r for r in met if r["name"] in have]
print(f"{CONTENT}: {len(met)} PVS, crop @({X},{Y}) frame {FRAME}", flush=True)

# --- extract aligned crops (ref once + one dis per PVS) ---
ref = f"{OUT}/_ref.png"
if not os.path.exists(ref): ff(SRC, ref, scale=False)
crop = {}
for r in met:
    p = f"{OUT}/{r['name']}.png"
    if not os.path.exists(p): ff(f"{DEC}/{r['name']}.decoded.mkv", p, scale=True)
    crop[r["name"]] = p
mos = {r["name"]: float(r["mos"]) for r in met}
vmaf = {r["name"]: float(r["vmaf"]) for r in met}
print("crops prontos.", flush=True)

# --- sample pairs (reproducible) ---
random.seed(42)
allp = list(itertools.combinations([r["name"] for r in met], 2))
pairs = random.sample(allp, min(N_PAIRS, len(allp)))

PAIR_SCHEMA = {"type": "object", "properties": {
    "choice": {"type": "string", "enum": ["A", "B"]}, "reason": {"type": "string"}},
    "required": ["choice", "reason"], "additionalProperties": False}
SYS = ("You compare two compressed reconstructions (A and B) of the SAME reference image "
       "patch. Decide which reconstruction is MORE FAITHFUL to the reference (better "
       "perceptual quality): consider fine texture, edges, color and compression artifacts.")
PROMPT = ("The reference patch is shown first, then Reconstruction A, then Reconstruction B. "
          "Which reconstruction is more faithful to the reference? Return JSON only: "
          "choice ('A' or 'B'), reason (one short sentence).")

def b64(p): return base64.standard_b64encode(open(p, "rb").read()).decode()

def claude(ref, a, b):
    import anthropic
    c = anthropic.Anthropic()
    img = lambda p: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(p)}}
    content = [{"type": "text", "text": "REFERENCE patch:"}, img(ref),
               {"type": "text", "text": "Reconstruction A:"}, img(a),
               {"type": "text", "text": "Reconstruction B:"}, img(b),
               {"type": "text", "text": PROMPT}]
    r = c.messages.create(model=J.CLAUDE_MODEL, max_tokens=512, system=SYS,
                          messages=[{"role": "user", "content": content}],
                          output_config={"format": {"type": "json_schema", "schema": PAIR_SCHEMA}})
    t = next(x.text for x in r.content if x.type == "text")
    return json.loads(t)["choice"]

def gpt(ref, a, b):
    from openai import OpenAI
    c = OpenAI()
    uri = lambda p: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(p)}", "detail": "high"}}
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [
        {"type": "text", "text": "REFERENCE patch:"}, uri(ref),
        {"type": "text", "text": "Reconstruction A:"}, uri(a),
        {"type": "text", "text": "Reconstruction B:"}, uri(b),
        {"type": "text", "text": PROMPT}]}]
    kw = dict(model=J.GPT_MODEL, response_format={"type": "json_object"}, messages=msgs)
    if J.GPT_MODEL.startswith(("gpt-4o", "gpt-4.1")): kw["temperature"] = 0; kw["max_tokens"] = 256
    else: kw["max_completion_tokens"] = 512
    r = c.chat.completions.create(**kw)
    return json.loads(r.choices[0].message.content)["choice"]

def gemini(ref, a, b):
    key = os.environ.get("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{J.GEMINI_MODEL}:generateContent"
    inl = lambda p: {"inline_data": {"mime_type": "image/png", "data": b64(p)}}
    body = {"contents": [{"parts": [{"text": SYS + "\nREFERENCE patch:"}, inl(ref),
            {"text": "Reconstruction A:"}, inl(a), {"text": "Reconstruction B:"}, inl(b),
            {"text": PROMPT}]}], "generationConfig": {"temperature": 0, "response_mime_type": "application/json"}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    return json.loads(out["candidates"][0]["content"]["parts"][0]["text"])["choice"]

MODELS = {"claude": claude, "gpt": gpt, "gemini": gemini}

def judge_pair(fn, p, q, flip):
    # flip controls which physical PVS is labeled "A" (position-bias mitigation)
    a, b = (q, p) if flip else (p, q)
    choice = fn(ref, crop[a], crop[b])
    picked = a if choice == "A" else b
    return picked

def run_model(name, fn):
    correct = correctclear = nclear = 0
    rows = []
    def task(i_pq):
        i, (p, q) = i_pq
        flip = bool(i % 2)
        for _ in range(2):
            try:
                picked = judge_pair(fn, p, q, flip)
                truth = p if mos[p] > mos[q] else q
                return (p, q, picked, picked == truth, abs(mos[p] - mos[q]))
            except Exception as e:
                last = e
        print(f"  [{name}] falha {p}|{q}: {type(last).__name__}: {str(last)[:60]}", file=sys.stderr)
        return None
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(task, enumerate(pairs)):
            if not res: continue
            p, q, picked, ok, dm = res
            rows.append(res)
            correct += ok
            if dm >= 0.5:
                nclear += 1; correctclear += ok
    n = len(rows)
    acc = correct / n if n else float("nan")
    accc = correctclear / nclear if nclear else float("nan")
    return name, n, acc, nclear, accc

# VMAF concordance on the same pairs
def vmaf_acc():
    c = cc = nc = 0
    for p, q in pairs:
        truth = p if mos[p] > mos[q] else q
        vpick = p if vmaf[p] > vmaf[q] else q
        c += (vpick == truth)
        if abs(mos[p] - mos[q]) >= 0.5:
            nc += 1; cc += (vpick == truth)
    return len(pairs), c / len(pairs), nc, cc / nc

print(f"\npares avaliados: {len(pairs)}  (de {len(allp)} possíveis)\n", flush=True)
print(f"{'preditor':<10}{'n':>5}{'acc_todos':>11}{'n_claros':>10}{'acc_claros':>12}")
nn, va, ncv, vac = vmaf_acc()
print(f"{'VMAF':<10}{nn:>5}{va:>11.3f}{ncv:>10}{vac:>12.3f}")
for name, fn in MODELS.items():
    nm, nn2, acc, ncl, accc = run_model(name, fn)
    print(f"{nm:<10}{nn2:>5}{acc:>11.3f}{ncl:>10}{accc:>12.3f}", flush=True)
print("\n(acc = % de pares em que o preditor ordenou igual ao humano; 'claros' = |ΔMOS|>=0.5)")
