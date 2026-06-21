"""
Step 2 (PARALLEL) — run the VLM judges (claude, gpt, gemini) over the crop pairs.

API calls are I/O-bound, so we fan out with a thread pool (--workers). Each
(crop pair, judge, repeat) is an independent task. Resumable: existing
(name,frame,crop_id,judge,rep,prompt) rows are skipped. CSV writes are guarded
by a lock; per-task retries with backoff handle transient rate limits.

Usage:
  source ~/.paper51_keys.env
  python 02_run_judges.py --manifest frames_test/manifest.json --out pilot_scores.csv \
      --judges claude gpt gemini --repeats 1 --prompt frozen --workers 16
"""
import argparse, csv, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import judges as J

FIELDS = ["name", "content", "codec", "cls", "res", "qp", "frame", "crop_id", "rep",
          "judge", "prompt", "mos", "ci", "score", "artifact", "justification", "model_version"]
META = ["name", "content", "codec", "cls", "res", "qp", "frame", "crop_id", "mos", "ci"]

def load_done(path):
    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add((r["name"], r["frame"], r["crop_id"], r["judge"], r["rep"], r["prompt"]))
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="judge_scores.csv")
    ap.add_argument("--judges", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--prompt", default="frozen", choices=list(J.PROMPTS.keys()))
    ap.add_argument("--two-pass", action="store_true",
                    help="2 calls/pair: 'describe' (artifact) + 'anchored' (score)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=4)
    a = ap.parse_args()

    plabel = "twopass" if a.two_pass else a.prompt
    pairs = json.load(open(a.manifest))
    done = load_done(a.out)

    tasks = []
    for p in pairs:
        for jname in a.judges:
            for rep in range(a.repeats):
                key = (p["name"], str(p["frame"]), str(p["crop_id"]), jname, str(rep), plabel)
                if key not in done:
                    tasks.append((p, jname, rep))
    print(f"tarefas a rodar: {len(tasks)} (já feitas: {len(done)}) | workers={a.workers}", flush=True)
    if not tasks:
        print("nada a fazer."); return

    def work(task):
        p, jname, rep = task
        fn = J.JUDGES[jname]
        for attempt in range(a.max_retries):
            try:
                v = (J.judge_twopass(fn, p["ref"], p["dis"]) if a.two_pass
                     else fn(p["ref"], p["dis"], prompt=a.prompt))
                row = {k: p.get(k, "") for k in META}
                row.update({"rep": rep, "judge": jname, "prompt": plabel,
                            "score": v.score, "artifact": v.artifact,
                            "justification": v.justification, "model_version": v.model_version})
                return row
            except Exception as e:
                wait = 2 ** attempt
                print(f"[{jname}] {p['name']} r{rep} tentativa {attempt+1}: "
                      f"{type(e).__name__}: {str(e)[:120]} -- retry {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
        return None

    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new: w.writeheader()
    lock = threading.Lock()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            with lock:
                if row:
                    w.writerow(row); fh.flush(); ok += 1
                else:
                    err += 1
            if i % 10 == 0 or i == len(tasks):
                print(f"  progresso: {i}/{len(tasks)}  (ok={ok} falhas={err})", flush=True)
    fh.close()
    print(f"done -> {a.out}  (ok={ok}, falhas={err})")

if __name__ == "__main__":
    main()
