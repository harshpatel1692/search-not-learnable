"""Probe: BASE model native accuracy on cryptarithm_deduce, stratified by gold QUERY-OP family.
Hypothesis: base pattern-spots concat-query rows (delete-the-operator) but fails value-query rows.
20 concat-query rows (7 val + 13 extra from train_categorized) + 30 value-query rows (from val).
Sequential OpenRouter free-tier calls (cap ~50/day). Results -> pipeline/data/crypt_native_probe.jsonl
"""
import csv, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openrouter
from solvers.common import verify

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pipeline/data/crypt_native_probe.jsonl"
openrouter.set_experiment("cryptarithm_native_probe")

def fam(qop):
    if qop is None: return "other"
    if qop.startswith("concat"): return "concat"
    return "value"

meta = {r["id"]: r for r in (json.loads(l) for l in open(ROOT/"pipeline/data/cryptarithm_gold_meta.jsonl"))
        if r["cat"] == "cryptarithm_deduce"}
val = [r for r in (json.loads(l) for l in open(ROOT/"pipeline/data/val.jsonl"))
       if r["category"] == "cryptarithm_deduce"]
train = {r["id"]: r for r in csv.DictReader(open(ROOT/"competition_dataset/train_categorized.csv"))
         if r["category"] == "cryptarithm_deduce"}

val_ids = {r["id"] for r in val}
rng = random.Random(0)

# concat stratum: all 7 val concat + 13 extra from train_categorized
val_concat = [r for r in val if fam(meta[r["id"]].get("qop")) == "concat"]
extra_pool = sorted(i for i, m in meta.items() if fam(m.get("qop")) == "concat" and i not in val_ids)
extra_ids = rng.sample(extra_pool, 13)
concat_rows = val_concat + [{"id": i, "prompt": train[i]["prompt"], "answer": train[i]["answer"]} for i in extra_ids]

# value stratum: 30 random from val's value rows
val_value = [r for r in val if fam(meta[r["id"]].get("qop")) == "value"]
value_rows = rng.sample(val_value, 30)

todo = [(r, "concat") for r in concat_rows] + [(r, "value") for r in value_rows]
done = set()
if OUT.exists():
    done = {json.loads(l)["id"] for l in open(OUT)}
print(f"plan: {len(concat_rows)} concat + {len(value_rows)} value; already done: {len(done)}", flush=True)

for r, f in todo:
    if r["id"] in done: continue
    qop = meta[r["id"]].get("qop")
    out = None
    for att in range(4):
        out = openrouter.ask(r["prompt"], max_tokens=7000, meta={"id": r["id"], "fam": f, "qop": qop})
        if "error" in out and "429" in out["error"]:
            print(f"{r['id']} 429, sleep 75s (att {att})", flush=True); time.sleep(75); continue
        break
    if out is None or "error" in out:
        print(f"{r['id']} FAILED: {out.get('error') if out else '?'}", flush=True)
        if out and "429" in str(out.get("error", "")):
            print("persistent 429 -- assuming daily cap, stopping.", flush=True); break
        continue
    ok = verify(r["answer"], out["answer"])
    rec = {"id": r["id"], "fam": f, "qop": qop, "gold": r["answer"], "pred": out["answer"],
           "ok": ok, "finish": out.get("finish"),
           "completion_tokens": out.get("usage", {}).get("completion_tokens"),
           "reasoning_len": len(out.get("reasoning") or ""), "content_len": len(out.get("content") or "")}
    with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
    print(f"{r['id']} {f:6s} {str(qop):12s} ok={ok} finish={out.get('finish')} "
          f"ctok={rec['completion_tokens']} pred={out['answer']!r} gold={r['answer']!r}", flush=True)
    time.sleep(3)

# summary
res = [json.loads(l) for l in open(OUT)] if OUT.exists() else []
for f in ("concat", "value"):
    sub = [x for x in res if x["fam"] == f]
    if sub:
        print(f"{f}: {sum(x['ok'] for x in sub)}/{len(sub)} = {sum(x['ok'] for x in sub)/len(sub):.3f}", flush=True)
