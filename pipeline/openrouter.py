"""Grader-matched OpenRouter client (FREE models only — account has NO credits, ~50/day cap).
EVERY call is logged to openrouter_logs/<experiment>.jsonl (full request + response) so all
OpenRouter usage is traceable. Set the experiment with set_experiment('name')."""
import os, json, re, time, urllib.request
from pathlib import Path
BASE="https://openrouter.ai/api/v1/chat/completions"
BOX="\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
LOGDIR=Path(__file__).resolve().parent.parent/"openrouter_logs"; LOGDIR.mkdir(exist_ok=True)
_EXP="misc"
def set_experiment(name): 
    global _EXP; _EXP=name
def _log(rec):
    rec["ts"]=time.strftime("%Y-%m-%d %H:%M:%S"); rec["experiment"]=_EXP
    with (LOGDIR/f"{_EXP}.jsonl").open("a") as f: f.write(json.dumps(rec)+"\n")
def extract_final_answer(text):
    if not text: return 'NOT_FOUND'
    bs=list(re.finditer(r'\\boxed\{', text)); ms=[]
    for i,m in enumerate(bs):
        s=m.end(); e=bs[i+1].start() if i+1<len(bs) else len(text)
        seg=text[s:e]; lb=seg.rfind('}'); ms.append(seg[:lb] if lb!=-1 else seg)
    if ms:
        ne=[m.strip() for m in ms if m.strip()]; return ne[-1] if ne else ms[-1].strip()
    nums=re.findall(r'-?\d+(?:\.\d+)?',text); return nums[-1] if nums else (text.strip().splitlines() or ['NOT_FOUND'])[-1]
def ask(prompt, model="nvidia/nemotron-3-nano-30b-a3b:free", max_tokens=7680, key=None, retries=2, meta=None):
    key=key or os.environ["OPENROUTER_API_KEY"]
    payload={"model":model,"messages":[{"role":"user","content":prompt+BOX}],
             "temperature":0,"top_p":1.0,"max_tokens":max_tokens,"reasoning":{"enabled":True}}
    for a in range(retries+1):
        try:
            req=urllib.request.Request(BASE, json.dumps(payload).encode(),
                 {"Authorization":f"Bearer {key}","Content-Type":"application/json"})
            r=json.load(urllib.request.urlopen(req, timeout=600))
            m=r["choices"][0]["message"]
            out={"content":m.get("content"),"reasoning":m.get("reasoning") or "",
                 "answer":extract_final_answer(m.get("content")),"finish":r["choices"][0].get("finish_reason"),
                 "usage":r.get("usage",{})}
            _log({"model":model,"max_tokens":max_tokens,"meta":meta,"prompt":prompt,
                  "content":out["content"],"reasoning":out["reasoning"],"answer":out["answer"],
                  "finish":out["finish"],"usage":out["usage"]})
            return out
        except Exception as e:
            msg=str(e)[:200]
            if '429' in msg:  # rate-limited: do NOT retry (wastes quota), fail fast
                _log({"model":model,"meta":meta,"prompt":prompt,"error":msg}); return {"error":msg}
            if a<retries: time.sleep(5); continue
            _log({"model":model,"meta":meta,"prompt":prompt,"error":msg})
            return {"error":msg}
