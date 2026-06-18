"""NVIDIA build API teacher client (integrate.api.nvidia.com), STREAMING (avoids gateway 504).
Separate quota from OpenRouter. Logs every call to nvidia_logs/<experiment>.jsonl."""
import os, json, re, time, urllib.request
from pathlib import Path
BASE="https://integrate.api.nvidia.com/v1/chat/completions"
BOX="\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
LOGDIR=Path(__file__).resolve().parent.parent/"nvidia_logs"; LOGDIR.mkdir(exist_ok=True)
_EXP="misc"
def set_experiment(n):
    global _EXP; _EXP=n
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
        ne=[x.strip() for x in ms if x.strip()]; return ne[-1] if ne else ms[-1].strip()
    nums=re.findall(r'-?\d+(?:\.\d+)?',text); return nums[-1] if nums else (text.strip().splitlines() or ['NOT_FOUND'])[-1]
def ask(prompt, model="nvidia/nemotron-3-super-120b-a12b", max_tokens=10000, temperature=0.6, key=None, meta=None, retries=4, system=None, add_box=True, top_p=0.95, reasoning_budget=None):
    key=key or os.environ["NVIDIA_API_KEY"]
    msgs=[]
    if system: msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt+(BOX if add_box else "")})
    body=json.dumps({"model":model,"messages":msgs,
                     "temperature":temperature,"top_p":top_p,"max_tokens":max_tokens,"stream":True,
                     "chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":(reasoning_budget if reasoning_budget is not None else max_tokens)}).encode()
    last="?"
    for att in range(retries+1):
        try:
            req=urllib.request.Request(BASE, body, {"Authorization":f"Bearer {key}","Content-Type":"application/json","Accept":"text/event-stream"})
            resp=urllib.request.urlopen(req, timeout=600)
            reasoning=[]; content=[]; finish=None; usage={}
            for raw in resp:
                line=raw.decode('utf-8','ignore').strip()
                if not line.startswith('data:'): continue
                data=line[5:].strip()
                if data=='[DONE]': break
                try: obj=json.loads(data)
                except: continue
                ch=(obj.get('choices') or [{}])[0]; d=ch.get('delta',{})
                if d.get('reasoning_content'): reasoning.append(d['reasoning_content'])
                if d.get('content'): content.append(d['content'])
                if ch.get('finish_reason'): finish=ch['finish_reason']
                if obj.get('usage'): usage=obj['usage']
            content=''.join(content); reasoning=''.join(reasoning)
            if not content and not reasoning:   # empty stream = treat as transient
                last="empty stream"; 
                if att<retries: time.sleep(15*(att+1)); continue
            out={"content":content,"reasoning":reasoning,"answer":extract_final_answer(content or reasoning),"finish":finish,"usage":usage}
            _log({"model":model,"meta":meta,"system":system,"prompt":prompt,"content":content,"reasoning":reasoning,"answer":out["answer"],"finish":finish,"usage":usage,"attempts":att+1})
            return out
        except urllib.error.HTTPError as e:
            last=f"HTTP {e.code}"
            if e.code in (429,500,502,503,504) and att<retries: time.sleep(15*(att+1)); continue
            _log({"model":model,"meta":meta,"error":last}); return {"error":last}
        except Exception as e:
            last=str(e)[:150]
            if att<retries: time.sleep(15*(att+1)); continue
            _log({"model":model,"meta":meta,"error":last}); return {"error":last}
    _log({"model":model,"meta":meta,"error":last}); return {"error":last}
