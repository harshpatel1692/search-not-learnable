"""STaR / rejection-sampling harvester (Approach 1) for the HARD search categories.
Sample a model's OWN CoT on real-train problems, keep ONLY verified-correct traces -> transferable SFT data.
Same-family teachers (nemotron-3 super-120b / ultra-550b) give imitable style; the base-30b gives self-STaR.
Usage: python3 pipeline/star_harvest.py <model> <category> <n> <K> [workers]
  model: short key (base|super|ultra|ds|qwen) or full id ; K = samples per problem (keep first correct)
Writes correct traces -> pipeline/data/star_<category>.jsonl ; prints solve rate. GPU-FREE (API only)."""
import sys, os, csv, json, re, math, time
sys.path.insert(0, os.path.dirname(__file__))
import nvidia_api as N
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS={"base":"nvidia/nemotron-3-nano-30b-a3b","super":"nvidia/nemotron-3-super-120b-a12b",
        "ultra":"nvidia/nemotron-3-ultra-550b-a55b","reason":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "ds":"deepseek-ai/deepseek-v4-pro","qwen":"qwen/qwen3.5-397b-a17b","gptoss":"openai/gpt-oss-120b"}

def verify(s,p):
    s,p=str(s).strip(),str(p).strip()
    if re.fullmatch(r'[01]+',s): return p.lower()==s.lower()
    try: return math.isclose(float(s),float(p),rel_tol=1e-2,abs_tol=1e-5)
    except: return p.lower()==s.lower()

def rows(cat,n):
    r=[x for x in csv.DictReader(open('competition_dataset/train_categorized.csv')) if x['category']==cat][:n]
    return r

BUDGET=7680          # grader generates up to 7680 tokens (max_model_len 8192); harvested CoT MUST fit this
KEEP_MAX=7000        # keep margin: a trace the small base must reproduce within budget under greedy

def attempt(model, prompt, gold, K, temp):
    """try up to K samples; cap gen at BUDGET; return (correct, trace_or_None, n_completion_tokens, finish)."""
    best=None
    for k in range(K):
        out=N.ask(prompt, model=model, max_tokens=BUDGET, temperature=temp, meta={"k":k})
        if out.get("error"): continue
        ct=(out.get("usage") or {}).get("completion_tokens",0)
        fin=out.get("finish")
        if verify(gold, out.get("answer","")):
            tr={"prompt":prompt,"reasoning":out.get("reasoning",""),"content":out.get("content",""),
                "answer":out.get("answer"),"gold":gold,"model":model,"k":k,"ctoks":ct,"finish":fin}
            # only KEEP if it both finished (not truncated) and fits the reproduce-budget
            if fin!="length" and ct<=KEEP_MAX:
                return True, tr, ct, fin
            best=(True, None, ct, fin)   # solved but over-budget/truncated -> count as solve, don't keep
    if best: return best
    return False, None, 0, None

def main():
    mkey=sys.argv[1]; cat=sys.argv[2]; n=int(sys.argv[3]); K=int(sys.argv[4] if len(sys.argv)>4 else 1)
    workers=int(sys.argv[5]) if len(sys.argv)>5 else 6
    temp=0.6
    model=MODELS.get(mkey,mkey)
    N.set_experiment(f"star_{cat}_{mkey}")
    rs=rows(cat,n)
    out_path=f"pipeline/data/star_{cat}.jsonl"
    ok=0; done=0; kept=[]; overlen=0; clens=[]; t0=time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(attempt, model, r['prompt'], r['answer'].strip(), K, temp): r for r in rs}
        for fut in as_completed(futs):
            cor,trace,ct,fin=fut.result(); done+=1
            if cor:
                ok+=1
                if trace is not None: kept.append(trace); clens.append(ct)
                else: overlen+=1          # solved but truncated/over-budget -> not reproducible
            if done%5==0 or done==len(rs):
                print(f"  {done}/{len(rs)} solved={ok} kept={len(kept)} overlen={overlen} {round(time.time()-t0)}s", flush=True)
    if kept:
        with open(out_path,"a") as f:
            for t in kept: f.write(json.dumps(t)+"\n")
    import statistics as st
    ld = f"min/med/max ctoks={min(clens)}/{int(st.median(clens))}/{max(clens)}" if clens else "n/a"
    print(f"[STAR] {cat} model={mkey} K={K}: SOLVED {ok}/{len(rs)}={100*ok/len(rs):.0f}% | "
          f"KEPT(<= {KEEP_MAX}tok, finished) {len(kept)} | over-budget solves {overlen} | {ld} | +->{out_path} | {round(time.time()-t0)}s")

if __name__=="__main__": main()
