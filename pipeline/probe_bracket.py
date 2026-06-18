"""Base probe: can the 30B execute the MAGNITUDE-BRACKET mul deduction (single-digit recall + comparison,
NO 2-digit multiply, NO factoring)? Worked example spells out: result-length->op; for mul, P 4-digit with
leading digit a1 (shared symbol) -> bracket a1*b1 <= floor(P/100) < (a1+1)(b1+1) via single-digit recall ->
b1 shortlist; units (a0*b0)mod10=P%10 -> b0; all-different finishes. Greedy, measured on mul-heavy crypt rows.
  python3 pipeline/probe_bracket.py [n]
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0,'pipeline/solvers'); sys.path.insert(0,'analysis/crypt_struct')
import nvidia_api as NV, cryptarithm2 as C2, propagation as P
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODEL='nvidia/nemotron-3-nano-30b-a3b'
OPPH={'add':'a+b','sub_signed':'a-b','rsub_signed':'b-a','mul':'a*b','absdiff':'|a-b|','neg_absdiff':'-|a-b|',
      'add_m1':'a+b-1','add_p1':'a+b+1','mul_m1':'a*b-1','mul_p1':'a*b+1'}

def worked(prompt, ops, rev, m, ans):
    eqs,qL=C2.parse(prompt); oc={L[2] for L,R in eqs}|{qL[2]}
    out=["Decode by result length: 4 symbols => multiply (or concat if the result reuses the operand symbols); "
         "1-2 symbols => subtract/absdiff. Then crack the digits by single-digit recall + all-different (each symbol a distinct digit). No 2-digit multiplications, no factoring."]
    out.append("Operators: "+", ".join(f"'{g}'={OPPH.get(ops[g],ops[g])}" for g in sorted(oc))+".")
    for (L,R) in eqs:
        op=ops[L[2]]
        if op!='mul': continue
        a=(m[L[1]]*10+m[L[0]]) if rev else (m[L[0]]*10+m[L[1]]); b=(m[L[4]]*10+m[L[3]]) if rev else (m[L[3]]*10+m[L[4]])
        Pn=a*b; a1,b1,a0,b0=a//10,b//10,a%10,b%10; hi=Pn//100
        out.append(f"Eq {''.join(L)}={R} (mul): result is {len(R)} symbols. With operand tens a1={a1} (leading symbol), "
                   f"hundreds-up of P is {hi}. Single-digit recall: which b1 give {a1}*b1 <= {hi} < {a1+1}*(b1+1)? "
                   f"-> b1={b1}. Units: P ends in {Pn%10}, a0={a0}, so {a0}*b0 ends in {Pn%10} -> b0={b0}.")
    out.append("All-different across equations -> Map: "+", ".join(f"{g}={m[g]}" for g in sorted(m))+".")
    qa,qb,op,qc,qd=qL; o=ops[op]
    a=(m[qb]*10+m[qa]) if rev else (m[qa]*10+m[qb]); b=(m[qd]*10+m[qc]) if rev else (m[qc]*10+m[qd])
    out.append(f"Query {''.join(qL)}: {a} {OPPH.get(o,o)} = computed -> map back -> {ans}")
    out.append(f"\\boxed{{{ans}}}")
    return "\n".join(out)

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 8
    NV.set_experiment('bracket_probe')
    rows=[r for r in __import__('csv').DictReader(open(os.path.join(ROOT,'competition_dataset/train_categorized.csv'))) if r['category']=='cryptarithm_deduce']
    random.Random(11).shuffle(rows)
    # pick mul-heavy, propagation-determined puzzles
    pool=[]
    for r in rows:
        try:
            res=C2.solve(r['prompt'],deadline_s=2.0)
            if not res: continue
            ops=(res[1] or {}).get('ops'); rev=(res[1] or {}).get('rev')
            if ops is None: continue
            eqs,qL=C2.parse(r['prompt']); oc={L[2] for L,R in eqs}|{qL[2]}
            if not any(ops[g]=='mul' for g in oc): continue
            e=P.Eng(eqs,qL,{g:[ops[g]] for g in oc}, oc<=set('+-*'), time.time()+2); e.propagate()
            if not all(len(e.dom[s])==1 for s in e.syms): continue
            pool.append((r,ops,rev,{s:next(iter(e.dom[s])) for s in e.syms}))
        except Exception: continue
        if len(pool)>n+1: break
    ex=pool[0]; w=worked(ex[0]['prompt'],ex[1],ex[2],ex[3],ex[0]['answer'].strip())
    print("=== WORKED EXAMPLE (bracket+units, non-teleport-ish) ===\n"+w+"\n"+"="*60, flush=True)
    BOX="Put the final answer inside \\boxed{}, e.g. \\boxed{...}."
    INSTR="Solve this symbol-cipher equation puzzle. Decode operators by result length, crack digits with single-digit recall + all-different (NO 2-digit multiply, NO factoring). "+BOX
    ok=0; test=pool[1:1+n]
    for r,ops,rev,m in test:
        prompt=INSTR+"\n\nWorked example:\n"+w+"\n\nNow solve:\n"+r['prompt']
        o=NV.ask(prompt, model=MODEL, max_tokens=7000, temperature=0.0, add_box=False, meta={'id':r['id']})
        pred=(o.get('answer') if isinstance(o,dict) else '') or ''
        ok+=(pred.strip()==r['answer'].strip())
        print(f"  {r['id']}: gold {r['answer']!r} | pred {pred!r:14s} | finish {o.get('finish') if isinstance(o,dict) else 'ERR'}", flush=True)
    print(f"\nRESULT: bracket-method base probe {ok}/{len(test)} correct on mul-heavy crypt", flush=True)

if __name__=='__main__': main()
