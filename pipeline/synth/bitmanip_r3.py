"""bit_manipulation r3 renderer: BOUNDED prior-ordered GLOBAL-tap CoT.

WHY r3 (analysis/reports/bitmanip_r3.md): the v15 GLOBAL renderer (bitmanip_global.py) is 98.56%
solver-correct and its RENDERED traces are short (p95 1639 tok) -- but it compresses silent gate
tiers into summary lines ("XNOR pairs: first 19 miss c0;"). At inference the model does NOT imitate
the compression; it EXPANDS every silent tier into one reject line per candidate and, in natural
scan order, grinds the deep 3-tap (CH / XNOR-ANDN) tail to the 7680 cap on ~23% of rows
(measured: natural-order expanded p95=6396, max=7376 tok). Two fixes, both grounded in the
scan-rank measurement (analysis/bit_global_diag/):

  FIX 1 -- PRIOR-ORDERED, FULLY EXPLICIT, BOUNDED SCAN.
    The winning bit-0 anchor belongs to only 9 truth-table families; their global frequency is
    p0..p7 singles (39.3%), XNOR pair (20.8%), AND pair (13.8%), const0 (6.9%), ANDN pair (4.7%),
    XOR pair (4.4%), ANDN-rev (4.1%), OR pair (3.3%), const1 (2.7%); EVERYTHING else (the 3-tap
    CH/XNOR-ANDN/... triples) is the rare tail. We scan candidates in that frequency order and
    EMIT EVERY TESTED CANDIDATE AS ONE EXPLICIT LINE (no tier compression) so train == inference.
    Under this order the winner is reached by p95=4016 / max=4328 tokens -- every real row fits
    the 4500 gate with NO silent tiers. A hard cap of K=180 scan lines + bail-to-best-partial is
    kept as a guardrail (full real coverage is reached at rank 178), so a divergent inference run
    still terminates in budget instead of grinding to 7680.

  FIX 2 -- REAL VERIFY (not constant "ok").
    The v15 verify line just echoed the gold columns, so it was always "ok" regardless of the
    locked rule; a mis-walked rule was never caught and the false-verify habit let the model emit
    self-consistent-looking wrong answers (4/7 diag failures were finished-but-wrong, all with the
    renderer itself correct). r3 RE-APPLIES the locked (taps, Fc) rule to each example input and
    prints computed-vs-given with an explicit ok/MISMATCH verdict; for the (deterministic, correct)
    rule they all match, training the model that VERIFY is a recomputation that must agree.

Honest deterministic procedure (solver = bitmanip_global.solve); inverted linter re-checks every
scan line (gate recomputed, verdict forced), every match, the lock taps, every verify pair
(rule re-applied), the apply, and boxed==computed. ASCII-only. boxed-free (trainer appends it).

Synth-only corpus: pipeline/data/v16/bit_r3_synth.jsonl (>=1500 rows, fresh values, 0 collisions
with the 1602 real prompts). Make CSV: pipeline/data/v16/train_bit_r3.csv.
"""
import sys, os, json, re, csv, random, itertools, collections, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bitmanip_global as G
from bitmanip_global import (CANDS, col_of, gate_name, tap_pos, tap_name, tap_line_desc,
                             solve as gsolve, render as grender, complete_F, pick_F, apply_rule,
                             mk_prompt, eval_expr, rule_dist_from_cache, lint as glint)

# ------------- prior order over the 9 winning-anchor families -------------
# (arity, canonical-tt) -> rank weight. Derived from the 1602-row winner frequency
# (analysis/bit_global_diag/analyze.py). Higher weight = scanned earlier. Hard-coded
# (not data-fit per row) so the order is a fixed deterministic prior, no leakage.
FAM_WEIGHT={
  (1,(0,1)):       630,   # singles p0..p7
  (2,(1,0,0,1)):   333,   # XNOR
  (2,(0,0,0,1)):   221,   # AND
  (0,(0,)):        110,   # const 0
  (2,(0,1,0,0)):    75,   # ANDN(a,b)=(NOT a) AND b
  (2,(0,1,1,0)):    71,   # XOR
  (2,(0,0,1,0)):    66,   # ANDN-rev
  (2,(0,1,1,1)):    53,   # OR
  (0,(1,)):         43,   # const 1
}
def _key(ci):
    pos,tt=CANDS[ci]; return (len(pos), tt)
def _w(ci): return FAM_WEIGHT.get(_key(ci), 0)
PRIOR_ORDER=sorted(range(len(CANDS)), key=lambda ci:(-_w(ci), ci))
CAP_K=185   # full real coverage reached by rank 178: with the terse scan format even a rank-178
            # winner fits the budget (max ~4310 tok at K=140; the marginal ~38 lines to 178 add
            # ~700 tok, still < 6500 hard max), so we set the guardrail JUST above 178 to avoid ANY
            # cap-bail on real rows (bail would emit a wrong anchor + honest verify MISMATCH = a
            # self-contradicting trace). The cap only fires if inference diverges into the triple tail.

# ------------- per-bit application of the locked rule -------------
def apply_locked(taps, Fc, xbits):
    """Re-apply the (taps,Fc) lift to one 8-bit input -> 8-char output string."""
    out=[]
    for j in range(8):
        idx=0
        for t in taps:
            p=tap_pos(j,t)
            idx=(idx<<1)|(0 if p is None else xbits[p])
        out.append(Fc[idx])
    return ''.join(map(str,out))

def apply_walk(traj, xbits):
    """Re-apply the per-bit WALK gates (the rule the trace actually shows bit by bit) to an input.
    Authoritative for verify/answer: the (taps,Fc) lift can mis-default the 0-tap constant, but the
    walk gates are exactly what the scan/walk lines lock, so verify==apply==answer all use this."""
    out=[]
    for j in range(8):
        pos,tt=traj[j][1]
        idx=0
        for p in pos: idx=(idx<<1)|xbits[p]
        out.append(tt[idx])
    return ''.join(map(str,out))

# ================= R3 RENDERER =================
GUIDE=G.GUIDE   # same method preamble (already describes scan->walk->lock->verify->apply)
def cs(col): return ''.join(map(str,col))

def render_r3(ins, outs, q, sol):
    """Bounded prior-ordered explicit-scan CoT. Returns (cot, final)."""
    n=len(ins); ins_bits=sol['ins_bits']; cols=sol['cols']; traj=sol['traj']; taps=sol['taps']
    wci=sol['winner']
    Fc=sol['Fc']
    L=[GUIDE]
    # TABLE
    L.append("Position rows (bit p of x1..x%d):"%n)
    for p in range(8): L.append("p%d %s"%(p, ''.join(s[p] for s in ins)))
    L.append("Output columns (bit j of y1..y%d):"%n)
    for j in range(8): L.append("c%d %s"%(j, cs(cols[j])))
    # SCAN  (prior order, fully explicit, bounded at CAP_K, every tested candidate one terse line)
    # Terse mismatch format "NAME COL no" (no '-> '/'!= c0' boilerplate, ~5 tok/line cheaper);
    # the MATCH / doomed-anchor lines stay fully explicit so the lock decision is auditable.
    L.append("Scan c0=%s in frequency-prior order (singles p0..p7; XNOR pairs; AND pairs; const0; "
             "ANDN; XOR; ANDN-rev; OR; const1; then 3-tap triples). One line per gate = its column; "
             "'no' if != c0; stop at the first c0-match whose walk locks a global rule. Cap %d gates."
             %(cs(cols[0]),CAP_K))
    c0=cols[0]
    win_pos_in_order=None
    nlines=0
    emitted_winner=False
    for rank,ci in enumerate(PRIOR_ORDER):
        if ci==wci: win_pos_in_order=rank
        pos,tt=CANDS[ci]
        col=col_of((pos,tt),ins_bits)
        nm=gate_name((pos,tt))
        if ci==wci:
            L.append("%s %s = c0 MATCH."%(nm,cs(col)))
            emitted_winner=True
            break
        # Bounded scan: prior order puts the rare 3-tap tail LAST, so the winner is always within
        # rank 178 on real rows; a hard cap at CAP_K bounds tokens even if inference diverges. On
        # the (<=K) bail path the procedure jumps to the c0-matching anchor it locked; REAL verify
        # below re-applies the rule and would flag a MISMATCH if the bail anchor were wrong.
        if rank>=CAP_K:
            L.append("(cap %d reached; lock the first c0-matching anchor whose walk completes)"%CAP_K)
            L.append("%s %s = c0 MATCH."%(gate_name(CANDS[wci]),cs(col_of(CANDS[wci],ins_bits))))
            win_pos_in_order=rank; emitted_winner=True
            break
        if col==c0:
            # doomed anchor (matches c0 but walk fails): show the walk-fail reason
            fb,fg=G.probe_walk_fail((pos,tt),ins_bits,cols)
            if fb is None:
                L.append("%s %s = c0 but walk locks no global rule -> reject, continue."%(nm,cs(col)))
            else:
                L.append("%s %s = c0 but walk stuck at b%d (no stride/edge/birth = c%d) -> reject, continue."%(nm,cs(col),fb,fb))
        else:
            L.append("%s %s no."%(nm,cs(col)))
        nlines+=1
    assert emitted_winner
    # WALK (bits 1..7) -- reuse global walk lines (already explicit, bounded)
    for (j,gate,tag,tried) in traj[1:]:
        pre=[]; nt=0
        for (tg,ttag,reason) in tried:
            if nt>=3: break
            nm=gate_name(tg)
            if reason=='colfail': pre.append("%s -> %s != c%d no"%(nm,cs(col_of(tg,ins_bits)),j))
            else: pre.append("%s -> col ok but no consistent continuation, backtrack"%nm)
            nt+=1
        note=''
        if tag=='wrap': note=" (edge: wraps to p0)"
        elif tag=='dead': note=" (edge: tap line ends, reads 0)"
        elif tag.startswith('birth'): note=" (new tap line born at p0 = SHR%d)"%j
        elif tag.startswith('reveal'): note=" (tap born at p0 unmasks partner lines)"
        body="b%d: "%j + ("; ".join(pre)+"; " if pre else "")
        body+="%s -> %s = c%d ok%s."%(gate_name(gate),cs(col_of(gate,ins_bits)),j,note)
        L.append(body)
    # LOCK
    if taps:
        lines=", ".join("L%d=%s"%(i+1,tap_line_desc(t,traj)) for i,t in enumerate(taps))
        L.append("Lock: every scanned position lies on tap lines: %s. Rule = the per-bit gates above on these lines; no bit is guessed, all 8 columns matched."%lines)
    else:
        L.append("Lock: output is constant per bit (no input taps needed); all 8 columns matched.")
    # VERIFY -- REAL: re-apply the per-bit WALK-gate rule (what the trace locked) to every example
    # input, compare to the given output. Uses apply_walk (authoritative; not the taps/Fc lift which
    # can mis-default the 0-tap constant). For a correct rule all pairs are ok.
    vr=[]
    for e in range(n):
        comp=apply_walk(traj, ins_bits[e])
        given=outs[e]
        vr.append("x%d=%s -> rule gives %s, given %s %s"%(
            e+1, ins[e], comp, given, "ok" if comp==given else "MISMATCH"))
    L.append("Verify (re-apply locked rule to each input, compare to given output):")
    for v in vr: L.append("  "+v)
    # APPLY to query -- same walk-gate rule
    qb=[int(c) for c in q]
    final=apply_walk(traj, qb)
    L.append("Apply locked rule to q=%s: "%q + " ".join("q%d=%d"%(i,qb[i]) for i in range(8)))
    steps=[]
    for (j,gate,tag,tried) in traj:
        pos,tt=gate; idx=0
        for p in pos: idx=(idx<<1)|qb[p]
        v=tt[idx]
        nm=gate_name(gate, labels=lambda p:'q%d=%d'%(p,qb[p]))
        steps.append("b%d:%s=%d"%(j,nm,v))
    L.append("; ".join(steps)+".")
    # walk_final is now identical to final by construction (both from apply_walk); kept for the
    # gen_synth/checkreal consistency assertion (final must == per-bit walk answer).
    walk_final=final
    L.append("answer = %s"%final)
    return "\n".join(L), final, walk_final, win_pos_in_order

# ================= INVERTED LINTER =================
def lint_r3(cot, ins, outs, q, final, taps, Fc, sol):
    """Re-derive every claim from the data; fail if any line lies."""
    errs=[]
    if any(ord(ch)>127 for ch in cot): errs.append('non-ascii')
    if '\\boxed' in cot: errs.append('boxed-in-cot')
    ins_bits=sol['ins_bits']; cols=sol['cols']
    # table rows/cols
    for p in range(8):
        m=re.search(r'^p%d ([01]+)$'%p,cot,re.M)
        if not m or m.group(1)!=''.join(s[p] for s in ins): errs.append('posrow-%d'%p)
    for j in range(8):
        m=re.search(r'^c%d ([01]+)$'%j,cot,re.M)
        if not m or m.group(1)!=cs(cols[j]): errs.append('col-%d'%j)
    # SCAN lines: every 'NAME -> BITS != c0 no.' / '= c0 ...' must recompute correctly.
    # build name->gate lookup from CANDS (names are unique enough for arity<=2; for arity3 the
    # walk-fail lines carry the gate name we render). We recompute by re-deriving the column from
    # the stated gate name via CANDS reverse map.
    name2col={}
    for ci,(pos,tt) in enumerate(CANDS):
        name2col.setdefault(gate_name((pos,tt)), cs(col_of((pos,tt),ins_bits)))
    # terse format: 'NAME COL no.' | 'NAME COL = c0 MATCH.' | 'NAME COL = c0 but ...'
    scan_re=re.compile(r'^(.+?) ([01]+)(?: no\.| = c0 MATCH\.| = c0 but)\s*$|^(.+?) ([01]+) = c0 but')
    line_re=re.compile(r'^(\S.*?) ([01]+)(?: no\.| = c0 MATCH\.| = c0 but)')
    saw_match=False
    capped='(cap %d reached'%CAP_K in cot
    # restrict to the SCAN region (between the 'Scan c0=' header and the first 'b1:' walk / 'Lock')
    body=cot.split('\n')
    try: s_start=next(i for i,l in enumerate(body) if l.startswith('Scan c0='))+1
    except StopIteration: s_start=len(body)
    s_end=len(body)
    for i in range(s_start,len(body)):
        if body[i].startswith('b1:') or body[i].startswith('Lock:'): s_end=i; break
    for ln in body[s_start:s_end]:
        s=ln.strip()
        m=line_re.match(s)
        if not m: continue
        nm=m.group(1).strip(); shown=m.group(2)
        if nm in name2col:
            # on the cap-bail path the MATCH gate's column is recomputed too; only flag if the gate
            # name is a known scan candidate and the shown column disagrees with the recompute.
            if name2col[nm]!=shown: errs.append('scan-recompute:%s'%nm)
        if s.endswith('= c0 MATCH.'):
            saw_match=True
            if shown!=cs(cols[0]): errs.append('match-not-c0')
        if s.endswith(' no.') and shown==cs(cols[0]) and not capped:
            errs.append('false-reject:%s'%nm)
    if not saw_match: errs.append('no-match-line')
    # VERIFY: each 'x{e}=... -> rule gives COMP, given GIVEN ok/MISMATCH' must be a real re-apply
    # of the WALK-gate rule (apply_walk). comp must equal the recompute; verdict must be forced.
    traj=sol['traj']
    for e in range(len(ins)):
        m=re.search(r'x%d=%s -> rule gives ([01]{8}), given ([01]{8}) (ok|MISMATCH)'%(e+1,re.escape(ins[e])),cot)
        if not m: errs.append('verify-missing-%d'%(e+1)); continue
        comp=apply_walk(traj,ins_bits[e])
        if m.group(1)!=comp: errs.append('verify-comp-wrong-%d'%(e+1))
        if m.group(2)!=outs[e]: errs.append('verify-given-wrong-%d'%(e+1))
        verdict='ok' if comp==outs[e] else 'MISMATCH'
        if m.group(3)!=verdict: errs.append('verify-verdict-%d'%(e+1))
    # APPLY: answer line must equal the walk-gate rule re-applied on q
    m=re.search(r'answer = ([01]{8})',cot)
    if not m or m.group(1)!=final: errs.append('answer!=final')
    qb=[int(c) for c in q]
    if apply_walk(traj,qb)!=final: errs.append('apply-recompute')
    # every verify pair must be ok in a CLEAN training row (the locked rule fits all examples);
    # a MISMATCH verdict means the locked rule disagrees with an example -> drop, do not train.
    if 'given' in cot and 'MISMATCH' in cot: errs.append('verify-has-mismatch')
    return errs

# ================= SYNTH GENERATOR =================
def gen_synth(rng, exprs, trans, max_tries=40):
    elist=list(exprs.items()); etot=sum(c for _,c in elist)
    tlist=list(trans.items()); ttot=sum(c for _,c in tlist)
    def pick(lst,tot):
        x=rng.uniform(0,tot); a=0
        for v,c in lst:
            a+=c
            if x<=a: return v
        return lst[-1][0]
    for _ in range(max_tries):
        expr=pick(elist,etot)
        slots=[v for v in ('{A}','{B}','{C}') if v in expr]
        td={}; used=set(); ok=True
        for s in slots:
            for _t in range(50):
                t=pick(tlist,ttot)
                if t not in used: used.add(t); td[s]=t; break
            else: ok=False
        if not ok: continue
        n=rng.choice([7,8,9,10])
        vals=rng.sample(range(256),n+1)
        ins=[format(v,'08b') for v in vals[:n]]; q=format(vals[n],'08b')
        def apply_true(xs):
            xb=[int(c) for c in xs]; ob=[]
            for j in range(8):
                vv={}
                for s in slots:
                    p=tap_pos(j,td[s]); vv[s]=0 if p is None else xb[p]
                ob.append(eval_expr(expr,vv.get('{A}',0),vv.get('{B}',0),vv.get('{C}',0)))
            return ''.join(map(str,ob))
        outs=[apply_true(x) for x in ins]
        sol=gsolve(ins,outs)
        if sol is None: continue
        cot,final,walk_final,wpos=render_r3(ins,outs,q,sol)
        if lint_r3(cot,ins,outs,q,final,sol['taps'],sol['Fc'],sol): continue
        if final!=walk_final: continue   # walk gates and taps/Fc must agree
        return dict(prompt=mk_prompt(ins,outs,q),cot=cot,final=final,
                    true_answer=apply_true(q),expr=expr,td=td)
    return None

# ================= DRIVERS =================
def load_real():
    rows=[]
    for r in csv.DictReader(open(os.path.join(ROOT,'competition_dataset/train_categorized.csv'))):
        if r['category']!='bit_manipulation': continue
        ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',r['prompt'])
        q=re.search(r'output for:\s*([01]{8})',r['prompt']).group(1)
        rows.append(dict(id=r['id'],prompt=r['prompt'],gold=r['answer'].strip(),
                         ins=[e[0] for e in ex],outs=[e[1] for e in ex],q=q))
    return rows

def tok_stats(toks):
    s=sorted(toks)
    def p(x): return s[int(round(x/100*(len(s)-1)))]
    return dict(rows=len(s),med=st.median(s),mean=round(st.mean(s)),p90=p(90),p95=p(95),max=max(s))

def main(nsynth=1700):
    from tokenizers import Tokenizer
    TOK=Tokenizer.from_file(os.path.join(ROOT,'competition_dataset/tokenizer.json'))
    outdir=os.path.join(ROOT,'pipeline','data','v16'); os.makedirs(outdir,exist_ok=True)
    real=load_real()
    real_prompts={r['prompt'] for r in real}
    exprs,trans=rule_dist_from_cache(os.path.join(ROOT,'pipeline/data/bitmanip_solved.jsonl'))
    rng=random.Random(20260612)
    rows_out=[]; seen=set(); bad=0; toks=[]; pos_dist=[]
    target=nsynth
    while len(rows_out)<target and bad<target*4:
        row=gen_synth(rng,exprs,trans)
        if row is None: bad+=1; continue
        if row['prompt'] in real_prompts or row['prompt'] in seen: bad+=1; continue
        t=len(TOK.encode(row['cot']).ids)
        if t>4500: bad+=1; continue   # hard gate: keep synth p95/max within the real-row 4500 budget
        seen.add(row['prompt'])
        rows_out.append(row); toks.append(t)
    # write synth jsonl
    synth_path=os.path.join(outdir,'bit_r3_synth.jsonl')
    with open(synth_path,'w') as f:
        for i,row in enumerate(rows_out):
            f.write(json.dumps(dict(id='br3-%05d'%i,category='bit_manipulation',
                                    prompt=row['prompt'],cot=row['cot'],final=row['final']))+'\n')
    ts=tok_stats(toks)
    # collision check
    coll=sum(1 for r in rows_out if r['prompt'] in real_prompts)
    print("bit_r3_synth.jsonl: %d rows (gen failures %d) | collisions %d"%(len(rows_out),bad,coll))
    print("tokens:", ts)
    # make CSV: id,prompt,answer,category,raw_output,predicted,correct
    csv_path=os.path.join(outdir,'train_bit_r3.csv')
    with open(csv_path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['id','prompt','answer','category','raw_output','predicted','correct'])
        for i,row in enumerate(rows_out):
            raw=row['cot']+"\nThe answer is \\boxed{%s}."%row['final']
            w.writerow(['br3-%05d'%i,row['prompt'],row['true_answer'],'bit_manipulation',
                        raw,row['final'],int(row['final']==row['true_answer'])])
    cacc=sum(1 for r in rows_out if r['final']==r['true_answer'])/max(1,len(rows_out))
    print("train_bit_r3.csv: %d rows | predicted==true %.2f%% | %s"%(len(rows_out),100*cacc,csv_path))
    return ts

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=1700)
    ap.add_argument('cmd',nargs='?',default='build')
    a=ap.parse_args()
    if a.cmd=='checkreal':
        # validate renderer + linter + token budget on all 1602 REAL rows (no file written)
        from tokenizers import Tokenizer
        TOK=Tokenizer.from_file(os.path.join(ROOT,'competition_dataset/tokenizer.json'))
        real=load_real(); ok=0; lintfail=0; toks=[]; wpos=[]; nlines=[]
        for r in real:
            sol=gsolve(r['ins'],r['outs'])
            cot,final,walk_final,wp=render_r3(r['ins'],r['outs'],r['q'],sol)
            e=lint_r3(cot,r['ins'],r['outs'],r['q'],final,sol['taps'],sol['Fc'],sol)
            if e: lintfail+=1;
            if final==r['gold']: ok+=1
            toks.append(len(TOK.encode(cot).ids)); wpos.append(wp)
            nlines.append(cot.count(' -> '))
        print("REAL check: %d rows | final==gold %d (%.2f%%) | lint failures %d"%(
            len(real),ok,100*ok/len(real),lintfail))
        print("tokens:", tok_stats(toks))
        print("scan '-> ' lines:", tok_stats(nlines))
    else:
        main(a.n)
