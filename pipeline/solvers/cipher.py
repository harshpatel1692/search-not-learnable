"""cipher: monoalphabetic substitution over a fixed 77-word vocab.
Constraint-propagation solver (bijection + vocab) -> 100% on train."""
import re
def build_vocab(rows):
    v=set()
    for r in rows:
        for line in r['prompt'].splitlines():
            if '->' in line:
                for w in line.split('->',1)[1].split(): v.add(w.lower())
        for w in r['answer'].split(): v.add(w.lower())
    return v
def _example_map(prompt):
    m={}
    for line in prompt.splitlines():
        if '->' not in line: continue
        l,r=line.split('->',1)
        for cw,pw in zip(l.split(), r.split()):
            if len(cw)==len(pw):
                for a,b in zip(cw,pw): m[a]=b
    return m
def solve(prompt, vocab):
    q=re.search(r'decrypt the following text:\s*(.+)$', prompt, re.M).group(1).strip().split()
    m=_example_map(prompt); inv={v:k for k,v in m.items()}
    # constraint propagation: lock query words whose vocab match is unique under the bijection
    changed=True
    while changed:
        changed=False
        for cw in q:
            if all(c in m for c in cw): continue
            cands=[]
            for w in vocab:
                if len(w)!=len(cw): continue
                ok=True; mm={}
                for i,c in enumerate(cw):
                    if c in m and m[c]!=w[i]: ok=False; break
                    if c not in m and w[i] in inv and inv[w[i]]!=c: ok=False; break
                    if c in mm and mm[c]!=w[i]: ok=False; break
                    mm[c]=w[i]
                if ok: cands.append(w)
            if len(set(cands))==1:
                for i,c in enumerate(cw):
                    if c not in m: m[c]=cands[0][i]; inv[cands[0][i]]=c; changed=True
    ans=' '.join(''.join(m.get(c,'?') for c in cw) for cw in q)
    # CONCISE worked CoT: one grounding alignment, the consolidated query-letter map, then word decode.
    qletters=set(c for cw in q for c in cw)
    ground=""
    for line in prompt.splitlines():
        if '->' not in line: continue
        l,r=line.split('->',1)
        done=False
        for cw,pw in zip(l.split(), r.split()):
            if len(cw)==len(pw) and any(c in qletters for c in cw):
                ground=f"'{cw}'→'{pw}' gives "+",".join(f"{a}→{b}" for a,b in zip(cw,pw)); done=True; break
        if done: break
    mapstr=", ".join(f"{c}→{m[c]}" for c in sorted(qletters) if c in m)
    words="; ".join(f"'{cw}'→'{''.join(m.get(c,'?') for c in cw)}'" for cw in q)
    cot=(f"Each ciphertext letter maps to one fixed plaintext letter — align example word-pairs by "
         f"position (e.g. {ground}). The mappings for the query's letters are: {mapstr}. "
         f"Decode each query word: {words}. Result: {ans}.")
    return ans, cot
