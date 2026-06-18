"""Base-probe (NVIDIA API): does the 30B LOOP less when it reasons in clean
letter-tokens (Ali-style A,B,C abstraction) vs raw cipher glyphs (our trace_we_need)?
Ali's letter form converged 96%; our glyph form looped 69%. Test it directly:
few-shot the base in each representation, greedy, measure truncation + correctness.

  python3 pipeline/probe_repr.py [n]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solvers'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'analysis', 'crypt_struct'))
import cryptarithm2 as C2
import nvidia_api as NV
import propagation as P
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = 'nvidia/nemotron-3-nano-30b-a3b'
OPPH = {'add': 'a+b', 'sub_signed': 'a-b', 'mul': 'a*b', 'mul_p1': 'a*b+1', 'mul_m1': 'a*b-1',
        'add_p1': 'a+b+1', 'add_m1': 'a+b-1', 'absdiff': '|a-b|', 'neg_absdiff': '-|a-b|',
        'concat_fwd': 'concat', 'concat_rev': 'concat'}


def solve_map(prompt):
    eqs, qL = C2.parse(prompt)
    res = C2.solve(prompt, deadline_s=4.0)
    if not res:
        return None
    ops = (res[1] or {}).get('ops'); rev = (res[1] or {}).get('rev')
    if ops is None:
        return None
    oc = {L[2] for L, R in eqs} | {qL[2]}
    try:
        e = P.Eng(eqs, qL, {g: [ops[g]] for g in oc}, oc <= set('+-*'), time.time() + 5); e.propagate()
    except Exception:
        return None
    if any(len(e.dom[s]) != 1 for s in e.syms):
        return None
    return eqs, qL, ops, rev, {s: next(iter(e.dom[s])) for s in e.syms}, res[0]


def render_letter(prompt, worked=True):
    """letter-abstraction solve: rename glyphs A,B,C... (by first appearance), reason in
    letters + numbers, map back to glyphs. (the representation the model handled at 96%.)"""
    out = solve_map(prompt)
    if out is None:
        return None
    eqs, qL, ops, rev, m, ans = out
    glyphs = list(dict.fromkeys(c for L, R in eqs for c in ''.join(L[:2] + L[3:]) + R if c not in '+-*'))
    alias = {g: chr(ord('A') + i) for i, g in enumerate(glyphs)}
    opal = {L[2]: L[2] for L, R in eqs}
    lines = ["Assign a letter to each symbol (first appearance): " +
             ", ".join(f"{g}->{a}" for g, a in alias.items()) + "."]
    lines.append("Operators: " + ", ".join(f"'{g}'={OPPH.get(op, op)}" for g, op in
                 {L[2]: ops[L[2]] for L, R in eqs}.items()) + ".")
    lines.append("Digits (solved): " + ", ".join(f"{alias[g]}={m[g]}" for g in glyphs if g in m) + ".")
    lines.append("Verify:")
    for (L, R) in eqs:
        op = ops[L[2]]
        if op in C2.CONCATS:
            lines.append(f"  {''.join(L)} = {R}: concatenation, ok"); continue
        A = (m[L[1]]*10+m[L[0]]) if rev else (m[L[0]]*10+m[L[1]])
        B = (m[L[4]]*10+m[L[3]]) if rev else (m[L[3]]*10+m[L[4]])
        v = C2.OPS[op](A, B); sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
        lines.append(f"  {''.join(L)} = {R}: {A} {OPPH.get(op,op)} = {sv}, ok")
    qa, qb, qop, qc, qd = qL; qt = ops[qop]
    A = (m[qb]*10+m[qa]) if rev else (m[qa]*10+m[qb]); B = (m[qd]*10+m[qc]) if rev else (m[qc]*10+m[qd])
    v = C2.OPS[qt](A, B); sv = abs(v) if (qt in C2.SIGNED or qt in C2.NEGPRE) else v
    lines.append(f"Query {''.join(qL)}: {A} {OPPH.get(qt,qt)} = {sv} -> map digits back to symbols -> {ans}")
    lines.append(f"\\boxed{{{ans}}}")
    return "\n".join(lines), ans


GLYPH_INSTR = ("Solve the symbol cipher. Find the digit for each symbol so every example holds, "
               "then apply to the query. Keep symbols spaced. End with \\boxed{}.")
LETTER_INSTR = ("Solve the symbol cipher by abstracting each symbol to a letter, finding the digits, "
                "then mapping back. End with \\boxed{}.")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    NV.set_experiment('repr_probe')
    val = [json.loads(l) for l in open(os.path.join(ROOT, 'pipeline/data/val.jsonl'))
           if json.loads(l)['category'] == 'cryptarithm_deduce']
    # build one worked exemplar of each kind from a solvable held-out-ish synth-like train row
    import random
    rng = random.Random(1)
    exemplar_g = exemplar_l = None
    for r in val[50:]:
        lr = render_letter(r['prompt'])
        if lr:
            exemplar_l = (r['prompt'], lr[0]); break
    sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
    import crypt_twn as T
    for r in val[50:]:
        out = solve_map(r['prompt'])
        if out:
            cot, ans, kind = T.render(r['prompt'], out[2], out[3])
            if cot and kind == 'solve':
                exemplar_g = (r['prompt'], cot); break
    rows = val[:n]
    for tag, instr, ex in (('GLYPH', GLYPH_INSTR, exemplar_g), ('LETTER', LETTER_INSTR, exemplar_l)):
        if ex is None:
            print(f"[{tag}] no exemplar built -- skipped", flush=True); continue
        trunc = ok = err = 0
        for r in rows:
            prompt = (instr + "\n\nWorked example:\n" + ex[0] + "\n\n" + ex[1] +
                      "\n\nNow solve:\n" + r['prompt'])
            o = NV.ask(prompt, model=MODEL, max_tokens=6000, temperature=0.0, add_box=False,
                       meta={'id': r['id'], 'repr': tag})
            if not isinstance(o, dict) or 'error' in o:
                err += 1; continue
            text = (o.get('reasoning') or '') + (o.get('content') or '')
            pred = o.get('answer')
            trunc += (o.get('finish') == 'length') or ('\\boxed' not in text)
            ok += (pred == r['answer'])
        print(f"[{tag}] {len(rows)} rows: truncated(finish=length/no-box) {trunc} | "
              f"correct {ok} | api-err {err}", flush=True)


if __name__ == '__main__':
    main()
