"""r9 cryptarithm_deduce renderer — the purified/terse/spaced best-shot for Tinker.

Lessons folded in (vs r1-r8 which scored 0.03-0.07):
 - PURIFY: only emit rows the engine FULLY solves honestly. Free-digit rows (22%,
   no honest answer) and deep-stall rows auto-drop (engine doesn't fully force).
   No fabrication target ever enters the corpus.
 - GLYPH-SPACING: every cipher glyph is space-separated in the CoT (the tokenizer
   merges adjacent glyphs e.g. '>[' -> 1 token; spacing -> 1 token/glyph). Final
   \boxed{} is UNspaced to match the grader.
 - HONEST WITNESS per digit: each pin shows the equation that forces it with real
   arithmetic (single-unknown-given-known); injectivity pins say so. No teleport.
 - VERIFY: full real-arithmetic check of every example under the found map.
 - ENCODE RITUAL: compute the query value, map digits back to glyphs one per line,
   then box (fights the Mamba copy-weakness / glyph-garbling).

  python3 pipeline/synth/crypt_r9.py demo [n]
  python3 pipeline/synth/crypt_r9.py build [n] [seed]   # -> data/crypt_r9/crypt_deduce_synth.jsonl
"""
import os, sys, time, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
import propagation as P
import cryptarithm2 as C2
import cryptarithm_cot as CC
from gav_render import op_expr, op_phrase, _decode   # reuse tested helpers


def sp(s):
    """space out a glyph string: '&>[&' -> '& > [ &' (one token per glyph)."""
    return ' '.join(s)


def operands(L, rev):
    """return (op1_glyphs, op2_glyphs) in value order (high,low)."""
    a = (L[1], L[0]) if rev else (L[0], L[1])
    b = (L[4], L[3]) if rev else (L[3], L[4])
    return a, b


def _cue(op, rlen):
    """operator-family hint from the RESULT length (no digit values needed)."""
    if op in C2.CONCATS:
        return "the result is the operands written side by side, so it is concatenation"
    if rlen >= 4:
        return "the result has 4 symbols; only multiplying two 2-digit numbers reaches 4 digits, so it is a multiplication"
    if rlen == 3:
        return "the result has 3 symbols; two 2-digit numbers reach 3 digits by adding, so try a sum"
    if rlen == 1:
        return "the result has 1 symbol, a small value, so it is a difference / remainder"
    return "the result has 2 symbols, so it is a small sum or a difference"


def render_v2(prompt, ops, rev, deadline_s=10.0):
    """trace_we_need.txt format: spaced restatement -> operator cues from result
    length -> reading-direction note -> witnessed digit derivation (most-constrained
    equation first) -> map -> verify -> spaced encode. Same purify filter as render().
    returns (cot, boxed) or (None, None)."""
    eqs, qL = C2.parse(prompt)
    glyphs = {L[2] for L, R in eqs} | {qL[2]}
    canonical = glyphs <= set('+-*')
    eng = P.Eng(eqs, qL, {g: [ops[g]] for g in glyphs}, canonical, time.time() + deadline_s)
    eng.propagate()
    if any(len(eng.dom[s]) != 1 for s in eng.syms):
        return None, None
    m = {s: next(iter(eng.dom[s])) for s in eng.syms}
    ans, _ = eng.answer()
    if ans is None:
        return None, None

    L = ["We need to infer the transformation rule from the examples. I rewrite "
         "everything spaced so I can track single glyphs."]
    for (Lh, Rh) in eqs:
        L.append(f"  {sp(''.join(Lh))}  =  {sp(Rh)}")
    L.append(f"Query:  {sp(''.join(qL))}")
    L.append("")
    L.append("Each left side is 5 symbols: two digit-symbols, an operator (middle), two "
             "digit-symbols. Read the operator of each from the result shape, no digit "
             "values needed yet:")
    seen = set()
    for i, (Lh, Rh) in enumerate(eqs, 1):
        g = Lh[2]
        if g in seen:
            continue
        seen.add(g)
        L.append(f"  EQ{i} '{g}': {_cue(ops[g], len(Rh))}, so '{g}' = {op_phrase(ops[g])}.")
    L.append("")
    L.append("Reading direction: " + ("the puzzle is little-endian (every number's digits "
             "reversed) -- standard order makes an example fail; reverse all numbers."
             if rev else "standard (left digit = tens)."))
    L.append("")
    # derivation: pins ordered as the engine forced them, each witnessed where possible
    L.append("Pin the digits, each forced by an example:")
    known = {}
    for ev in eng.sev:
        if ev[0] != 'pin':
            continue
        _, s, d, why = ev
        wit = None
        for (Lh, Rh), nr in zip(eqs, _norms(eqs, rev)):
            if nr is None or ops[Lh[2]] in C2.CONCATS:
                continue
            egl = {nr[0], nr[1], nr[2], nr[3], *nr[4]}
            if s not in egl or (egl - set(known) - {s}):
                continue
            op = ops[Lh[2]]
            a, b = operands(Lh, rev)
            full = dict(known); full[s] = d
            A, B = _decode(a, full), _decode(b, full)
            v = C2.OPS[op](A, B); sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
            wit = f"from {sp(''.join(Lh))} = {sp(Rh)}: {op_expr(op, A, B)} = {sv}, so '{s}' = {d}"
            break
        if wit is None:
            wit = (f"'{s}' = {d} ({d} is the only unused digit)" if why and why[0] in ('only_left', 'bij')
                   else f"'{s}' = {d} (only value the examples allow)")
        L.append("  " + wit)
        known[s] = d
    L.append("")
    L.append("Map:  " + ",  ".join(f"{s} = {m[s]}" for s in sorted(m)))
    L.append("Verify every example:")
    for (Lh, Rh) in eqs:
        op = ops[Lh[2]]
        if op in C2.CONCATS:
            L.append(f"  {sp(''.join(Lh))} = {sp(Rh)}: concatenation, ok")
            continue
        a, b = operands(Lh, rev)
        A, B = _decode(a, m), _decode(b, m)
        v = C2.OPS[op](A, B); sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
        neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
        L.append(f"  {sp(''.join(Lh))} = {sp(Rh)}: {op_expr(op, A, B)} = {neg}{sv}, ok")
    qa, qb, qop, qc, qd = qL
    qt = ops[qop]
    if qt in C2.CONCATS:
        L.append(f"Query is concatenation -> {sp(ans)}")
    else:
        a, b = operands(qL, rev)
        A, B = _decode(a, m), _decode(b, m)
        v = C2.OPS[qt](A, B); sv = abs(v) if (qt in C2.SIGNED or qt in C2.NEGPRE) else v
        neg = '-' if (qt in C2.NEGPRE or (qt in C2.SIGNED and v < 0)) else ''
        inv = {dd: s for s, dd in m.items()}
        steps = ", ".join(f"{ch}->{inv.get(int(ch),'(new)')}" for ch in str(sv))
        L.append(f"Apply to the query {sp(''.join(qL))}: {op_expr(qt, A, B)} = {neg}{sv}. "
                 f"Encode back to glyphs ({steps}){', reversed for little-endian' if rev else ''} -> {sp(ans)}")
    L.append(f"\\boxed{{{ans}}}")
    return "\n".join(L), ans


def render(prompt, ops, rev, deadline_s=10.0):
    """returns (cot, boxed) or (None, None) if not honestly fully solved."""
    eqs, qL = C2.parse(prompt)
    glyphs = {L[2] for L, R in eqs} | {qL[2]}
    canonical = glyphs <= set('+-*')
    eng = P.Eng(eqs, qL, {g: [ops[g]] for g in glyphs}, canonical, time.time() + deadline_s)
    eng.propagate()
    if any(len(eng.dom[s]) != 1 for s in eng.syms):
        return None, None                      # not fully forced -> drop (purify)
    m = {s: next(iter(eng.dom[s])) for s in eng.syms}

    L = []
    L.append("Each example: 5 symbols, the middle one is the operator, the outer two "
             "pairs are 2-digit numbers. I rewrite everything spaced so I can track "
             "single glyphs.")
    for (Lh, Rh) in eqs:
        L.append(f"  {sp(''.join(Lh))}  =  {sp(Rh)}")
    L.append(f"Query:  {sp(''.join(qL))}")

    # operator statement (committed hypothesis)
    L.append("Operators: " + "; ".join(
        f"'{g}' = {op_phrase(ops[g])}" for g in sorted(glyphs)) +
        (".  Reading little-endian (digits reversed)." if rev else ".  Reading normally."))

    # honest digit derivation: replay engine pin order, witness each
    L.append("Pin digits (each forced by an example):")
    pin_order = [(ev[1], ev[2], ev[3]) for ev in eng.sev if ev[0] == 'pin']
    known = {}
    used = set()
    for s, d, why in pin_order:
        wit = None
        # try: an example equation where s is the ONLY unknown given known
        for (Lh, Rh), nr in zip(eqs, _norms(eqs, rev)):
            if nr is None:
                continue
            op = ops[Lh[2]]
            egl = {nr[0], nr[1], nr[2], nr[3], *nr[4]}
            if s not in egl:
                continue
            if egl - set(known) - {s}:
                continue                       # other unknowns remain -> not a clean witness
            if op in C2.CONCATS:
                continue
            a, b = operands(Lh, rev)
            full = dict(known); full[s] = d
            A, B = _decode(a, full), _decode(b, full)
            v = C2.OPS[op](A, B)
            sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
            wit = f"from {sp(''.join(Lh))}={sp(Rh)}: {op_expr(op, A, B)} = {sv}, so '{s}' = {d}"
            break
        if wit is None:
            if why and why[0] in ('only_left', 'bij'):
                wit = f"'{s}' = {d} ({d} is the only digit left)"
            else:
                wit = f"'{s}' = {d} (only value consistent with the examples)"
        L.append("  " + wit)
        known[s] = d; used.add(d)

    # full verify (real arithmetic, spaced)
    L.append("Verify all examples:")
    for (Lh, Rh) in eqs:
        op = ops[Lh[2]]
        if op in C2.CONCATS:
            L.append(f"  {sp(''.join(Lh))} = {sp(Rh)}: concatenation, ok")
            continue
        a, b = operands(Lh, rev)
        A, B = _decode(a, m), _decode(b, m)
        v = C2.OPS[op](A, B); sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
        neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
        L.append(f"  {sp(''.join(Lh))} = {sp(Rh)}: {op_expr(op, A, B)} = {neg}{sv}, ok")

    # query + encode ritual (spaced)
    qa, qb, qop, qc, qd = qL
    qt = ops[qop]
    if qt in C2.CONCATS:
        ans = (qa + qb + qc + qd) if qt == 'concat_fwd' else (qc + qd + qa + qb)
        L.append(f"Query is concatenation -> {sp(ans)}")
    else:
        a, b = operands(qL, rev)
        A, B = _decode(a, m), _decode(b, m)
        v = C2.OPS[qt](A, B); sv = abs(v) if (qt in C2.SIGNED or qt in C2.NEGPRE) else v
        neg = '-' if (qt in C2.NEGPRE or (qt in C2.SIGNED and v < 0)) else ''
        ans_full, _ = eng.answer()
        if ans_full is None:
            return None, None
        ans = ans_full
        # encode: map each result digit back to its glyph, spaced, one per step
        inv = {d: s for s, d in m.items()}
        digits = (neg + str(sv))
        steps = []
        for ch in digits:
            if ch == '-':
                continue
            g = inv.get(int(ch))
            steps.append(f"{ch}->{g}" if g else f"{ch}->(new)")
        L.append(f"Query: {op_expr(qt, A, B)} = {neg}{sv}. Encode digits back to glyphs: "
                 + ", ".join(steps) + f" -> {sp(ans)}")
    L.append(f"\\boxed{{{ans}}}")
    return "\n".join(L), ans


def _norms(eqs, rev):
    return [C2.normalize(L, R, rev, L[2]) for L, R in eqs]


def demo(n=3):
    rng = random.Random(5)
    shown = tried = 0
    while shown < n and tried < 6000:
        tried += 1
        p = CC.gen_puzzle_r5(rng)
        if p is None:
            continue
        try:
            cot, ans = render(p['prompt'], p['ops'], p['rev'])
        except Exception:
            continue
        if cot is None or ans != p['answer']:
            continue
        shown += 1
        ntok = len(CC.tokenizer().encode(cot).ids)
        print('=' * 72)
        print(f"[demo {shown}] ops={p['ops']} rev={p['rev']} gold={p['answer']!r} ntok={ntok}")
        print(cot)
        print(f"--> matches gold: {ans == p['answer']}")


def build(n=3000, seed=900):
    rng = random.Random(seed)
    out_dir = os.path.join(ROOT, 'pipeline', 'data', 'crypt_r9')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    tried = 0
    seen = set()
    tok = CC.tokenizer()
    while len(rows) < n and tried < n * 40:
        tried += 1
        p = CC.gen_puzzle_r5(rng)
        if p is None or p['prompt'] in seen:
            continue
        try:
            cot, ans = render(p['prompt'], p['ops'], p['rev'])
        except Exception:
            continue
        if cot is None or ans != p['answer']:
            continue
        ntok = len(tok.encode(cot).ids)
        if ntok > 4000:
            continue
        seen.add(p['prompt'])
        rows.append({'id': f'r9-{seed}-{len(rows):05d}', 'category': 'cryptarithm_deduce',
                     'prompt': p['prompt'], 'cot': cot, 'final': ans, 'ntok': ntok})
    out = os.path.join(out_dir, 'crypt_deduce_synth.jsonl')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    import statistics
    nt = [r['ntok'] for r in rows]
    print(f"wrote {len(rows)} rows (tried {tried}, keep rate {len(rows)/tried:.2f}) -> {out}")
    print(f"ntok median {int(statistics.median(nt))} p95 {int(sorted(nt)[int(len(nt)*.95)])} max {max(nt)}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    if cmd == 'demo':
        demo(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == 'build':
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 3000,
              int(sys.argv[3]) if len(sys.argv) > 3 else 900)
