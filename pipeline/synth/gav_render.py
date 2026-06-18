"""r9 Generate-and-Verify renderer (accept path, milestone 1).

Grammar (the unlock vs r1-r8): SEARCH over the OPERATOR (base-doable), then the
digit map FALLS OUT as a forced chain (committed-op propagation forces all digits
for ~69% of trainable rows). Each pin is justified by a column and, the moment an
equation's glyphs are all pinned, that equation is VERIFIED with real 2-digit
arithmetic inline (anti-teleport anchor r8 lacked). Ends with a full verify sweep
+ explicit answer encode.

This module renders the ACCEPT path only (committed correct operator, fully/near
forced map). Operator enumeration + bounded-search rejects = milestone 3.

  python3 pipeline/synth/gav_render.py demo [n]   # render n synth accept traces
"""
import os, sys, time, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
import propagation as P
import cryptarithm2 as C2
import cryptarithm_cot as CC

OPW = {'add': '+', 'sub': '-', 'mul': '*', 'floordiv': 'floordiv', 'mod': 'mod',
       'rsub': 'rsub', 'rfloordiv': 'rfloordiv', 'rmod': 'rmod',
       'concat_fwd': 'concatenation', 'concat_rev': 'reverse-concatenation'}
# human phrase for the obfuscated families (a op b then +/-k, etc.)
OP_DESC = {
    'add': 'a + b', 'sub': 'a - b', 'mul': 'a * b',
    'add_p1': 'a + b + 1', 'add_m1': 'a + b - 1', 'add_m2': 'a + b - 2',
    'add_p2': 'a + b + 2', 'mul_p1': 'a * b + 1', 'mul_m1': 'a * b - 1',
    'mul_p2': 'a * b + 2', 'mul_m2': 'a * b - 2', 'lcm': 'lcm(a, b)',
    'absdiff': '|a - b|', 'absdiff_p1': '|a - b| + 1', 'absdiff_m1': '|a - b| - 1',
    'absdiff_p2': '|a - b| + 2', 'absdiff_m2': '|a - b| - 2',
    'sub_signed': 'a - b (signed)', 'rsub_signed': 'b - a (signed)',
    'neg_absdiff': '-|a - b|', 'mod': 'a mod b', 'floordiv': 'a // b',
}


def op_phrase(op):
    return OP_DESC.get(op, op)


def op_expr(op, A, B):
    """concrete arithmetic expression with the actual operand values substituted."""
    base = {'add': f'{A} + {B}', 'sub': f'{A} - {B}', 'mul': f'{A} * {B}',
            'sub_signed': f'{A} - {B}', 'rsub_signed': f'{B} - {A}',
            'rsub': f'{B} - {A}', 'absdiff': f'|{A} - {B}|',
            'neg_absdiff': f'-|{A} - {B}|', 'lcm': f'lcm({A}, {B})',
            'mod': f'{A} mod {B}', 'floordiv': f'{A} // {B}',
            'rfloordiv': f'{B} // {A}', 'rmod': f'{B} mod {A}'}
    if op in base:
        return base[op]
    for stem, sym in (('mul', '*'), ('add', '+'), ('absdiff', None)):
        if op.startswith(stem + '_'):
            tail = op[len(stem) + 1:]
            delta = {'p1': ' + 1', 'm1': ' - 1', 'p2': ' + 2', 'm2': ' - 2'}.get(tail, '')
            core = f'|{A} - {B}|' if stem == 'absdiff' else f'{A} {sym} {B}'
            return core + delta
    return f'{op}({A}, {B})'


def _decode(seq, m):
    v = 0
    for g in seq:
        if g not in m:
            return None
        v = v * 10 + m[g]
    return v


def render_accept(prompt, ops, rev, deadline_s=10.0):
    """render the committed-operator forcing-chain + inline verify + answer.
    returns (cot_lines, boxed) or (None, None) if not fully forced (deep search)."""
    eqs, qL = C2.parse(prompt)
    glyphs = {L[2] for L, R in eqs} | {qL[2]}
    canonical = glyphs <= set('+-*')
    vocab = {g: [ops[g]] for g in glyphs}     # COMMIT the operator hypothesis
    eng = P.Eng(eqs, qL, vocab, canonical, time.time() + deadline_s)
    eng.propagate()
    # require full forcing for the clean accept grammar (milestone 1)
    if any(len(eng.dom[s]) != 1 for s in eng.syms):
        return None, None
    m = {s: next(iter(eng.dom[s])) for s in eng.syms}

    L = []
    # 1) operator hypothesis stated
    opstr = ', '.join(f"'{g}' = {op_phrase(ops[g])}" for g in sorted(glyphs))
    L.append(f"Assume the operators are: {opstr}.")
    L.append(f"Reading order: {'reversed (little-endian)' if rev else 'normal'}.")
    L.append("Now pin the digits column by column; each equation is checked as soon "
             "as its letters are all known.")

    # 2) forcing chain, with inline equation verification
    eq_glyphs = []
    for (Lh, Rh) in eqs:
        gs = set(Lh[0] + Lh[1] + Lh[3] + Lh[4] + Rh)
        eq_glyphs.append((Lh, Rh, gs))
    verified = [False] * len(eqs)
    known = set()

    def emit_ready_checks():
        for i, (Lh, Rh, gs) in enumerate(eq_glyphs):
            if verified[i] or not gs <= known:
                continue
            op = ops[Lh[2]]
            if op in C2.CONCATS:
                want = (Lh[0]+Lh[1]+Lh[3]+Lh[4]) if op == 'concat_fwd' else (Lh[3]+Lh[4]+Lh[0]+Lh[1])
                L.append(f"  check {''.join(Lh)} = {Rh}: concatenation gives {want} = {Rh}. ok")
            else:
                A = _decode(Lh[1]+Lh[0] if rev else Lh[0]+Lh[1], m)
                B = _decode(Lh[4]+Lh[3] if rev else Lh[3]+Lh[4], m)
                val = C2.OPS[op](A, B)
                sval = abs(val) if (op in C2.SIGNED or op in C2.NEGPRE) else val
                neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and val < 0)) else ''
                L.append(f"  check {''.join(Lh)} = {Rh}: {op_expr(op, A, B)} = "
                         f"{neg}{sval}, reads as {Rh}. ok")
            verified[i] = True

    for ev in eng.sev:
        if ev[0] != 'pin':
            continue
        _, s, d, why = ev
        if why[0] == 'fit':
            reason = "forced by the column arithmetic"
        elif why[0] == 'only_left':
            reason = "the only digit still free"
        elif why[0] == 'bij':
            reason = "the last unused digit"
        elif why[0] == 'split':
            reason = "the only value that verifies"
        else:
            reason = "forced"
        L.append(f"  '{s}' = {d}: {reason}.")
        known.add(s)
        emit_ready_checks()
    # any concat / structural equations with no arithmetic glyphs
    known |= set(m)
    emit_ready_checks()

    # 3) answer encode (explicit: value -> sign -> endianness -> glyphs)
    ans, stall = eng.answer()
    if ans is None:
        return None, None
    qa, qb, qop, qc, qd = qL
    if ops[qop] in C2.CONCATS:
        L.append(f"Query {qa+qb}{qop}{qc+qd}: concatenation -> {ans}.")
    else:
        A = _decode((qb+qa) if rev else (qa+qb), m)
        B = _decode((qd+qc) if rev else (qc+qd), m)
        val = C2.OPS[ops[qop]](A, B)
        sval = abs(val) if (ops[qop] in C2.SIGNED or ops[qop] in C2.NEGPRE) else val
        neg = '-' if (ops[qop] in C2.NEGPRE or (ops[qop] in C2.SIGNED and val < 0)) else ''
        L.append(f"Query {qa+qb}{qop}{qc+qd}: {op_expr(ops[qop], A, B)} = {neg}{sval}.")
        L.append(f"Encode {neg}{sval} back through the digit map -> {ans}.")
    L.append(f"\\boxed{{{ans}}}")
    return L, ans


def demo(n=3):
    rng = random.Random(7)
    shown = 0
    tried = 0
    while shown < n and tried < 4000:
        tried += 1
        p = CC.gen_puzzle_r5(rng)
        if p is None:
            continue
        try:
            lines, ans = render_accept(p['prompt'], p['ops'], p['rev'])
        except Exception:
            continue
        if lines is None or ans != p['answer']:
            continue
        shown += 1
        print('=' * 70)
        print('PROMPT:')
        print('  ' + p['prompt'].split('examples:')[1].strip().replace('\n', '\n  '))
        print(f"(gold {p['answer']!r}, canonical={p['prompt'].count(chr(10))})")
        print('--- GAV accept trace ---')
        print('\n'.join(lines))
        print(f"--> boxed matches gold: {ans == p['answer']}")


def dump(n, path):
    """render n synthetic accept traces and SAVE them to a file for review."""
    rng = random.Random(11)
    out = []
    tried = 0
    canon = scr = 0
    while len(out) < n and tried < 40000:
        tried += 1
        p = CC.gen_puzzle_r5(rng)
        if p is None:
            continue
        try:
            lines, ans = render_accept(p['prompt'], p['ops'], p['rev'])
        except Exception:
            continue
        if lines is None or ans != p['answer']:
            continue
        is_canon = set(p['ops']) <= set('+-*')
        # keep a rough canonical/scrambled balance (real ~0.58 canonical)
        if is_canon and canon > 0.62 * n:
            continue
        canon += is_canon
        scr += (not is_canon)
        body = ['=' * 72,
                f"[sample {len(out)+1}]  ops={p['ops']}  rev={p['rev']}  "
                f"{'canonical' if is_canon else 'scrambled'}",
                'PROMPT:',
                '  ' + p['prompt'].split('examples:')[1].strip().replace('\n', '\n  '),
                f"GOLD: {p['answer']!r}",
                '--- GAV accept trace ---',
                '\n'.join(lines),
                f"--> boxed matches gold: {ans == p['answer']}", '']
        out.append('\n'.join(body))
    hdr = (f"# r9 GAV accept-path renders ({len(out)} samples; "
           f"{canon} canonical / {scr} scrambled)\n"
           f"# renderer: pipeline/synth/gav_render.py  (milestone 1: accept path only)\n"
           f"# every arithmetic line is real; digit pins are in propagation-forced order.\n\n")
    with open(path, 'w') as f:
        f.write(hdr + '\n'.join(out))
    print(f"wrote {len(out)} samples -> {path}  ({canon} canonical, {scr} scrambled)")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    if cmd == 'demo':
        demo(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == 'dump':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
            ROOT, 'pipeline', 'data', 'crypt_r9', 'gav_accept_samples.txt')
        dump(n, path)
