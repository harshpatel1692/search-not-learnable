"""equation_numeric v15: synthetic generator + CoT renderer for the SEQUENTIAL STRICT policy.

Policy (= pipeline/eq_fit.py policy_seq, measured on all 732 train rows):
  deduce 567/596 = 95.1%, guess 28/136 = 20.6% (adapter reference: 90.6% / 15.4%).

Generator model (fitted, 0 unexplained rows):
  - line = AB<g>CD = R ; per-glyph op from a 13-op library; one GLOBAL mode per row:
      REV   (P~0.665): operand digits reversed before computing AND result string reversed
      PLAIN (P~0.335): as written
  - negative results print the operator glyph as the minus sign:
      PLAIN          -> glyph in front
      REV, glyph '-' -> in front (canonical digits+'-' then string-reversed)
      REV, sub       -> in front
      REV, neg/rsub  -> after the digits
CoT discipline (cod-emit): the elimination, the lock line, the verification and the apply
block all reference the SAME (mode, op); final == the value computed by the apply block;
ASCII only; no \\boxed in the cot (the trainer appends "The answer is \\boxed{final}.").

Run:
  python3 pipeline/synth/equation_cot.py real    -> pipeline/data/v15/eq_deduce_real.jsonl
  python3 pipeline/synth/equation_cot.py synth   -> pipeline/data/v15/eq_deduce_synth.jsonl
  python3 pipeline/synth/equation_cot.py guess   -> pipeline/data/v15/eq_guess.jsonl
  python3 pipeline/synth/equation_cot.py stats   -> token statistics (tokenizer.json)
"""
import sys, os, json, random, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import eq_fit as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, 'pipeline', 'data', 'v15')
HDR = ("In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
       "Below are a few examples:")

MODENAME = {'rev': 'REV', 'id': 'PLAIN'}
OPNAME = {'add': 'add', 'add_m1': 'add-1', 'add_p1': 'add+1',
          'mul': 'mul', 'mul_m1': 'mul-1', 'mul_p1': 'mul+1',
          'sub': 'sub', 'negabsdiff': 'neg', 'absdiff': 'abs', 'rsub': 'rsub',
          'concat': 'concat', 'maxmodmin': 'max%min', 'rconcat': 'rconcat'}

# ------------------------------------------------------------------ target parsing
def parse_target(r, glyph, mode):
    """Printed result -> (value:int|None, string:str|None, pos:'front'|'end'|None).
    Numeric ops must match `string` (str(v) == string); sign results match (value, pos)."""
    if glyph in r and not r.isdigit():            # sign-marked negative
        digits = r.replace(glyph, '')
        pos = 'front' if r[0] == glyph else 'end'
        if mode == 'rev':
            digits = digits[::-1]
            pos = 'front' if pos == 'front' else 'end'   # printed-domain position kept
        return -int(digits), None, pos
    s = r[::-1] if mode == 'rev' else r
    return None, s, None

def expected_pos(mode, op, glyph):
    """PRINTED sign position the generator uses (strict model)."""
    if mode == 'id':
        return 'front'
    if glyph == '-' or op == 'sub':
        return 'front'
    return 'end'

def op_value(op, a, b, sa, sb):
    try:
        return E.OPS[op](a, b, sa, sb)
    except Exception:
        return None

def matches(op, a, b, sa, sb, glyph, mode, tval, tstr, tpos):
    v = op_value(op, a, b, sa, sb)
    if v is None:
        return False
    if isinstance(v, str):                         # concat family: compare canonical strings
        return tstr is not None and v == tstr
    if v < 0:
        return tval is not None and v == tval and tpos == expected_pos(mode, op, glyph)
    return tstr is not None and str(v) == tstr

# ------------------------------------------------------------------ trace emission
FAMS = [('addfam', ['add', 'add_m1', 'add_p1']),
        ('mulfam', ['mul', 'mul_m1', 'mul_p1']),
        ('difffam', ['sub', 'negabsdiff', 'absdiff', 'rsub']),
        ('concat', ['concat']),
        ('maxmodmin', ['maxmodmin']),
        ('rconcat', ['rconcat'])]
assert [op for _, mem in FAMS for op in mem] == E.SEQ_ORDER

def ops_of(sa, sb, mode):
    if mode == 'rev':
        sa, sb = sa[::-1], sb[::-1]
    return int(sa), int(sb), sa, sb

def tdesc(tval, tstr, tpos, r, mode):
    if tstr is not None:
        return f"target rev('{r}')='{tstr}'" if mode == 'rev' else f"target '{tstr}'"
    side = 'symbol in front' if tpos == 'front' else 'symbol at the end'
    return f"target {tval} ({side})"

def calc_text(op, a, b, sa, sb):
    v = op_value(op, a, b, sa, sb)
    if op == 'add':       return f"{a}+{b}={v}", v
    if op == 'add_m1':    return f"{a}+{b}-1={v}", v
    if op == 'add_p1':    return f"{a}+{b}+1={v}", v
    if op == 'mul':       return f"{a}*{b}={v}", v
    if op == 'mul_m1':    return f"{a}*{b}-1={v}", v
    if op == 'mul_p1':    return f"{a}*{b}+1={v}", v
    if op == 'sub':       return f"{a}-{b}={v}", v
    if op == 'negabsdiff':return f"-|{a}-{b}|={v}", v
    if op == 'absdiff':   return f"|{a}-{b}|={v}", v
    if op == 'rsub':      return f"{b}-{a}={v}", v
    if op == 'concat':    return f"'{sa}'+'{sb}'='{v}'", v
    if op == 'rconcat':   return f"'{sb}'+'{sa}'='{v}'", v
    if op == 'maxmodmin':
        hi, lo = max(a, b), min(a, b)
        return (f"max%min: {hi}%{lo}={v}", v) if v is not None else (f"max%min: min=0 undefined", None)
    raise KeyError(op)

def fam_line(fam, members, exa, glyph, mode, tval, tstr, tpos):
    """One compact trial line for a family on the FIRST example. Returns (text, hit_op)."""
    a, b, sa, sb = exa
    hit = next((op for op in members
                if matches(op, a, b, sa, sb, glyph, mode, tval, tstr, tpos)), None)
    if fam in ('addfam', 'mulfam'):
        if members[0] in ('add', 'mul'):            # full family: compact base +-1 form
            sym = '+' if fam == 'addfam' else '*'
            base = members[0]
            v = op_value(base, a, b, sa, sb)
            if hit == base:
                return f"{OPNAME[base]}: {a}{sym}{b}={v} ok", hit
            if hit:                                 # -1 / +1 variant
                hv = op_value(hit, a, b, sa, sb)
                return f"{OPNAME[base]}: {a}{sym}{b}={v} no, {OPNAME[hit][-2:]}={hv} ok", hit
            return f"{OPNAME[base]}: {a}{sym}{b}={v} no ({v-1}/{v+1} no)", None
        # partial family (after a resume): one verdict per remaining member
        parts = []
        for op in members:
            txt, _ = calc_text(op, a, b, sa, sb)
            if op == hit:
                parts.append(f"{txt} ok")
                break
            parts.append(f"{txt} no")
        return ', '.join(parts), hit
    if fam == 'difffam':
        d = a - b
        parts = [f"diff: {a}-{b}={d}."]
        for op in members:
            v = op_value(op, a, b, sa, sb)
            if op == hit:
                side = ''
                if v < 0:
                    side = ' (symbol in front)' if expected_pos(mode, op, glyph) == 'front' \
                        else ' (symbol at the end)'
                parts.append(f"{OPNAME[op]} {v}{side} ok")
                break
            # value coincidence with wrong sign side -> say why rejected
            if tval is not None and v == tval:
                parts.append(f"{OPNAME[op]} {v} wrong side no")
            else:
                parts.append(f"{OPNAME[op]} {v} no")
        return parts[0] + ' ' + ', '.join(parts[1:]), hit
    # singleton families
    op = members[0]
    txt, v = calc_text(op, a, b, sa, sb)
    if op in ('concat', 'rconcat'):
        txt = f"{OPNAME[op]}: {txt}"            # max%min already self-prefixes
    return f"{txt} {'ok' if hit else 'no'}", hit

def search_glyph_trace(exs, glyph, mode, label):
    """Mirror of eq_fit.fit_glyph_strict with a recorded trace.
    Returns (op_or_None, lines)."""
    lines = []
    sa0, sb0, r0 = exs[0]
    a, b, csa, csb = ops_of(sa0, sb0, mode)
    t0 = parse_target(r0, glyph, mode)
    intro = f"{label} {sa0}{glyph}{sb0}={r0} -> operands {csa},{csb}, {tdesc(*t0, r0, mode)}."
    lines.append(intro)
    tried = set()
    for fam, members in FAMS:
        todo = [op for op in members if op not in tried]
        if not todo:
            continue
        line, hit = fam_line(fam, todo, (a, b, csa, csb), glyph, mode, *t0)
        lines.append(line + ('.' if not line.endswith('.') else ''))
        tried.update(todo[:todo.index(hit) + 1] if hit else todo)
        while hit:
            # verify the candidate on the remaining examples of this glyph
            bad = False
            for sa, sb, r in exs[1:]:
                a2, b2, csa2, csb2 = ops_of(sa, sb, mode)
                t = parse_target(r, glyph, mode)
                txt, _ = calc_text(hit, a2, b2, csa2, csb2)
                if matches(hit, a2, b2, csa2, csb2, glyph, mode, *t):
                    lines.append(f"check {sa}{glyph}{sb}={r}: operands {csa2},{csb2}, "
                                 f"{tdesc(*t, r, mode)}. {txt} ok.")
                else:
                    lines.append(f"check {sa}{glyph}{sb}={r}: operands {csa2},{csb2}, "
                                 f"{tdesc(*t, r, mode)}. {txt} no -> drop {OPNAME[hit]}.")
                    bad = True
                    break
            if not bad:
                lines.append(f"'{glyph}' = {OPNAME[hit]} fits all its examples.")
                return hit, lines
            # resume: next op (SEQ order) that matches example 1; show the rejects too
            rest = [op for op in E.SEQ_ORDER if op not in tried]
            hit = None
            rejects = []
            for op in rest:
                tried.add(op)
                txt, _ = calc_text(op, a, b, csa, csb)
                if matches(op, a, b, csa, csb, glyph, mode, *t0):
                    pre = ('; '.join(f"{t} no" for t in rejects) + '; ') if rejects else ''
                    lines.append(f"resume on the first example: {pre}{txt} ok.")
                    hit = op
                    break
                rejects.append(txt)
            if hit is None:
                if rejects:
                    lines.append("resume on the first example: "
                                 + '; '.join(f"{t} no" for t in rejects) + '.')
                break
    lines.append(f"No rule fits '{glyph}'.")
    return None, lines

MODE_INTRO = {
    'rev': ("Mode REV first: reverse the digits of each operand, compute, and write the "
            "result digits-reversed. A negative result uses the operator symbol as its "
            "minus sign: in front for sub (and always when the symbol is '-'), after the "
            "digits for neg/rsub."),
    'id': ("Mode PLAIN: operands and result exactly as written; a negative result carries "
           "the operator symbol in front of the digits."),
}

def render_apply(mode, op, qa, qg, qb):
    """Apply block lines + final printed answer (must equal eq_fit.apply_rule)."""
    a, b, csa, csb = ops_of(qa, qb, mode)
    txt, v = calc_text(op, a, b, csa, csb)
    intro = f"Apply to {qa}{qg}{qb}: operands {'reversed ' if mode == 'rev' else ''}{csa},{csb}. {txt}."
    if v is None:
        return [intro, "Undefined."], None
    if isinstance(v, str):
        out = v[::-1] if mode == 'rev' else v
        line = f"Reverse the string: '{out}'." if mode == 'rev' else f"Result string: '{out}'."
        return [intro, line], out
    if v < 0:
        digits = str(-v)[::-1] if mode == 'rev' else str(-v)
        pos = expected_pos(mode, op, qg)
        out = (qg + digits) if pos == 'front' else (digits + qg)
        side = 'in front of' if pos == 'front' else 'after'
        rev_note = 'reversed digits' if mode == 'rev' else 'digits'
        return [intro, f"Negative: symbol '{qg}' {side} the {rev_note}: '{out}'."], out
    out = str(v)[::-1] if mode == 'rev' else str(v)
    line = f"Write digits reversed: '{out}'." if mode == 'rev' else f"Result: '{out}'."
    return [intro, line], out

def emit_trace(byop, q):
    """Full CoT for one row (deduce or guess). Returns (cot, final, (mode, op))."""
    qa, qg, qb = q
    lines = ["Group the examples by operator symbol."]
    for g, exs in byop.items():
        tag = "   <- query symbol" if g == qg else ""
        lines.append(f"'{g}': " + "; ".join(f"{sa}{g}{sb}={r}" for sa, sb, r in exs) + tag)
    lines.append(f"Query: {qa}{qg}{qb}, symbol '{qg}'.")
    is_guess = qg not in byop
    if is_guess:
        lines.append(f"'{qg}' never appears in the examples, so its rule cannot be deduced. "
                     "Pin the mode from the known symbols, then use the most likely rule.")
    locked = None
    for mode in E.MODES:
        lines.append("")
        lines.append(MODE_INTRO[mode])
        if not is_guess:
            qop, qlines = search_glyph_trace(byop[qg], qg, mode, f"Rule for '{qg}':")
            lines += qlines
            if qop is None:
                lines.append(f"Mode {MODENAME[mode]} cannot explain the query symbol.")
                continue
        else:
            qop = 'sub'
        others = [g for g in byop if g != qg]
        bad = None
        if others:
            lines.append(f"{MODENAME[mode]} must fit the other symbols too." if not is_guess
                         else "Check every symbol under this mode.")
            for g in others:
                gop, glines = search_glyph_trace(byop[g], g, mode, f"Symbol '{g}':")
                lines += glines
                if gop is None:
                    bad = g
                    break
        else:
            lines.append("No other symbols to check.")
        if bad is not None:
            lines.append(f"Mode {MODENAME[mode]} fails on '{bad}'.")
            continue
        lines.append(f"Mode {MODENAME[mode]} confirmed.")
        if is_guess:
            lines.append("Unseen symbol: the most common rule is sub (a-b).")
        locked = (mode, qop)
        break
    if locked is None:                                  # fallback (never on explained rows)
        for mode in E.MODES:
            qop = E.fit_glyph_strict(byop.get(qg, []), qg, mode) if qg in byop else 'sub'
            if qop:
                locked = (mode, qop)
                lines.append(f"Fall back to mode {MODENAME[mode]} from the query symbol alone.")
                break
    if locked is None:
        return None, None, None
    mode, qop = locked
    lines.append(f"Lock: mode {MODENAME[mode]}, '{qg}' = {OPNAME[qop]}.")
    apply_lines, out = render_apply(mode, qop, qa, qg, qb)
    lines += apply_lines
    if out is None:
        return None, None, None
    lines.append(f"So the result for {qa}{qg}{qb} is {out}.")
    return "\n".join(lines), out, locked

# ------------------------------------------------------------------ linter (cod-emit)
def lint(cot, final, locked, qg):
    mode, op = locked
    assert all(ord(c) < 128 for c in cot), "non-ASCII in cot"
    assert '\\boxed' not in cot, "cot must not contain \\boxed (trainer appends it)"
    lock_line = [l for l in cot.split('\n') if l.startswith('Lock: ')]
    assert len(lock_line) == 1, "exactly one Lock line"
    assert lock_line[0] == f"Lock: mode {MODENAME[mode]}, '{qg}' = {OPNAME[op]}.", "lock mismatch"
    assert cot.rstrip().endswith(f"is {final}."), "final differs from the trace conclusion"
    # the locked op must be the last candidate that survived for the query glyph (deduce)
    if f"'{qg}' = " in cot:
        surv = [l for l in cot.split('\n') if l.startswith(f"'{qg}' = ") and 'fits all' in l]
        if surv:
            assert surv[-1].startswith(f"'{qg}' = {OPNAME[op]} "), "verified rule != locked rule"
    # exactly one Apply block referencing the locked computation
    assert cot.count('Apply to ') == 1, "exactly one apply block"

# ------------------------------------------------------------------ synthetic generator
OP_MASS = {'add': 274, 'mul': 265, 'concat': 207.5, 'sub': 198.4, 'absdiff': 123.4,
           'negabsdiff': 122.4, 'add_p1': 121, 'mul_p1': 120, 'mul_m1': 116,
           'maxmodmin': 108.4, 'add_m1': 99, 'rsub': 76.4, 'rconcat': 32.5}
GLYPHS_HOT = '+-*'
GLYPHS_COLD = [c for c in '!"#$%&\'()/:<>?@[\\]^`{|}' if c not in GLYPHS_HOT]
P_HOT = 0.484
P_REV = 0.665
P_SMALL_OPERAND = 0.0628

def draw_operand(rng):
    v = rng.randint(0, 9) if rng.random() < P_SMALL_OPERAND else rng.randint(10, 99)
    return f"{v:02d}"

def draw_glyphs(rng, n):
    out = []
    while len(out) < n:
        g = rng.choice(GLYPHS_HOT) if rng.random() < P_HOT else rng.choice(GLYPHS_COLD)
        if g not in out:
            out.append(g)
    return out

def draw_op(rng):
    r = rng.random() * sum(OP_MASS.values())
    for op, m in OP_MASS.items():
        r -= m
        if r <= 0:
            return op
    return 'add'

def gen_row(rng, kind):
    """One synthetic row rendered with the strict generator model. Returns (byop, q, gold)."""
    while True:
        mode = 'rev' if rng.random() < P_REV else 'id'
        if kind == 'deduce':
            n_glyph = rng.choices([3, 2, 1], weights=[308, 273, 15])[0]
            glyphs = draw_glyphs(rng, n_glyph)
            qg = rng.choice(glyphs)
        else:
            n_glyph = rng.choices([2, 1], weights=[107, 29])[0]
            glyphs = draw_glyphs(rng, n_glyph + 1)
            qg = glyphs[-1]
            glyphs = glyphs[:-1]
        ops = {g: draw_op(rng) for g in glyphs}
        qop = ops[qg] if kind == 'deduce' else draw_op(rng)
        n_ex = rng.choice([3, 4, 5])
        counts = {g: 1 for g in glyphs}
        for _ in range(n_ex - n_glyph):
            counts[rng.choice(glyphs)] += 1
        byop, ok = collections.OrderedDict(), True
        for g in glyphs:
            exs = []
            for _ in range(counts[g]):
                sa, sb = draw_operand(rng), draw_operand(rng)
                sp = E.default_signpos(mode, ops[g], g)
                r = E.apply_rule(mode, ops[g], sa, sb, g, sp)
                v = op_value(ops[g], *ops_of(sa, sb, mode))
                if r is None or (isinstance(v, int) and v < 0
                                 and ops[g] not in ('sub', 'negabsdiff', 'rsub')):
                    ok = False
                    break
                exs.append((sa, sb, r))
            if not ok:
                break
            byop[g] = exs
        if not ok:
            continue
        qa, qb = draw_operand(rng), draw_operand(rng)
        sp = E.default_signpos(mode, qop, qg)
        gold = E.apply_rule(mode, qop, qa, qb, qg, sp)
        qv = op_value(qop, *ops_of(qa, qb, mode))
        if gold is None or (isinstance(qv, int) and qv < 0
                            and qop not in ('sub', 'negabsdiff', 'rsub')):
            continue
        return byop, (qa, qg, qb), gold

def make_prompt(byop, q, rng):
    lines = [HDR]
    exlines = [(sa, g, sb, r) for g, exs in byop.items() for sa, sb, r in exs]
    # interleave like the real generator (examples are not grouped in the prompt)
    rng.shuffle(exlines)
    for sa, g, sb, r in exlines:
        lines.append(f"{sa}{g}{sb} = {r}")
    lines.append(f"Now, determine the result for: {q[0]}{q[1]}{q[2]}")
    return "\n".join(lines)

# ------------------------------------------------------------------ commands
def val_ids():
    ids = set()
    for line in open(os.path.join(ROOT, 'pipeline', 'data', 'val.jsonl')):
        d = json.loads(line)
        if str(d.get('category', '')).startswith('equation_numeric'):
            ids.add(d['id'])
    return ids

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f"wrote {len(rows)} rows -> {path}")

def cmd_real():
    excl = val_ids()
    rows = E.load_rows()
    out, kept, skipped_val, wrong = [], 0, 0, 0
    for row in rows:
        if row['cat'] != 'equation_numeric_deduce':
            continue
        ans, _ = E.policy_seq(row)
        if ans != row['gold']:
            wrong += 1
            continue
        if row['id'] in excl:
            skipped_val += 1
            continue
        cot, final, locked = emit_trace(row['byop'], row['q'])
        assert final == row['gold'], (row['id'], final, row['gold'])
        lint(cot, final, locked, row['q'][1])
        out.append(dict(id=row['id'], category=row['cat'], prompt=row['prompt'],
                        cot=cot, final=final))
    print(f"deduce real: policy-correct {len(out)+skipped_val}/596, "
          f"excluded {skipped_val} val ids, dropped {wrong} policy-wrong")
    write_jsonl(os.path.join(OUTDIR, 'eq_deduce_real.jsonl'), out)

def cmd_synth(n=650, seed=15):
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        byop, q, gold = gen_row(rng, 'deduce')
        ans, _ = E.policy_seq(dict(byop=byop, q=q))
        if ans != gold:
            continue                                  # tie went the other way: drop
        cot, final, locked = emit_trace(byop, q)
        if final != gold:
            continue
        lint(cot, final, locked, q[1])
        out.append(dict(id=f"synth-eq-{len(out):04d}", category='equation_numeric_deduce',
                        prompt=make_prompt(byop, q, rng), cot=cot, final=final))
    print(f"synth deduce: kept {len(out)}/{tries} generated (policy-agreement filter)")
    write_jsonl(os.path.join(OUTDIR, 'eq_deduce_synth.jsonl'), out)

def cmd_guess(n_synth=92, seed=16):
    excl = val_ids()
    rows = E.load_rows()
    out, skipped_val = [], 0
    for row in rows:
        if row['cat'] != 'equation_numeric_guess':
            continue
        ans, _ = E.policy_seq(row)
        if ans != row['gold']:
            continue
        if row['id'] in excl:
            skipped_val += 1
            continue
        cot, final, locked = emit_trace(row['byop'], row['q'])
        assert final == row['gold'], (row['id'], final, row['gold'])
        lint(cot, final, locked, row['q'][1])
        out.append(dict(id=row['id'], category=row['cat'], prompt=row['prompt'],
                        cot=cot, final=final))
    n_real = len(out)
    rng = random.Random(seed)
    tries = 0
    while len(out) < n_real + n_synth:
        tries += 1
        byop, q, gold = gen_row(rng, 'guess')
        ans, _ = E.policy_seq(dict(byop=byop, q=q))
        if ans != gold:
            continue
        cot, final, locked = emit_trace(byop, q)
        if final != gold:
            continue
        lint(cot, final, locked, q[1])
        out.append(dict(id=f"synth-eqg-{len(out)-n_real:04d}", category='equation_numeric_guess',
                        prompt=make_prompt(byop, q, rng), cot=cot, final=final))
    print(f"guess: {n_real} real policy-correct (excl {skipped_val} val) + "
          f"{len(out)-n_real} synth (kept of {tries})")
    write_jsonl(os.path.join(OUTDIR, 'eq_guess.jsonl'), out)

def cmd_stats():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(ROOT, 'competition_dataset', 'tokenizer.json'))
    for name in ('eq_deduce_real', 'eq_deduce_synth', 'eq_guess'):
        path = os.path.join(OUTDIR, name + '.jsonl')
        if not os.path.exists(path):
            continue
        lens = []
        for line in open(path):
            d = json.loads(line)
            lens.append(len(tk.encode(d['cot']).ids))
        lens.sort()
        n = len(lens)
        print(f"{name}: n={n} cot tokens min={lens[0]} median={lens[n//2]} "
              f"p95={lens[int(n*0.95)]} max={lens[-1]}")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'real'
    if cmd == 'real':
        cmd_real()
    elif cmd == 'synth':
        cmd_synth(int(sys.argv[2]) if len(sys.argv) > 2 else 650)
    elif cmd == 'guess':
        cmd_guess()
    elif cmd == 'stats':
        cmd_stats()
    elif cmd == 'sample':
        rows = E.load_rows()
        for row in rows:
            if row['id'] == sys.argv[2]:
                cot, final, locked = emit_trace(row['byop'], row['q'])
                print(cot)
                print('FINAL:', final, '| GOLD:', row['gold'])
