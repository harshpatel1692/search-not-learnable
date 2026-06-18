"""equation_numeric generator forensics: fit the (mode, op) library against every train row.

Generator model (established by ali_validation forensics, completed here from data):
  - each example line is  AB<g>CD = R  with A..D digits, g an arbitrary glyph (1-3 distinct per row)
  - per-glyph op drawn from a library; one GLOBAL mode per row:
      identity : operands read as printed, result printed as computed
      reverse  : each operand's 2 digits reversed BEFORE the op, and the final result STRING reversed
  - negatives: the minus sign is rendered AS THE OPERATOR GLYPH, prefix; under reverse mode the
    whole string (sign included) is reversed, so the glyph lands as a suffix.

Usage:
  python3 pipeline/eq_fit.py fit        # fit every row, report library completeness + prior table
  python3 pipeline/eq_fit.py policy     # simulate the gold-free policy on deduce + guess
"""
import csv, re, sys, json, collections, itertools, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'competition_dataset', 'train_categorized.csv')
EXLINE = re.compile(r'^(\d\d)(.)(\d\d) = (\S+)$')

# ---------------------------------------------------------------- op library
# Each op takes (a, b, sa, sb): ints and the (possibly digit-reversed) 2-char strings.
# Returns an int (renderer handles sign/glyph) or a string (digit-string ops), or None (undefined).
def _maxmodmin(a, b, sa, sb):
    lo, hi = min(a, b), max(a, b)
    return hi % lo if lo else None

# COMPLETE library: leave-one-out over all 732 rows shows each of these 13 is REQUIRED
# (some row is only explained with it) and nothing else is (mod/div/gcd/digit-ops were
# tested and are pure tie artifacts -- never uniquely needed).
OPS = {
    'add':        lambda a, b, sa, sb: a + b,
    'sub':        lambda a, b, sa, sb: a - b,
    'rsub':       lambda a, b, sa, sb: b - a,
    'absdiff':    lambda a, b, sa, sb: abs(a - b),
    'negabsdiff': lambda a, b, sa, sb: -abs(a - b),
    'mul':        lambda a, b, sa, sb: a * b,
    'concat':     lambda a, b, sa, sb: sa + sb,
    'rconcat':    lambda a, b, sa, sb: sb + sa,
    'add_p1':     lambda a, b, sa, sb: a + b + 1,
    'add_m1':     lambda a, b, sa, sb: a + b - 1,
    'mul_p1':     lambda a, b, sa, sb: a * b + 1,
    'mul_m1':     lambda a, b, sa, sb: a * b - 1,
    'maxmodmin':  _maxmodmin,
}

MODES = ('rev', 'id')
SIGNPOS = ('suf', 'pre')   # canonical sign-glyph position BEFORE the mode reversal

def render(v, glyph, mode, signpos='suf'):
    """Render op output v (int or digit-string) to the printed result string.
    Canonical string: digits (+ sign glyph at `signpos` if v<0); rev mode reverses the WHOLE string."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v
    elif v < 0:
        s = (glyph + str(-v)) if signpos == 'pre' else (str(-v) + glyph)
    else:
        s = str(v)
    return s[::-1] if mode == 'rev' else s

def operands(sa, sb, mode):
    if mode == 'rev':
        sa, sb = sa[::-1], sb[::-1]
    return int(sa), int(sb), sa, sb

def apply_rule(mode, op, sa, sb, glyph, signpos='suf'):
    a, b, ra, rb = operands(sa, sb, mode)
    try:
        v = OPS[op](a, b, ra, rb)
    except Exception:
        return None
    return render(v, glyph, mode, signpos)

# ---------------------------------------------------------------- parsing
def parse(prompt):
    exs, q = [], None
    for ln in prompt.split('\n'):
        m = EXLINE.match(ln.strip())
        if m:
            exs.append(m.groups())   # (sa, glyph, sb, result)
            continue
        if ln.startswith('Now, determine the result for:'):
            q = ln.split(':', 1)[1].strip()
    byop = collections.OrderedDict()
    for sa, g, sb, r in exs:
        byop.setdefault(g, []).append((sa, sb, r))
    return byop, (q[:2], q[2], q[3:])

def load_rows():
    out = []
    for r in csv.DictReader(open(CSV)):
        if r['category'].startswith('equation_numeric'):
            byop, q = parse(r['prompt'])
            out.append(dict(id=r['id'], cat=r['category'], prompt=r['prompt'],
                            gold=r['answer'].strip(), byop=byop, q=q))
    return out

# ---------------------------------------------------------------- fitting
def ops_consistent(examples, glyph, mode, signpos):
    """All ops that reproduce every example of one glyph under (mode, signpos)."""
    out = []
    for op in OPS:
        if all(apply_rule(mode, op, sa, sb, glyph, signpos) == r for sa, sb, r in examples):
            out.append(op)
    return out

def fit_row(row):
    """All (mode, signpos, per-glyph-op-sets) fits; for deduce the query glyph must also hit gold."""
    qa, qg, qb = row['q']
    fits = []
    for mode in MODES:
        for signpos in SIGNPOS:
            per = {}
            ok = True
            for g, exs in row['byop'].items():
                cand = ops_consistent(exs, g, mode, signpos)
                if g == qg:  # deduce: must also reproduce the gold
                    cand = [op for op in cand if apply_rule(mode, op, qa, qb, qg, signpos) == row['gold']]
                if not cand:
                    ok = False
                    break
                per[g] = cand
            if ok and qg not in row['byop']:  # guess: gold fixes the query op directly
                cand = [op for op in OPS if apply_rule(mode, op, qa, qb, qg, signpos) == row['gold']]
                if not cand:
                    ok = False
                else:
                    per['?query'] = cand
            if ok:
                fits.append((mode, signpos, per))
    return fits

def cmd_fit():
    rows = load_rows()
    unexplained = []
    mode_count = collections.Counter()
    sign_count = collections.Counter()         # rows that PIN the sign position (only one survives)
    sign_pin = collections.Counter()           # (mode, signpos) for sign-pinned rows
    op_count = collections.Counter()           # query-glyph op draw (deduce, unambiguous only)
    op_count_all = collections.Counter()       # every glyph slot, fractional over ties
    amb_modes = amb_ops = 0
    for row in rows:
        fits = fit_row(row)
        if not fits:
            unexplained.append(row)
            continue
        modes = {m for m, sp, _ in fits}
        signs = {sp for m, sp, _ in fits}
        if len(modes) > 1:
            amb_modes += 1
        mode_count[fits[0][0] if len(modes) == 1 else 'ambiguous'] += 1
        sign_count[fits[0][1] if len(signs) == 1 else 'free'] += 1
        if len(signs) == 1:
            sign_pin[(fits[0][0] if len(modes) == 1 else '?', fits[0][1])] += 1
        # query op prior: use the first fitting (mode, signpos)'s candidate set
        mode, signpos, per = fits[0]
        qg = row['q'][1]
        qcand = per.get(qg) or per.get('?query')
        if len(qcand) == 1:
            op_count[qcand[0]] += 1
        else:
            amb_ops += 1
            op_count['tie:' + '|'.join(sorted(qcand))] += 1
        for g, cand in per.items():
            for op in cand:
                op_count_all[op] += 1.0 / len(cand)
    print(f'rows: {len(rows)}  unexplained: {len(unexplained)}')
    for row in unexplained[:20]:
        print('--- UNEXPLAINED', row['id'], row['cat'])
        print(row['prompt'])
        print('gold:', row['gold'])
    print('mode (unique fits):', dict(mode_count), f'(+{amb_modes} mode-ambiguous rows)')
    print('signpos (pinned only):', dict(sign_count))
    print('sign-pinned (mode, signpos):', dict(sign_pin))
    print('query-op draws (unique):')
    for op, n in op_count.most_common():
        print(f'  {op:30s} {n}')
    print('all-glyph fractional op mass:')
    for op, n in op_count_all.most_common():
        print(f'  {op:30s} {n:.1f}')

# ---------------------------------------------------------------- gold-free policy
# Enumeration order: rev first (generator prior 476:239), common ops first.
OP_ORDER = ['concat', 'rconcat', 'add', 'mul', 'sub', 'negabsdiff', 'absdiff', 'rsub',
            'add_p1', 'add_m1', 'mul_p1', 'mul_m1', 'maxmodmin']

def fit_glyph(exs, glyph, mode):
    """Ops consistent with the glyph's examples under mode, with the sign-render position
    fitted per op. Returns list of (op, signpos_or_None) -- signpos pinned only if some
    example output is negative."""
    out = []
    has_neg = any(not r.isdigit() for _, _, r in exs)
    for op in OPS:
        if mode == 'id':
            if all(apply_rule('id', op, sa, sb, glyph, 'pre') == r for sa, sb, r in exs):
                out.append((op, 'pre' if has_neg else None))
        else:
            for sp in SIGNPOS:
                if all(apply_rule('rev', op, sa, sb, glyph, sp) == r for sa, sb, r in exs):
                    out.append((op, sp if has_neg else None))
                    break
    return out

def default_signpos(mode, op, glyph):
    """Measured render prior when the glyph's examples never go negative:
    id -> printed prefix (canonical pre), 93:0 in pinned rows;
    rev + '-' glyph -> printed prefix (canonical suf), 125:0;
    rev + sub -> printed prefix (canonical suf), 25:0 pure + 14:5 in ties;
    rev + rsub/negabsdiff -> printed suffix (canonical pre), 20:0 / 4:1."""
    if mode == 'id':
        return 'pre'
    if glyph == '-':
        return 'suf'
    return 'suf' if op == 'sub' else 'pre'

# Measured tie-break priority (argmax-gold per candidate set over the 596 deduce rows):
#   sub > negabsdiff > absdiff > rsub > maxmodmin, EXCEPT the pure {absdiff, maxmodmin}
#   pair where maxmodmin wins 6/6 (absdiff 5/6). Render preference comes first: an op whose
#   FITTED sign render matches its generator default beats one that needed the other render.
TIE_ORDER = ['concat', 'rconcat', 'add', 'mul', 'add_p1', 'add_m1', 'mul_p1', 'mul_m1',
             'sub', 'negabsdiff', 'absdiff', 'rsub', 'maxmodmin']

def choose_op(cand, mode, glyph):
    """cand: {op: fitted_signpos_or_None}. Returns the locked op."""
    if set(cand) == {'absdiff', 'maxmodmin'}:
        return 'maxmodmin'
    def key(op):
        sp = cand[op]
        render_mismatch = sp is not None and sp != default_signpos(mode, op, glyph)
        return (render_mismatch, TIE_ORDER.index(op))
    return min(cand, key=key)

def policy_deduce(row, global_mode=True):
    """Gold-free: pin the global mode (rev first), lock the best op surviving the query
    glyph's examples, apply with the fitted/default sign render."""
    qa, qg, qb = row['q']
    exs = row['byop'][qg]
    modes = list(MODES)
    if global_mode:
        # the mode is global: keep only modes under which EVERY glyph has some consistent op
        valid = [m for m in MODES
                 if all(fit_glyph(e, g, m) for g, e in row['byop'].items())]
        if valid:
            modes = valid
    for mode in modes:
        cand = dict(fit_glyph(exs, qg, mode))
        if not cand:
            continue
        op = choose_op(cand, mode, qg)
        sp = cand[op] or default_signpos(mode, op, qg)
        return apply_rule(mode, op, qa, qb, qg, sp), (mode, op, sp)
    return None, None

# ---------------------------------------------------------------- THE policy (CoT-exact)
# Sequential first-hit trial order, fused families, STRICT default sign render.
# Frequency-ordered (expected-token greedy) subject to measured tie-break constraints:
#   sub before negabsdiff before absdiff before rsub; absdiff before maxmodmin.
SEQ_ORDER = ['add', 'add_m1', 'add_p1', 'mul', 'mul_m1', 'mul_p1',
             'sub', 'negabsdiff', 'absdiff', 'rsub', 'concat', 'maxmodmin', 'rconcat']

def fit_glyph_strict(exs, glyph, mode):
    """First op in SEQ_ORDER consistent with all examples under mode w/ DEFAULT sign render."""
    for op in SEQ_ORDER:
        sp = default_signpos(mode, op, glyph)
        if all(apply_rule(mode, op, sa, sb, glyph, sp) == r for sa, sb, r in exs):
            return op
    return None

def policy_seq(row):
    """The exact procedure the CoT performs. Returns (answer, (mode, op))."""
    qa, qg, qb = row['q']
    exs = row['byop'].get(qg, [])
    for mode in MODES:                      # REV first = generator prior
        qop = fit_glyph_strict(exs, qg, mode) if exs else 'sub'   # guess: best fixed op
        if qop is None:
            continue                        # query glyph kills this mode
        if all(fit_glyph_strict(e, g, mode) for g, e in row['byop'].items() if g != qg):
            sp = default_signpos(mode, qop, qg)
            return apply_rule(mode, qop, qa, qb, qg, sp), (mode, qop)
    # fallback (never hit on train): query-glyph-only lock under rev
    for mode in MODES:
        qop = fit_glyph_strict(exs, qg, mode) if exs else 'sub'
        if qop:
            sp = default_signpos(mode, qop, qg)
            return apply_rule(mode, qop, qa, qb, qg, sp), (mode, qop)
    return None, None

def cmd_seq():
    rows = load_rows()
    for cat in ('equation_numeric_deduce', 'equation_numeric_guess'):
        sub = [r for r in rows if r['cat'] == cat]
        ok = 0
        fails = []
        for r in sub:
            ans, why = policy_seq(r)
            if ans == r['gold']:
                ok += 1
            else:
                fails.append((r['id'], ans, why, r['gold'], ''.join(r['q'])))
        print(f'{cat}: {ok}/{len(sub)} = {100*ok/len(sub):.2f}%')
        for f in fails[:35]:
            print('   FAIL', f)

def cmd_policy():
    rows = load_rows()
    ded = [r for r in rows if r['cat'] == 'equation_numeric_deduce']
    gue = [r for r in rows if r['cat'] == 'equation_numeric_guess']
    for gm in (False, True):
        ok = 0
        fails = []
        for r in ded:
            ans, why = policy_deduce(r, global_mode=gm)
            if ans == r['gold']:
                ok += 1
            else:
                fails.append((r, ans, why))
        print(f'deduce global_mode={gm}: {ok}/{len(ded)} = {100*ok/len(ded):.1f}%')
        if gm:
            for r, ans, why in fails[:30]:
                print(f'  FAIL {r["id"]} pred={ans!r} ({why}) gold={r["gold"]!r} q={"".join(r["q"])}')
    # guess: fixed (mode, op) policies, mode optionally fitted from the OTHER glyphs
    print('\nguess fixed policies (mode fitted globally where possible):')
    scores = collections.Counter()
    for mode_strategy in ('fit', 'rev', 'id'):
        for op in OPS:
            ok = 0
            for r in gue:
                qa, qg, qb = r['q']
                if mode_strategy == 'fit':
                    valid = [m for m in MODES
                             if all(fit_glyph(e, g, m) for g, e in r['byop'].items())]
                    mode = valid[0] if valid else 'rev'
                else:
                    mode = mode_strategy
                sp = default_signpos(mode, op, qg)
                if apply_rule(mode, op, qa, qb, qg, sp) == r['gold']:
                    ok += 1
            scores[(mode_strategy, op)] = ok
    for (ms, op), n in scores.most_common(12):
        print(f'  mode={ms:4s} op={op:12s} {n}/{len(gue)} = {100*n/len(gue):.1f}%')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    if cmd == 'fit':
        cmd_fit()
    elif cmd == 'policy':
        cmd_policy()
    elif cmd == 'seq':
        cmd_seq()
