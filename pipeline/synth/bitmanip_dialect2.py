"""bitmanip_dialect2: bit_manipulation training dialect v2 (autopsy-driven rebuild).

WHY (analysis/bit_autopsy/classification.json + verify_report.json, 12 transcripts,
28 wrong bits, all mechanically verified): the v16 dialect's arithmetic atoms are
100% executable (ZERO errors in table transcription, popcount hashes, cmatch
suffixes, rule application, assembly).  Every error lives in three structural
steps whose verdicts did not restate their evidence:
  E-shift (6 bits): a verified stride run is committed to Tentative with one
    anchor DROPPED, shifting the whole run -1 (run placement was positional
    recall, not transcription).
  E-acc (6) / C-rej (6): a run accept/reject verdict contradicts the member's
    own table line / inverted index (the verdict was bare - no witness).
  H-tie (3): multiple verified fits disagree on the query and the model's
    tie-break diverges from bit_stride's longest-run prior (the prior was
    invisible - baked into the data, never rendered).

THE THREE FIXES (everything else byte-compatible with bitmanip_alifix's dialect;
see its docstring for the base grammar):

1. PLACEMENT-AS-TRANSCRIPTION.  Between "Truncated ..." and "Tentative" two new
   blocks derive every per-bit assignment in-line from the committed run lists:
       Place left (anchor 0 step +1)        Place right (anchor 7 step -1)
       0+0=0 I1                             7-0=7 AND60
       0+1=1 NOT2                           7-1=6 AND57 ...
   k-th line = k-th member of the Truncated run; the bit index is arithmetic,
   not recall.  "Tentative" is then a pure copy of the Place lines (+ pending).

2. EXTENSION VERDICT WITNESS.  Run lines in the Left/Right blocks change from
   bare member lists ("61 50 47 36x") to witnessed steps:
       61 m7 7 ok, 50 m6 6 ok, 47 m5 5 ok, 36 m- 4 x
   Each step re-quotes the member's FULL match set (the 'match' suffixes of its
   own table line; 'm-' = empty) and the required bit; the verdict is forced:
   ok iff the required bit is in the quoted set, x terminates the run.  Both
   accept and reject outcomes occur abundantly (truthful variation preserved).
   To pay for the witnesses, symmetric-order duplicate anchors (AND/OR/XOR/
   XOR-NOT listed both ways in the match lists) are deduped - each symmetric
   run is rendered once.

3. EXPLICIT TIE-BREAK.  A policy line right after the preamble:
       Tie rule: longest cyclic stride run, then constant, unary, pair, then
       first listed.
   and, at EVERY choice point with >1 distinct verified candidate (family-side
   Best run, cross-family Best left/right, Matched per-bit fill, Recheck pick),
   an explicit tie line:
       Tie 7: AND03 r4, AND35 r1 -> AND03 bits 7 0 1 2
   rN = length of the maximal cyclic stride run of that rule through the bit
   (exactly bit_stride's longest-run cover metric; ties by param class
   C < unary < pair = bit_stride's rank, then first listed).  The winner's run
   bits are quoted so the verdict restates its evidence.  The her-rule warm-
   start prior and the perfect-family override of dialect v1 are REMOVED:
   selection is now a deterministic function of the trace itself, so model
   choice == prior choice by construction.

Eligibility, gates and discipline are inherited from bitmanip_alifix: val.jsonl
ids excluded, bit_stride coverage == 8 required, the rendered procedure's answer
must equal train.csv gold (computed, never copied) else the row is DROPPED,
ASCII-only, and every emitted row passes lint2 (af.lint inherited checks + the
inverted re-derivation of the three fixes below).  Token gate: hard max 4999
(7680 inference cap; targets med <= ~4100, p95 <= 4500).

Outputs (one dialect, replaces BOTH v1 files - never mix):
  python3 bitmanip_dialect2.py passed -> pipeline/data/v16/bit_alipass2.jsonl
  python3 bitmanip_dialect2.py        -> pipeline/data/v16/bit_alifix2.jsonl
Row schema identical to v1 (id, category, prompt, cot, final, src, n_tokens
[, tier]).  Tamper gate: analysis/bit_autopsy/tamper_dialect2.py.
Report: analysis/reports/bit_dialect2.md.
"""
import csv, json, os, re, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
sys.path.insert(0, HERE)
import bit_stride as bs
import bitmanip_alifix as af
from bitmanip_alifix import (FAMS, SYM, Tables, bitc, comp, hashv, rname, reval,
                             shift, same, run_from, parse_rule, RULE_RE)

TIE_RULE = 'Tie rule: longest cyclic stride run, then constant, unary, pair, then first listed.'
TOKEN_GATE = 4999

def rank(r):
    return 0 if r[0] == 'C' else (1 if r[0] in ('I', 'NOT') else 2)

def rtok(r):
    if r[0] in ('I', 'NOT', 'C'): return str(r[1])
    return '%d%d' % (r[1], r[2])

def rule_col(ex, r):
    return ''.join(str(reval(r, i)) for i, _ in ex)

def outcols(ex):
    return [''.join(str(bitc(o, j)) for _, o in ex) for j in range(8)]

def mset(ex, ocols, r):
    col = rule_col(ex, r)
    return [t for t in range(8) if col == ocols[t]]

def msettxt(ms):
    return 'm' + (''.join(map(str, ms)) if ms else '-')

def cyc(ex, r, j):
    """Maximal cyclic stride run of rule r through output bit j.
    Returns (length, bits in cyclic walk order).  bit_stride's run metric."""
    base = shift(r, -j) if r[0] != 'C' else r
    cons = [all(reval(shift(base, t), i) == bitc(o, t) for i, o in ex) for t in range(8)]
    if all(cons): return 8, list(range(8))
    back = []
    t = j
    while cons[(t - 1) % 8]:
        t = (t - 1) % 8
        back.append(t)
    bits = back[::-1] + [j]
    t = j
    while cons[(t + 1) % 8] and (t + 1) % 8 not in bits:
        t = (t + 1) % 8
        bits.append(t)
    return len(bits), bits

def choose(ex, cands, j):
    """Deterministic tie-break: max cyclic run length, then param class
    C < unary < pair, then first listed.  Returns (deduped cands, infos, best)
    where infos = [(rule, len, bits)] and best is one of them."""
    ded = []
    for c in cands:
        if not any(same(c, d) for d in ded): ded.append(c)
    infos = []
    for c in ded:
        l, bits = cyc(ex, c, j)
        infos.append((c, l, bits))
    best = infos[0]
    for inf in infos[1:]:
        if (inf[1], -rank(inf[0])) > (best[1], -rank(best[0])): best = inf
    return ded, infos, best

def tie_text(j, infos, best):
    return 'Tie %d: %s -> %s bits %s' % (
        j, ', '.join('%s r%d' % (rname(c), l) for c, l, _ in infos),
        rname(best[0]), ' '.join(map(str, best[2])))

def run_line2(ex, ocols, rules, fail, j0, step):
    """Witnessed run line (fix 2): every member re-quotes its match set and the
    required bit; verdict ok iff bit in set, x terminates."""
    steps = []
    for k, r in enumerate(rules):
        steps.append('%s %s %d ok' % (rtok(r), msettxt(mset(ex, ocols, r)), j0 + step * k))
    if fail is not None:
        b = j0 + step * len(rules)
        if 0 <= b <= 7:
            steps.append('%s %s %d x' % (rtok(fail), msettxt(mset(ex, ocols, fail)), b))
    return ', '.join(steps)

# ---------------- the renderer (dialect v2) ----------------
def render2(ex, q):
    T = Tables(ex)
    N = T.N
    ocols = T.outcol
    L = []
    add = L.append
    add('We need to deduce the transformation by matching the example outputs.')
    add(TIE_RULE)
    add('')
    add('Output bit columns (with bitsum as hash)')
    for j in range(8): add('%d %s %s' % (j, T.outcol[j], hashv(T.outcol[j])))
    add('')
    add('Output bit columns complemented')
    for j in range(8): add('%d %s %s' % (j, T.coutcol[j], hashv(T.coutcol[j])))
    add('')

    def suffixes(col):
        s = ''
        m = [j for j in range(8) if col == T.outcol[j]]
        if m: s += ' match ' + ' '.join(map(str, m))
        return s, [j for j in range(8) if col == T.coutcol[j]]

    def csuffix(cm):
        return (' cmatch ' + ' '.join(map(str, cm))) if cm else ''

    def matching_block(fam):
        add('Matching output with %s' % fam)
        for j in range(8):
            es = T.match[fam][j]
            if not es:
                add('%d absent' % j)
            elif fam in ('Identity', 'NOT'):
                add('%d %s' % (j, ' '.join(str(r[1]) for r in es)))
            elif fam == 'Constant':
                add('%d %s' % (j, ' '.join('C%d' % r[1] for r in es)))
            else:
                add('%d %s' % (j, ' '.join('%d%d' % (r[1], r[2]) for r in es)))
        add('')

    best = {}                     # (fam, side) -> (rules, fail) or None
    def lr_blocks(fam):
        for side, j0, step in (('Left', 0, 1), ('Right', 7, -1)):
            add(side)
            anchors = []
            for a in T.match[fam][j0]:
                if not any(same(a, b) for b in anchors): anchors.append(a)
            runs = []
            for anchor in anchors:
                rules, fail = run_from(T, fam, anchor, j0, step)
                runs.append((rules, fail))
                add(run_line2(ex, ocols, rules, fail, j0, step))
            if not anchors: add('none')
            b = None
            if runs:
                mx = max(len(r[0]) for r in runs)
                top = [r for r in runs if len(r[0]) == mx]
                if len(top) > 1:
                    ded, infos, bi = choose(ex, [r[0][0] for r in top], j0)
                    add(tie_text(j0, infos, bi))
                    b = next(r for r in top if same(r[0][0], bi[0]))
                else:
                    b = top[0]
            best[(fam, side)] = b
            add('Best: %s' % ('%s: %d' % (' '.join(rname(r) for r in b[0]), len(b[0])) if b else 'none'))
            add('')

    # Identity
    add('Identity')
    for a in range(8):
        s, cm = suffixes(T.incol[a])
        add('%d %s %s%s%s' % (a, T.incol[a], hashv(T.incol[a]), s, csuffix(cm)))
    add('')
    lr_blocks('Identity')
    # NOT (derived)
    add('NOT (Identity cmatch)')
    matching_block('NOT'); lr_blocks('NOT')
    # Constant
    add('Constant')
    for c in (0, 1):
        col = str(c) * N
        s, _ = suffixes(col)
        add('%d %s a%s' % (c, col, s))
    add('')
    lr_blocks('Constant')
    # AND OR XOR
    for fam in ('AND', 'OR', 'XOR'):
        add(fam)
        for grp in af.sym_pairs():
            for a, b in grp:
                col = T.paircol[(fam, a, b)]
                s, cm = suffixes(col)
                cs = csuffix(cm) if fam == 'XOR' else ''
                add('%d%d %d%d %s %s%s%s' % (a, b, b, a, col, hashv(col), s, cs))
            add('')
        lr_blocks(fam)
    # AND-NOT
    add('AND-NOT')
    for grp in af.asym_pairs():
        for a, b in grp:
            col = T.paircol[('AND-NOT', a, b)]
            s, cm = suffixes(col)
            add('%d%d %s %s%s%s' % (a, b, col, hashv(col), s, csuffix(cm)))
        add('')
    lr_blocks('AND-NOT')
    # OR-NOT / XOR-NOT (derived)
    add('OR-NOT (AND-NOT cmatch, swapped)')
    matching_block('OR-NOT'); lr_blocks('OR-NOT')
    add('XOR-NOT (XOR cmatch)')
    matching_block('XOR-NOT'); lr_blocks('XOR-NOT')

    # ---------------- Selecting ----------------
    add('Selecting')
    add('')
    def cross(side, j0):
        cands = [best[(f, side)] for f in FAMS if best[(f, side)]]
        if not cands: return None, None
        mx = max(len(r[0]) for r in cands)
        top = [r for r in cands if len(r[0]) == mx]
        if len(top) == 1: return top[0], None
        ded, infos, bi = choose(ex, [r[0][0] for r in top], j0)
        win = next(r for r in top if same(r[0][0], bi[0]))
        return win, tie_text(j0, infos, bi)
    bl, tiel = cross('Left', 0)
    br, tier_ = cross('Right', 7)
    ll = len(bl[0]) if bl else 0
    rl = len(br[0]) if br else 0
    add('Left longest: %d' % ll)
    add('Right longest: %d' % rl)
    add('')
    if tiel: add(tiel)
    if tier_: add(tier_)
    order = ('left', 'right') if ll >= rl else ('right', 'left')
    bestline = {'left': bl, 'right': br}
    for s in order:
        b = bestline[s]
        add('Best %s: %s' % (s, '%s: %d' % (' '.join(rname(r) for r in b[0]), len(b[0])) if b else 'none'))
    lrules = list(bl[0]) if bl else []
    rrules = list(br[0]) if br else []
    if len(lrules) + len(rrules) > 8:
        if order[0] == 'left': rrules = rrules[:8 - len(lrules)]
        else: lrules = lrules[:8 - len(rrules)]
    trunc = {'left': lrules, 'right': rrules}
    for s in order:
        rs = trunc[s]
        add('Truncated %s: %s' % (s, ('%s: %d' % (' '.join(rname(r) for r in rs), len(rs))) if rs else 'none'))
    add('')
    # ---------------- Place (fix 1): placement as transcription ----------------
    assign = [None] * 8
    if lrules:
        add('Place left (anchor 0 step +1)')
        for k, r in enumerate(lrules):
            add('0+%d=%d %s' % (k, k, rname(r)))
            assign[k] = r
    if rrules:
        add('Place right (anchor 7 step -1)')
        for k, r in enumerate(rrules):
            add('7-%d=%d %s' % (k, 7 - k, rname(r)))
            assign[7 - k] = r
    add('Tentative')
    for j in range(8): add('%d %s' % (j, rname(assign[j]) if assign[j] else 'pending'))
    add('')
    # placeholders continue the winning run through all pending bits
    win = order[0]
    wrun = trunc[win] if trunc[win] else trunc[order[1]]
    if not wrun:                                  # no runs at all
        ph = {j: None for j in range(8) if assign[j] is None}
    else:
        wanchor, w0 = (wrun[0], 0) if (win == 'left' or not trunc[win]) else (wrun[0], 7)
        if not trunc[win]:
            win = order[1]; w0 = 0 if win == 'left' else 7
        ph = {}
        for j in range(8):
            if assign[j] is None: ph[j] = shift(wanchor, j - w0)
    def phtok(r):
        if r is None: return '?'
        if r[0] in ('I', 'NOT'): return '?%d?' % r[1]
        if r[0] == 'C': return '?C%d' % r[1]
        return '?%d%d' % (r[1], r[2])
    add('Preferred from %s' % win)
    for j in range(8): add('%d %s' % (j, rname(assign[j]) if assign[j] else phtok(ph[j])))
    add('')

    # ---------------- Matching at placeholders ----------------
    def ph_check(fam, j, p):
        if fam == 'Constant':
            for r in T.match['Constant'][j]: return r
            return None
        if p is None or p[0] == 'C': return None
        a = p[1]; b = p[2] if len(p) > 2 else None
        if fam in ('Identity', 'NOT'):
            r = ('I' if fam == 'Identity' else 'NOT', a)
            return r if T.has(fam, j, r) else None
        if b is None:
            for r in T.match[fam][j]:
                if a in (r[1], r[2]): return r
            return None
        for r in ((fam, a, b), (fam, b, a)):
            if T.has(fam, j, r): return r
        return None

    pend = sorted(ph)
    hits = {}
    add('Matching')
    for j in range(8):
        if j not in ph:
            add('%d %s' % (j, rname(assign[j])))
            continue
        hits[j] = {}
        parts = []
        for fam in FAMS:
            h = ph_check(fam, j, ph[j])
            if h is None:
                parts.append('%s absent' % fam)
            else:
                hits[j][fam] = h
                parts.append('Constant C%d' % h[1] if fam == 'Constant' else rname(h))
        if not hits[j]: parts = ['all absent']
        add('%d %s - %s' % (j, phtok(ph[j]), ', '.join(parts)))
    add('')
    add('Perfect match')
    for fam in FAMS:
        ok = bool(pend) and all(fam in hits[j] for j in pend)
        add('%s %s' % (fam, 'yes' if ok else 'no'))
    add('')
    add('Matched')
    for j in range(8):
        if j in ph and hits[j]:
            cands = [hits[j][fam] for fam in FAMS if fam in hits[j]]
            ded, infos, bi = choose(ex, cands, j)
            if len(ded) > 1: add(tie_text(j, infos, bi))
            assign[j] = bi[0]
        add('%d %s' % (j, rname(assign[j]) if assign[j] else 'none'))
    add('')

    # ---------------- Recheck (continue the search, never punt) ------
    add('Recheck pending')
    rest = [j for j in range(8) if assign[j] is None]
    if not rest: add('none')
    for j in rest:
        parts = []
        cands = []
        for fam in FAMS:
            es = T.match[fam][j]
            if not es:
                parts.append('%s absent' % fam)
                continue
            if fam in ('Identity', 'NOT'):
                parts.append('%s %s' % (fam, ' '.join(str(r[1]) for r in es)))
            elif fam == 'Constant':
                parts.append('%s %s' % (fam, ' '.join('C%d' % r[1] for r in es)))
            else:
                parts.append('%s %s' % (fam, ' '.join('%d%d' % (r[1], r[2]) for r in es)))
            cands += es
        if not cands: return None           # 3-input bit: row not renderable
        ded, infos, bi = choose(ex, cands, j)
        assign[j] = bi[0]
        if len(ded) > 1:
            add('%d %s -> %s' % (j, ', '.join(parts), tie_text(j, infos, bi)))
        else:
            add('%d %s -> %s' % (j, ', '.join(parts), rname(assign[j])))
    add('')
    add('Selected')
    for j in range(8): add('%d %s' % (j, rname(assign[j])))
    add('')

    # ---------------- Apply ----------------
    add('Applying to %s' % format(q, '08b'))
    add('Input')
    for p in range(8): add('%d %d' % (p, bitc(q, p)))
    add('Output')
    bits = []
    for j in range(8):
        r = assign[j]
        v = reval(r, q)
        bits.append(v)
        if r[0] == 'I': add('%d %s = %d' % (j, rname(r), v))
        elif r[0] == 'NOT': add('%d %s = NOT(%d) = %d' % (j, rname(r), bitc(q, r[1]), v))
        elif r[0] == 'C': add('%d %s = %d' % (j, rname(r), v))
        elif r[0] in SYM:
            add('%d %s = %s(%d,%d) = %d' % (j, rname(r), r[0], bitc(q, r[1]), bitc(q, r[2]), v))
        else:
            base = r[0][:-4] if r[0].endswith('-NOT') else r[0]
            add('%d %s = %s(%d,NOT(%d)) = %d' % (j, rname(r), base, bitc(q, r[1]), bitc(q, r[2]), v))
    ans = ''.join(map(str, bits))
    add('')
    add('I will now return the answer in \\boxed{}')
    add('The answer in \\boxed is')
    add('\\boxed{%s}' % ans)
    return '\n'.join(L), ans, assign

# ---------------- linter (inverted re-derivation of the three fixes) ----------------
FAM_HEADERS = {'Identity': 'Identity', 'Constant': 'Constant', 'AND': 'AND',
               'OR': 'OR', 'XOR': 'XOR', 'AND-NOT': 'AND-NOT'}
STEP_RE = re.compile(r'^(\d{1,2}) m(\d+|-) (\d) (ok|x)$')
TIE_RE = re.compile(r'Tie (\d): (.+?) -> (\S+) bits ((?:\d ?)+)$')
CAND_RE = re.compile(r'^(\S+) r(\d)$')
PLACE_L_RE = re.compile(r'^0\+(\d)=(\d) (\S+)$')
PLACE_R_RE = re.compile(r'^7-(\d)=(\d) (\S+)$')

def _tok2rule(fam, tok):
    if fam == 'Identity': return ('I', int(tok))
    if fam == 'NOT': return ('NOT', int(tok))
    if fam == 'Constant': return ('C', int(tok))
    assert len(tok) == 2, (fam, tok)
    return (fam, int(tok[0]), int(tok[1]))

def _names(spec):
    """'I1 NOT2: 2' -> ['I1','NOT2']; 'none' -> []"""
    spec = spec.strip()
    if spec == 'none': return []
    return spec.rsplit(':', 1)[0].split()

def lint2(cot, prompt, final):
    af.lint(cot, prompt, final)               # inherited v1 invariants
    assert TIE_RULE in cot, 'tie rule line missing'
    ex, q = bs.parse(prompt)
    ocols = outcols(ex)

    lines = cot.split('\n')
    fam = None
    side = None
    in_runs = False
    j0 = step = None
    for ln in lines:
        s = ln.strip()
        if s in FAM_HEADERS or s.startswith(('NOT (', 'OR-NOT (', 'XOR-NOT (')):
            fam = s.split(' ')[0].split('(')[0].strip()
            in_runs = False
            continue
        if s == 'Selecting':
            fam = None; in_runs = False
            continue
        if s in ('Left', 'Right') and fam is not None:
            side = s; in_runs = True
            j0, step = (0, 1) if s == 'Left' else (7, -1)
            continue
        if in_runs:
            if s.startswith('Best:') or s == '':
                in_runs = s != ''
                if s.startswith('Best:'): in_runs = False
                continue
            if s == 'none' or s.startswith('Tie '):
                continue
            # witnessed run line (fix 2)
            steps = s.split(', ')
            anchor = None
            for k, st in enumerate(steps):
                m = STEP_RE.match(st)
                assert m, 'bad run step %r in fam %s' % (st, fam)
                tok, msetq, bq, verdict = m.group(1), m.group(2), int(m.group(3)), m.group(4)
                r = _tok2rule(fam, tok)
                if k == 0: anchor = r
                # member must be the stride-shifted anchor
                assert r == shift(anchor, step * k) or (r[0] not in ('I', 'NOT', 'C') and
                       (r[0], r[2], r[1]) == shift(anchor, step * k)), \
                    'run member %s not stride-consistent' % st
                # quoted match set must be the TRUE match set
                true_ms = mset(ex, ocols, r)
                quoted = [] if msetq == '-' else [int(c) for c in msetq]
                assert quoted == true_ms, 'forged match set %r (true %s)' % (st, true_ms)
                # required bit arithmetic
                assert bq == j0 + step * k, 'wrong required bit in %r' % st
                # verdict forced by the quoted evidence
                assert (verdict == 'ok') == (bq in quoted), 'verdict contradicts witness %r' % st
                # x terminates the run
                assert verdict == 'ok' or k == len(steps) - 1, 'run continues past x'
            continue

    # ---- fix 3: every Tie line re-derived ----
    nties = 0
    for ln in lines:
        m = TIE_RE.search(ln)
        if not m: continue
        nties += 1
        j = int(m.group(1))
        cands = []
        for cs in m.group(2).split(', '):
            cm = CAND_RE.match(cs.strip())
            assert cm, 'bad tie candidate %r' % cs
            r = parse_rule(cm.group(1))
            l, bits = cyc(ex, r, j)
            assert l == int(cm.group(2)), 'forged run length %r (true %d)' % (cs, l)
            cands.append((r, l, bits))
        best = cands[0]
        for inf in cands[1:]:
            if (inf[1], -rank(inf[0])) > (best[1], -rank(best[0])): best = inf
        win = parse_rule(m.group(3))
        assert same(win, best[0]), 'tie-break winner %s != policy %s' % (m.group(3), rname(best[0]))
        qbits = [int(x) for x in m.group(4).split()]
        assert sorted(qbits) == sorted(best[2]) and j in qbits, 'tie winner bits wrong'

    # ---- fix 1: Truncated -> Place -> Tentative chain ----
    def grab(pat):
        m = re.search(pat, cot)
        return m.group(1) if m else None
    tl = _names(grab(r'\nTruncated left: ([^\n]+)') or 'none')
    tr = _names(grab(r'\nTruncated right: ([^\n]+)') or 'none')
    bl = _names(grab(r'\nBest left: ([^\n]+)') or 'none')
    brn = _names(grab(r'\nBest right: ([^\n]+)') or 'none')
    assert tl == bl[:len(tl)] and tr == brn[:len(tr)], 'Truncated not a prefix of Best'
    expected = {}
    for k, name in enumerate(tl): expected[k] = name
    for k, name in enumerate(tr): expected[7 - k] = name
    placed = {}
    mpl = re.search(r'\nPlace left \(anchor 0 step \+1\)\n((?:0\+\d=\d \S+\n)+)', cot)
    if tl:
        assert mpl, 'Place left missing'
        pls = mpl.group(1).strip().split('\n')
        assert len(pls) == len(tl), 'Place left length != Truncated left'
        for k, pl in enumerate(pls):
            g = PLACE_L_RE.match(pl)
            assert g and int(g.group(1)) == k and int(g.group(2)) == k, 'Place left arithmetic %r' % pl
            assert g.group(3) == tl[k], 'Place left rule %r != run member %s' % (pl, tl[k])
            placed[k] = g.group(3)
    else:
        assert not mpl, 'Place left present without a left run'
    mpr = re.search(r'\nPlace right \(anchor 7 step -1\)\n((?:7-\d=\d \S+\n)+)', cot)
    if tr:
        assert mpr, 'Place right missing'
        prs = mpr.group(1).strip().split('\n')
        assert len(prs) == len(tr), 'Place right length != Truncated right'
        for k, pl in enumerate(prs):
            g = PLACE_R_RE.match(pl)
            assert g and int(g.group(1)) == k and int(g.group(2)) == 7 - k, 'Place right arithmetic %r' % pl
            assert g.group(3) == tr[k], 'Place right rule %r != run member %s' % (pl, tr[k])
            placed[7 - k] = g.group(3)
    else:
        assert not mpr, 'Place right present without a right run'
    assert placed == expected, 'Place block != Truncated run lists'
    mt = re.search(r'\nTentative\n((?:\d \S+\n){8})', cot)
    assert mt, 'no Tentative block'
    tent = {}
    for tlne in mt.group(1).strip().split('\n'):
        jj, name = tlne.split(' ', 1)
        tent[int(jj)] = name
    for j in range(8):
        assert tent[j] == expected.get(j, 'pending'), \
            'Tentative bit %d = %s, placement says %s' % (j, tent[j], expected.get(j, 'pending'))
    # committed bits survive into Selected unchanged (no silent re-selection)
    msel = re.search(r'\nSelected\n((?:\d .+\n){8})', cot)
    sel = {}
    for slne in msel.group(1).strip().split('\n'):
        jj, name = slne.split(' ', 1)
        sel[int(jj)] = name
    for j, name in expected.items():
        assert sel[j] == name, 'Selected bit %d = %s overrides committed %s' % (j, sel[j], name)
    # Matched lines agree with their tie winners
    mm = re.search(r'\nMatched\n((?:.+\n)+?)\n', cot)
    if mm:
        last_tie = None
        for mlne in mm.group(1).strip().split('\n'):
            tm = TIE_RE.search(mlne)
            if tm:
                last_tie = (int(tm.group(1)), tm.group(3))
                continue
            jj, name = mlne.split(' ', 1)
            if last_tie and int(jj) == last_tie[0]:
                assert name == last_tie[1], 'Matched bit %s != tie winner %s' % (mlne, last_tie[1])
                last_tie = None
    return nties

# ---------------- build ----------------
def build(mode):
    csv.field_size_limit(10 ** 9)
    sub = 'passed' if mode == 'passed' else 'failed'
    src_dir = os.path.join(ROOT, 'analysis/ali_validation/%s/bit_manipulation' % sub)
    ids = sorted(f[:-4] for f in os.listdir(src_dir) if f.endswith('.txt'))
    val_ids = {json.loads(l)['id'] for l in open(os.path.join(ROOT, 'pipeline/data/val.jsonl'))}
    rows = {r['id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
            if r['category'] == 'bit_manipulation'}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, 'competition_dataset/tokenizer.json'))

    out_rows = []
    drops = {'val': 0, 'cov<8': 0, 'answer!=gold': 0, 'render_fail': 0, 'tokens>%d' % TOKEN_GATE: 0}
    drop_ids = {k: [] for k in drops}
    toks, tiers, nties_total = [], {}, 0
    for rid in ids:
        if rid in val_ids:
            drops['val'] += 1; drop_ids['val'].append(rid); continue
        r = rows[rid]
        ex, q = bs.parse(r['prompt'])
        gold = r['answer'].strip()
        assign_s, cov = bs.solve_bits(ex)
        if cov < 8:
            drops['cov<8'] += 1; drop_ids['cov<8'].append(rid); continue
        res = render2(ex, q)
        if res is None:
            drops['render_fail'] += 1; drop_ids['render_fail'].append(rid); continue
        cot, ans, assign = res
        if ans != gold:
            drops['answer!=gold'] += 1; drop_ids['answer!=gold'].append(rid); continue
        nties_total += lint2(cot, r['prompt'], ans)        # 0-fail gate: raises on any violation
        nt = len(tok.encode(cot).ids)
        if nt > TOKEN_GATE:
            drops['tokens>%d' % TOKEN_GATE] += 1; drop_ids['tokens>%d' % TOKEN_GATE].append(rid); continue
        d = {'id': rid, 'category': 'bit_manipulation', 'prompt': r['prompt'],
             'cot': cot, 'final': ans, 'src': 'alipass' if mode == 'passed' else 'ali_failed_bit',
             'n_tokens': nt}
        if mode != 'passed':
            t = af.classify(os.path.join(src_dir, rid + '.txt'))
            d['tier'] = t
            tiers[t] = tiers.get(t, 0) + 1
        toks.append(nt)
        out_rows.append(d)
    out_name = 'bit_alipass2.jsonl' if mode == 'passed' else 'bit_alifix2.jsonl'
    out_path = os.path.join(ROOT, 'pipeline/data/v16', out_name)
    with open(out_path, 'w') as f:
        for d in out_rows:
            f.write(json.dumps(d) + '\n')
    toks_s = sorted(toks)
    stats = dict(n=len(toks), median=int(statistics.median(toks_s)) if toks else 0,
                 p95=toks_s[int(0.95 * len(toks_s))] if toks else 0,
                 max=max(toks_s) if toks else 0)
    print('mode:', mode, 'emitted:', len(out_rows), 'of', len(ids), 'drops:', drops)
    if tiers: print('tiers:', tiers)
    print('tokens:', stats)
    print('tie lines rendered (linted):', nties_total)
    for k in drops:
        if drop_ids[k] and k != 'val': print('drop %s ids (first 10): %s' % (k, drop_ids[k][:10]))
    print('wrote', out_path)
    return out_rows, drops, stats

if __name__ == '__main__':
    build('passed' if 'passed' in sys.argv[1:] else 'failed')
