#!/usr/bin/env python3
"""Per-line three-axis fidelity audit of crypt-boot-r8 greedy val transcripts.

For every structured line in each transcript OUTPUT, classify the line type
(r8 grammar families, tolerant) and check three independent axes:
  T TRANSCRIPTION  numbers/glyphs copied from the prompt or earlier trace
  A ARITHMETIC     every visible computation recomputed
  V VERDICT        does the stated conclusion follow from the line's OWN numbers
Axis values: 1 ok / 0 fail / '' not applicable-or-uncheckable.

Outputs: line_audit.csv + fidelity_report.md (same dir) + stdout summary.
"""
import csv, glob, os, re, statistics, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TDIR = os.path.join(HERE, 'r8_val_transcripts')

# ------------------------------------------------------------ rule machinery
def _safe(f):
    def g(a, b):
        try:
            return f(a, b)
        except ZeroDivisionError:
            return None
    return g

RULE_F = {
    'a+b': lambda a, b: a + b, 'a+b+1': lambda a, b: a + b + 1,
    'a+b-1': lambda a, b: a + b - 1, 'a+b+2': lambda a, b: a + b + 2,
    'a+b-2': lambda a, b: a + b - 2,
    'a*b': lambda a, b: a * b, 'a*b+1': lambda a, b: a * b + 1,
    'a*b-1': lambda a, b: a * b - 1, 'a*b+2': lambda a, b: a * b + 2,
    'a*b-2': lambda a, b: a * b - 2,
    'a-b': lambda a, b: a - b, 'b-a': lambda a, b: b - a,
    '|a-b|': lambda a, b: abs(a - b), '-|a-b|': lambda a, b: -abs(a - b),
    '|a-b|+1': lambda a, b: abs(a - b) + 1, '|a-b|-1': lambda a, b: abs(a - b) - 1,
    '|a-b|+2': lambda a, b: abs(a - b) + 2, '|a-b|-2': lambda a, b: abs(a - b) - 2,
    'a mod b': _safe(lambda a, b: a % b), 'b mod a': _safe(lambda a, b: b % a),
    'a div b': _safe(lambda a, b: a // b), 'b div a': _safe(lambda a, b: b // a),
    'a*a+b': lambda a, b: a * a + b, 'a+b*b': lambda a, b: a + b * b,
}
import math
RULE_F['gcd(a,b)'] = lambda a, b: math.gcd(a, b)
RULE_F['lcm(a,b)'] = _safe(lambda a, b: a * b // math.gcd(a, b))

NONNEG = {'a+b', 'a+b+1', 'a+b+2', 'a*b', 'a*b+1', 'a*b+2', '|a-b|', '|a-b|+1',
          '|a-b|+2', 'a mod b', 'b mod a', 'gcd(a,b)', 'lcm(a,b)', 'a div b',
          'b div a', 'a*a+b', 'a+b*b'}
# a+b-1 >= 19 over 2-digit operands -> effectively nonneg too
NONNEG |= {'a+b-1', 'a+b-2', 'a*b-1', 'a*b-2'}
ALWAYSNEG = {'-|a-b|'}          # 0 possible at a==b but renderer treats as neg-pre
MAYNEG = {'a-b', 'b-a'}

# prior order (MEASURED_OP_COUNTS, cryptarithm2.py)
PRIOR = ['a*b', 'a+b', 'a-b', '|a-b|', '-|a-b|', 'a+b-1', 'a*b-1', 'a+b+1',
         'a*b+1', 'b-a', 'a mod b', 'b mod a']
FAMA = {'+': ['a+b', 'a+b-1', 'a+b+1'], '*': ['a*b', 'a*b-1', 'a*b+1'],
        '-': ['a-b', '|a-b|', '-|a-b|', 'a mod b', 'b mod a']}
OPB = ['a+b', 'a*b', 'a-b', 'b-a', '|a-b|', '-|a-b|', 'a+b+1', 'a+b-1',
       'a*b+1', 'a*b-1', 'a mod b', 'b mod a']

RULE_WORDS = sorted(RULE_F, key=len, reverse=True)
RULE_ALT = '|'.join(re.escape(w) for w in RULE_WORDS)


def rule_eval(w, a, b):
    f = RULE_F.get(w)
    return None if f is None else f(a, b)


# ------------------------------------------------------------ state machine
class St:
    def __init__(self, eqs, qL, canonical):
        self.eqs = eqs              # list of (L,R)
        self.qL = qL
        self.sym2let = {}
        self.op2let = {}
        self.pins = {}              # letter -> digit
        self.excl = defaultdict(set)
        self.mode = None            # None / 'std' / 'le'
        self.rules = {}             # oplet -> rule word
        ops = []
        for L, _ in eqs:
            if L[2] not in ops:
                ops.append(L[2])
        if qL[2] not in ops:
            ops.append(qL[2])
        self.opglyphs = ops
        self.canonical = canonical
        self.rdom = {}              # oplet -> set of rule words (filled when opmap known)
        self.journal = []           # (kind, payload) for rollback
        self.guesses = []           # open guess markers: (jidx, desc, kind, who, val)
        self.snapshot = None        # for Case A/B
        self.leading = set()        # letters in leading positions
        self.killed_concat = False

    def init_rdom(self):
        for g, ol in self.op2let.items():
            fam = FAMA.get(g) if self.canonical else None
            self.rdom[ol] = set(fam if fam else OPB)

    def compute_leading(self):
        for L, R in self.eqs + [(self.qL, None)]:
            for s in (L[0], L[3]):
                if s in self.sym2let:
                    self.leading.add(self.sym2let[s])
            if R:
                mag, _ = strip_sign(R, L[2])
                if len(mag) >= 2 and mag[0] in self.sym2let:
                    self.leading.add(self.sym2let[mag[0]])

    def dom(self, let):
        d = set(range(10)) - set(self.pins.values()) - self.excl[let]
        if let in self.leading:
            d -= {0}
        if let in self.pins:
            return {self.pins[let]}
        return d

    # journaled mutations -------------------------------------------------
    def jpin(self, let, d):
        self.journal.append(('pin', let, self.pins.get(let)))
        self.pins[let] = d

    def jmode(self, m):
        self.journal.append(('mode', self.mode))
        self.mode = m

    def jrule(self, ol, w):
        self.journal.append(('rule', ol, self.rules.get(ol)))
        self.rules[ol] = w

    def jrkill(self, ol, ws):
        old = set(self.rdom.get(ol, set()))
        self.journal.append(('rdom', ol, old))
        self.rdom[ol] = old - set(ws)

    def jexcl(self, let, d):
        self.journal.append(('excl', let, set(self.excl[let])))
        self.excl[let].add(d)

    def push_guess(self, desc, kind, who, val):
        self.guesses.append((len(self.journal), desc, kind, who, val))

    def rollback_to(self, jidx):
        while len(self.journal) > jidx:
            ev = self.journal.pop()
            if ev[0] == 'pin':
                if ev[2] is None:
                    self.pins.pop(ev[1], None)
                else:
                    self.pins[ev[1]] = ev[2]
            elif ev[0] == 'mode':
                self.mode = ev[1]
            elif ev[0] == 'rule':
                if ev[2] is None:
                    self.rules.pop(ev[1], None)
                else:
                    self.rules[ev[1]] = ev[2]
            elif ev[0] == 'rdom':
                self.rdom[ev[1]] = ev[2]
            elif ev[0] == 'excl':
                self.excl[ev[1]] = ev[2]

    def die_last_guess(self):
        """kill the most recent open guess; return its marker (or None)."""
        if not self.guesses:
            return None
        m = self.guesses.pop()
        self.rollback_to(m[0])
        return m


def strip_sign(R, opglyphs):
    """(magnitude, sign_glyph_or_None). Leading char is a sign iff it is an op glyph."""
    if len(R) >= 2 and R[0] in opglyphs:
        return R[1:], R[0]
    return R, None


def num_of(syms, pins, sym2let, mode):
    """value of a 2-symbol operand under mode ('std'/'le'); None if unpinned."""
    ds = []
    for s in syms:
        let = sym2let.get(s)
        if let is None or let not in pins:
            return None
        ds.append(pins[let])
    if mode == 'le':
        ds = ds[::-1]
    v = 0
    for d in ds:
        v = v * 10 + d
    return v


def res_value(mag, pins, sym2let, mode):
    ds = []
    for s in mag:
        let = sym2let.get(s)
        if let is None or let not in pins:
            return None
        ds.append(pins[let])
    if mode == 'le':
        ds = ds[::-1]
    v = 0
    for d in ds:
        v = v * 10 + d
    return v


def digit_at(v, j, n, mode):
    """digit at printed result position j (1-based) of |v| printed with n digits."""
    s = str(abs(v))
    if len(s) != n:
        return None
    if mode == 'le':
        j = n + 1 - j
    return int(s[j - 1])


# ------------------------------------------------------------ record keeping
class Rec:
    __slots__ = ('tid', 'ln', 'ctr', 'typ', 'fam', 't', 'a', 'v', 'note', 'text')

    def __init__(self, tid, ln, ctr, typ, fam, t, a, v, note, text):
        self.tid, self.ln, self.ctr = tid, ln, ctr
        self.typ, self.fam = typ, fam
        self.t, self.a, self.v = t, a, v
        self.note, self.text = note, text


class Ax:
    """axis accumulator: collect (ok, why) items per axis."""
    def __init__(self):
        self.t = []
        self.a = []
        self.v = []

    def T(self, ok, why=''):
        self.t.append((bool(ok), why))

    def A(self, ok, why=''):
        self.a.append((bool(ok), why))

    def V(self, ok, why=''):
        self.v.append((bool(ok), why))

    def out(self):
        def red(xs):
            if not xs:
                return None
            return all(ok for ok, _ in xs)
        notes = [w for xs in (self.t, self.a, self.v) for ok, w in xs if not ok and w]
        return red(self.t), red(self.a), red(self.v), ' | '.join(notes[:4])


# ------------------------------------------------------------ clause parsers
ONES_DSC = re.compile(
    r"(?:([A-Z])\s*(?:=\s*(\d)|in \{([\d,]+)\})|(\d))")
DSC_PAT = r"(?:[A-Z]\s*=\s*\d|[A-Z] in \{[\d,]+\}|[A-Z]|\d)"
OPL = '[vwxyz]'
OPLSEQ = 'xyzwv'


def parse_dsc(txt):
    """-> (letter|None, fixed|None, domain|None) for `B=3` / `B in {..}` / `3`."""
    m = ONES_DSC.fullmatch(txt.strip())
    if not m:
        return None
    let, fx, dom, bare = m.groups()
    if bare is not None:
        return (None, int(bare), None)
    if fx is not None:
        return (let, int(fx), None)
    return (let, None, set(int(x) for x in dom.split(',')))


PAIR_RE = re.compile(
    r"^(?:(\d)\s*([+*-])\s*(\d)|\|(\d)-(\d)\|)(?: then ([+-]\d))? ends (\d)(?: or (\d))?$")


def eval_pair(m):
    """recompute a `p op q [then +1] ends o [or o2]` pair -> (set stated, set true, p, q)."""
    p1, op, q1, ap, aq, off, o1, o2 = m.groups()
    off = int(off) if off else 0
    if ap is not None:
        p, q = int(ap), int(aq)
        true = {(abs(p - q) + off) % 10, (abs(q - p) + off) % 10}
    else:
        p, q = int(p1), int(q1)
        base = p + q if op == '+' else p * q if op == '*' else p - q
        true = {(base + off) % 10}
    stated = {int(o1)} | ({int(o2)} if o2 else set())
    return stated, true, p, q


def parse_ones_header(seg):
    """`EQk ones: a ends DSC, b ends DSC, result ends ED` -> dict or None.
    ED may be DSC or `a's ones symbol` / `b's ones symbol`."""
    m = re.match(rf"^EQ(\d) ones: a ends ({DSC_PAT}), b ends ({DSC_PAT}),"
                 r" result ends (.+)$", seg)
    if not m:
        return None
    k = int(m.group(1))
    a = parse_dsc(m.group(2))
    b = parse_dsc(m.group(3))
    ed = m.group(4).strip()
    if a is None or b is None:
        return None
    if ed in ("a's ones symbol", "b's ones symbol"):
        tgt = ('opnd', ed[0])
    else:
        t = parse_dsc(ed)
        if t is None:
            return None
        tgt = ('dsc', t)
    return {'k': k, 'a': a, 'b': b, 'tgt': tgt}


def check_ones_witness(seg, ax, st, mode, ruleword=None, pin_concl=None):
    """Full check of an ones/onesenum witness clause: header + pair list +
    `-> none matches` (kill) or unique-match pin. seg excludes the kill/pin tail
    handling: caller passes conclusion via pin_concl ('kill' or (L,d))."""
    parts = seg.split('; ')
    hd = parse_ones_header(parts[0] + (', ' + parts[1] if parse_ones_header(parts[0]) is None and len(parts) > 1 else ''))
    hd = parse_ones_header(parts[0])
    pi = 1
    if hd is None:
        return False
    k, A, B, tgt = hd['k'], hd['a'], hd['b'], hd['tgt']
    if k - 1 >= len(st.eqs):
        ax.T(False, f'EQ{k} does not exist')
        return True
    L, R = st.eqs[k - 1]
    mag, sgn = strip_sign(R, L[2])
    mm = mode or st.mode
    # transcription: ones symbols/letters/domains/pins
    for side, dsc, syms in (('a', A, (L[0], L[1])), ('b', B, (L[3], L[4]))):
        ones_sym = syms[1] if mm != 'le' else syms[0]
        let_true = st.sym2let.get(ones_sym)
        let, fx, dom = dsc
        if let is not None:
            ax.T(let == let_true, f'{side}-ones letter {let} != table {let_true}')
        if fx is not None and let is not None:
            ax.T(st.pins.get(let) == fx,
                 f'{side}-ones {let}={fx} but pinned {st.pins.get(let)}')
        elif fx is not None and let is None:
            tl = let_true
            ax.T(tl is not None and st.pins.get(tl) == fx,
                 f'{side}-ones digit {fx} != pin of {tl} ({st.pins.get(tl) if tl else "?"})')
        if dom is not None and let is not None:
            ax.T(dom <= st.dom(let), f'{side}-ones dom {sorted(dom)} contains digits'
                 f' outside tracked {sorted(st.dom(let))}')
    r_ones_sym = mag[-1] if mm != 'le' else mag[0]
    r_let_true = st.sym2let.get(r_ones_sym)
    if tgt[0] == 'opnd':
        opnd_syms = (L[0], L[1]) if tgt[1] == 'a' else (L[3], L[4])
        o_sym = opnd_syms[1] if mm != 'le' else opnd_syms[0]
        ax.T(r_ones_sym == o_sym,
             f"result ones {r_ones_sym} != {tgt[1]}'s ones {o_sym}")
    else:
        let, fx, dom = tgt[1]
        if let is not None:
            ax.T(let == r_let_true, f'result-ones letter {let} != table {r_let_true}')
        if fx is not None and let is not None:
            ax.T(st.pins.get(let) == fx,
                 f'result-ones {let}={fx} but pinned {st.pins.get(let)}')
        if dom is not None and let is not None:
            ax.T(dom <= st.dom(let), f'result-ones dom exceeds tracked')
    # pairs
    pairs = []
    for p in parts[pi:]:
        m = PAIR_RE.match(p.strip())
        if m:
            stated, true, pv, qv = eval_pair(m)
            ax.A(stated <= true or stated == true,
                 f'pair {p.strip()!r} recomputes to {sorted(true)}')
            pairs.append((pv, qv, stated))
        elif p.strip():
            return False
    # verdict: matches + completeness over the STATED domains
    def tgt_digits(pv, qv):
        if tgt[0] == 'opnd':
            return {pv if tgt[1] == 'a' else qv}
        let, fx, dom = tgt[1]
        if fx is not None:
            return {fx}
        return dom or set()

    n_match = 0
    for pv, qv, stated in pairs:
        if stated & tgt_digits(pv, qv):
            n_match += 1
    aL, afx, adom = A
    bL, bfx, bdom = B
    aset = adom if adom is not None else ({afx} if afx is not None else None)
    bset = bdom if bdom is not None else ({bfx} if bfx is not None else None)
    complete = None
    if aset is not None and bset is not None:
        want = set()
        for pv in aset:
            for qv in bset:
                if aL is not None and bL is not None:
                    if aL == bL and pv != qv:
                        continue
                    if aL != bL and pv == qv:
                        continue
                want.add((pv, qv))
        got = {(pv, qv) for pv, qv, _ in pairs}
        complete = want <= got
    if pin_concl == 'kill':
        ok = n_match == 0
        ax.V(ok, f'{n_match} stated pair(s) actually match the target')
        if complete is not None:
            ax.V(complete, 'pair enumeration incomplete vs stated domains')
    elif isinstance(pin_concl, tuple):
        Lp, dp = pin_concl
        ax.V(n_match == 1, f'{n_match} stated pair(s) match (need exactly 1)')
        if complete is not None:
            ax.V(complete, 'pair enumeration incomplete vs stated domains')
        if n_match == 1:
            pv, qv, _ = next(x for x in pairs if x[2] & tgt_digits(x[0], x[1]))
            # the forced letter must be the enumerated operand's ones letter
            exp = {}
            if aL is not None and (adom is not None):
                exp[aL] = pv
            if bL is not None and (bdom is not None):
                exp[bL] = qv
            ax.V(exp.get(Lp) == dp,
                 f'unique match forces {exp}, line pins {Lp}={dp}')
    return True


CLASH_RES = [
    ('ndig', re.compile(r"^(\d+) digit\(s\) but the result has (\d+)$")),
    ('samepos', re.compile(r"^result positions (\d) and (\d) are the same symbol"
                           r" but digits (\d) != (\d)$")),
    ('dpin', re.compile(r"^result digit (\d) would be (\d) but ([A-Z]) = (\d)$")),
    ('dtaken', re.compile(r"^result digit (\d) would be (\d) but (\d) is taken"
                          r" by ([A-Z])$")),
    ('reads', re.compile(r"^but the result reads (-?\d+)$")),
    ('negglyph', re.compile(r"^negative but the result has no sign glyph$")),
    ('posglyph', re.compile(r"^positive but the result carries the sign glyph$")),
]


def check_clash(txt, v, k, st, ax, mode=None, hypo=None):
    """check one clash tail given computed/stated value v on EQk. Returns True if parsed."""
    txt = txt.strip().rstrip('.')
    if k - 1 >= len(st.eqs):
        return True
    L, R = st.eqs[k - 1]
    mag, sgn = strip_sign(R, L[2])
    n = len(mag)
    mm = mode or st.mode
    for kind, rx in CLASH_RES:
        m = rx.match(txt)
        if not m:
            continue
        if kind == 'ndig':
            nd, rl = int(m.group(1)), int(m.group(2))
            if v is not None:
                ax.A(len(str(abs(v))) == nd, f'{v} has {len(str(abs(v)))} digits not {nd}')
            ax.T(rl == n, f'result has {n} digits not {rl}')
            ax.V(nd != rl, 'digit counts equal -> no clash')
        elif kind == 'samepos':
            j1, j2, d1, d2 = map(int, m.groups())
            ok = j1 <= n and j2 <= n and mag[j1 - 1] == mag[j2 - 1]
            ax.T(ok, f'positions {j1},{j2} of {mag!r} not the same symbol')
            if v is not None:
                t1, t2 = digit_at(v, j1, n, mm), digit_at(v, j2, n, mm)
                if t1 is not None:
                    ax.A(t1 == d1 and t2 == d2,
                         f'digits at {j1},{j2} of {v} are {t1},{t2} not {d1},{d2}')
            ax.V(d1 != d2, f'stated digits {d1}=={d2} -> no clash')
        elif kind == 'dpin':
            j, want, Lt, have = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            if j <= n:
                ax.T(st.sym2let.get(mag[j - 1]) == Lt,
                     f'result pos {j} letter is {st.sym2let.get(mag[j-1])} not {Lt}')
            ax.T(st.pins.get(Lt) == have, f'{Lt} pinned {st.pins.get(Lt)} not {have}')
            if v is not None:
                tj = digit_at(v, j, n, mm)
                if tj is not None:
                    ax.A(tj == want, f'digit {j} of {v} is {tj} not {want}')
            ax.V(want != have, 'stated digits equal -> no clash')
        elif kind == 'dtaken':
            j, want, d, Lt = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            ax.T(st.pins.get(Lt) == d, f'{d} not pinned to {Lt} ({st.pins.get(Lt)})')
            if j <= n:
                jl = st.sym2let.get(mag[j - 1])
                ax.V(jl != Lt, f'result pos {j} letter IS {Lt} itself -> digit'
                     f' {d} there is consistent, not a clash')
            if v is not None:
                tj = digit_at(v, j, n, mm)
                if tj is not None:
                    ax.A(tj == want, f'digit {j} of {v} is {tj} not {want}')
            ax.V(want == d, f'stated {want} != taken digit {d} -> clash text broken')
        elif kind == 'reads':
            rv = int(m.group(1))
            tv = res_value(mag, st.pins, st.sym2let, mm or 'std')
            if tv is not None:
                if sgn:
                    tv = -tv
                ax.T(rv == tv, f'result reads {tv} from pins, not {rv}')
            if v is not None:
                ax.V(v != rv, f'{v} == {rv} -> no clash')
        elif kind == 'negglyph':
            if v is not None:
                ax.A(v < 0, f'{v} not negative')
            ax.T(sgn is None, 'result DOES carry a sign glyph')
            ax.V(True)
        elif kind == 'posglyph':
            if v is not None:
                ax.A(v >= 0, f'{v} negative')
            ax.T(sgn is not None, 'result has no sign glyph')
            ax.V(True)
        return True
    if txt == 'undefined':
        return True
    return False


def parse_case_eval(txt):
    """`a=41, b=40: a*b = 1640` or `a = 41, b = 40, a*b = 1640` -> (A,B,rw,v,rest)."""
    m = re.match(r"^a\s*=\s*(\d+), b\s*=\s*(\d+)[:,] (" + RULE_ALT +
                 r") = (-?\d+)(?:[,;] (.+))?$", txt)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), (m.group(5) or '')
    m = re.match(r"^a\s*=\s*(\d+), b\s*=\s*(\d+)[:,]? undefined$", txt)
    if m:
        return int(m.group(1)), int(m.group(2)), None, None, 'undefined'
    return None


# ------------------------------------------------------------ line checkers
def split_cases(body):
    """split `L = 4 fails (..); L = 5 fails (..)` respecting parens."""
    out, depth, cur = [], 0, ''
    i = 0
    while i < len(body):
        c = body[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0 and body.startswith('; ', i):
            out.append(cur)
            cur = ''
            i += 2
            continue
        cur += c
        i += 1
    if cur:
        out.append(cur)
    return out


def check_fail_clause(inner, st, ax, mode=None, hypo=None):
    """one `(...)` fail reason: possibly several `EQj:`-scoped case evals.
    hypo=(letter, digit): the candidate assignment under test (treated as a
    temporary pin so case-local references to it are not false transcription
    fails). Returns True if at least one sub-clause parsed."""
    restore = None
    if hypo is not None and hypo[0] not in st.pins:
        st.pins[hypo[0]] = hypo[1]
        restore = hypo[0]
    try:
        return _check_fail_clause(inner, st, ax, mode, hypo)
    finally:
        if restore is not None:
            st.pins.pop(restore, None)


def _check_fail_clause(inner, st, ax, mode=None, hypo=None):
    any_ok = False
    for sub in split_cases(inner):
        sub = sub.strip()
        k = None
        m = re.match(r"^EQ(\d): (.+)$", sub)
        if m:
            k = int(m.group(1))
            sub = m.group(2)
        mm = re.match(r"^(standard|little-endian): (.+)$", sub)
        cmode = mode
        if mm:
            cmode = 'std' if mm.group(1) == 'standard' else 'le'
            sub = mm.group(2)
        ce = parse_case_eval(sub)
        if ce:
            A, B, rw, v, rest = ce
            if rw:
                tv = rule_eval(rw, A, B)
                ax.A(tv == v, f'{A} {rw} {B} = {tv} not {v}')
            if rest and rest != 'undefined' and k:
                check_clash(rest, v, k, st, ax, cmode, hypo)
            any_ok = True
            continue
        m = re.match(r"^(\d) is taken by ([A-Z])$", sub)
        if m:
            d, Lt = int(m.group(1)), m.group(2)
            ax.T(st.pins.get(Lt) == d, f'{d} not pinned to {Lt} ({st.pins.get(Lt)})')
            if hypo is not None:
                ax.V(Lt != hypo[0], f'{d} "taken by" the candidate letter itself'
                     f' -> no clash')
            any_ok = True
            continue
        m = re.match(r"^ones (.+) ends (\d) not (\d)$", sub)
        if m:
            pm = PAIR_RE.match(f"{m.group(1)} ends {m.group(2)}")
            if pm:
                stated, true, _, _ = eval_pair(pm)
                ax.A(stated == true or stated <= true,
                     f'ones {m.group(1)} recomputes {sorted(true)} not {m.group(2)}')
            ax.V(m.group(2) != m.group(3), 'stated ones digits equal -> no clash')
            any_ok = True
            continue
        m = re.match(r"^with (.+): (" + RULE_ALT + r") (<=|>=) (-?\d+), the result"
                     r" needs (-?\d+)\.\.(-?\d+)$", sub)
        if m:
            bound, lo, hi = int(m.group(4)), int(m.group(5)), int(m.group(6))
            if m.group(3) == '<=':
                ax.V(bound < lo, f'<= {bound} does not miss {lo}..{hi}')
            else:
                ax.V(bound > hi, f'>= {bound} does not miss {lo}..{hi}')
            any_ok = True
            continue
        if 'ones' in sub and 'carry' in sub:
            any_ok = True   # tens-family clause: arithmetic checked loosely below
            for am in re.finditer(r"(\d+)\+(\d+)(?:\+(\d+))? = (\d+)", sub):
                x = int(am.group(1)) + int(am.group(2)) + int(am.group(3) or 0)
                ax.A(x == int(am.group(4)), f'{am.group(0)} recomputes to {x}')
            continue
    return any_ok


# ------------------------------------------------------------ main transcript loop
def parse_transcript(path):
    txt = open(path).read()
    hdr = txt.splitlines()[0]
    tid = re.search(r"id=(\S+)", hdr).group(1)
    gold = re.search(r"gold=(\S+)", hdr).group(1)
    ntok = int(re.search(r"ntok=(\d+)", hdr).group(1))
    prompt = txt.split('## PROMPT', 1)[1].split('## OUTPUT', 1)[0]
    out = txt.split('## OUTPUT', 1)[1]
    eqs = []
    for ln in prompt.splitlines():
        m = re.match(r"^(\S{5}) = (\S+)$", ln.strip())
        if m:
            eqs.append((m.group(1), m.group(2)))
    qm = re.search(r"result for: (\S+)", prompt)
    qL = qm.group(1)
    return tid, gold, ntok, eqs, qL, out


PROSE = (
    'We need to infer', 'Build the symbol table', 'Digit symbols in order',
    'Operators:', 'Deduce the digit map;', '</think>',
)


def audit_one(path):
    tid, gold, ntok, eqs, qL, out = parse_transcript(path)
    opglyphs = []
    for L, _ in eqs + [(qL, '')]:
        if L[2] not in opglyphs:
            opglyphs.append(L[2])
    canonical = all(g in '+-*/' for g in opglyphs)
    st = St(eqs, qL, canonical)
    recs = []
    lines = out.splitlines()
    last_ctr = 0
    expected_letters = {}
    # ground-truth first-appearance walk for the union table
    walk = []
    for L, R in eqs:
        mag, sgn = strip_sign(R, L[2])
        for s in (L[0], L[1], L[3], L[4]) + tuple(mag):
            if s not in walk:
                walk.append(s)
    for s in (qL[0], qL[1], qL[3], qL[4]):
        if s not in walk:
            walk.append(s)
    opwalk = []
    for L, _ in eqs + [(qL, '')]:
        if L[2] not in opwalk:
            opwalk.append(L[2])
    boxed_first = None
    compute_val = None
    qops_vals = None
    encode_str = None
    in_think = True

    def add(ln_no, ctr, typ, fam, ax, text):
        t, a, v, note = ax.out()
        recs.append(Rec(tid, ln_no, ctr, typ, fam,
                        1 if t else 0 if t is not None else '',
                        1 if a else 0 if a is not None else '',
                        1 if v else 0 if v is not None else '',
                        note, text))

    for ln_no, raw in enumerate(lines, 1):
        text = raw.strip()
        if not text:
            continue
        if text == '</think>':
            in_think = False
            continue
        if any(text.startswith(p) for p in PROSE):
            continue
        ax = Ax()
        ctr = ''
        body = text
        sm = re.match(r"^s(\d+): (.*)$", text)
        if sm:
            ctr = int(sm.group(1))
            body = sm.group(2)
            cax = Ax()
            cax.T(ctr == last_ctr + 1, f's{ctr} after s{last_ctr}')
            add(ln_no, ctr, 'try-counter', '', cax, f's{ctr}:')
            last_ctr = ctr

        try:
            typ, fam = classify_and_check(body, st, ax, text,
                                          dict(eqs=eqs, qL=qL, walk=walk,
                                               opwalk=opwalk, opglyphs=opglyphs))
        except Exception as e:
            typ, fam = 'malformed', f'exception:{type(e).__name__}'
            ax = Ax()
        if typ is None:
            continue
        add(ln_no, ctr, typ, fam, ax, text)
    return tid, gold, ntok, recs, out


# the big dispatcher ------------------------------------------------------
def classify_and_check(body, st, ax, raw, ctx):
    eqs, qL, walk, opwalk = ctx['eqs'], ctx['qL'], ctx['walk'], ctx['opwalk']
    opglyphs = ctx['opglyphs']

    # ---- table scan
    m = re.match(r"^EQ(\d) (\S+) chars: (.+?) -> op (\S) ; digits (\S) (\S) (\S) (\S)"
                 r" ; RHS (\S+) -> (.+)$", body)
    if m:
        k = int(m.group(1))
        Ls, idx, opc = m.group(2), m.group(3), m.group(4)
        dg = [m.group(i) for i in range(5, 9)]
        Rs, tail = m.group(9), m.group(10)
        if k - 1 < len(eqs):
            L, R = eqs[k - 1]
            ax.T(Ls == L, f'LHS {Ls!r} != prompt {L!r}')
            ax.T(Rs == R, f'RHS {Rs!r} != prompt {R!r}')
            chars = re.findall(r"(\d)=(\S)", idx)
            ax.T(len(chars) == 5 and all(int(i) <= len(L) and L[int(i) - 1] == c
                 for i, c in chars), 'indexed chars != prompt LHS')
            # verdict: conclusions vs the line's OWN indexed chars
            stated = {int(i): c for i, c in chars}
            ax.V(stated.get(3) == opc, f'op {opc} != own stated pos-3 char'
                 f' {stated.get(3)}')
            ax.V(dg == [stated.get(1), stated.get(2), stated.get(4), stated.get(5)],
                 'digit list != own stated chars 1,2,4,5')
            mag, sgn = strip_sign(R, L[2])
            sm2 = re.match(r"^leading (\S) is the op glyph \(sign\), digits (.+)$", tail)
            if sm2:
                smag, ssgn = strip_sign(Rs, sm2.group(1))
                ax.V(len(Rs) >= 2 and Rs[0] == sm2.group(1),
                     'sign annotation does not match own stated RHS')
                ax.V(sm2.group(2).split() == list(Rs[1:]),
                     'RHS digit list != own stated RHS minus sign')
                ax.T(sgn is not None, 'prompt RHS carries no sign glyph')
            else:
                dm = re.match(r"^digits (.+)$", tail)
                if dm:
                    ax.T(sgn is None, 'missed leading sign glyph of prompt RHS')
                    ax.V(dm.group(1).split() == list(Rs), 'RHS digit list != stated RHS')
        else:
            ax.T(False, f'EQ{k} does not exist in prompt')
        return 'table-scan', ''

    m = re.match(r"^Query (\S+) chars: (.+?) -> op (\S) ; digits (\S) (\S) (\S) (\S)$", body)
    if m:
        Ls, idx, opc = m.group(1), m.group(2), m.group(3)
        dg = [m.group(i) for i in range(4, 8)]
        ax.T(Ls == qL, f'query {Ls!r} != prompt {qL!r}')
        chars = re.findall(r"(\d)=(\S)", idx)
        ax.T(len(chars) == 5 and all(int(i) <= len(Ls) and Ls[int(i) - 1] == c
             for i, c in chars), 'indexed chars != stated query')
        ax.V(len(Ls) == 5 and opc == Ls[2], 'op != pos-3 char')
        ax.V(len(Ls) == 5 and dg == [Ls[0], Ls[1], Ls[3], Ls[4]], 'digit list wrong')
        return 'table-scan', 'query'

    # ---- union table entries
    m = re.match(r"^(\S) (?:\(in EQ\d\) )?-> ([A-Z])$", body)
    if m:
        s, let = m.group(1), m.group(2)
        ax.T(s in walk, f'symbol {s!r} not a digit symbol of the prompt')
        # expected letter = position in first-appearance walk of DIGIT symbols
        if s in walk:
            exp = chr(ord('A') + walk.index(s))
            ax.V(let == exp, f'{s!r} should map to {exp} (first-appearance), got {let}')
        if s in st.sym2let:
            ax.T(False, f'symbol {s!r} mapped twice')
        st.sym2let[s] = let
        return 'union', 'digit-map'
    m = re.match(r"^(\S) (?:\(in EQ\d\) )?-> ([vwxyz])$", body)
    if m:
        g, ol = m.group(1), m.group(2)
        ax.T(g in opwalk, f'{g!r} not an operator glyph')
        if g in opwalk:
            exp = OPLSEQ[opwalk.index(g)] if opwalk.index(g) < len(OPLSEQ) else '?'
            ax.V(ol == exp, f'{g!r} should be {exp}, got {ol}')
        st.op2let[g] = ol
        if set(st.op2let) == set(opwalk):
            st.init_rdom()
            st.compute_leading()
        return 'union', 'op-map'

    m = re.match(r"^Op glyphs: (.+?) (?:\(the literal arithmetic signs\) )?; digit-position"
                 r" glyphs: (.+?) ; intersection: (.+?) -> (.+)$", body)
    if m:
        ops = m.group(1).split()
        digs = m.group(2).split()
        ax.T(ops == opwalk, f'op set {ops} != {opwalk}')
        ax.T(digs == walk, 'digit-glyph set != first-appearance set')
        inter = set(ops) & set(digs)
        ax.V((m.group(3) == 'none') == (not inter), 'intersection claim wrong')
        if not st.rdom:
            st.init_rdom()
            st.compute_leading()
        return 'union', 'disjoint'

    # ---- routing
    if body.startswith('Routing check on the query operator'):
        return 'routing', 'header'
    m = re.match(r"^Query: (\S+) -> ([A-Z])([A-Z]) ([vwxyz]) ([A-Z])([A-Z])$", body)
    if m:
        ax.T(m.group(1) == qL, 'query echoed wrong')
        ll = [st.sym2let.get(qL[0]), st.sym2let.get(qL[1]),
              st.sym2let.get(qL[3]), st.sym2let.get(qL[4])]
        ax.V([m.group(2), m.group(3), m.group(5), m.group(6)] == ll,
             f'query letters != table {ll}')
        ax.V(st.op2let.get(qL[2]) == m.group(4), 'query op letter wrong')
        return 'routing', 'query-letters'
    m = re.match(r"^EQ(\d) RHS (\S+) vs (\S+) \(in order\) / (\S+) \(swapped\)"
                 r" -> (.+)\.$", body)
    if m:
        k = int(m.group(1))
        Rs, ino, sw, verd = m.group(2), m.group(3), m.group(4), m.group(5)
        if k - 1 < len(eqs):
            L, R = eqs[k - 1]
            ax.T(Rs == R, f'RHS {Rs!r} != prompt {R!r}')
            ax.T(ino == L[0] + L[1] + L[3] + L[4], 'in-order operand string wrong')
            ax.T(sw == L[3] + L[4] + L[0] + L[1], 'swapped operand string wrong')
            ax.T(L[2] == qL[2], f'EQ{k} op {L[2]!r} is not the query op {qL[2]!r}')
        truth = ('matches in order' if Rs == ino else
                 'matches swapped' if Rs == sw else 'neither')
        if Rs == ino and Rs == sw:
            truth = 'matches in order'
        ax.V(verd.startswith(truth) or (truth != 'neither' and truth in verd),
             f'stated {verd!r}, own strings say {truth!r}')
        return 'routing', 'example'
    m = re.match(r"^EQ(\d) RHS (\S+) has (\d+) symbols, the operands have 4"
                 r" -> not a juxtaposition\.$", body)
    if m:
        k, Rs, n = int(m.group(1)), m.group(2), int(m.group(3))
        if k - 1 < len(eqs):
            L, R = eqs[k - 1]
            ax.T(Rs == R, f'RHS {Rs!r} != prompt {R!r}')
            ax.T(L[2] == qL[2], f'EQ{k} op is not the query op')
        ax.A(len(Rs) == n, f'{Rs!r} has {len(Rs)} symbols not {n}')
        ax.V(n != 4, 'n==4 does not refute juxtaposition')
        return 'routing', 'example'
    m = re.match(r"^Every ([vwxyz]) example matches (in order|swapped) -> \1 is"
                 r" concatenation \((in order|swapped)\)\.$", body)
    if m:
        st.jrule(m.group(1), 'concat')
        ax.V(m.group(2) == m.group(3), 'order mismatch in conclusion')
        return 'routing', 'conclusion'
    if re.match(r"^No single juxtaposition order survives -> [vwxyz] is a value rule\.$", body):
        st.killed_concat = True
        return 'routing', 'conclusion'
    m = re.match(r"^Both orders match every ([vwxyz]) example .*\(guess\)\.$", body)
    if m:
        st.jrule(m.group(1), 'concat')
        return 'routing', 'conclusion'
    m = re.match(r"^Apply to the query: operands (\S+) and (\S+) -> (\S+)\.$", body)
    if m:
        a, b, r = m.groups()
        ax.T(a == qL[0] + qL[1], f'operand {a!r} != query {qL[0]+qL[1]!r}')
        ax.T(b == qL[3] + qL[4], f'operand {b!r} != query {qL[3]+qL[4]!r}')
        ax.V(r in (a + b, b + a), f'{r!r} != juxtaposition of stated operands')
        return 'routing', 'apply'

    # ---- policy lines
    if re.match(r"^the operator glyphs are the literal", body):
        ax.T(st.canonical, 'op glyphs are NOT literal signs')
        return 'policy', 'literal'
    if re.match(r"^the operator glyphs are arbitrary symbols", body):
        ax.T(not st.canonical, 'op glyphs ARE literal signs')
        return 'policy', 'arbitrary'
    if re.match(r"^no number starts with 0", body):
        return 'policy', 'lead0'

    # ---- norepl
    m = re.match(r"^rules are drawn without replacement -> ([vwxyz]) is not"
                 r" ((?:" + RULE_ALT + r")(?:, (?:" + RULE_ALT + r"))*)\.$", body)
    if m:
        import re as _re
        ol = m.group(1)
        rws = _re.findall(RULE_ALT, m.group(2))
        for rw in rws:
            owner = [o for o, w in st.rules.items() if w == rw and o != ol]
            forced = [o for o, d in st.rdom.items() if d == {rw} and o != ol]
            ax.T(bool(owner or forced), f'{rw} not assigned to another operator')
        st.jrkill(ol, rws)
        return 'kill', 'norepl'

    # ---- kill lines  `g != r1, r2: BODY -> drop.`
    m = re.match(r"^([vwxyz]) != ((?:" + RULE_ALT + r")(?:, (?:" + RULE_ALT + r"))*):"
                 r" (.+) -> drop\.$", body)
    if m:
        ol, rl, kb = m.group(1), m.group(2), m.group(3)
        rws = re.findall(RULE_ALT, rl)
        fam = check_kill_body(kb, rws, ol, st, ax)
        st.jrkill(ol, rws)
        return 'kill', fam

    # ---- modekill
    m = re.match(r"^(standard|little-endian) order dies on ([vwxyz]): (.+) -> none matches"
                 r" -> (standard|little-endian) order\.$", body)
    if m:
        died = 'std' if m.group(1) == 'standard' else 'le'
        ol, inner = m.group(2), m.group(3)
        surv = 'std' if m.group(4) == 'standard' else 'le'
        ax.V(died != surv, 'survivor == dead mode')
        # inner: `rule: ones-clause` repeated
        segs = re.split(r"; (?=(?:" + RULE_ALT + r"): EQ\d)", inner)
        shown_rules = []
        parsed_any = False
        for seg in segs:
            mm = re.match(r"^(" + RULE_ALT + r"): (.+)$", seg)
            if not mm:
                continue
            shown_rules.append(mm.group(1))
            if check_ones_witness(mm.group(2), ax, st, died, pin_concl='kill'):
                parsed_any = True
        live = st.rdom.get(ol, set()) if not st.rules.get(ol) else {st.rules[ol]}
        ax.V(live <= set(shown_rules),
             f'modekill omits live rule(s) {sorted(live - set(shown_rules))}')
        st.jmode(surv)
        if not parsed_any:
            return 'malformed', 'modekill'
        return 'kill', 'modekill'

    # ---- pin lines (ordered most-specific first)
    m = re.match(r"^([A-Z]) in \{([\d,]+)\} on EQ(\d): (.+) -> ([A-Z]) = (\d)\.$", body)
    if m:
        Lp, pre, k, cases, L2, d = m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5), int(m.group(6))
        pred = set(int(x) for x in pre.split(','))
        ax.T(pred <= st.dom(Lp), f'pre-domain {sorted(pred)} contains digits outside'
             f' tracked {sorted(st.dom(Lp))}')
        ax.T(Lp == L2, 'pinned letter != enumerated letter')
        killed = set()
        ok_parse = True
        for case in split_cases(cases):
            cm = re.match(r"^([A-Z]) = (\d) fails \((.+)\)$", case.strip())
            if not cm:
                ok_parse = False
                continue
            killed.add(int(cm.group(2)))
            ax.T(cm.group(1) == Lp, 'case letter mismatch')
            check_fail_clause(cm.group(3), st, ax, hypo=(Lp, int(cm.group(2))))
        ax.V(killed == pred - {d}, f'killed {sorted(killed)} != pre minus survivor'
             f' {sorted(pred - {d})}')
        ax.V(d in pred, f'survivor {d} not in stated pre-domain')
        st.jpin(Lp, d)
        return ('pin', 'enumd') if ok_parse else ('malformed', 'pin/enumd')

    m = re.match(r"^free digits for ([A-Z]): \{([\d,]+)\}: (.+) -> ([A-Z]) = (\d)\.$", body)
    if m:
        Lp, fr, cases, L2, d = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        frd = set(int(x) for x in fr.split(','))
        free_tr = set(range(10)) - set(st.pins.values()) - st.excl[Lp]
        ax.T(frd <= free_tr, f'free set {sorted(frd)} contains digits outside'
             f' tracked free {sorted(free_tr)}')
        ax.T(Lp == L2, 'pinned letter != scanned letter')
        killed = set()
        ok_parse = True
        for case in split_cases(cases):
            cm = re.match(r"^([A-Z]) = (\d) fails \((.+)\)$", case.strip())
            if not cm:
                ok_parse = False
                continue
            killed.add(int(cm.group(2)))
            check_fail_clause(cm.group(3), st, ax, hypo=(Lp, int(cm.group(2))))
        ax.V(killed == frd - {d}, f'killed {sorted(killed)} != free minus survivor')
        ax.V(d in frd, f'survivor {d} not in stated free set')
        st.jpin(Lp, d)
        return ('pin', 'eliml') if ok_parse else ('malformed', 'pin/eliml')

    m = re.match(r"^EQ(\d): (\d+) (" + RULE_ALT + r") (\d+) = (-?\d+)[;,] result digit"
                 r" (\d) is (\d) -> (\d) is taken by ([A-Z])\.$", body)
    if m:
        # copy-form with a CLASH conclusion (model hybrid) -> val kill
        k, A, rw, B, v, j, dj = (int(m.group(1)), int(m.group(2)), m.group(3),
                                 int(m.group(4)), int(m.group(5)), int(m.group(6)),
                                 int(m.group(7)))
        d, Lt = int(m.group(8)), m.group(9)
        tv = rule_eval(rw, A, B)
        ax.A(tv == v, f'{A} {rw} {B} = {tv} not {v}')
        s = str(abs(v))
        if j <= len(s):
            ax.A(int(s[j - 1 if (st.mode or "std") != "le" else len(s) - j]) == dj,
                 f'digit {j} of {v} != {dj}')
        ax.T(st.pins.get(Lt) == d, f'{Lt} pinned {st.pins.get(Lt)} not {d}')
        ax.V(dj == d, 'clash digit != extracted digit')
        return 'kill', 'val'

    m = re.match(r"^EQ(\d): (\d+) (" + RULE_ALT + r") (\d+) = (-?\d+)[;,] result digit"
                 r" (\d) is (\d) -> ([A-Z]) = (\d)\.$", body)
    if m:
        k, A, rw, B, v, j, dj, Lp, d = (int(m.group(1)), int(m.group(2)), m.group(3),
                                        int(m.group(4)), int(m.group(5)), int(m.group(6)),
                                        int(m.group(7)), m.group(8), int(m.group(9)))
        if k - 1 < len(st.eqs):
            L, R = st.eqs[k - 1]
            mag, sgn = strip_sign(R, L[2])
            av = num_of((L[0], L[1]), st.pins, st.sym2let, st.mode or 'std')
            bv = num_of((L[3], L[4]), st.pins, st.sym2let, st.mode or 'std')
            if av is not None:
                ax.T(av == A, f'a from pins = {av} not {A}')
            else:
                ax.T(False, f'a of EQ{k} not fully pinned, value {A} asserted')
            if bv is not None:
                ax.T(bv == B, f'b from pins = {bv} not {B}')
            else:
                ax.T(False, f'b of EQ{k} not fully pinned, value {B} asserted')
            tv = rule_eval(rw, A, B)
            ax.A(tv == v, f'{A} {rw} {B} = {tv} not {v}')
            s = str(abs(v))
            mm0 = st.mode or 'std'
            if j <= len(s):
                tj = int(s[j - 1] if mm0 != 'le' else s[len(s) - j])
                ax.A(tj == dj, f'digit {j} of {v} = {tj} not {dj}')
            else:
                ax.A(False, f'{v} has no digit {j}')
            ax.V(len(s) == len(mag), f'value {v} has {len(s)} digits, result'
                 f' has {len(mag)} -> pin does not follow')
            if j <= len(mag):
                jl = st.sym2let.get(mag[j - 1])
                ax.V(jl == Lp, f'result pos {j} letter is {jl}, line pins {Lp}')
            ax.V(dj == d, 'pinned digit != extracted digit')
        st.jpin(Lp, d)
        if k - 1 < len(st.eqs):
            eop = st.op2let.get(st.eqs[k - 1][0][2])
            if eop:
                st.jrule(eop, rw)
        return 'pin', 'copy'

    m = re.match(r"^EQ(\d) ones: a ends (\d), b ends (\d) -> (.+) ends (\d)"
                 r" -> ([A-Z]) = (\d)\.$", body)
    if m:
        k, a1, b1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        expr, o, Lp, d = m.group(4), int(m.group(5)), m.group(6), int(m.group(7))
        pm = PAIR_RE.match(f"{expr} ends {o}")
        if pm:
            stated, true, _, _ = eval_pair(pm)
            ax.A(stated <= true, f'{expr} recomputes {sorted(true)} not {o}')
        if k - 1 < len(st.eqs):
            L, R = st.eqs[k - 1]
            mag, _ = strip_sign(R, L[2])
            mm0 = st.mode or 'std'
            for side, syms, val in (('a', (L[0], L[1]), a1), ('b', (L[3], L[4]), b1)):
                osym = syms[1] if mm0 != 'le' else syms[0]
                olet = st.sym2let.get(osym)
                ax.T(olet is not None and st.pins.get(olet) == val,
                     f'{side}-ones pin is {st.pins.get(olet) if olet else "?"} not {val}')
            rsym = mag[-1] if mm0 != 'le' else mag[0]
            ax.V(st.sym2let.get(rsym) == Lp, f'result ones letter is'
                 f' {st.sym2let.get(rsym)}, line pins {Lp}')
        ax.V(o == d, 'pinned digit != computed ones digit')
        st.jpin(Lp, d)
        return 'pin', 'ones_res'

    m = re.match(r"^EQ(\d) ones: (a|b) ends (\d), result ends ([A-Z])"
                 r"(?:=(\d)| in \{([\d,]+)\}); the only"
                 r" digit ([A-Z]) with (.+) ending (\d) is (\d) -> ([A-Z]) = (\d)\.$", body)
    if m:
        k, oth_side, oth1 = int(m.group(1)), m.group(2), int(m.group(3))
        Er, e_s, edom = m.group(4), m.group(5), m.group(6)
        Lu, expr, e2, dsol, Lp, d = (m.group(7), m.group(8), int(m.group(9)),
                                     int(m.group(10)), m.group(11), int(m.group(12)))
        if e_s is not None:
            e = int(e_s)
            ax.T(st.pins.get(Er) == e, f'result-ones {Er} pinned {st.pins.get(Er)} not {e}')
            ax.T(e == e2, 'target digit restated differently')
        else:
            e = e2
            ax.T(False, f'result-ones {Er} not pinned (domain {{{edom}}}) ->'
                 f' target {e2} is an arbitrary choice')
            ax.V(False, 'pin from an unpinned result-ones target does not follow')
        # solve expr with unknown = Lu over Lu's domain
        em = re.match(r"^(?:([A-Z])|(\d))\s*([+*-])\s*(?:([A-Z])|(\d))(?: then ([+-]\d))?$",
                      expr.replace(' ', ''))
        if em:
            op = em.group(3)
            off = int(em.group(6)) if em.group(6) else 0
            sols = []
            for cand in sorted(st.dom(Lu) | ({d} if True else set())):
                p = cand if em.group(1) == Lu else (int(em.group(2)) if em.group(2) else oth1)
                q = cand if em.group(4) == Lu else (int(em.group(5)) if em.group(5) else oth1)
                base = p + q if op == '+' else p * q if op == '*' else p - q
                if (base + off) % 10 == e:
                    sols.append(cand)
            ax.A(dsol in sols if sols else False,
                 f'{dsol} does not produce ending {e}')
            ax.V(sols == [dsol], f'solutions over domain = {sols}, line claims only {dsol}')
        ax.V(Lp == Lu and d == dsol, 'pin != solved unknown')
        st.jpin(Lp, d)
        return 'pin', 'ones_op'

    # onesenum body with unique-match pin tail (model hybrid + tens variants)
    m = re.match(r"^(?:[vwxyz] != (?:" + RULE_ALT + r")(?:, (?:" + RULE_ALT
                 + r"))*: )?(EQ\d ones: .+?) -> only (.+?) matches ->"
                 r" (?:(tens|ones) digit )?([A-Z]) = (\d)\.$", body)
    if m:
        inner = m.group(1)
        slot, Lp, d = m.group(3), m.group(4), int(m.group(5))
        ok = check_ones_witness(inner, ax, st, None, pin_concl=(Lp, d))
        st.jpin(Lp, d)
        return ('pin', 'ones_op') if ok else ('malformed', 'pin/ones_op')

    m = re.match(r"^EQ(\d): (a|b) = (\d+), result = (-?\d+); the only (a|b) in"
                 r" 10\.\.99 with (" + RULE_ALT + r") = (-?\d+) is (\d+) ->"
                 r" (tens|ones) digit ([A-Z]) = (\d)\.$", body)
    if m:
        k, known_side, kv, rv = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        unk_side, rw, rv2, xval = m.group(5), m.group(6), int(m.group(7)), int(m.group(8))
        slot, Lp, d = m.group(9), m.group(10), int(m.group(11))
        ax.T(rv == rv2, 'result value restated differently')
        ax.T(known_side != unk_side, 'known and unknown operand are the same side')
        if k - 1 < len(st.eqs):
            L, R = st.eqs[k - 1]
            mag, sgn = strip_sign(R, L[2])
            mm0 = st.mode or 'std'
            ksyms = (L[0], L[1]) if known_side == 'a' else (L[3], L[4])
            tv = num_of(ksyms, st.pins, st.sym2let, mm0)
            ax.T(tv == kv, f'{known_side} from pins = {tv} not {kv}')
            rvv = res_value(mag, st.pins, st.sym2let, mm0)
            if rvv is not None:
                if sgn:
                    rvv = -rvv
                ax.T(rvv == rv, f'result from pins = {rvv} not {rv}')
            sols = []
            for cand in range(10, 100):
                a = kv if known_side == 'a' else cand
                b = cand if known_side == 'a' else kv
                if rule_eval(rw, a, b) == rv:
                    sols.append(cand)
            ax.A((xval in sols) if sols else False, f'{xval} does not satisfy'
                 f' {rw} = {rv}' if xval not in sols else '')
            ax.V(sols == [xval], f'inverse solutions in 10..99 = {sols}')
            if 10 <= xval <= 99:
                want = xval // 10 if slot == 'tens' else xval % 10
                ax.V(want == d, f'{slot} digit of {xval} is {want} not {d}')
            usyms = (L[0], L[1]) if unk_side == 'a' else (L[3], L[4])
            usym = (usyms[0] if slot == 'tens' else usyms[1]) if mm0 != 'le' else \
                   (usyms[1] if slot == 'tens' else usyms[0])
            ax.V(st.sym2let.get(usym) == Lp, f'{slot} letter of {unk_side} is'
                 f' {st.sym2let.get(usym)}, line pins {Lp}')
        st.jpin(Lp, d)
        return 'pin', 'inv'

    m = re.match(r"^every digit but (\d) is taken \((.+)\) -> ([A-Z]) = (\d)\.$", body)
    if m:
        d0, lst, Lp, d = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        pairs = re.findall(r"([A-Z])=(\d)", lst)
        seen = []
        for Lt, dv in pairs:
            dv = int(dv)
            ax.T(st.pins.get(Lt) == dv, f'listed {Lt}={dv} but pinned {st.pins.get(Lt)}')
            seen.append(dv)
        ax.V(sorted(seen) == sorted(set(range(10)) - {d0}),
             f'listed digits {sorted(seen)} != all-but-{d0}')
        ax.V(d0 == d, 'pinned digit != the free digit')
        ax.V(Lp not in [p[0] for p in pairs], 'pinned letter appears in taken list')
        st.jpin(Lp, d)
        return 'pin', 'taken'

    m = re.match(r"^only (\d) and (\d) are free and ([A-Z]) leads a number \(not 0\)"
                 r" -> ([A-Z]) = (\d)\.$", body)
    if m:
        da, db, Ll, Lp, d = (int(m.group(1)), int(m.group(2)), m.group(3),
                             m.group(4), int(m.group(5)))
        free = set(range(10)) - set(st.pins.values())
        ax.T(free == {da, db}, f'free digits are {sorted(free)} not {{{da},{db}}}')
        ax.T(Ll in st.leading, f'{Ll} is not in a leading position')
        ax.V(0 in (da, db), 'neither free digit is 0 -> not-0 reasoning unsupported')
        nz = db if da == 0 else da
        ax.V(Ll == Lp and d == nz, 'conclusion does not follow from premise')
        st.jpin(Lp, d)
        return 'pin', 'taken0'

    m = re.match(r"^all ten digits are used and no other open symbol can take (\d)"
                 r" \((.+)\) -> ([A-Z]) = (\d)\.$", body)
    if m:
        d0, lst, Lp, d = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        ok_v = True
        for seg in lst.split('; '):
            sm2 = re.match(r"^([A-Z]) in \{([\d,]+)\}$", seg.strip())
            if sm2:
                Lt = sm2.group(1)
                dom = set(int(x) for x in sm2.group(2).split(','))
                ax.T(dom <= st.dom(Lt), f'{Lt} domain {sorted(dom)} exceeds'
                     f' tracked {sorted(st.dom(Lt))}')
                if d0 in dom:
                    ok_v = False
        ax.V(ok_v, f'{d0} IS available to a listed open symbol')
        ax.V(d0 == d, 'pinned digit != orphan digit')
        st.jpin(Lp, d)
        return 'pin', 'bijl'

    m = re.match(r"^EQ(\d): ones (.+?) -> carry (\d); tens: (.+) -> ([A-Z]) = (\d)\.$", body)
    if m:
        k, onespart, c, tenspart, Lp, d = (int(m.group(1)), m.group(2), int(m.group(3)),
                                           m.group(4), m.group(5), int(m.group(6)))
        am = re.match(r"^(\d)\+(\d)([+-]\d)? = (\d+)$", onespart)
        if am:
            tot = int(am.group(1)) + int(am.group(2)) + int(am.group(3) or 0)
            ax.A(tot == int(am.group(4)), f'ones sum {tot} != {am.group(4)}')
            ax.A((int(am.group(4)) // 10) == c, f'carry of {am.group(4)} is'
                 f' {int(am.group(4)) // 10} not {c}')
        for tm in re.finditer(r"(\d+)\+(\d+)\+(\d+) = (\d+)", tenspart):
            x = int(tm.group(1)) + int(tm.group(2)) + int(tm.group(3))
            ax.A(x == int(tm.group(4)), f'tens sum {x} != {tm.group(4)}')
        vm = re.search(r"-> result tens digit (\d)$", tenspart)
        if vm:
            ax.V(int(vm.group(1)) == d, 'pinned digit != derived tens digit')
        mm2 = re.match(r"^([A-Z])\+(\d)\+(\d) must end (\d)$", tenspart)
        if mm2:
            want = (int(mm2.group(4)) - int(mm2.group(2)) - int(mm2.group(3))) % 10
            ax.A(want == d, f'solving gives {want} not {d}')
        st.jpin(Lp, d)
        return 'pin', 'tens_op'

    # ---- stalls / guesses / contradictions / terminal moves
    m = re.match(r"^forced deduction stalls: (.+)$", body)
    if m:
        rest = m.group(1)
        fam = 'stall'
        gm = re.search(r"-> assume ([A-Z]) = (\d) \(the smallest candidate\)\.$", rest)
        if 'interchangeable' in rest:
            fam = 'stall-interchangeable'
            if gm:
                Lp, d = gm.group(1), int(gm.group(2))
                ax.T(Lp not in st.pins, f'{Lp} already pinned to {st.pins.get(Lp)}'
                     f' (degenerate re-guess)')
                dm = st.dom(Lp)
                ax.V(d in dm, f'{d} not in live candidates'
                     f' {sorted(dm)} of {Lp}')
                st.push_guess(f'{Lp} = {d}', 'digit', Lp, d)
                st.jpin(Lp, d)
            else:
                ax.T(False, 'stall line truncated / no assumption emitted')
        elif rest.startswith('both digit orders'):
            fam = 'stall-mode'
            ax.V(st.mode is None, f'mode already decided ({st.mode})')
        elif rest.startswith('query-op-ambiguous'):
            fam = 'stall-op'
            qol = st.op2let.get(qL[2])
            ax.V(st.rules.get(qol) is None, f'query op already assigned'
                 f' {st.rules.get(qol)}')
        elif rest.startswith('query-digit-unpinned'):
            fam = 'stall-digit'
            qlets = [st.sym2let.get(s) for s in (qL[0], qL[1], qL[3], qL[4])]
            ax.V(any(l not in st.pins for l in qlets if l), 'all query letters pinned')
        return 'bail', fam

    if re.match(r"^no single case split settles it", body):
        return 'bail', 'nosplit'

    m = re.match(r"^best guess at the (order|rule|digit) level: (.+) -> assume (.+?)"
                 r" \(guess\)\.$", body)
    if m:
        lvl, rat, what = m.groups()
        if lvl == 'order':
            om = re.match(r"^(standard|little-endian) order$", what)
            if om:
                md = 'std' if om.group(1) == 'standard' else 'le'
                ax.V(st.mode is None, 'mode already decided')
                st.push_guess(what, 'order', None, md)
                st.jmode(md)
        elif lvl == 'rule':
            rm2 = re.match(r"^([vwxyz]) = (" + RULE_ALT + r")$", what)
            if rm2:
                ol, rw = rm2.groups()
                dom = st.rdom.get(ol, set())
                ax.T(rw in dom, f'{rw} not in live rule set {sorted(dom)} of {ol}')
                live_sorted = [w for w in PRIOR if w in dom]
                if live_sorted:
                    ax.V(rw == live_sorted[0], f'highest-prior remaining is'
                         f' {live_sorted[0]}, not {rw}')
                st.push_guess(what, 'rule', ol, rw)
                st.jrule(ol, rw)
        else:
            dm2 = re.match(r"^([A-Z]) = (\d)$", what)
            if dm2:
                Lp, d = dm2.group(1), int(dm2.group(2))
                ax.T(Lp not in st.pins, f'{Lp} already pinned ({st.pins.get(Lp)})')
                dom = st.dom(Lp)
                ax.V(d in dom, f'{d} not in live candidates {sorted(dom)}')
                st.push_guess(what, 'digit', Lp, d)
                st.jpin(Lp, d)
        return 'bail', f'guess-{lvl}'

    m = re.match(r"^contradiction: (.+?) -> the (.+?) guess dies; exclude it\.$", body)
    if m:
        inner, gdesc = m.groups()
        fam = 'guess-die'
        # transcription: the dying guess must be the most recent open one
        if st.guesses:
            last = st.guesses[-1]
            ax.T(gdesc.strip() == last[1].strip(),
                 f'dying guess {gdesc!r} != most recent open guess {last[1]!r}')
        else:
            ax.T(False, 'no open guess to die')
        check_contradiction(inner, st, ax)
        mark = st.die_last_guess()
        if mark:
            _, desc, kind, who, val = mark
            if kind == 'digit':
                st.jexcl(who, val)
            elif kind == 'rule':
                st.jrkill(who, [val])
            elif kind == 'order':
                st.jmode('le' if val == 'std' else 'std')
        return 'bail', fam

    m = re.match(r"^contradiction: (.+?) -> backtrack; the ([A-Z]) = (\d) guess dies,"
                 r" so \2 = (\d)\.$", body)
    if m:
        inner, Lp, d, d2 = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        check_contradiction(inner, st, ax)
        if st.guesses:
            last = st.guesses[-1]
            ax.T(last[2] == 'digit' and last[3] == Lp and last[4] == d,
                 f'dying guess != most recent open guess {last[1]!r}')
        mark = st.die_last_guess()
        if mark and mark[2] == 'digit':
            st.jexcl(mark[3], mark[4])
        dom = st.dom(Lp)
        ax.V(d2 in dom, f'{d2} not in live candidates {sorted(dom)}')
        st.push_guess(f'{Lp} = {d2}', 'digit', Lp, d2)
        st.jpin(Lp, d2)
        return 'bail', 'guess-die'

    m = re.match(r"^contradiction: (.+?) -> case ([AB]) dies\.$", body)
    if m:
        check_contradiction(m.group(1), st, ax)
        return 'split', 'case-die'

    m = re.match(r"^contradiction: (.+?) -> the assumed reading of the operator"
                 r" glyphs .+$", body)
    if m:
        check_contradiction(m.group(1), st, ax)
        # regime revision: reset digit state, keep table
        st.pins.clear()
        st.excl.clear()
        st.guesses.clear()
        st.journal.clear()
        if 'little-endian order' in body:
            st.mode = 'le'
        elif 'standard digit order' in body:
            st.mode = 'std'
        return 'bail', 'revise'

    m = re.match(r"^only (.+?) remains -> forced\.$", body)
    if m:
        what = m.group(1)
        om = re.match(r"^(standard|little-endian) order$", what)
        if om:
            md = 'std' if om.group(1) == 'standard' else 'le'
            ax.V(st.mode is None or st.mode == md, 'mode already fixed differently')
            st.jmode(md)
            return 'bail', 'forced-order'
        dm2 = re.match(r"^([A-Z]) = (\d)$", what)
        if dm2:
            Lp, d = dm2.group(1), int(dm2.group(2))
            dom = st.dom(Lp)
            ax.V(dom == {d}, f'domain of {Lp} is {sorted(dom)}, not exactly {{{d}}}')
            st.jpin(Lp, d)
            return 'bail', 'forced-digit'
        rm2 = re.match(r"^([vwxyz]) = (" + RULE_ALT + r")$", what)
        if rm2:
            ol, rw = rm2.groups()
            ax.V(st.rdom.get(ol, set()) == {rw}, f'live rules of {ol} ='
                 f' {sorted(st.rdom.get(ol, set()))}')
            st.jrule(ol, rw)
            return 'bail', 'forced-rule'
        return 'bail', 'forced'

    m = re.match(r"^refinement exhausted -> assemble the most constrained reading:"
                 r" (.+) \(guess\)\.$", body)
    if m:
        parts = m.group(1).split('; ')
        for p in parts:
            rm2 = re.match(r"^([vwxyz]) = (" + RULE_ALT + r") \(highest prior"
                           r" remaining\)$", p)
            if rm2:
                ol, rw = rm2.groups()
                dom = st.rdom.get(ol, set())
                ax.T(rw in dom, f'{rw} not in live set of {ol} {sorted(dom)}')
                live_sorted = [w for w in PRIOR if w in dom]
                if live_sorted:
                    ax.V(rw == live_sorted[0],
                         f'highest prior remaining is {live_sorted[0]} not {rw}')
                st.jrule(ol, rw)
                continue
            dm2 = re.match(r"^([A-Z]) = (\d) \(smallest remaining candidate\)$", p)
            if dm2:
                Lp, d = dm2.group(1), int(dm2.group(2))
                dom = st.dom(Lp)
                ax.V(d in dom, f'{Lp} = {d}: {d} not in live candidates'
                     f' {sorted(dom)} (taken or excluded)')
                st.jpin(Lp, d)
        return 'bail', 'assemble'

    m = re.match(r"^try a case split: (.+)$", body)
    if m:
        st.snapshot = (dict(st.pins), {k: set(v) for k, v in st.excl.items()},
                       st.mode, dict(st.rules),
                       {k: set(v) for k, v in st.rdom.items()})
        return 'split', 'open'

    m = re.match(r"^Case ([AB]) \((.+)\):$", body)
    if m:
        ci, desc = m.groups()
        if ci == 'B' and st.snapshot:
            st.pins, st.excl2, st.mode, st.rules, st.rdom = (
                dict(st.snapshot[0]), None, st.snapshot[2], dict(st.snapshot[3]),
                {k: set(v) for k, v in st.snapshot[4].items()})
            st.excl = defaultdict(set, {k: set(v) for k, v in st.snapshot[1].items()})
        if st.snapshot is None:
            st.snapshot = (dict(st.pins), {k: set(v) for k, v in st.excl.items()},
                           st.mode, dict(st.rules),
                           {k: set(v) for k, v in st.rdom.items()})
        om = re.match(r"^(standard|little-endian) order$", desc)
        if om:
            st.mode = 'std' if om.group(1) == 'standard' else 'le'
        rm2 = re.match(r"^([vwxyz]) = (" + RULE_ALT + r")$", desc)
        if rm2:
            st.rules[rm2.group(1)] = rm2.group(2)
        dm2 = re.match(r"^([A-Z]) = (\d)$", desc)
        if dm2:
            st.pins[dm2.group(1)] = int(dm2.group(2))
        return 'split', 'case-open'

    m = re.match(r"^(?:Case ([AB]) resolved|Resolved): (standard|little-endian) order;"
                 r" ([vwxyz]) = (" + RULE_ALT + r"); (.+)\.$", body)
    if m:
        ci, md, ol, rw, lst = m.groups()
        md = 'std' if md == 'standard' else 'le'
        if st.mode is not None:
            ax.T(st.mode == md, f'order restated as {md}, tracked {st.mode}')
        if st.rules.get(ol):
            ax.T(st.rules[ol] == rw, f'{ol} restated {rw}, tracked {st.rules[ol]}')
        else:
            st.jrule(ol, rw)
        pairs = re.findall(r"([A-Z])=(\d)", lst)
        vals = {}
        for Lt, dv in pairs:
            dv = int(dv)
            if Lt in st.pins:
                ax.T(st.pins.get(Lt) == dv, f'{Lt}={dv} but pinned {st.pins.get(Lt)}')
            else:
                ax.T(False, f'{Lt}={dv} never pinned in the trace')
            vals.setdefault(dv, []).append(Lt)
        ax.V(all(len(v) == 1 for v in vals.values()),
             'two query letters share one digit (collision)')
        qlets = set(st.sym2let.get(s) for s in (qL[0], qL[1], qL[3], qL[4]))
        ax.V(set(p[0] for p in pairs) >= {l for l in qlets if l},
             f'listed letters miss query letters {sorted(l for l in qlets if l)}')
        ax.V(ol == st.op2let.get(qL[2]), 'resolved op is not the query op')
        return ('Resolved', 'case' if ci else '')

    m = re.match(r"^Best guess: (standard|little-endian) order; ([vwxyz]) ="
                 r" (" + RULE_ALT + r"); (.+)\.$", body)
    if m:
        md, ol, rw, lst = m.groups()
        md = 'std' if md == 'standard' else 'le'
        if st.mode is not None:
            ax.T(st.mode == md, f'order restated {md}, tracked {st.mode}')
        if st.rules.get(ol):
            ax.T(st.rules[ol] == rw, f'{ol} restated {rw}, tracked {st.rules[ol]}')
        pairs = re.findall(r"([A-Z])=(\d)", lst)
        vals = {}
        for Lt, dv in pairs:
            dv = int(dv)
            if Lt in st.pins:
                ax.T(st.pins.get(Lt) == dv, f'{Lt}={dv} but pinned/guessed'
                     f' {st.pins.get(Lt)}')
            else:
                st.jpin(Lt, dv)
            vals.setdefault(dv, []).append(Lt)
        ax.V(all(len(v) == 1 for v in vals.values()), 'digit collision in summary')
        ax.V(ol == st.op2let.get(qL[2]), 'summarized op is not the query op')
        return 'bail', 'best-guess-summary'

    if re.match(r"^Best guess: the operand symbols juxtaposed -> (\S+)\.$", body):
        r = re.match(r"^Best guess: the operand symbols juxtaposed -> (\S+)\.$", body).group(1)
        ax.V(r in (qL[0] + qL[1] + qL[3] + qL[4], qL[3] + qL[4] + qL[0] + qL[1]),
             'juxtaposition string wrong')
        return 'bail', 'juxtapose'

    m = re.match(r"^Only case ([AB]) survives and it is fully forced\.$", body)
    if m:
        return 'split', 'survivor'
    m = re.match(r"^Both cases give the same result (\S+) -> ", body)
    if m:
        return 'split', 'agree'

    # ---- final block
    m = re.match(r"^(?:Query operands|Read the query operands little-endian \(reverse"
                 r" the digits\)): (\S+) -> (\d+), (\S+) -> (\d+)\.$", body)
    if m:
        a_s, a_v, b_s, b_v = m.group(1), m.group(2), m.group(3), m.group(4)
        le = body.startswith('Read the query operands little-endian')
        ax.T(a_s == qL[0] + qL[1], f'operand {a_s!r} != query a {qL[0]+qL[1]!r}')
        ax.T(b_s == qL[3] + qL[4], f'operand {b_s!r} != query b {qL[3]+qL[4]!r}')
        if st.mode is not None:
            ax.T(le == (st.mode == 'le'), 'endianness of read != decided mode')
        exp_a = num_of((qL[0], qL[1]), st.pins, st.sym2let, 'le' if le else 'std')
        exp_b = num_of((qL[3], qL[4]), st.pins, st.sym2let, 'le' if le else 'std')
        ax.V(exp_a is not None and int(a_v) == exp_a,
             f'a should read {exp_a}, line says {a_v}')
        ax.V(exp_b is not None and int(b_v) == exp_b,
             f'b should read {exp_b}, line says {b_v}')
        st.qvals = (int(a_v), int(b_v))
        return 'query-compute', 'operands'

    m = re.match(r"^Compute: (-?\d+) (" + RULE_ALT + r") (-?\d+) = (-?\d+)\.$", body)
    if m:
        A, rw, B, v = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        qv = getattr(st, 'qvals', None)
        if qv:
            ax.T((A, B) == qv, f'operands {(A, B)} != read values {qv}')
        qol = st.op2let.get(qL[2])
        if st.rules.get(qol):
            ax.T(rw == st.rules[qol], f'rule {rw} != decided {st.rules[qol]}')
        tv = rule_eval(rw, A, B)
        ax.A(tv == v, f'{A} {rw} {B} = {tv} not {v}')
        st.cval = v
        return 'query-compute', 'compute'

    m = re.match(r"^Encode digit by digit: (.+?) -> (\S+?)"
                 r"(?:; negative -> sign glyph (\S) in front: (.+?))?"
                 r"(?:; little-endian -> reverse to (.+?))?"
                 r"(?:; negative -> sign glyph (\S) in front: (.+?))?\.$", body)
    if m:
        maps, first, sg, signed, rev, sg2, signed2 = m.groups()
        if sg2:
            sg, signed = sg2, signed2
        cv = getattr(st, 'cval', None)
        pairs = re.findall(r"(\d)=(\S)", maps)
        if cv is not None:
            ax.T(''.join(p[0] for p in pairs) == str(abs(cv)),
                 f'digit list != computed value {cv}')
        inv = {d: s for s, l in st.sym2let.items() for ltr, d2 in st.pins.items()
               if ltr == l for d in [d2]}
        for d_s, sym in pairs:
            d = int(d_s)
            exp = inv.get(d)
            ax.T(exp == sym, f'digit {d} encodes as {exp!r} per pins, line wrote {sym!r}')
        ax.V(first == ''.join(p[1] for p in pairs), 'encoded string != listed symbols')
        if rev:
            ax.V(rev == first[::-1], 'reversal wrong')
        if signed:
            base = rev if (rev and sg2) else first
            ax.V(signed == sg + base, 'signed string != glyph + magnitude')
            if cv is not None:
                ax.A(cv < 0, f'value {cv} is not negative')
        st.encoded = signed or rev or first
        return 'encode', ''

    m = re.match(r"^\\boxed\{(.*)\}$", body)
    if m:
        bx = m.group(1)
        enc = getattr(st, 'encoded', None)
        if getattr(st, 'boxed1', None) is not None:
            ax.T(bx == st.boxed1, f'final boxed {bx!r} != in-think boxed {st.boxed1!r}')
        elif enc is not None:
            ax.T(bx == enc, f'boxed {bx!r} != encoded {enc!r}')
        st.boxed1 = bx if getattr(st, 'boxed1', None) is None else st.boxed1
        return 'boxed', ''

    m = re.match(r"^contradiction: (.+?)\.? -> (.+)$", body)
    if m:
        check_contradiction(m.group(1), st, ax)
        fb = re.search(r"fall back to the default rule set for ([vwxyz]): \{(.+?)\}",
                       m.group(2))
        if fb:
            st.jrkill(fb.group(1), [])
            st.rdom[fb.group(1)] = set(re.findall(RULE_ALT, fb.group(2)))
        return 'bail', 'contra-other'

    return 'malformed', guess_nearest(body)


def check_contradiction(inner, st, ax):
    inner = inner.rstrip('.')
    m = re.match(r"^every rule for ([vwxyz]) is eliminated$", inner)
    if m:
        ol = m.group(1)
        dom = st.rdom.get(ol, set())
        ax.V(not dom, f'live rules remain for {ol}: {sorted(dom)}')
        return
    m = re.match(r"^([A-Z]) = (\d) and ([A-Z]) = (\d) collide$", inner)
    if m:
        L1, d1, L2, d2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        ax.T(st.pins.get(L1) == d1 and st.pins.get(L2) == d2,
             f'pins are {L1}={st.pins.get(L1)} {L2}={st.pins.get(L2)}')
        ax.V(d1 == d2 and L1 != L2, 'stated digits do not collide')
        return
    m = re.match(r"^([A-Z]) collides: \1 = (\d) and \1 = (\d)$", inner)
    if m:
        ax.V(m.group(2) != m.group(3), 'same digit twice is not a collision')
        return
    m = re.match(r"^([A-Z]) collides with ([A-Z])$", inner)
    if m:
        L1, L2 = m.groups()
        ax.V(L1 in st.pins and st.pins.get(L1) == st.pins.get(L2),
             f'{L1}={st.pins.get(L1)} {L2}={st.pins.get(L2)} do not collide')
        return
    m = re.match(r"^([A-Z])'s only remaining digit (\d) is taken by ([A-Z])$", inner)
    if m:
        L1, d, L2 = m.group(1), int(m.group(2)), m.group(3)
        ax.T(st.pins.get(L2) == d, f'{L2} pinned {st.pins.get(L2)} not {d}')
        return
    m = re.match(r"^no digit fits ([A-Z]) on EQ(\d): (.+)$", inner)
    if m:
        check_fail_clause(m.group(3), st, ax)
        return


def check_kill_body(kb, rws, ol, st, ax):
    """classify + check a kill body; returns family name."""
    # 2mode
    if kb.startswith('standard: '):
        m = re.match(r"^standard: (.+?); little-endian: (EQ\d .+)$", kb)
        if m:
            ok1 = check_kill_clause(m.group(1), rws, st, ax, 'std')
            ok2 = check_kill_clause(m.group(2), rws, st, ax, 'le')
            return '2mode' if (ok1 and ok2) else '2mode'
        return '2mode'
    check_kill_clause(kb, rws, st, ax, None)
    return kill_family(kb)


def kill_family(kb):
    if 'sign glyph' in kb and ('never go' in kb or 'always write' in kb):
        return 'sign'
    if re.match(r"^EQ\d result (?:has|is negative with) \d+ digit", kb):
        return 'range'
    if 'result has exactly' in kb:
        return 'ident'
    if 'readings:' in kb:
        return 'enum'
    if re.match(r"^EQ\d ones:", kb):
        return 'onesenum' if ' in {' in kb else 'ones'
    if re.match(r"^EQ\d: a = ", kb):
        return 'val'
    if 'carry' in kb:
        return 'tens'
    return 'other'


def check_kill_clause(kb, rws, st, ax, mode):
    fam = kill_family(kb)
    if fam == 'sign':
        m = re.match(r"^EQ(\d) result carries the sign glyph but (.+) never"
                     r" goes? negative$", kb)
        if m:
            k = int(m.group(1))
            if k - 1 < len(st.eqs):
                L, R = st.eqs[k - 1]
                mag, sgn = strip_sign(R, L[2])
                ax.T(sgn is not None, f'EQ{k} result has NO sign glyph')
            ax.V(all(w in NONNEG for w in rws),
                 f'a listed rule CAN go negative: {[w for w in rws if w not in NONNEG]}')
            return True
        m = re.match(r"^EQ(\d) result (?:has|carries) no sign glyph but (.+) always"
                     r" writes? one$", kb)
        if m:
            k = int(m.group(1))
            if k - 1 < len(st.eqs):
                L, R = st.eqs[k - 1]
                mag, sgn = strip_sign(R, L[2])
                ax.T(sgn is None, f'EQ{k} result DOES carry a sign glyph')
            ax.V(all(w in ALWAYSNEG for w in rws),
                 f'a listed rule does not always write a sign: '
                 f'{[w for w in rws if w not in ALWAYSNEG]}')
            return True
        return False
    if fam == 'range':
        m = re.match(r"^EQ(\d) result (has|is negative with) (\d+) digit\(s\) \(value"
                     r" (-?\d+)\.\.(-?\d+)\) but with a (?:in (\d+)\.\.(\d+)|= (\d+)),"
                     r" b (?:in (\d+)\.\.(\d+)|= (\d+)): (.+)$", kb)
        if not m:
            return False
        k, neg, rl = int(m.group(1)), m.group(2) == 'is negative with', int(m.group(3))
        lo, hi = int(m.group(4)), int(m.group(5))
        A = (int(m.group(8)), int(m.group(8))) if m.group(8) else (int(m.group(6)), int(m.group(7)))
        B = (int(m.group(11)), int(m.group(11))) if m.group(11) else (int(m.group(9)), int(m.group(10)))
        bounds = m.group(12)
        if k - 1 < len(st.eqs):
            L, R = st.eqs[k - 1]
            mag, sgn = strip_sign(R, L[2])
            ax.T(len(mag) == rl, f'EQ{k} result has {len(mag)} digits not {rl}')
            ax.T((sgn is not None) == neg, 'sign flag vs prompt RHS wrong')
            for side, iv, syms in (('a', A, (L[0], L[1])), ('b', B, (L[3], L[4]))):
                exp = derived_interval(syms, st)
                # stated interval must CONTAIN the pin-derived one (looser-but-
                # true interval = deviation, not a false copy)
                ax.T(iv[0] <= exp[0] and exp[1] <= iv[1],
                     f'{side} interval {iv} excludes pin-derived {exp}')
        tlo = 0 if rl == 1 else 10 ** (rl - 1)
        thi = 10 ** rl - 1
        ax.A((lo, hi) == ((-thi, -tlo) if neg else (tlo, thi)),
             f'target interval {lo}..{hi} != {rl}-digit range')
        seen = {}
        for bm in re.finditer(r"(" + RULE_ALT + r") (<=|>=) (-?\d+)", bounds):
            rw, dr, bd = bm.group(1), bm.group(2), int(bm.group(3))
            seen[rw] = (dr, bd)
            vals = [rule_eval(rw, a, b) for a in range(A[0], A[1] + 1)
                    for b in range(B[0], B[1] + 1)]
            vals = [v for v in vals if v is not None]
            if vals:
                # SOUNDNESS: the stated bound must be a true bound (renderer
                # prints sound interval bounds, not exact extremes)
                sound = (bd >= max(vals)) if dr == '<=' else (bd <= min(vals))
                ax.A(sound, f'{rw} {dr} {bd} is FALSE (true extreme'
                     f' {max(vals) if dr == "<=" else min(vals)})')
            kill = (bd < lo) if dr == '<=' else (bd > hi)
            ax.V(kill, f'{rw} {dr} {bd} does not miss {lo}..{hi}')
        ax.V(all(w in seen for w in rws),
             f'no bound shown for {[w for w in rws if w not in seen]}')
        return True
    if fam in ('ones', 'onesenum'):
        m = re.match(r"^(EQ\d ones: .+?) -> none matches$", kb)
        if m:
            return check_ones_witness(m.group(1), ax, st, mode, pin_concl='kill')
        # single form without trailing none-matches: `...; X+Y ends o not e`
        m = re.match(r"^EQ(\d) ones: (.+); (.+) ends (\d) not (\d)$", kb)
        if m:
            hd = parse_ones_header(f"EQ{m.group(1)} ones: {m.group(2)}")
            pm = PAIR_RE.match(f"{m.group(3)} ends {m.group(4)}")
            if pm:
                stated, true, _, _ = eval_pair(pm)
                ax.A(stated <= true, f'{m.group(3)} recomputes {sorted(true)}')
            ax.V(m.group(4) != m.group(5), 'stated digits equal -> no kill')
            return True
        return False
    if fam == 'ident':
        m = re.match(r"^EQ(\d) result has exactly (a|b)'s symbols -> needs"
                     r" (.+) = \2, impossible for two 2-digit numbers$", kb)
        if not m:
            return False
        k, side = int(m.group(1)), m.group(2)
        if k - 1 < len(st.eqs):
            L, R = st.eqs[k - 1]
            mag, _ = strip_sign(R, L[2])
            opnd = L[0] + L[1] if side == 'a' else L[3] + L[4]
            ax.T(mag == opnd, f'result {mag!r} != {side} symbols {opnd!r}')
        for rw in rws:
            poss = any(rule_eval(rw, a, b) == (a if side == 'a' else b)
                       for a in range(10, 100) for b in range(10, 100))
            ax.V(not poss, f'{rw} CAN equal {side} for 2-digit operands')
        return True
    if fam == 'enum':
        m = re.match(r"^EQ(\d) readings: (.+)$", kb)
        if not m:
            return False
        k = int(m.group(1))
        for seg in split_cases(m.group(2)):
            ce = parse_case_eval(seg.strip())
            if ce:
                A2, B2, rw, v, rest = ce
                if rw:
                    tv = rule_eval(rw, A2, B2)
                    ax.A(tv == v, f'{A2} {rw} {B2} = {tv} not {v}')
                if rest and rest != 'undefined':
                    check_clash(rest, v, k, st, ax, mode)
        return True
    if fam == 'val':
        m = re.match(r"^EQ(\d): a = (\d+), b = (\d+); (" + RULE_ALT + r") ="
                     r" (-?\d+)(?:, (.+))?$", kb)
        if not m:
            m2 = re.match(r"^EQ(\d): a = (\d+), b = (\d+); (" + RULE_ALT + r") is"
                          r" undefined", kb)
            if m2:
                k, A2, B2, rw = (int(m2.group(1)), int(m2.group(2)), int(m2.group(3)),
                                 m2.group(4))
                ax.A(rule_eval(rw, A2, B2) is None, f'{rw}({A2},{B2}) IS defined')
                return True
            return False
        k, A2, B2, rw, v = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            m.group(4), int(m.group(5)))
        if k - 1 < len(st.eqs):
            L, R = st.eqs[k - 1]
            av = num_of((L[0], L[1]), st.pins, st.sym2let, mode or st.mode or 'std')
            bv = num_of((L[3], L[4]), st.pins, st.sym2let, mode or st.mode or 'std')
            if av is not None:
                ax.T(av == A2, f'a from pins {av} != {A2}')
            if bv is not None:
                ax.T(bv == B2, f'b from pins {bv} != {B2}')
        tv = rule_eval(rw, A2, B2)
        ax.A(tv == v, f'{A2} {rw} {B2} = {tv} not {v}')
        if m.group(6):
            check_clash(m.group(6), v, k, st, ax, mode)
        return True
    if fam == 'tens':
        for am in re.finditer(r"(\d+)\+(\d+)(?:\+(\d+))? = (\d+)", kb):
            x = int(am.group(1)) + int(am.group(2)) + int(am.group(3) or 0)
            ax.A(x == int(am.group(4)), f'{am.group(0)} recomputes to {x}')
        em = re.search(r"ends (\d+) not (\d)$", kb)
        if em:
            ax.V(em.group(1)[-1] != em.group(2), 'stated tens digits equal -> no kill')
        return True
    return False


def derived_interval(syms, st):
    """pin-derived operand interval as the renderer states it."""
    mm = st.mode or 'std'
    t_sym, o_sym = (syms[0], syms[1]) if mm != 'le' else (syms[1], syms[0])
    tl, ol = st.sym2let.get(t_sym), st.sym2let.get(o_sym)
    td, od = st.pins.get(tl), st.pins.get(ol)
    if td is not None and od is not None:
        v = td * 10 + od
        return (v, v)
    if td is not None:
        return (td * 10, td * 10 + 9)
    return (10, 99)


def guess_nearest(body):
    if re.match(r"^[vwxyz] !=", body) or '-> drop' in body:
        return 'kill'
    if re.search(r"-> [A-Z] = \d\.?$", body) or body.startswith('free digits') \
            or re.match(r"^only \d and \d are free", body):
        return 'pin'
    if 'guess' in body or body.startswith('contradiction') \
            or 'backtrack' in body or 'stalls' in body:
        return 'bail'
    if 'chars' in body or re.match(r"^Query \S+ ->", body):
        return 'table-scan'
    if re.match(r"^Query: ", body) or 'RHS' in body or 'juxtaposition' in body:
        return 'routing'
    if body.startswith('Encode'):
        return 'encode'
    if body.startswith('\\boxed'):
        return 'boxed'
    if re.match(r"^(\S|.+?) -> [A-Za-z]$", body):
        return 'union'
    if ' ones:' in body:
        return 'kill'
    return 'other'


# ------------------------------------------------------------ driver
def main():
    files = sorted(glob.glob(os.path.join(TDIR, '*.txt')))
    all_recs = []
    firsts = []          # (tid, type, fam, axis, ln, tokens_before)
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(ROOT, 'competition_dataset',
                                               'tokenizer.json'))
    except Exception:
        tok = None
    for path in files:
        try:
            tid, gold, ntok, recs, out = audit_one(path)
        except Exception as e:
            print(f'AUDIT-CRASH {path}: {type(e).__name__} {e}', file=sys.stderr)
            continue
        all_recs.extend(recs)
        first = None
        for r in recs:
            fails = [ax for ax, v in (('transcription', r.t), ('arithmetic', r.a),
                                      ('verdict', r.v)) if v == 0]
            if fails:
                first = (r, fails[0])
                break
        if first:
            r, axis = first
            lines = out.splitlines()
            prefix = '\n'.join(lines[:r.ln - 1])
            ntb = len(tok.encode(prefix).ids) if tok else len(prefix) // 4
            firsts.append((tid, r.typ, r.fam, axis, r.ln, ntb, r.text, r.note))
        else:
            total = len(tok.encode(out).ids) if tok else len(out) // 4
            firsts.append((tid, None, None, None, None, total, '', ''))

    # ---- CSV
    with open(os.path.join(HERE, 'line_audit.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'line_no', 'counter', 'type', 'family',
                    'transcription', 'arithmetic', 'verdict', 'note', 'text'])
        for r in all_recs:
            w.writerow([r.tid, r.ln, r.ctr, r.typ, r.fam, r.t, r.a, r.v,
                        r.note, r.text])

    # ---- aggregate
    def key(r):
        return f'{r.typ}/{r.fam}' if r.fam else r.typ

    agg = defaultdict(lambda: [0, [0, 0], [0, 0], [0, 0]])
    for r in all_recs:
        a = agg[key(r)]
        a[0] += 1
        for i, v in ((1, r.t), (2, r.a), (3, r.v)):
            if v != '':
                a[i][1] += 1
                a[i][0] += v
    rows = []
    for k in sorted(agg, key=lambda k: -agg[k][0]):
        n, T, A, V = agg[k]
        def pct(x):
            return f'{100*x[0]/x[1]:.1f}' if x[1] else '-'
        rows.append((k, n, pct(T), f'{T[1]}', pct(A), f'{A[1]}', pct(V), f'{V[1]}'))

    print(f'{"line type":34s} {"n":>5s} {"T_ok%":>7s} {"(nT)":>6s} {"A_ok%":>7s}'
          f' {"(nA)":>6s} {"V_ok%":>7s} {"(nV)":>6s}')
    for k, n, t, nt, a, na, v, nv in rows:
        print(f'{k:34s} {n:5d} {t:>7s} {nt:>6s} {a:>7s} {na:>6s} {v:>7s} {nv:>6s}')

    print('\nFIRST-FALSE-LINE distribution:')
    cnt = Counter((f'{t}/{fm}' if fm else t, ax) for _, t, fm, ax, _, _, _, _ in firsts
                  if t is not None)
    for (k, ax), c in cnt.most_common():
        print(f'  {c:3d}  {k}  [{ax}]')
    clean = [f for f in firsts if f[1] is None]
    print(f'  {len(clean):3d}  <no false line>')
    toks = sorted(f[5] for f in firsts if f[1] is not None)
    if toks:
        print(f'\ntokens before first false line: median {statistics.median(toks)},'
              f' p25 {toks[len(toks)//4]}, p75 {toks[3*len(toks)//4]}, '
              f'min {toks[0]}, max {toks[-1]}  (n={len(toks)})')

    # dump first-false details for report
    with open(os.path.join(HERE, 'first_false.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'type', 'family', 'axis', 'line_no', 'tokens_before',
                    'text', 'note'])
        for row in firsts:
            w.writerow(row)
    print(f'\n{len(all_recs)} line records -> line_audit.csv; first_false.csv written')


if __name__ == '__main__':
    main()
