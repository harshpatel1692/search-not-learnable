"""cryptarithm CoT — synthetic puzzle generator + tiered trace renderer (v15 workstream).

Generator: samples fresh puzzles per the measured train distributions
(ops/mode/#examples/#glyphs/sign-end; pipeline/data/cryptarithm_gold_meta.jsonl).
Renderer: emits a self-consistent deduction trace driven by the solver's config:
  Tier1 concat    : Ali-grammar (relabel -> transcribe -> classify -> apply), ASCII-only.
  Tier2 add/sub   : cue classification, domain propagation + branch try/reject,
                    lock map, verify on EVERY example, decode query, re-encode.
  Tier3 mul/exotic: same engine; result-length pruning enters via the op-candidate cues.
Every trace is linted: single \\boxed, boxed == solver answer == in-trace computation,
the locked rules are the rules verified and applied, pure ASCII.

r2 trace contract (defect fixes over v15):
  * verify lines DECODE the RHS through the locked map first, then compute, then match
    (an echoed "expected" can never pass; lint re-derives from the solver mapping).
  * deduce LOCK lines render the pinning constraint with numbers (compute / units
    congruence / leading-1 / explicit per-digit try-reject), never "is only satisfiable".
  * "Distinct check:" line after the locked map; injectivity asserted by renderer + lint.
  * final decode is digit-by-digit DIRECTLY to symbols (no letter hop).

r3 trace contract (truthful conditionals; fixes "match"/"ok" learned as constants):
  * WRONG-BRANCH EPISODES: at rule-pin time, candidates still narratively live (never
    explicitly eliminated in a rendered line) are tried most-common-first; genuinely
    wrong ones FAIL with real numbers ("105 vs 2745. no -> mistake, backtrack") before
    the true rule locks with a real "match". Fully-forced traces get ONE injected
    double-check of the most common rule (a+b / a*b) with real numbers. Tier-1 concat
    traces refute the wrong concat order with the real symbol strings.
  * DISTINCT-COLLISION EPISODES (subset of traces): genuine alldiff prunes ("only digit
    left after F = 9") and units-congruence "taken" exclusions render as
    "try G = 9: distinct check: F=9 and G=9 collide. conflict -> backtrack -> G = 2."
  * LINT IS TRUTHFULNESS-BASED: every "X vs Y. no|match" comparator is re-parsed and the
    terminal token must be CORRECT for the two values on the line; collide lines must
    show equal digits on distinct letters consistent with the final map; pin-try lines
    are re-derived (arithmetic + RHS decode through the solver mapping); every trace
    must contain at least one truthful "no". Final verify block stays all-"match"
    (the locked solution is correct by construction). All r2 checks kept.

r4 (policy-driven; see analysis/reports/cryptarithm_r4.md):
  * the renderer EXECUTES a deterministic truth-free policy (PolicyP2 DFS +
    tiered std/rev x plain/offset passes); gold only gates the final boxed answer.
  * tried-set bookkeeping on every retry, next-choice policy lines, genuine
    exhaustion + level-up diagnosis episodes; distinct-ops generator constraint.
  * lint_r4 = r3 lint + duplicate-try-line / tried-set growth / refuted-then-
    locked / meta-free arithmetic re-derivation. r3 paths left intact.

Usage:
  python3 pipeline/synth/cryptarithm_cot.py render-real              # r3 -> data/crypt_r3/crypt_deduce_real.jsonl
  python3 pipeline/synth/cryptarithm_cot.py render-synth 3000 15 OUT # r3 -> data/crypt_r3/crypt_deduce_synth.jsonl
  python3 pipeline/synth/cryptarithm_cot.py render-real4             # r4 -> data/crypt_r4/crypt_deduce_real.jsonl
  python3 pipeline/synth/cryptarithm_cot.py render-synth4 750 151 OUT
  python3 pipeline/synth/cryptarithm_cot.py coverage 8 [out.jsonl]   # r4 pillar-2 gate measurement
  python3 pipeline/synth/cryptarithm_cot.py gen-test
"""
import csv, json, math, os, random, sys, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
from solvers import cryptarithm2 as C2

csv.field_size_limit(10 ** 8)

# ---------------------------------------------------------------- measured distributions
DIGIT_POOL = list("!\"#$%&'()/:<>?@[\\]^`{|}")          # 23 chars seen as digit symbols
OP_EXTRA = list("*+-")                                   # ops also draw from these
N_EX_W = {3: 236, 4: 301, 5: 286}
N_GLYPH_W = {1: 13, 2: 302, 3: 508}
P_REV = 305 / 798
P_SUF_GIVEN_REV_SIGN = 27 / 198
HDR = ("In Alice's Wonderland, a secret set of transformation rules is applied to "
       "equations. Below are a few examples:")
QSTR = "Now, determine the result for: "

LETTERS = "ABCDEFGHIJ"
OPLETTERS = "xyzw"

ADD_FAM = {'add', 'add_p1', 'add_m1', 'add_p2', 'add_m2'}
MUL_FAM = {'mul', 'mul_p1', 'mul_m1', 'mul_p2', 'mul_m2'}
SUB_FAM = {'sub_signed', 'rsub_signed', 'neg_absdiff', 'absdiff', 'absdiff_p1',
           'absdiff_m1', 'absdiff_p2', 'absdiff_m2'}
SMALL_FAM = {'mod', 'rmod', 'gcd', 'fdiv', 'rdiv'}
# render-vocabulary: ops the trace actually searches (99%+ of measured glyph draws).
RVOCAB = ['add', 'sub_signed', 'mul', 'rsub_signed', 'absdiff', 'neg_absdiff',
          'add_p1', 'add_m1', 'mul_p1', 'mul_m1', 'add_p2', 'add_m2',
          'mul_p2', 'mul_m2', 'absdiff_p1', 'absdiff_m1', 'absdiff_p2', 'absdiff_m2',
          'mod', 'rmod']
# r4 tier-A vocabulary: the plain (offset-free) rules, prior mass ~57% per glyph
# draw. The policy searches these FIRST so that a high-prior plain config is
# found before any offset config can hijack the map (mirrors the vote's prior).
PLAIN_VOCAB = ['add', 'sub_signed', 'mul', 'rsub_signed', 'absdiff', 'neg_absdiff']

OP_WORD = {
    'add': 'a+b', 'add_p1': 'a+b+1', 'add_m1': 'a+b-1', 'add_p2': 'a+b+2', 'add_m2': 'a+b-2',
    'mul': 'a*b', 'mul_p1': 'a*b+1', 'mul_m1': 'a*b-1', 'mul_p2': 'a*b+2', 'mul_m2': 'a*b-2',
    'sub_signed': 'a-b', 'rsub_signed': 'b-a',
    'absdiff': '|a-b|', 'absdiff_p1': '|a-b|+1', 'absdiff_m1': '|a-b|-1',
    'absdiff_p2': '|a-b|+2', 'absdiff_m2': '|a-b|-2', 'neg_absdiff': '-|a-b|',
    'a2_plus_b': 'a*a+b', 'a_plus_b2': 'a+b*b',
    'mod': 'a mod b', 'rmod': 'b mod a', 'gcd': 'gcd(a,b)', 'lcm': 'lcm(a,b)',
    'fdiv': 'a div b', 'rdiv': 'b div a',
    'concat_fwd': 'concatenation', 'concat_rev': 'reverse concatenation',
}

def w_choice(rng, wd):
    ks = list(wd); ws = [wd[k] for k in ks]
    return rng.choices(ks, weights=ws, k=1)[0]

# ---------------------------------------------------------------- r3 episode knobs
PIN_WRONG_CAP = 3      # max wrong-branch (failed-try) episodes per trace
P_DIST_EP = 0.38       # per-opportunity gate for distinct-collision episodes (cap 1/trace)

def _rhs_parts(r, M, let, rev):
    """(Rr, opch, side, sign, dd, exp_str) for a value row under locked letter map M:
    raw RHS string, op glyph, sign side, sign flag, decoded digits (raw order), and the
    signed expected value string (msb order)."""
    Lr, Rr = r.raw
    opch = Lr[2]
    sign = False; side = None; mag = Rr
    if len(Rr) > 1 and Rr[0] == opch:
        sign = True; side = 'leading'; mag = Rr[1:]
    elif len(Rr) > 1 and Rr[-1] == opch:
        sign = True; side = 'trailing'; mag = Rr[:-1]
    dd = [M[let[c]] for c in mag]
    msb = dd[::-1] if rev else dd
    exp_str = ('-' if sign else '') + str(int(''.join(map(str, msb))))
    return Rr, opch, side, sign, dd, exp_str

def _got_str(op, a, b):
    """rendered result string of op(a,b) (verify-block convention), or None."""
    v = C2.OPS[op](a, b)
    if v is None:
        return None
    return ('-' if (v < 0 or op in C2.NEGPRE) else '') + str(abs(v))

# ---------------------------------------------------------------- synthetic generator
def render_eq(a, b, op, smap, opch, rev, sign_end):
    """values a,b -> (lhs5, rhs) strings under cipher smap{digit->sym}, mode, sign end."""
    v = C2.OPS[op](a, b)
    if v is None:
        return None
    sign = False
    if op in C2.SIGNED:
        if v < 0:
            sign = True; v = -v
    elif op in C2.NEGPRE:
        sign = True; v = abs(v)
    elif v < 0:
        return None
    da = [a // 10, a % 10]; db = [b // 10, b % 10]
    dv = []
    x = v
    if x == 0:
        dv = [0]
    while x:
        dv.append(x % 10); x //= 10
    dv = dv[::-1]
    if rev:
        da = da[::-1]; db = db[::-1]; dv = dv[::-1]
    lhs = smap[da[0]] + smap[da[1]] + opch + smap[db[0]] + smap[db[1]]
    rhs = ''.join(smap[d] for d in dv)
    if sign:
        rhs = (rhs + opch) if sign_end == 'suf' else (opch + rhs)
    return lhs, rhs

def gen_puzzle(rng, force_guess=False):
    """sample one synthetic puzzle. returns dict prompt/answer/config or None (resample)."""
    rev = rng.random() < P_REV
    sign_end = 'suf' if (rev and rng.random() < P_SUF_GIVEN_REV_SIGN) else 'pre'
    nex = w_choice(rng, N_EX_W)
    ng = min(w_choice(rng, N_GLYPH_W), nex)
    digs = rng.sample(DIGIT_POOL, 10)
    smap = {d: digs[d] for d in range(10)}
    pool = [c for c in DIGIT_POOL + OP_EXTRA if c not in digs]
    glyphs = rng.sample(pool, ng + (1 if force_guess else 0))
    opw = {o: c for o, c in C2.MEASURED_OP_COUNTS.items() if c >= 2}
    gops = {}
    pool_ops = dict(opw)
    for gl in glyphs:
        op = w_choice(rng, pool_ops)
        gops[gl] = op
        pool_ops.pop(op)   # r4: measured — glyph ops are drawn WITHOUT replacement
    qglyph = glyphs[-1] if force_guess else rng.choice(glyphs)
    exglyphs = glyphs[:-1] if force_guess else glyphs
    asg = list(exglyphs) + [rng.choice(exglyphs) for _ in range(nex - len(exglyphs))]
    rng.shuffle(asg)
    lines = []
    for gl in asg:
        op = gops[gl]
        for _ in range(40):
            a = rng.randint(10, 99); b = rng.randint(10, 99)
            if op in ('concat_fwd', 'concat_rev'):
                da = [a // 10, a % 10]; db = [b // 10, b % 10]
                if rev:
                    da = da[::-1]; db = db[::-1]
                lhs = smap[da[0]] + smap[da[1]] + gl + smap[db[0]] + smap[db[1]]
                rhs = (lhs[0] + lhs[1] + lhs[3] + lhs[4]) if op == 'concat_fwd' else \
                      (lhs[3] + lhs[4] + lhs[0] + lhs[1])
                lines.append((lhs, rhs)); break
            r = render_eq(a, b, op, smap, gl, rev, sign_end)
            if r is not None:
                lines.append(r); break
        else:
            return None
    qop = gops[qglyph]
    for _ in range(60):
        a = rng.randint(10, 99); b = rng.randint(10, 99)
        if qop in ('concat_fwd', 'concat_rev'):
            da = [a // 10, a % 10]; db = [b // 10, b % 10]
            if rev:
                da = da[::-1]; db = db[::-1]
            qL = smap[da[0]] + smap[da[1]] + qglyph + smap[db[0]] + smap[db[1]]
            ans = (qL[0] + qL[1] + qL[3] + qL[4]) if qop == 'concat_fwd' else \
                  (qL[3] + qL[4] + qL[0] + qL[1])
            break
        r = render_eq(a, b, qop, smap, qglyph, rev, sign_end)
        if r is not None:
            qL, ans = r
            break
    else:
        return None
    prompt = HDR + "\n" + "\n".join(f"{l} = {r}" for l, r in lines) + "\n" + QSTR + qL
    return {'prompt': prompt, 'answer': ans, 'rev': rev, 'ops': dict(gops),
            'map': dict(smap), 'sign_end': sign_end, 'qglyph': qglyph, 'qop': qop,
            'category': 'cryptarithm_guess' if force_guess else 'cryptarithm_deduce'}

# ---------------------------------------------------------------- letter-space model
class Row:
    __slots__ = ('gl', 'a0', 'a1', 'b0', 'b1', 'res', 'sign', 'rl', 'raw', 'conc', 'k')
    def __init__(self, gl, a0, a1, b0, b1, res, sign, rl, raw, conc, k):
        self.gl, self.a0, self.a1, self.b0, self.b1 = gl, a0, a1, b0, b1
        self.res, self.sign, self.rl, self.raw, self.conc, self.k = res, sign, rl, raw, conc, k
    def lhs(self):
        return f"{self.a0}{self.a1} {self.gl} {self.b0}{self.b1}"
    def rstr(self):
        return ('-' if self.sign else '') + ''.join(self.res)

def letterize(prompt, rev):
    """letter-space rows for a given mode. Letters by first appearance, ops x/y/z."""
    eqs, qL = C2.parse(prompt)
    opchars = {L[2] for L, R in eqs} | {qL[2]}
    let = {}
    for L, R in eqs:
        for c in L + R:
            if c not in opchars and c not in let:
                let[c] = LETTERS[len(let)]
    for c in qL:
        if c not in opchars and c not in let:
            let[c] = LETTERS[len(let)]
    olet = {}
    for L, R in eqs:
        if L[2] not in olet:
            olet[L[2]] = OPLETTERS[min(len(olet), 3)]
    if qL[2] not in olet:
        olet[qL[2]] = OPLETTERS[min(len(olet), 3)]
    rows = []
    for k, (L, R) in enumerate(eqs, 1):
        nr = C2.normalize(L, R, rev, L[2])
        l0, l1, r0, r1, res, sign, rl, end = nr
        conc = C2.concat_patterns(L, R, sign)
        rows.append(Row(olet[L[2]], let[l0], let[l1], let[r0], let[r1],
                        tuple(let[s] for s in res), sign, rl, (L, R), conc, k))
    return rows, let, olet, eqs, qL

# ---------------------------------------------------------------- planner (domain propagation)
class Contra(Exception):
    def __init__(self, reason):
        self.reason = reason     # renderable string
        super().__init__(reason)

def initial_cands(rows_g, vocab):
    """sign/length filtered candidates for one glyph (concat handled by caller)."""
    lo, hi = 10, 99
    out = []
    for op in vocab:
        ok = True
        for r in rows_g:
            mn, mx, can_s, must_s = C2.op_bounds(op, lo, hi)
            if r.sign and not can_s: ok = False; break
            if not r.sign and must_s: ok = False; break
            vlo = 0 if r.rl == 1 else 10 ** (r.rl - 1)
            if mx < vlo or mn > 10 ** r.rl - 1: ok = False; break
        if ok:
            out.append(op)
    return out

class P2:
    """domain-propagation deduction engine over letter equations.
    Truth-guided only at BRANCH digit choice; every emitted step is checkable."""
    def __init__(self, vrows, cands, truth=None):
        self.rows = vrows                          # value rows only
        letters = set()
        for r in vrows:
            letters |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
        self.letters = sorted(letters)
        self.dom = {l: set(range(10)) for l in self.letters}
        self.cands = {g: list(c) for g, c in cands.items()}
        self.truth = truth                         # letter -> digit (solver config)
        self.steps = []                            # renderable steps
        self.nbranch = 0
        self.ep = None                             # r3 episode state (main line only)

    def clone(self):
        p = P2.__new__(P2)
        p.rows = self.rows
        p.letters = self.letters
        p.dom = {l: set(d) for l, d in self.dom.items()}
        p.cands = {g: list(c) for g, c in self.cands.items()}
        p.truth = self.truth
        p.steps = []
        p.nbranch = self.nbranch
        p.ep = None
        return p

    def sing(self, l):
        return len(self.dom[l]) == 1
    def val(self, l):
        return next(iter(self.dom[l]))
    def assigned(self):
        return {l: self.val(l) for l in self.letters if self.sing(l)}

    def set_dom(self, l, ds, why):
        nd = self.dom[l] & ds
        if nd == self.dom[l]:
            return False
        if not nd:
            raise Contra(f"{why} leaves no digit for {l}")
        self.dom[l] = nd
        if len(nd) == 1:
            d = next(iter(nd))
            self.steps.append(('LOCK', l, d, why))
            # alldiff
            for l2 in self.letters:
                if l2 != l and d in self.dom[l2]:
                    nd2 = self.dom[l2] - {d}
                    if not nd2:
                        raise Contra(f"{why} forces {l} = {d}, but then no digit is left for {l2}")
                    self.dom[l2] = nd2
                    if len(nd2) == 1:
                        self.steps.append(('LOCK', l2, next(iter(nd2)),
                                           f"only digit left after {l} = {d}"))
        return True

    # ---------- the single rule: per-equation generalized arc consistency
    def gac_eq(self, r):
        """project equation r onto its letters; filter glyph candidates. Raises Contra."""
        import itertools
        g = r.gl
        lets = []
        for l in (r.a0, r.a1, r.b0, r.b1) + r.res:
            if l not in lets:
                lets.append(l)
        op_lets = []
        for l in (r.a0, r.a1, r.b0, r.b1):
            if l not in op_lets:
                op_lets.append(l)
        prod = 1
        for l in op_lets:
            prod *= len(self.dom[l])
        if prod > 25000:
            return False    # too loose; revisit after other equations narrow it
        proj = {l: set() for l in lets}
        surv_ops = set()
        singles = {l: self.val(l) for l in self.letters if self.sing(l)}
        full = all(self.sing(x) for x in op_lets)
        tries = []
        for combo in itertools.product(*[sorted(self.dom[l]) for l in op_lets]):
            if len(set(combo)) != len(combo):
                continue
            asn = dict(zip(op_lets, combo))
            bad = False
            for l, d in asn.items():
                for l2, d2 in singles.items():
                    if d2 == d and l2 != l:
                        bad = True; break
                if bad:
                    break
            if bad:
                continue
            a = asn[r.a0] * 10 + asn[r.a1]
            b = asn[r.b0] * 10 + asn[r.b1]
            for op in self.cands[g]:
                if full and all(op != t[0] for t in tries):
                    tries.append((op, C2.OPS[op](a, b)))
                fv = self._fit_vals(op, r, a, b, asn, singles)
                if fv is None:
                    continue
                surv_ops.add(op)
                for l, d in fv.items():
                    proj[l].add(d)
        if not surv_ops:
            c = Contra(self.infeasible_reason(r, tries if full else None))
            # r4: structured payload so the renderer can show the op-level
            # exhaustion (try-list with real arithmetic) + level diagnosis.
            if full:
                ex = ('-' if r.sign else '') + ''.join(
                    str(self.val(x)) if self.sing(x) else x for x in r.res)
                c.opexh = (r, self.val(r.a0) * 10 + self.val(r.a1),
                           self.val(r.b0) * 10 + self.val(r.b1), list(tries), ex)
            raise c
        ch = False
        if len(surv_ops) < len(self.cands[g]):
            gone = [op for op in self.cands[g] if op not in surv_ops]
            if full and len(surv_ops) <= 2:
                a = self.val(r.a0) * 10 + self.val(r.a1)
                b = self.val(r.b0) * 10 + self.val(r.b1)
                vals = [(op, C2.OPS[op](a, b)) for op in gone]
                ep = self.ep
                emitted = False
                if (ep is not None and len(surv_ops) == 1 and 1 <= len(gone) <= 2
                        and len(gone) <= ep.get('wrong_left', 0)
                        and all(self.sing(x) for x in r.res)):
                    # r3 ELIM2: genuine wrong-branch episode — operands AND result
                    # digits known here, so each competing candidate can be tried
                    # with real numbers and refuted truthfully.
                    Mloc = {x: self.val(x) for x in
                            {r.a0, r.a1, r.b0, r.b1} | set(r.res)}
                    exp = _rhs_parts(r, Mloc, ep['let'], ep['rev'])[5]
                    ok2 = all(v is not None for _, v in vals)
                    for op, v in vals:
                        gs = _got_str(op, a, b) if v is not None else None
                        if gs is None or gs == exp or int(gs) == int(exp):
                            ok2 = False
                    if ok2:
                        ep['wrong_left'] -= len(gone)
                        ep['n_elim2'] = ep.get('n_elim2', 0) + len(gone)
                        self.steps.append(('ELIM2', r, a, b, vals,
                                           next(iter(surv_ops)), Mloc))
                        emitted = True
                if not emitted:
                    self.steps.append(('ELIM', r, a, b, vals, sorted(surv_ops)))
            elif len(surv_ops) <= 3:
                self.steps.append(('OPSHRINK', r, gone, sorted(surv_ops)))
            self.cands[g] = [op for op in self.cands[g] if op in surv_ops]
            ch = True
        for l in lets:
            nd = self.dom[l] & proj[l]
            target = next(iter(nd)) if len(nd) == 1 and len(self.dom[l]) > 1 else None
            why = self.lock_reason(r, l, target)
            ch |= self.set_dom(l, proj[l], why)
        return ch

    def _fit_vals(self, op, r, a, b, asn, singles):
        """letter->digit dict for the whole equation if op fits, else None."""
        v = C2.OPS[op](a, b)
        if v is None:
            return None
        if op in C2.SIGNED:
            if (v < 0) != r.sign:
                return None
            v = abs(v)
        elif op in C2.NEGPRE:
            if not r.sign:
                return None
            v = abs(v)
        else:
            if r.sign or v < 0:
                return None
        if v == 0:
            if r.rl != 1:
                return None
        elif not (10 ** (r.rl - 1) <= v < 10 ** r.rl):
            return None
        digs = C2._to_digits(v, 10, r.rl)
        out = dict(asn)
        for k, l in enumerate(r.res):
            d = digs[k]
            if d not in self.dom[l]:
                return None
            if l in out:
                if out[l] != d:
                    return None
            else:
                # injectivity: d must not collide with other letters in this equation
                if d in (x for y, x in out.items() if y != l):
                    return None
                for l2, d2 in singles.items():
                    if d2 == d and l2 != l:
                        return None
                out[l] = d
        return out

    def _units_expr(self, op, lu, ru, sign):
        """terse arithmetic string + units digit set for known operand units, or None."""
        d = {'_p1': 1, '_m1': -1, '_p2': 2, '_m2': -2}.get(op[-3:], 0)
        suf = {1: '+1', -1: '-1', 2: '+2', -2: '-2'}.get(d, '')
        if op.startswith('add'):
            v = lu + ru + d
            return f"{lu}+{ru}{suf} = {v}", {v % 10}
        if op.startswith('mul'):
            v = lu * ru + d
            return f"{lu}*{ru}{suf} = {v}", {v % 10}
        if op in ('sub_signed', 'rsub_signed'):
            if op == 'sub_signed':
                v = (ru - lu) if sign else (lu - ru)
                ex = f"{ru}-{lu}" if sign else f"{lu}-{ru}"
            else:
                v = (lu - ru) if sign else (ru - lu)
                ex = f"{lu}-{ru}" if sign else f"{ru}-{lu}"
            return f"({ex}) mod 10 = {v % 10}", {v % 10}
        return None

    def _bunits_expr(self, op, known_u, l_is_a1, sign):
        """expr text in unknown digit d for the backward units template, or None."""
        d = {'_p1': 1, '_m1': -1, '_p2': 2, '_m2': -2}.get(op[-3:], 0)
        suf = {1: '+1', -1: '-1', 2: '+2', -2: '-2'}.get(d, '')
        if op.startswith('add'):
            return f"d+{known_u}{suf}"
        if op.startswith('mul'):
            return f"d*{known_u}{suf}"
        if op == 'sub_signed':
            a, b = ('d', known_u) if l_is_a1 else (known_u, 'd')
            return f"{b}-{a}" if sign else f"{a}-{b}"
        if op == 'rsub_signed':
            a, b = ('d', known_u) if l_is_a1 else (known_u, 'd')
            return f"{a}-{b}" if sign else f"{b}-{a}"
        if op in ('absdiff', 'neg_absdiff'):
            return f"|d-{known_u}|{suf}"
        if op.startswith('absdiff'):
            return f"|d-{known_u}|{suf}"
        return None

    def lock_reason(self, r, l, target=None):
        """numeric, derivable justification for pinning letter l (target = pinned digit).
        Never teleports: each branch states the constraint arithmetic that does the pin."""
        ops = self.cands[r.gl]
        opw = '/'.join(OP_WORD[o] for o in ops[:3])
        full = all(self.sing(x) for x in (r.a0, r.a1, r.b0, r.b1))
        # (a) operands fully known -> compute the result outright
        if full:
            a = self.val(r.a0) * 10 + self.val(r.a1)
            b = self.val(r.b0) * 10 + self.val(r.b1)
            vals = set()
            for op in ops:
                fv = self._fit(op, r, a, b)
                if fv is not None:
                    vals.add(('-' if r.sign else '') + ''.join(str(x) for x in fv[1]))
            if len(vals) == 1:
                vs = vals.pop()
                res_l = ('-' if r.sign else '') + ''.join(r.res)
                return f"EQ{r.k}: {a} {opw} {b} = {vs}, so {res_l} = {vs}"
        # (b) forward units congruence: result units from known operand units
        if (target is not None and l == r.res[-1] and len(ops) == 1
                and self.sing(r.a1) and self.sing(r.b1)):
            ue = self._units_expr(ops[0], self.val(r.a1), self.val(r.b1), r.sign)
            if ue and ue[1] == {target}:
                return f"EQ{r.k} units: {ue[0]} -> result ends in {target}"
        # (c) backward units congruence: operand units from known result units
        if (target is not None and l in (r.a1, r.b1) and len(ops) == 1
                and self.sing(r.res[-1])):
            other = r.b1 if l == r.a1 else r.a1
            if self.sing(other) and other != l:
                t = self.val(r.res[-1]); ov = self.val(other)
                surv = []
                for d in range(10):
                    lu, ru = (d, ov) if l == r.a1 else (ov, d)
                    u = C2.units_set(ops[0], lu, ru, 10, r.sign)
                    if u is not None and t in u:
                        surv.append(d)
                ex = self._bunits_expr(ops[0], ov, l == r.a1, r.sign)
                if ex and target in surv:
                    inset = [d for d in surv if d in self.dom[l]]
                    if inset == [target]:
                        excl = [d for d in surv if d != target]
                        taken = {self.val(x) for x in self.letters if self.sing(x) and x != l}
                        note = ''
                        if excl:
                            word = 'taken' if all(d in taken for d in excl) else 'ruled out'
                            note = f"; {','.join(map(str, excl))} {word}"
                            ep = self.ep
                            if (ep and ep.get('dist_left', 0) > 0 and len(excl) == 1
                                    and excl[0] in taken
                                    and ep['rng'].random() < ep['p']):
                                e = excl[0]
                                holder = next((x for x in self.letters if self.sing(x)
                                               and x != l and self.val(x) == e), None)
                                if holder:
                                    ep['dist_left'] -= 1
                                    ep['n_collide'] = ep.get('n_collide', 0) + 1
                                    note = (f"; try {l} = {e}: distinct check: {holder}={e}"
                                            f" and {l}={e} collide. conflict -> backtrack")
                        return (f"EQ{r.k} units: {ex} must end in {t} -> d in "
                                f"{{{','.join(map(str, surv))}}}{note}")
        # (d) 3-digit add result -> leading digit 1
        if (target == 1 and l == r.res[0] and r.rl == 3
                and all(o in ADD_FAM for o in ops)):
            return (f"EQ{r.k}: sum of two 2-digit numbers < 200, so the 3-digit"
                    f" result starts with 1")
        # (e) fallback: explicit per-digit try/reject over the current domain
        if target is not None:
            bad = sorted(self.dom[l] - {target})
            if len(bad) <= 4:
                badtxt = ", ".join(f"{l}={d} no fit" for d in bad)
            else:
                badtxt = f"{l} in {{{','.join(map(str, bad))}}} all fail"
            return (f"EQ{r.k}: {self.lhs_vals(r)} = {r.rstr()} ({opw}): try each digit:"
                    f" {badtxt}; only {l}={target} admits consistent digits")
        return f"EQ{r.k}: {self.lhs_vals(r)} = {r.rstr()} ({opw}) constrains {l}"

    def infeasible_reason(self, r, tries):
        if tries:
            seen = {}
            for op, v in tries:
                if op not in seen:
                    seen[op] = v
            bits = [f"{OP_WORD[op]}={v if v is not None else '?'}"
                    for op, v in list(seen.items())[:6]]
            return (f"EQ{r.k}: {self.lhs_vals(r)} must give {r.rstr()} but "
                    + ", ".join(bits) + " all fail")
        # units-level explanation if available
        if self.sing(r.a1) and self.sing(r.b1):
            rel = set()
            for op in self.cands[r.gl]:
                u = C2.units_set(op, self.val(r.a1), self.val(r.b1), 10, r.sign)
                if u is None:
                    rel = None; break
                rel |= u
            if rel is not None and not (rel & self.dom[r.res[-1]]):
                return (f"EQ{r.k} units: would need units in {sorted(rel)} but"
                        f" {r.res[-1]} can only be {sorted(self.dom[r.res[-1]])}")
        return f"EQ{r.k}: no consistent digit values remain"

    def lhs_vals(self, r):
        def two(x, y):
            if self.sing(x) and self.sing(y):
                return f"{self.val(x)}{self.val(y)}"
            return f"{x}{y}"
        return f"{two(r.a0, r.a1)} {r.gl} {two(r.b0, r.b1)}"

    def _fit(self, op, r, a, b):
        v = C2.OPS[op](a, b)
        if v is None:
            return None
        if op in C2.SIGNED:
            if (v < 0) != r.sign:
                return None
            v = abs(v)
        elif op in C2.NEGPRE:
            if not r.sign:
                return None
            v = abs(v)
        else:
            if r.sign or v < 0:
                return None
        if v == 0:
            if r.rl != 1:
                return None
        elif not (10 ** (r.rl - 1) <= v < 10 ** r.rl):
            return None
        digs = C2._to_digits(v, 10, r.rl)
        seen = {}
        for k, l in enumerate(r.res):
            if digs[k] not in self.dom[l]:
                return None
            if seen.setdefault(l, digs[k]) != digs[k]:
                return None
        if len(set(seen.values())) != len(seen):
            return None
        # injectivity vs other singletons
        sm = {l: self.val(l) for l in self.letters if self.sing(l)}
        for l, d in seen.items():
            if not self.sing(l):
                for l2, d2 in sm.items():
                    if d2 == d and l2 != l:
                        return None
        return v, digs

    def propagate(self, cap=40):
        n = 0
        while n < cap:
            n += 1
            ch = False
            for r in self.rows:
                ch |= self.gac_eq(r)
            if not ch:
                break

    def done(self, need):
        return all(self.sing(l) for l in need)

    # ---------- truth-guided branching with rendered refutations
    def branch(self, need, depth=0, max_depth=5):
        self.propagate()
        if self.done(need):
            return True
        if depth >= max_depth or self.truth is None:
            return False
        # rank branch letters: units positions first, smallest domain, most rows touched
        cnt = {}
        for r in self.rows:
            for l in (r.a1, r.b1, r.res[-1]):
                cnt[l] = cnt.get(l, 0) + 2
            for l in (r.a0, r.b0) + tuple(r.res[:-1]):
                cnt[l] = cnt.get(l, 0) + 1
        cands = sorted((l for l in need if not self.sing(l)),
                       key=lambda l: (len(self.dom[l]), -cnt.get(l, 0), l))
        for l in cands[:6]:
            td = self.truth.get(l)
            if td is None or td not in self.dom[l]:
                continue
            trials = []
            ok = True
            for d in sorted(self.dom[l]):
                if d == td:
                    continue
                sub = self.clone()
                try:
                    sub.set_dom(l, {d}, f"try {l} = {d}")
                    sub.propagate(cap=40)
                except Contra as e:
                    trials.append((d, e.reason, sub.steps))
                    continue
                ok = False
                break
            if not ok:
                continue
            # all wrong digits refuted; commit the true digit
            self.steps.append(('BRANCH', l, sorted(self.dom[l]), td, trials))
            self.set_dom(l, {td}, f"only consistent choice for {l}")
            self.nbranch += 1
            return self.branch(need, depth + 1, max_depth)
        return False

# ---------------------------------------------------------------- step rendering
import re as _re
_ALLDIFF_WHY = _re.compile(r"^only digit left after ([A-J]) = (\d)$")

def render_steps(steps, L, ep=None):
    i = 0
    last_branch_letter = None
    while i < len(steps):
        st = steps[i]
        k = st[0]
        if k == 'LOCK':
            _, l, d, why = st
            if l == last_branch_letter and why.startswith('only consistent choice'):
                i += 1
                continue   # BRANCH block already rendered this lock
            m = _ALLDIFF_WHY.match(why) if ep else None
            if (m and ep.get('dist_left', 0) > 0 and ep['rng'].random() < ep['p']):
                # r3 distinct-collision episode: the alldiff prune, shown truthfully
                partner, dp = m.group(1), int(m.group(2))
                lo, hi = sorted((d, dp))
                ep['dist_left'] -= 1
                ep['n_collide'] = ep.get('n_collide', 0) + 1
                L.append(f"{l} in {{{lo},{hi}}}: try {l} = {dp}: distinct check:"
                         f" {partner}={dp} and {l}={dp} collide. conflict ->"
                         f" backtrack -> {l} = {d}.")
                i += 1
                continue
            grp = [(l, d)]
            while (i + 1 < len(steps) and steps[i + 1][0] == 'LOCK'
                   and steps[i + 1][3] == why):
                grp.append((steps[i + 1][1], steps[i + 1][2]))
                i += 1
            L.append(f"{why} -> " + ", ".join(f"{x} = {y}" for x, y in grp) + ".")
            i += 1
            continue
        last_branch_letter = st[1] if k == 'BRANCH' else None
        _render_step_other(st, L, ep)
        i += 1

def _render_step_other(st, L, ep=None):
        k = st[0]
        if k == 'ELIM2':
            # r3 wrong-branch episode: decode the RHS first, then tentatively lock each
            # competing candidate; the wrong ones FAIL with real numbers and backtrack.
            _, r, a, b, vals, opk, Mloc = st
            let, rev = ep['let'], ep['rev']
            Rr, opch, side, sign, dd, exp = _rhs_parts(r, Mloc, let, rev)
            note = f" ({side} {opch} = minus)" if sign else ""
            dec = " ".join(str(d) for d in dd)
            revw = "reversed " if rev else ""
            L.append(f"EQ{r.k} RHS {Rr}{note} -> {dec} -> {revw}{exp}.")
            for op, _v in vals:
                got = _got_str(op, a, b)
                L.append(f"Try {r.gl} = {OP_WORD[op]} on EQ{r.k}: {a} {OP_WORD[op]} {b} ="
                         f" {got}. {got} vs {exp}. no -> mistake, backtrack.")
            got = _got_str(opk, a, b)
            L.append(f"Try {r.gl} = {OP_WORD[opk]} on EQ{r.k}: {a} {OP_WORD[opk]} {b} ="
                     f" {got}. {got} vs {exp}. match -> lock {r.gl} = {OP_WORD[opk]}.")
        elif k == 'ELIM':
            _, r, a, b, tries, surv = st
            parts = []
            for op, fv in tries:
                if op in surv:
                    continue
                v = C2.OPS[op](a, b)
                parts.append(f"{OP_WORD[op]}={v if v is not None else '?'} no")
            keep = " / ".join(OP_WORD[op] for op in surv)
            L.append(f"EQ{r.k}: {a} {r.gl} {b} vs {r.rstr()}: " + "; ".join(parts[:6])
                     + f" -> {r.gl} is {keep}.")
        elif k == 'OPSHRINK':
            _, r, gone, surv = st
            L.append(f"EQ{r.k}: {r.lhs()} = {r.rstr()} cannot come from "
                     + "/".join(OP_WORD[o] for o in gone[:5])
                     + (" (no digit values fit)" if len(gone) <= 5 else
                        f" or {len(gone)-5} other rules (no digit values fit)")
                     + f" -> {r.gl} is " + " or ".join(OP_WORD[o] for o in surv) + ".")
        elif k == 'BRANCH':
            _, l, dom, td, trials = st
            L.append(f"Try each digit for {l} (candidates {dom}):")
            for d, reason, substeps in trials:
                subs = [s for s in substeps if s[0] == 'LOCK'
                        and not (s[1] == l and s[3].startswith('try '))]
                pre = "; ".join(f"{s[3]} -> {s[1]} = {s[2]}" for s in subs[:2])
                line = f"  {l} = {d}: "
                if pre:
                    line += pre + "; "
                line += reason + " -> no, backtrack."
                L.append(line)
            L.append(f"  {l} = {td}: consistent. Lock {l} = {td}.")

# ---------------------------------------------------------------- trace renderer
def fam_cue(g, rs, cands):
    lens = sorted({r.rl for r in rs})
    signed = any(r.sign for r in rs)
    if all(r.conc for r in rs) and set.intersection(*[set(r.conc) for r in rs]):
        pat = sorted(set.intersection(*[set(r.conc) for r in rs]))[0]
        return f"{g}: every result reuses the operand symbols in order -> {OP_WORD[pat]}"
    if signed:
        return (f"{g}: some results carry the operator glyph as a minus sign ->"
                f" subtraction-like (a-b, b-a or -|a-b|)")
    if 4 in lens:
        return f"{g}: 4-digit results -> multiplication-like (only mul of two 2-digit numbers reaches 4 digits)"
    if lens == [1]:
        return f"{g}: 1-digit results -> small-value op (difference or mod)"
    if 3 in lens and 2 in lens:
        return f"{g}: 2- and 3-digit results, no sign -> add-like (mul would need >= 3 digits always)"
    if 3 in lens:
        return f"{g}: 3-digit results -> add-like or mul-like"
    return f"{g}: short unsigned results -> add/sub/diff/mod family"

def build_cands(rows, vocab=None):
    """vocab: op list for every glyph, or (r5) a dict keyed by glyph LETTER ->
    per-glyph op list (canonical-regime family vocabularies)."""
    byg = {}
    for r in rows:
        byg.setdefault(r.gl, []).append(r)
    cands = {}
    conc = {}
    for g, rs in byg.items():
        pats = set.intersection(*[set(r.conc) for r in rs]) if all(r.conc for r in rs) else set()
        if pats:
            conc[g] = sorted(pats)[0]
            cands[g] = [conc[g]]
        else:
            v = vocab.get(g, RVOCAB) if isinstance(vocab, dict) else vocab
            cands[g] = initial_cands(rs, v if v is not None else RVOCAB)
    return byg, cands, conc

def pin_episodes(byg, conc, narrative_live, pins, survs, M, let, rev, gone_r,
                 cap=PIN_WRONG_CAP, have_no=0):
    """r3 wrong-branch episodes at rule-pin time. Candidates that are still narratively
    live (never explicitly eliminated in a rendered line) are tried most-common-first;
    genuinely wrong ones FAIL with real numbers and backtrack, the true rule locks with
    a real match. If no genuine failed try is renderable anywhere, inject ONE
    double-check of the most common rule (a+b / a*b / a-b). Returns (lines, n_no)."""
    lines = []
    seen_dec = set()
    n_no = 0

    def dec_line(r):
        if r.k in seen_dec:
            return
        seen_dec.add(r.k)
        Rr, opch, side, sign, dd, exp = _rhs_parts(r, M, let, rev)
        note = f" ({side} {opch} = minus)" if sign else ""
        dec = " ".join(str(d) for d in dd)
        revw = "reversed " if rev else ""
        lines.append(f"EQ{r.k} RHS {Rr}{note} -> {dec} -> {revw}{exp}.")

    def try_line(g, op, r):
        a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
        got = _got_str(op, a, b)
        exp = _rhs_parts(r, M, let, rev)[5]
        if got == exp:
            lines.append(f"Try {g} = {OP_WORD[op]} on EQ{r.k}: {a} {OP_WORD[op]} {b} ="
                         f" {got}. {got} vs {exp}. match -> lock {g} = {OP_WORD[op]}.")
            return 'match'
        lines.append(f"Try {g} = {OP_WORD[op]} on EQ{r.k}: {a} {OP_WORD[op]} {b} ="
                     f" {got}. {got} vs {exp}. no -> mistake, backtrack.")
        return 'no'

    def wrong_row(g, op):
        for r in byg[g]:
            a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
            got = _got_str(op, a, b)
            if got is None:
                continue
            exp = _rhs_parts(r, M, let, rev)[5]
            if got != exp and int(got) != int(exp):
                return r
        return None

    for g in sorted(narrative_live):
        if g in conc or n_no >= cap:
            continue
        order = sorted(narrative_live[g],
                       key=lambda o: -C2.MEASURED_OP_COUNTS.get(o, 1))
        episode = []
        for op in order:
            if op == pins[g]:
                break
            if op in survs.get(g, ()):
                continue                  # tie op (fits everywhere): tie lines handle it
            if len(episode) >= 2 or n_no + len(episode) >= cap:
                continue
            r = wrong_row(g, op)
            if r is not None:
                episode.append((op, r))
        if not episode:
            continue
        lines.append(f"{g} could still be " + " / ".join(OP_WORD[o] for o in order[:4])
                     + ". Try the most common first.")
        for op, r in episode:
            dec_line(r)
            try_line(g, op, r)
            n_no += 1
        r0 = episode[-1][1]
        dec_line(r0)
        try_line(g, pins[g], r0)
    if n_no + have_no == 0:
        # fully forced search: inject ONE failed double-check with real numbers
        for g in sorted(narrative_live):
            if g in conc:
                continue
            for w in ('add', 'mul', 'sub_signed'):
                if w == pins[g] or w in survs.get(g, ()) or w in gone_r.get(g, set()):
                    continue
                r = wrong_row(g, w)
                if r is None:
                    continue
                lines.append(f"Double-check {g} against the most common rules"
                             f" before locking.")
                dec_line(r)
                try_line(g, w, r)
                try_line(g, pins[g], r)
                n_no += 1
                return lines, n_no
    return lines, n_no

def render_trace(prompt, meta, answer, ep_rng=None):
    """returns (cot_text, tier, info) or (None, tier, reason)."""
    rev = meta['rev']
    rows, let, olet, eqs, qL = letterize(prompt, rev)
    qop_ch = qL[2]
    qg = olet[qop_ch]
    inv_let = {v: k for k, v in let.items()}
    tmap = {let[s]: d for s, d in meta['mapping'].items() if d is not None and s in let}
    tops = {olet[ch]: op for ch, op in meta['ops'].items() if ch in olet and op}
    qtrue = meta['qop']
    is_guess = all(r.gl != qg for r in rows)
    tier = 1 if qtrue in ('concat_fwd', 'concat_rev') else \
           (3 if qtrue in MUL_FAM | SMALL_FAM | {'a2_plus_b', 'a_plus_b2', 'lcm', 'gcd'} else 2)

    L = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("")
    L.append("First, assign letters to each symbol:")
    for s, a in let.items():
        L.append(f"{s} -> {a}")
    L.append("Operators:")
    for ch, a in olet.items():
        L.append(f"{ch} -> {a}")
    L.append("")
    L.append("Examples in letter form (each left side: two digit-symbols, operator, two digit-symbols):")
    # transcription is mode-agnostic (raw order), letters only
    raw_lets = []
    for k, (Lr, Rr) in enumerate(eqs, 1):
        opch = Lr[2]
        mag = Rr; sign_note = ''
        if len(Rr) > 1 and Rr[0] == opch:
            mag = Rr[1:]; sign_note = 'minus sign in front'
        elif len(Rr) > 1 and Rr[-1] == opch:
            mag = Rr[:-1]; sign_note = 'minus sign at the end'
        ll = ''.join(let[c] for c in (Lr[0], Lr[1])) + f" {olet[opch]} " + \
             ''.join(let[c] for c in (Lr[3], Lr[4]))
        rl = ''.join(let[c] for c in mag)
        raw_lets.append((ll, rl, sign_note))
        L.append(f"EQ{k}: {Lr} = {Rr} -> {ll} = {rl}" + (f" ({sign_note})" if sign_note else ""))
    qlets = f"{let[qL[0]]}{let[qL[1]]} {qg} {let[qL[3]]}{let[qL[4]]}"
    L.append(f"Query: {qL} -> {qlets}")
    L.append("")
    # ---- classification
    byg, cands, conc = build_cands(rows)
    L.append("Classify each operator from cues:")
    for g, rs in byg.items():
        L.append(fam_cue(g, rs, cands[g]))
    if is_guess:
        L.append(f"{qg}: appears only in the query -> rule unknown, decide at the end.")
    L.append("")
    # ---- Tier 1 short-circuit
    if not is_guess and qtrue in ('concat_fwd', 'concat_rev'):
        L.append(f"The question operator is {qg}, which is {OP_WORD[qtrue]}.")
        # r3: truthful order check — refute the wrong concat order with real strings
        chk = None
        for r in byg.get(qg, []):
            Lr, Rr = r.raw
            fwd = Lr[0] + Lr[1] + Lr[3] + Lr[4]
            rv = Lr[3] + Lr[4] + Lr[0] + Lr[1]
            if fwd != rv:
                chk = (r.k, Rr, fwd, rv)
                break
        if chk is None:
            return None, tier, "concat: no order-discriminating example"
        k_, Rr_, fwd_, rv_ = chk
        wrongs, rights = (rv_, fwd_) if qtrue == 'concat_fwd' else (fwd_, rv_)
        wname, rname = (('swapped', 'in order') if qtrue == 'concat_fwd'
                        else ('in order', 'swapped'))
        if rights != Rr_:
            return None, tier, "concat order check inconsistent"
        L.append(f"Check the order on EQ{k_}: {wname} would give {wrongs}."
                 f" {wrongs} vs {Rr_}. no -> mistake, backtrack; {rname} gives"
                 f" {rights}. {rights} vs {Rr_}. match.")
        A2, B2 = qlets.split()[0], qlets.split()[2]
        s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
        if qtrue == 'concat_fwd':
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {A2} || {B2} = {A2}{B2}")
            syms = s1 + s2
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s1} then {s2} -> {syms}")
        else:
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {B2} || {A2} = {B2}{A2}")
            syms = s2 + s1
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s2} then {s1} -> {syms}")
        L.append("")
        L.append(f"\\boxed{{{syms}}}")
        if syms != answer:
            return None, tier, f"concat mismatch {syms!r} != {answer!r}"
        return "\n".join(L), tier, {}

    # ---- mode
    vrows = [r for r in rows if r.gl not in conc]
    if not vrows:
        return None, tier, "no value rows but non-concat query"
    need = set()
    for r in vrows:
        need |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
    if rev:
        # demonstrate the standard-mode failure, then re-read little-endian
        srows, _, _, _, _ = letterize(prompt, False)
        svrows = [r for r in srows if r.gl not in conc]
        sb, scands, sconc = build_cands(srows)
        sneed = set()
        for r in svrows:
            sneed |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
        mode_block = std_refute_block(svrows, scands, sneed)
        if mode_block is None:
            return None, tier, "rev row: no compact standard-mode refutation"
        L.append("Assume standard digit order (left digit = tens).")
        L.extend(mode_block)
        L.append("Standard reading fails -> the puzzle writes numbers LITTLE-ENDIAN"
                 " (digits reversed). Re-read every number MSB-first by reversing:")
        for r in rows:
            L.append(f"EQ{r.k}: {r.lhs()} = {r.rstr()}")
        L.append(f"Query reads: {let[qL[1]]}{let[qL[0]]} {qg} {let[qL[4]]}{let[qL[3]]}")
        L.append("")
    else:
        L.append("Assume standard digit order (left digit = tens); verify at the end.")
        L.append("")
    # ---- crack
    eng = P2(vrows, {g: c for g, c in cands.items() if g not in conc}, truth=tmap)
    ep = {'rng': ep_rng if ep_rng is not None else random.Random(12345),
          'p': P_DIST_EP, 'dist_left': 1, 'n_collide': 0,
          'wrong_left': PIN_WRONG_CAP, 'n_elim2': 0, 'let': let, 'rev': rev}
    eng.ep = ep
    L.append("Deduce the digits:")
    try:
        okp = eng.branch(need)
    except Contra as e:
        return None, tier, f"engine contradiction on true config: {e.reason}"
    if not okp:
        return None, tier, "engine: could not complete map within branch budget"
    render_steps(eng.steps, L, ep=ep)
    M = eng.assigned()
    # check against solver truth
    for l, d in tmap.items():
        if l in M and M[l] != d:
            return None, tier, f"engine map disagrees with solver at {l}"
    # pin ops
    pins = {}
    survs = {}
    tie_lines = []
    for g in eng.cands:
        surv = []
        for op in eng.cands[g]:
            okop = True
            for r in [r for r in vrows if r.gl == g]:
                a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
                fv = eng._fit(op, r, a, b)
                if fv is None:
                    okop = False; break
                v, digs = fv
                if any(M[r.res[k]] != digs[k] for k in range(r.rl)):
                    okop = False; break
            if okop:
                surv.append(op)
        if not surv:
            return None, tier, f"no op fits glyph {g} at final map"
        survs[g] = set(surv)
        choice = tops.get(g)
        if choice not in surv:
            choice = surv[0]
        pins[g] = choice
        if len(surv) > 1:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in surv)} on every"
                             f" example; keep the most common: {g} = {OP_WORD[choice]}.")
    for g, op in conc.items():
        pins[g] = op
    L.append("")
    L.append("Locked map: " + ", ".join(f"{l}={d}" for l, d in sorted(M.items())))
    # ---- injectivity (asserted, not just printed)
    if len(set(M.values())) != len(M):
        return None, tier, "injectivity violated in locked map"
    L.append("Distinct check: " + " ".join(str(d) for _, d in sorted(M.items()))
             + " - all different. ok")
    # ---- r3: wrong-branch episodes at rule-pin time (truthful failed tries).
    # narrative-live candidates = initial cue-consistent set minus everything a
    # rendered ELIM/OPSHRINK line explicitly disposed of (silent GAC prunes stay live).
    gone_r = {}
    narrative_live = {g: list(c) for g, c in cands.items() if g not in conc}
    for st in eng.steps:
        if st[0] == 'ELIM':
            _, r0, _a0, _b0, tries0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(
                op for op, _v in tries0 if op not in surv0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
        elif st[0] == 'ELIM2':
            _, r0, _a0, _b0, vals0, opk0, _M0 = st
            gone_r.setdefault(r0.gl, set()).update(op for op, _v in vals0)
            narrative_live[r0.gl] = [opk0]
        elif st[0] == 'OPSHRINK':
            _, r0, gone0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(gone0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
    byg2 = {}
    for r in vrows:
        byg2.setdefault(r.gl, []).append(r)
    pin_lines, n_no = pin_episodes(byg2, conc, narrative_live, pins, survs,
                                   M, let, rev, gone_r,
                                   cap=ep['wrong_left'], have_no=ep['n_elim2'])
    if n_no + ep['n_elim2'] == 0:
        return None, tier, "no truthful negative episode renderable"
    L.extend(pin_lines)
    L.extend(tie_lines)
    L.append("Locked rules: " + ", ".join(f"{g} = {OP_WORD[pins[g]]}" for g in sorted(pins)))
    L.append("")
    # ---- verify: decode the RHS through the locked map FIRST, then compute, then match
    L.append("Verify every example (decode RHS first, then compute):")
    for r in rows:
        op = pins[r.gl]
        Lr, Rr = r.raw
        opch_row = Lr[2]
        if op in ('concat_fwd', 'concat_rev'):
            pat = (Lr[0] + Lr[1] + Lr[3] + Lr[4] if op == 'concat_fwd'
                   else Lr[3] + Lr[4] + Lr[0] + Lr[1])
            if Rr != pat:
                return None, tier, f"concat verify failed EQ{r.k}"
            order = 'in order' if op == 'concat_fwd' else 'swapped'
            L.append(f"EQ{r.k}: RHS {Rr} = operand symbols {order}"
                     f" ({Lr[0]}{Lr[1]} and {Lr[3]}{Lr[4]}). match")
            continue
        sign = False; side = None
        mag = Rr
        if len(Rr) > 1 and Rr[0] == opch_row:
            sign = True; side = 'leading'; mag = Rr[1:]
        elif len(Rr) > 1 and Rr[-1] == opch_row:
            sign = True; side = 'trailing'; mag = Rr[:-1]
        dd = [M[let[c]] for c in mag]
        msb = dd[::-1] if rev else dd
        exp_num = int(''.join(map(str, msb)))
        exp_str = ('-' if sign else '') + str(exp_num)
        a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
        v = C2.OPS[op](a, b)
        if v is None:
            return None, tier, f"verify undefined EQ{r.k}"
        got = ('-' if (v < 0 or op in C2.NEGPRE) else '') + str(abs(v))
        if got != exp_str:
            return None, tier, f"verify failed EQ{r.k}: {got} != {exp_str}"
        note = f" ({side} {opch_row} = minus)" if sign else ""
        dec = " ".join(str(d) for d in dd)
        revw = "reversed " if rev else ""
        L.append(f"EQ{r.k}: RHS {Rr}{note} -> {dec} -> {revw}{exp_str};"
                 f" compute {a} {OP_WORD[op]} {b} = {got}. match")
    L.append("")
    # ---- apply
    qa_, qb_, qc_, qd_ = let[qL[0]], let[qL[1]], let[qL[3]], let[qL[4]]
    missing = [x for x in (qa_, qb_, qc_, qd_) if x not in M]
    if missing:
        free_digs = sorted(set(range(10)) - set(M.values()))
        if len(missing) == 1 and len(free_digs) == 1:
            M[missing[0]] = free_digs[0]
            L.append(f"{missing[0]} appears only in the query; the one unused digit is"
                     f" {free_digs[0]}, so {missing[0]} = {free_digs[0]}.")
        else:
            tm = {l: tmap.get(l) for l in missing}
            if any(v is None for v in tm.values()):
                return None, tier, "query letters not deducible"
            for l in missing:
                M[l] = tmap[l]
            L.append("Letters " + ", ".join(missing) + " appear only in the query;"
                     f" unused digits are {free_digs}. Take "
                     + ", ".join(f"{l} = {M[l]}" for l in missing) + ".")
    if is_guess:
        L.append(f"The query operator {qg} never appears in the examples."
                 f" The most common rule in these puzzles is subtraction, so bet {qg} = a-b.")
        qapply = 'sub_signed'
        if qtrue != 'sub_signed':
            return None, tier, "guess policy mismatch (row not renderable as policy hit)"
    else:
        qapply = pins[qg]
        if qapply != qtrue:
            return None, tier, f"pinned query op {qapply} != solver op {qtrue}"
    if rev:
        A = M[qb_] * 10 + M[qa_]; B = M[qd_] * 10 + M[qc_]
        L.append(f"Apply to the query (little-endian): {qb_}{qa_} {qg} {qd_}{qc_}"
                 f" = {A} {OP_WORD[qapply]} {B}")
    else:
        A = M[qa_] * 10 + M[qb_]; B = M[qc_] * 10 + M[qd_]
        L.append(f"Apply to the query: {qa_}{qb_} {qg} {qc_}{qd_} = {A} {OP_WORD[qapply]} {B}")
    v = C2.OPS[qapply](A, B)
    if v is None:
        return None, tier, "query value undefined"
    sign_needed = (qapply in C2.NEGPRE) or (qapply in C2.SIGNED and v < 0)
    if v < 0 and not sign_needed:
        return None, tier, "negative query value for unsigned op"
    mag = abs(v)
    digs = [int(c) for c in str(mag)]
    inv_d = {d: l for l, d in M.items()}
    if any(d not in inv_d for d in digs):
        return None, tier, f"answer digit unmapped: {mag}"
    dig2sym = {d: inv_let[inv_d[d]] for d in set(digs)}
    L.append(f"= {'-' if (v < 0 or qapply in C2.NEGPRE) else ''}{mag}")
    # decode digit-by-digit DIRECTLY to symbols (no letter hop)
    disp = digs[::-1] if rev else digs
    pairs = " ".join(f"{d}={dig2sym[d]}" for d in disp)
    core = ''.join(dig2sym[d] for d in disp)
    if rev:
        L.append(f"Little-endian, write digits reversed: {mag} ->"
                 f" {''.join(map(str, disp))}; decode digit by digit: {pairs} -> {core}")
    else:
        L.append(f"Decode digit by digit: {pairs} -> {core}")
    syms = core
    if sign_needed:
        s_end = meta.get('sign_end', 'pre')
        if s_end == 'suf':
            syms = core + qop_ch
            L.append(f"Negative -> append the operator glyph {qop_ch} as the sign"
                     f" (as in the examples) -> {syms}")
        else:
            syms = qop_ch + core
            L.append(f"Negative -> prefix the operator glyph {qop_ch} as the sign"
                     f" -> {syms}")
    L.append("")
    L.append(f"\\boxed{{{syms}}}")
    if syms != answer:
        return None, tier, f"final mismatch {syms!r} != {answer!r}"
    return "\n".join(L), tier, {'branches': eng.nbranch}

def std_refute_block(svrows, scands, sneed):
    """compact demonstration that standard mode is impossible: propagate; if direct
    contradiction, render it; else branch the best letter and require ALL its digits
    to refute. Returns lines or None."""
    eng = P2(svrows, {g: scands[g] for g in {r.gl for r in svrows}}, truth=None)
    try:
        eng.propagate()
    except Contra as e:
        lines = []
        render_steps(eng.steps[:3], lines)
        lines.append(f"Contradiction: {e.reason}.")
        return lines
    cnt = {}
    for r in svrows:
        for l in (r.a1, r.b1, r.res[-1]):
            cnt[l] = cnt.get(l, 0) + 2
    cands = sorted((l for l in sneed if not eng.sing(l)),
                   key=lambda l: (len(eng.dom[l]), -cnt.get(l, 0), l))
    for l in cands[:3]:
        trials = []
        allfail = True
        for d in sorted(eng.dom[l]):
            sub = eng.clone()
            try:
                sub.set_dom(l, {d}, f"try {l} = {d}")
                sub.propagate(cap=40)
            except Contra as e:
                trials.append((d, e.reason))
                continue
            allfail = False
            break
        if allfail and trials:
            lines = [f"No digit works for {l}:"]
            for d, reason in trials[:3]:
                lines.append(f"  {l} = {d}: {reason} -> no.")
            if len(trials) > 3:
                lines.append(f"  ... every remaining digit fails the same checks.")
            return lines
    return None

# ================================================================ r4: policy-driven search
# The r3 renderer was truth-guided: it knew the answer, so every rendered search
# succeeded within 1-3 tries and the model never saw an exhausted candidate level.
# r4 executes a DETERMINISTIC, truth-free policy (most-constrained letter first,
# digits ascending, ops by measured prior, distinct ops per row, standard mode
# first) and renders the search the policy actually performs — including failed
# digit pins, exhausted levels, level-up diagnosis and recovery. The same
# executor measures POLICY COVERAGE on the real train rows (no gold guidance:
# gold is only compared against the final boxed answer).

class Budget(Exception):
    pass

POLICY_NODE_CAP = 44    # digit-try budget per mode pass (rendered tries)
POLICY_DEPTH_CAP = 8

class PolicyP2(P2):
    """P2 with a deterministic truth-free DFS = the rendered search policy."""

    dup_ok = frozenset()    # r5: ops exempt from distinct-rules (canonical concat)

    def clone(self):
        p = P2.clone(self)
        p.__class__ = PolicyP2
        p.dup_ok = self.dup_ok
        return p

    def pick_letter(self, need):
        cnt = {}
        for r in self.rows:
            for l in (r.a1, r.b1, r.res[-1]):
                cnt[l] = cnt.get(l, 0) + 2
            for l in (r.a0, r.b0) + tuple(r.res[:-1]):
                cnt[l] = cnt.get(l, 0) + 1
        opn = [l for l in sorted(need) if not self.sing(l)]
        return min(opn, key=lambda l: (len(self.dom[l]), -cnt.get(l, 0), l))

    def _distinct_fail(self):
        """at a complete map: None if a distinct op assignment exists over the
        value glyphs, else (g1, g2, opword) of a forced duplicate."""
        sets = {}
        M = self.assigned()
        for g, ops in self.cands.items():
            surv = []
            for op in ops:
                fits = True
                for r in (r for r in self.rows if r.gl == g):
                    a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
                    fv = self._fit(op, r, a, b)
                    if fv is None or any(M[r.res[k]] != fv[1][k] for k in range(r.rl)):
                        fits = False; break
                if fits:
                    surv.append(op)
            if not surv:
                return None     # op-level inconsistency surfaces elsewhere
            sets[g] = [op for op in C2.PRIORITY if op in surv]
        gl = sorted(sets)
        if C2._best_distinct([sets[g] for g in gl], set(), self.dup_ok) is not None:
            return None
        # name a forced duplicate pair for the rendered diagnosis
        for i, g1 in enumerate(gl):
            for g2 in gl[i + 1:]:
                if len(sets[g1]) == 1 and len(sets[g2]) == 1 and sets[g1] == sets[g2]:
                    return (g1, g2, OP_WORD[sets[g1][0]])
        return (gl[0], gl[-1], OP_WORD[sets[gl[0]][0]])

    def policy_branch(self, need, budget, depth=0):
        """deterministic DFS. True = solved (state adopted), False = level exhausted
        (sound refutation of the current pins). Raises Budget when out of nodes."""
        self.propagate()
        if self.done(need):
            if len(self.cands) > 1:
                bad = self._distinct_fail()
                if bad is not None:
                    self.steps.append(('DISTFAIL', bad))
                    return False
            return True
        if depth >= POLICY_DEPTH_CAP:
            raise Budget('depth')
        l = self.pick_letter(need)
        dom = sorted(self.dom[l])
        self.steps.append(('NEXT', l, list(dom)))
        tried = []
        for d in dom:
            budget[0] -= 1
            if budget[0] <= 0:
                raise Budget('nodes')
            sub = self.clone()
            try:
                sub.set_dom(l, {d}, f"try {l} = {d}")
                sub.propagate()
            except Contra as e:
                tried.append(d)
                self.steps.append(('PTRY', l, d, e, sub.steps, list(tried)))
                continue
            if sub.policy_branch(need, budget, depth + 1):
                self.steps.append(('PENTER', l, d, list(tried), sub.steps))
                self.dom = sub.dom
                self.cands = sub.cands
                self.nbranch += 1
                return True
            tried.append(d)
            self.steps.append(('PDEEP', l, d, sub.steps, list(tried)))
        self.steps.append(('PEXH', l, list(tried)))
        return False

def _trset(l, tried):
    return f"tried {l}: " + ",".join(map(str, tried))

def _opexh_txt(opexh):
    """truthful op-level exhaustion: every candidate op tried with real arithmetic."""
    r, a, b, tries, exp = opexh
    bits = []
    for op, v in tries[:6]:
        got = _got_str(op, a, b) if v is not None else None
        bits.append(f"{OP_WORD[op]} {got if got is not None else '?'}")
    more = f" (+{len(tries)-6} more)" if len(tries) > 6 else ""
    return (f"EQ{r.k}: {a} {r.gl} {b} must give {exp}: " + ", ".join(bits) + more
            + " - none fits -> the op is not the problem, a digit pin is wrong")

def _contra_txt(e):
    opexh = getattr(e, 'opexh', None)
    if opexh is not None:
        return _opexh_txt(opexh)
    return e.reason

def render_steps4(steps, L, ep=None):
    """render engine + policy steps (r4): LOCK grouping as r3, plus the policy
    search events with tried-set bookkeeping and level diagnosis."""
    i = 0
    while i < len(steps):
        st = steps[i]
        k = st[0]
        if k == 'LOCK':
            _, l, d, why = st
            if why.startswith('try ') or why.startswith('only consistent choice'):
                i += 1
                continue   # the policy try/lock line renders this pin
            m = _ALLDIFF_WHY.match(why) if ep else None
            if (m and ep.get('dist_left', 0) > 0 and ep['rng'].random() < ep['p']):
                partner, dp = m.group(1), int(m.group(2))
                lo, hi = sorted((d, dp))
                ep['dist_left'] -= 1
                ep['n_collide'] = ep.get('n_collide', 0) + 1
                L.append(f"{l} in {{{lo},{hi}}}: try {l} = {dp}: distinct check:"
                         f" {partner}={dp} and {l}={dp} collide. conflict ->"
                         f" backtrack -> {l} = {d}.")
                i += 1
                continue
            grp = [(l, d)]
            while (i + 1 < len(steps) and steps[i + 1][0] == 'LOCK'
                   and steps[i + 1][3] == why):
                grp.append((steps[i + 1][1], steps[i + 1][2]))
                i += 1
            L.append(f"{why} -> " + ", ".join(f"{x} = {y}" for x, y in grp) + ".")
        elif k == 'NEXT':
            _, l, dom = st
            L.append(f"next: {l}, smallest domain {{{','.join(map(str, dom))}}}."
                     f" try digits ascending.")
        elif k == 'PTRY':
            _, l, d, e, substeps, tried = st
            pre = [s for s in substeps if s[0] == 'LOCK'
                   and not (s[1] == l and s[3].startswith('try '))][:2]
            ptxt = "; ".join(f"{s[3]} -> {s[1]} = {s[2]}" for s in pre)
            line = f"try {l} = {d}: " + (ptxt + "; " if ptxt else "")
            line += _contra_txt(e)
            line += f" -> no, un-pin {l} = {d}. {_trset(l, tried)}."
            L.append(line)
        elif k == 'PDEEP':
            _, l, d, substeps, tried = st
            L.append(f"try {l} = {d}: no conflict yet, go deeper.")
            render_steps4(substeps, L, None)   # off the truth path: no episode hooks
            L.append(f"dead end below -> the {l} = {d} pin is wrong, backtrack."
                     f" un-pin {l} = {d}. {_trset(l, tried)}.")
        elif k == 'PENTER':
            _, l, d, tried, substeps = st
            if tried:
                L.append(f"try {l} = {d}: consistent ({_trset(l, tried)} all failed,"
                         f" {d} holds). lock {l} = {d}.")
            else:
                L.append(f"try {l} = {d}: consistent. lock {l} = {d}.")
            render_steps4(substeps, L, ep)
        elif k == 'PEXH':
            _, l, tried = st
            L.append(f"all digits for {l} fail ({_trset(l, tried)}) ->"
                     f" {l} is not the problem, a pin one level up is wrong.")
        elif k == 'DISTFAIL':
            _, (g1, g2, opw) = st
            L.append(f"every letter is pinned, but {g1} and {g2} would both have to"
                     f" be {opw} - the rules in one puzzle are all different ->"
                     f" this assignment is wrong, backtrack.")
        else:
            _render_step_other(st, L, ep)
        i += 1

def infer_sign_end(eqs, opchars):
    """sign-glyph side policy, derived from the prompt examples (no truth)."""
    pre = suf = 0
    for L, R in eqs:
        opch = L[2]
        if len(R) > 1 and R[0] == opch:
            pre += 1
        elif len(R) > 1 and R[-1] == opch:
            suf += 1
    if suf > pre:
        return 'suf'
    return 'pre'

def pin_episodes_r4(byg, conc, narrative_live, pins, survs, M, let, rev, gone_r,
                    cap=PIN_WRONG_CAP, have_no=0, dbl_cands=None):
    """r4 rule-pin episodes: ops tried by measured prior with numbered tries +
    cumulative tried-set; candidates pinned to OTHER glyphs are excluded by the
    distinct-rules constraint (rendered); genuinely wrong candidates fail with
    real numbers; the true rule locks with a real match.
    dbl_cands (r5): optional {glyph: [ops]} for the injected double-check on
    fully-forced traces (canonical rows must double-check INSIDE the family)."""
    lines = []
    seen_dec = set()
    n_no = 0

    def dec_line(r):
        if r.k in seen_dec:
            return
        seen_dec.add(r.k)
        Rr, opch, side, sign, dd, exp = _rhs_parts(r, M, let, rev)
        note = f" ({side} {opch} = minus)" if sign else ""
        dec = " ".join(str(d) for d in dd)
        revw = "reversed " if rev else ""
        lines.append(f"EQ{r.k} RHS {Rr}{note} -> {dec} -> {revw}{exp}.")

    def try_line(g, op, r, tried):
        a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
        got = _got_str(op, a, b)
        exp = _rhs_parts(r, M, let, rev)[5]
        if got == exp:
            lines.append(f"Try {g} = {OP_WORD[op]} on EQ{r.k}: {a} {OP_WORD[op]} {b} ="
                         f" {got}. {got} vs {exp}. match -> lock {g} = {OP_WORD[op]}.")
            return 'match'
        tried.append(OP_WORD[op])
        lines.append(f"Try {g} = {OP_WORD[op]} on EQ{r.k}: {a} {OP_WORD[op]} {b} ="
                     f" {got}. {got} vs {exp}. no -> mistake, backtrack."
                     f" tried {g}: {', '.join(tried)}.")
        return 'no'

    def wrong_row(g, op):
        for r in byg[g]:
            a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
            got = _got_str(op, a, b)
            if got is None:
                continue
            exp = _rhs_parts(r, M, let, rev)[5]
            if got != exp and int(got) != int(exp):
                return r
        return None

    taken = {}
    for g2, op2 in pins.items():
        taken.setdefault(op2, g2)
    for g in sorted(narrative_live):
        if g in conc or n_no >= cap:
            continue
        order = sorted(narrative_live[g],
                       key=lambda o: -C2.MEASURED_OP_COUNTS.get(o, 1))
        # distinct-rules exclusions (rendered, then dropped from the try order)
        excl = [o for o in order if o != pins[g] and taken.get(o) not in (None, g)]
        order = [o for o in order if o not in excl]
        episode = []
        for op in order:
            if op == pins[g]:
                break
            if op in survs.get(g, ()):
                continue                  # tie op (fits everywhere): tie lines handle it
            if len(episode) >= 2 or n_no + len(episode) >= cap:
                continue
            r = wrong_row(g, op)
            if r is not None:
                episode.append((op, r))
        if not episode:
            continue
        head = (f"{g} could still be " + " / ".join(OP_WORD[o] for o in order[:4])
                + ". try by prior, most common first.")
        if excl:
            head += (" rules are distinct: " + ", ".join(
                f"{OP_WORD[o]} is taken by {taken[o]}" for o in excl[:2]) + ".")
        lines.append(head)
        tried = []
        for op, r in episode:
            dec_line(r)
            try_line(g, op, r, tried)
            n_no += 1
        r0 = episode[-1][1]
        dec_line(r0)
        try_line(g, pins[g], r0, tried)
    if n_no + have_no == 0:
        # fully forced search: inject ONE failed double-check with real numbers
        for g in sorted(narrative_live):
            if g in conc:
                continue
            wcands = (dbl_cands.get(g, ()) if dbl_cands is not None
                      else ('add', 'mul', 'sub_signed'))
            for w in wcands:
                if w == pins[g] or w in survs.get(g, ()) or w in gone_r.get(g, set()):
                    continue
                r = wrong_row(g, w)
                if r is None:
                    continue
                lines.append(f"Double-check {g} against the most common rules"
                             f" before locking.")
                dec_line(r)
                tried = []
                try_line(g, w, r, tried)
                try_line(g, pins[g], r, tried)
                n_no += 1
                return lines, n_no
    return lines, n_no

def render_trace_r4(prompt, answer, ep_rng=None, meta=None):
    """policy-driven r4 renderer. Executes the deterministic policy with NO truth
    guidance; `answer` (gold) is only compared against the final boxed string.
    meta (optional, data-gen only): solver/gold config for cross-checks.
    returns (cot_text, tier, info) or (None, tier, reason)."""
    rng = ep_rng if ep_rng is not None else random.Random(12345)
    if C2.parse(prompt) is None:
        return None, 0, "unparseable"
    rows, let, olet, eqs, qL = letterize(prompt, False)
    qop_ch = qL[2]
    qg = olet[qop_ch]
    inv_let = {v: k for k, v in let.items()}
    is_guess = all(r.gl != qg for r in rows)
    info = {'n_exh': 0, 'n_deep': 0, 'n_opexh': 0, 'mode_switch': 0}

    L = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("")
    L.append("First, assign letters to each symbol:")
    for s, a in let.items():
        L.append(f"{s} -> {a}")
    L.append("Operators:")
    for ch, a in olet.items():
        L.append(f"{ch} -> {a}")
    L.append("")
    L.append("Examples in letter form (each left side: two digit-symbols, operator, two digit-symbols):")
    for k, (Lr, Rr) in enumerate(eqs, 1):
        opch = Lr[2]
        mag = Rr; sign_note = ''
        if len(Rr) > 1 and Rr[0] == opch:
            mag = Rr[1:]; sign_note = 'minus sign in front'
        elif len(Rr) > 1 and Rr[-1] == opch:
            mag = Rr[:-1]; sign_note = 'minus sign at the end'
        ll = ''.join(let[c] for c in (Lr[0], Lr[1])) + f" {olet[opch]} " + \
             ''.join(let[c] for c in (Lr[3], Lr[4]))
        rl = ''.join(let[c] for c in mag)
        L.append(f"EQ{k}: {Lr} = {Rr} -> {ll} = {rl}" + (f" ({sign_note})" if sign_note else ""))
    qlets = f"{let[qL[0]]}{let[qL[1]]} {qg} {let[qL[3]]}{let[qL[4]]}"
    L.append(f"Query: {qL} -> {qlets}")
    L.append("")
    byg, cands, conc = build_cands(rows)
    L.append("Classify each operator from cues:")
    for g, rs in byg.items():
        L.append(fam_cue(g, rs, cands[g]))
    if is_guess:
        L.append(f"{qg}: appears only in the query -> rule unknown, decide at the end.")
    L.append("")

    # ---- Tier 1 short-circuit (structural concat; mode-free)
    if not is_guess and qg in conc:
        qtrue = conc[qg]
        tier = 1
        L.append(f"The question operator is {qg}, which is {OP_WORD[qtrue]}.")
        chk = None
        for r in byg.get(qg, []):
            Lr, Rr = r.raw
            fwd = Lr[0] + Lr[1] + Lr[3] + Lr[4]
            rv = Lr[3] + Lr[4] + Lr[0] + Lr[1]
            if fwd != rv:
                chk = (r.k, Rr, fwd, rv)
                break
        if chk is None:
            return None, tier, "concat: no order-discriminating example"
        k_, Rr_, fwd_, rv_ = chk
        wrongs, rights = (rv_, fwd_) if qtrue == 'concat_fwd' else (fwd_, rv_)
        wname, rname = (('swapped', 'in order') if qtrue == 'concat_fwd'
                        else ('in order', 'swapped'))
        if rights != Rr_:
            return None, tier, "concat order check inconsistent"
        L.append(f"Check the order on EQ{k_}: {wname} would give {wrongs}."
                 f" {wrongs} vs {Rr_}. no -> mistake, backtrack; {rname} gives"
                 f" {rights}. {rights} vs {Rr_}. match.")
        A2, B2 = qlets.split()[0], qlets.split()[2]
        s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
        if qtrue == 'concat_fwd':
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {A2} || {B2} = {A2}{B2}")
            syms = s1 + s2
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s1} then {s2} -> {syms}")
        else:
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {B2} || {A2} = {B2}{A2}")
            syms = s2 + s1
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s2} then {s1} -> {syms}")
        L.append("")
        L.append(f"\\boxed{{{syms}}}")
        if syms != answer:
            return None, tier, f"concat mismatch {syms!r} != {answer!r}"
        return "\n".join(L), tier, info

    # ---- assumption-revision policy, joint-prior order (measured):
    # standard+plain -> little-endian+plain -> standard+offsets -> little-endian+offsets
    sign_end = infer_sign_end(eqs, {L0[2] for L0, _ in eqs} | {qop_ch})
    eng = None
    rev = False
    ep = {'rng': rng, 'p': P_DIST_EP, 'dist_left': 1, 'n_collide': 0,
          'wrong_left': PIN_WRONG_CAP, 'n_elim2': 0, 'let': let, 'rev': False}
    PASSES = [(False, 'plain'), (True, 'plain'), (False, 'full'), (True, 'full')]
    BANNER = [
        "Assume standard digit order (left digit = tens) and the plain rules"
        " first (a+b, a*b, a-b, b-a, |a-b|, -|a-b|); offsets only if these fail.",
        "No assignment works: standard reading with plain rules is impossible."
        " Revise one assumption at a time, digit order first -> the puzzle may"
        " write numbers LITTLE-ENDIAN (digits reversed). Re-read every number"
        " MSB-first by reversing:",
        "Little-endian with plain rules fails too -> the rules are not all plain."
        " Allow the +1/-1/+2/-2 offset variants (and mod), back to standard"
        " digit order.",
        "Standard reading still fails with offsets -> re-read LITTLE-ENDIAN"
        " (offsets allowed):",
    ]
    for pidx, (mode_try, vocab_tier) in enumerate(PASSES):
        vocab = PLAIN_VOCAB if vocab_tier == 'plain' else RVOCAB
        mrows, mlet, molet, _, _ = letterize(prompt, mode_try)
        mbyg, mcands, mconc = build_cands(mrows, vocab=vocab)
        vrows = [r for r in mrows if r.gl not in mconc]
        if not vrows:
            return None, 2, "no value rows but non-concat query"
        if any(not mcands[g] for g in mcands):
            # a glyph has no candidate at all in this tier: cheap renderable skip
            g0 = next(g for g in mcands if not mcands[g])
            L.append(BANNER[pidx])
            L.append(f"{g0}: no plain rule fits its result shapes -> this pass"
                     f" cannot work, move on.")
            info['n_exh'] += 1
            if pidx == len(PASSES) - 1:
                return None, 2, "all policy passes refuted"
            continue
        need = set()
        for r in vrows:
            need |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
        e2 = PolicyP2(vrows, {g: c for g, c in mcands.items() if g not in mconc},
                      truth=None)
        ep['rev'] = mode_try
        # NOTE: e2.ep stays None during the SEARCH (no ELIM2/collide emission from
        # engine state that might not survive); episode hooks apply at RENDER time
        # and only on the successful pass / truth path.
        L.append(BANNER[pidx])
        if mode_try:
            for r in mrows:
                L.append(f"EQ{r.k}: {r.lhs()} = {r.rstr()}")
            L.append(f"Query reads: {let[qL[1]]}{let[qL[0]]} {qg} {let[qL[4]]}{let[qL[3]]}")
            L.append("")
        L.append("Deduce the digits:")
        budget = [POLICY_NODE_CAP]
        try:
            ok = e2.policy_branch(need, budget)
        except Budget as b:
            return None, 2, f"policy budget ({b}) pass {pidx} rev={mode_try} {vocab_tier}"
        except Contra as e:
            ok = False
            e2.steps.append(('ROOTCONTRA', e))
        if ok:
            rows, conc = mrows, mconc
            cands = mcands
            byg = mbyg
            rev = mode_try
            eng = e2
            info['mode_switch'] = 1 if rev else 0
            info['pass'] = pidx
            render_steps4(e2.steps, L, ep=ep)
            break
        # render the sound refutation of this pass (exhaustion or contradiction)
        st = e2.steps
        if st and st[-1][0] == 'ROOTCONTRA':
            render_steps4(st[:-1][:4], L, ep=None)
            L.append(f"Contradiction: {st[-1][1].reason}.")
        else:
            if st and st[-1][0] == 'PEXH':
                render_steps4(st[:-1], L, ep=None)
                _, l_, tried_ = st[-1]
                L.append(f"all digits for {l_} fail ({_trset(l_, tried_)}) -> no pin"
                         f" above {l_} to revise inside this reading -> an assumption"
                         f" is wrong.")
            else:
                render_steps4(st, L, ep=None)
        info['n_exh'] += 1
        if pidx == len(PASSES) - 1:
            return None, 2, "all policy passes refuted"
    if eng is None:
        return None, 2, "policy: no pass solved"
    # episode counters from the step tree
    def _count(steps):
        for s in steps:
            if s[0] == 'PTRY':
                if getattr(s[3], 'opexh', None) is not None:
                    info['n_opexh'] += 1
            elif s[0] == 'PDEEP':
                info['n_deep'] += 1
                _count(s[3])
            elif s[0] == 'PENTER':
                _count(s[4])
    _count(eng.steps)
    M = eng.assigned()
    if meta is not None:
        tm = {let[s]: d for s, d in meta['mapping'].items()
              if d is not None and s in let}
        for l, d in tm.items():
            if l in M and M[l] != d:
                return None, 2, f"policy map disagrees with solver at {l}"
        if bool(meta.get('rev')) != rev:
            return None, 2, "policy mode disagrees with solver"
    # ---- op pins: surviving ops at the final map; distinct assignment by prior
    vrows = [r for r in rows if r.gl not in conc]
    survs = {}
    for g in eng.cands:
        surv = []
        for op in eng.cands[g]:
            okop = True
            for r in [r for r in vrows if r.gl == g]:
                a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
                fv = eng._fit(op, r, a, b)
                if fv is None:
                    okop = False; break
                v, digs = fv
                if any(M[r.res[k]] != digs[k] for k in range(r.rl)):
                    okop = False; break
            if okop:
                surv.append(op)
        if not surv:
            return None, 2, f"no op fits glyph {g} at final map"
        survs[g] = [op for op in C2.PRIORITY if op in surv]
    # concat glyphs: structural patterns (can be both orders on degenerate rows)
    for g, rs in byg.items():
        if g in conc:
            pats = set.intersection(*[set(r.conc) for r in rs])
            survs[g] = [op for op in C2.PRIORITY if op in pats]
    glyph_order = sorted(survs)
    ba = C2._best_distinct([survs[g] for g in glyph_order], set())
    if ba is not None:
        pins = dict(zip(glyph_order, ba[1]))
    else:
        # concat ties are exempt: degenerate examples can make two concat glyphs
        # structurally identical; the VALUE rules still must be distinct.
        vord = [g for g in glyph_order if g not in conc]
        ba2 = C2._best_distinct([survs[g] for g in vord], set())
        if ba2 is None:
            return None, 2, "no distinct op assignment"
        pins = dict(zip(vord, ba2[1]))
        for g in glyph_order:
            if g in conc:
                pins[g] = survs[g][0]
    tie_lines = []
    taken = {op: g for g, op in pins.items()}
    for g in glyph_order:
        if g in conc or len(survs[g]) <= 1:
            continue
        top = survs[g][0]
        if pins[g] == top:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in survs[g])} on"
                             f" every example; keep the most common: {g} = {OP_WORD[top]}.")
        else:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in survs[g])};"
                             f" {OP_WORD[top]} is taken by {taken.get(top, '?')} and the"
                             f" rules are distinct -> {g} = {OP_WORD[pins[g]]}.")
    tier = 1 if (not is_guess and pins.get(qg) in ('concat_fwd', 'concat_rev')) else \
           (3 if (not is_guess and pins.get(qg) in MUL_FAM | SMALL_FAM
                  | {'a2_plus_b', 'a_plus_b2', 'lcm', 'gcd'}) else 2)
    L.append("")
    L.append("Locked map: " + ", ".join(f"{l}={d}" for l, d in sorted(M.items())))
    if len(set(M.values())) != len(M):
        return None, tier, "injectivity violated in locked map"
    L.append("Distinct check: " + " ".join(str(d) for _, d in sorted(M.items()))
             + " - all different. ok")
    # ---- rule-pin episodes (truthful failed tries with tried-set state)
    gone_r = {}
    narrative_live = {g: list(c) for g, c in cands.items() if g not in conc}
    for st in eng.steps:
        if st[0] == 'ELIM':
            _, r0, _a0, _b0, tries0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(
                op for op, _v in tries0 if op not in surv0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
        elif st[0] == 'ELIM2':
            _, r0, _a0, _b0, vals0, opk0, _M0 = st
            gone_r.setdefault(r0.gl, set()).update(op for op, _v in vals0)
            narrative_live[r0.gl] = [opk0]
        elif st[0] == 'OPSHRINK':
            _, r0, gone0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(gone0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
    byg2 = {}
    for r in vrows:
        byg2.setdefault(r.gl, []).append(r)
    pin_lines, n_no = pin_episodes_r4(byg2, conc, narrative_live, pins,
                                      {g: set(s) for g, s in survs.items()},
                                      M, let, rev, gone_r,
                                      cap=ep['wrong_left'], have_no=ep['n_elim2'])
    if n_no + ep['n_elim2'] == 0:
        return None, tier, "no truthful negative episode renderable"
    L.extend(pin_lines)
    L.extend(tie_lines)
    L.append("Locked rules: " + ", ".join(f"{g} = {OP_WORD[pins[g]]}" for g in sorted(pins)))
    L.append("")
    # ---- verify: decode the RHS through the locked map FIRST, then compute
    L.append("Verify every example (decode RHS first, then compute):")
    for r in rows:
        op = pins[r.gl]
        Lr, Rr = r.raw
        opch_row = Lr[2]
        if op in ('concat_fwd', 'concat_rev'):
            pat = (Lr[0] + Lr[1] + Lr[3] + Lr[4] if op == 'concat_fwd'
                   else Lr[3] + Lr[4] + Lr[0] + Lr[1])
            if Rr != pat:
                return None, tier, f"concat verify failed EQ{r.k}"
            order = 'in order' if op == 'concat_fwd' else 'swapped'
            L.append(f"EQ{r.k}: RHS {Rr} = operand symbols {order}"
                     f" ({Lr[0]}{Lr[1]} and {Lr[3]}{Lr[4]}). match")
            continue
        sign = False; side = None
        mag = Rr
        if len(Rr) > 1 and Rr[0] == opch_row:
            sign = True; side = 'leading'; mag = Rr[1:]
        elif len(Rr) > 1 and Rr[-1] == opch_row:
            sign = True; side = 'trailing'; mag = Rr[:-1]
        dd = [M[let[c]] for c in mag]
        msb = dd[::-1] if rev else dd
        exp_num = int(''.join(map(str, msb)))
        exp_str = ('-' if sign else '') + str(exp_num)
        a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
        v = C2.OPS[op](a, b)
        if v is None:
            return None, tier, f"verify undefined EQ{r.k}"
        got = ('-' if (v < 0 or op in C2.NEGPRE) else '') + str(abs(v))
        if got != exp_str:
            return None, tier, f"verify failed EQ{r.k}: {got} != {exp_str}"
        note = f" ({side} {opch_row} = minus)" if sign else ""
        dec = " ".join(str(d) for d in dd)
        revw = "reversed " if rev else ""
        L.append(f"EQ{r.k}: RHS {Rr}{note} -> {dec} -> {revw}{exp_str};"
                 f" compute {a} {OP_WORD[op]} {b} = {got}. match")
    L.append("")
    # ---- apply to the query
    qa_, qb_, qc_, qd_ = let[qL[0]], let[qL[1]], let[qL[3]], let[qL[4]]
    missing = [x for x in (qa_, qb_, qc_, qd_) if x not in M]
    if missing:
        free_digs = sorted(set(range(10)) - set(M.values()))
        ml = sorted(set(missing))
        if len(ml) == 1 and len(free_digs) == 1:
            M[ml[0]] = free_digs[0]
            L.append(f"{ml[0]} appears only in the query; the one unused digit is"
                     f" {free_digs[0]}, so {ml[0]} = {free_digs[0]}.")
        elif len(ml) <= len(free_digs) and len(ml) <= 3:
            # deterministic bet: no example constrains these letters; assign the
            # unused digits in order (policy line; correctness gated on gold)
            for l, d in zip(ml, free_digs):
                M[l] = d
            L.append("Letters " + ", ".join(ml) + " appear only in the query;"
                     f" unused digits are {{{','.join(map(str, free_digs))}}};"
                     " no example constrains them - take them in order: "
                     + ", ".join(f"{l} = {M[l]}" for l in ml) + ".")
        else:
            return None, tier, "query letters not deducible by policy"
    if is_guess:
        used = [OP_WORD[o] for o in pins.values()]
        qapply = next((o for o in C2.PRIORITY
                       if o not in pins.values() and o in RVOCAB + ['concat_fwd']), None)
        if qapply is None:
            return None, tier, "guess: no candidate op"
        L.append(f"The query operator {qg} never appears in the examples. The rules"
                 f" in one puzzle are all different, so {qg} is not"
                 f" {', '.join(used)}. The most common remaining rule is"
                 f" {OP_WORD[qapply]} -> bet {qg} = {OP_WORD[qapply]}.")
        if qapply in ('concat_fwd', 'concat_rev'):
            s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
            syms = s1 + s2 if qapply == 'concat_fwd' else s2 + s1
            L.append(f"Applying concatenation: take the query operand symbols:"
                     f" -> {syms}")
            L.append("")
            L.append(f"\\boxed{{{syms}}}")
            if syms != answer:
                return None, tier, "guess policy miss"
            return "\n".join(L), tier, info
    else:
        qapply = pins[qg]
    if rev:
        A = M[qb_] * 10 + M[qa_]; B = M[qd_] * 10 + M[qc_]
        L.append(f"Apply to the query (little-endian): {qb_}{qa_} {qg} {qd_}{qc_}"
                 f" = {A} {OP_WORD[qapply]} {B}")
    else:
        A = M[qa_] * 10 + M[qb_]; B = M[qc_] * 10 + M[qd_]
        L.append(f"Apply to the query: {qa_}{qb_} {qg} {qc_}{qd_} = {A} {OP_WORD[qapply]} {B}")
    v = C2.OPS[qapply](A, B)
    if v is None:
        return None, tier, "query value undefined"
    sign_needed = (qapply in C2.NEGPRE) or (qapply in C2.SIGNED and v < 0)
    if v < 0 and not sign_needed:
        return None, tier, "negative query value for unsigned op"
    mag = abs(v)
    digs = [int(c) for c in str(mag)]
    inv_d = {d: l for l, d in M.items()}
    miss_d = sorted({d for d in digs if d not in inv_d})
    if miss_d:
        unas = sorted(set(let.values()) - set(M))
        if len(miss_d) == 1 and len(unas) == 1:
            M[unas[0]] = miss_d[0]
            inv_d[miss_d[0]] = unas[0]
            L.append(f"Digit {miss_d[0]} never appeared in a value example; the only"
                     f" letter without a digit is {unas[0]} -> {unas[0]} = {miss_d[0]}.")
        elif len(miss_d) <= len(unas) and len(miss_d) <= 2:
            for d, l in zip(miss_d, unas):
                M[l] = d
                inv_d[d] = l
            L.append("Digits " + ",".join(map(str, miss_d)) + " never appeared in a"
                     " value example; letters without digits: " + ",".join(unas)
                     + " - take them in order: "
                     + ", ".join(f"{l} = {d}" for d, l in zip(miss_d, unas)) + ".")
        else:
            return None, tier, f"answer digit unmapped: {mag}"
    dig2sym = {d: inv_let[inv_d[d]] for d in set(digs)}
    L.append(f"= {'-' if (v < 0 or qapply in C2.NEGPRE) else ''}{mag}")
    disp = digs[::-1] if rev else digs
    pairs = " ".join(f"{d}={dig2sym[d]}" for d in disp)
    core = ''.join(dig2sym[d] for d in disp)
    if rev:
        L.append(f"Little-endian, write digits reversed: {mag} ->"
                 f" {''.join(map(str, disp))}; decode digit by digit: {pairs} -> {core}")
    else:
        L.append(f"Decode digit by digit: {pairs} -> {core}")
    syms = core
    if sign_needed:
        if sign_end == 'suf':
            syms = core + qop_ch
            L.append(f"Negative -> append the operator glyph {qop_ch} as the sign"
                     f" (as in the examples) -> {syms}")
        else:
            syms = qop_ch + core
            L.append(f"Negative -> prefix the operator glyph {qop_ch} as the sign"
                     f" -> {syms}")
    L.append("")
    L.append(f"\\boxed{{{syms}}}")
    if syms != answer:
        return None, tier, f"final mismatch {syms!r} != {answer!r}"
    info['branches'] = eng.nbranch
    info['n_no'] = n_no + ep['n_elim2']
    info['n_collide'] = ep['n_collide']
    return "\n".join(L), tier, info

# ---------------------------------------------------------------- lint
def official_extract(text):
    """replica of the competition metric's extract_final_answer boxed branch."""
    import re
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    return None

WORD_OP = {v: k for k, v in OP_WORD.items()}
import re as _re
# ---- r3 truthful-conditional grammars
_CMP_RE = _re.compile(r"(\S+) vs (\S+)\. (no|match)\b")
_COLLIDE_RE = _re.compile(r"([A-J])=(\d) and ([A-J])=(\d) collide\. conflict")
_PINDEC_RE = _re.compile(
    r"^EQ(\d+) RHS (\S+)((?: \((?:leading|trailing) \S = minus\))?) -> "
    r"((?:\d )*\d) -> (reversed )?(-?\d+)\.$")
_PINTRY_RE = _re.compile(
    r"^Try ([xyzw]) = (.+?) on EQ(\d+): (\d+) (.+?) (\d+) = (-?\d+)\. "
    r"(-?\d+) vs (-?\d+)\. (no|match)")
_ORDER_RE = _re.compile(
    r"^Check the order on EQ(\d+): (swapped|in order) would give (\S+)\.")

def _cmp_eq(x, y):
    try:
        return int(x) == int(y)
    except ValueError:
        return x == y

def _true_rhs(Lp, Rp, mapping, rev):
    """(decoded digits raw-order, signed expected int, sign side) from the SOLVER map,
    or None if a symbol is unmapped."""
    opch = Lp[2]
    sign = False; side = None; mag = Rp
    if len(Rp) > 1 and Rp[0] == opch:
        sign, side, mag = True, 'leading', Rp[1:]
    elif len(Rp) > 1 and Rp[-1] == opch:
        sign, side, mag = True, 'trailing', Rp[:-1]
    try:
        dd = [mapping[c] for c in mag]
    except KeyError:
        return None
    if any(t is None for t in dd):
        return None
    msb = dd[::-1] if rev else dd
    return dd, int(''.join(map(str, msb))) * (-1 if sign else 1), side

_VERIFY_RE = _re.compile(
    r"^EQ(\d+): RHS (\S+)((?: \((?:leading|trailing) (?:\S) = minus\))?) -> "
    r"((?:\d )*\d) -> (reversed )?(-?\d+); compute (\d+) (.+?) (\d+) = (-?\d+)\. match$")
_CONCAT_RE = _re.compile(
    r"^EQ(\d+): RHS (\S+) = operand symbols (in order|swapped) \((\S+) and (\S+)\)\. match$")
_DECODE_RE = _re.compile(r"[Dd]ecode digit by digit: ((?:\d=\S )*\d=\S) -> (\S+)\s*$")
_MAP_RE = _re.compile(r"^Locked map: (.*)$", _re.M)
_DIST_RE = _re.compile(r"^Distinct check: ((?:\d )*\d) - all different\. ok$", _re.M)

def lint(cot, answer, prompt=None, meta=None, skip_eq=None):
    """machine-check an emitted trace. Returns a reason string (drop) or None (pass).
    With prompt+meta: verify lines are re-derived from the SOLVER's mapping (so an
    echoed 'expected' can never pass), injectivity + distinct line are checked, and
    the final per-digit decode must equal the boxed answer which must equal gold.
    skip_eq (r6 bail traces only): this EQ index is the diagnosed-corrupt example;
    it is exempt from the verify-block completeness requirement (lint_r6 validates
    its truthful mismatch line separately)."""
    if any(ord(c) > 126 or (ord(c) < 32 and c != '\n') for c in cot):
        bad = sorted({c for c in cot if ord(c) > 126})
        return f"non-ascii: {bad}"
    nb = cot.count('\\boxed{')
    if nb != 1:
        return f"boxed count {nb}"
    got = official_extract(cot)
    if got != answer:
        return f"official-extract {got!r} != answer {answer!r}"
    # the trace must end with the boxed answer (nothing after the closing brace)
    if not cot.rstrip().endswith(f"\\boxed{{{answer}}}"):
        return "boxed not at end of trace"
    # ---- r3 TRUTHFULNESS: every comparison terminal must be CORRECT for the values
    # on its line. A line whose values differ MUST end "no"; equal values MUST end
    # "match". Collide lines must show equal digits on distinct letters.
    lines = cot.splitlines()
    n_no = 0
    for ln in lines:
        for m in _CMP_RE.finditer(ln):
            x, y, t = m.group(1), m.group(2), m.group(3)
            if _cmp_eq(x, y) != (t == 'match'):
                return f"untruthful comparator: {ln.strip()[:64]}"
            if t == 'no':
                n_no += 1
        for m in _COLLIDE_RE.finditer(ln):
            l1, d1, l2, d2 = m.groups()
            if l1 == l2 or d1 != d2:
                return f"untruthful collide: {ln.strip()[:64]}"
    if n_no == 0:
        return "no truthful negative (failure episode missing)"
    if prompt is None or meta is None:
        return None
    mapping = meta['mapping']; rev = meta['rev']
    # ---- injectivity + distinct line (whenever a map was locked)
    mm = _MAP_RE.search(cot)
    if mm:
        pairs = _re.findall(r"([A-J])=(\d)", mm.group(1))
        mdigs = [int(d) for _, d in pairs]
        if len(set(mdigs)) != len(mdigs):
            return "locked map not injective"
        dm = _DIST_RE.search(cot)
        if not dm:
            return "distinct-check line missing"
        if [int(x) for x in dm.group(1).split()] != mdigs:
            return "distinct-check digits != locked map digits"
        # map must agree with the solver's mapping (letter relabel is by appearance,
        # checked indirectly: every verify decode below uses the SOLVER mapping)
        # r3: collide episodes must be consistent with the FINAL locked map — the
        # holder (first letter) really holds that digit; the hypothesized letter
        # (second) really resolves to something else.
        mpd = {l: int(d) for l, d in pairs}
        for ln in lines:
            for cm in _COLLIDE_RE.finditer(ln):
                l1, d1, l2, d2 = cm.group(1), int(cm.group(2)), cm.group(3), int(cm.group(4))
                if mpd.get(l1) != d1:
                    return f"collide holder {l1}={d1} not in final map"
                if l2 not in mpd or mpd[l2] == d2:
                    return f"collide letter {l2} not actually excluded from {d2}"
    has_verify = 'Verify every example' in cot
    if mm and not has_verify:
        return "locked map without verify block"
    pr = C2.parse(prompt)
    if pr is None:
        return "prompt unparseable"
    eqs, qL = pr
    # ---- r3: pin-episode lines re-derived from the solver mapping
    pdec = {}
    for ln in lines:
        m = _PINDEC_RE.match(ln)
        if not m:
            continue
        k = int(m.group(1))
        if not (1 <= k <= len(eqs)):
            return f"pin decode: bad EQ index {k}"
        Lp, Rp = eqs[k - 1]
        if m.group(2) != Rp:
            return f"pin decode EQ{k}: rendered RHS != prompt RHS"
        tr = _true_rhs(Lp, Rp, mapping, rev)
        if tr is None:
            return f"pin decode EQ{k}: RHS symbol unmapped"
        dd, expv, side = tr
        note = m.group(3)
        if bool(note) != bool(side) or (side and side not in note):
            return f"pin decode EQ{k}: sign note wrong"
        if [int(x) for x in m.group(4).split()] != dd:
            return f"pin decode EQ{k}: decoded digits != true RHS decode"
        if bool(m.group(5)) != bool(rev):
            return f"pin decode EQ{k}: reversed flag mismatch"
        if int(m.group(6)) != expv:
            return f"pin decode EQ{k}: stated value != decoded value"
        pdec[k] = expv
    for ln in lines:
        m = _PINTRY_RE.match(ln)
        if not m:
            continue
        g_, opw1, k, a, opw2, b, gotv, x, y, term = m.groups()
        k = int(k); a = int(a); b = int(b); gotv = int(gotv)
        if opw1 != opw2:
            return f"pin try EQ{k}: op words differ"
        op = WORD_OP.get(opw1)
        if op is None:
            return f"pin try EQ{k}: unknown op {opw1!r}"
        if not (1 <= k <= len(eqs)):
            return f"pin try: bad EQ index {k}"
        if k not in pdec:
            return f"pin try EQ{k}: RHS not decoded first"
        v = C2.OPS[op](a, b)
        if v is None:
            return f"pin try EQ{k}: op undefined on operands"
        if op in C2.NEGPRE:
            v = -abs(v)
        if v != gotv:
            return f"pin try EQ{k}: arithmetic wrong ({a} {opw1} {b} != {gotv})"
        if int(x) != gotv or int(y) != pdec[k]:
            return f"pin try EQ{k}: comparator values not derived"
        Lp, Rp = eqs[k - 1]
        l0, l1_, r0, r1 = ((Lp[1], Lp[0], Lp[4], Lp[3]) if rev
                           else (Lp[0], Lp[1], Lp[3], Lp[4]))
        if (a != mapping[l0] * 10 + mapping[l1_]
                or b != mapping[r0] * 10 + mapping[r1]):
            return f"pin try EQ{k}: operand decode mismatch"
    # ---- r3: tier-1 order-check claims recomputed from the prompt
    for ln in lines:
        m = _ORDER_RE.match(ln)
        if not m:
            continue
        k = int(m.group(1))
        if not (1 <= k <= len(eqs)):
            return f"order check: bad EQ index {k}"
        Lp, Rp = eqs[k - 1]
        want = (Lp[3] + Lp[4] + Lp[0] + Lp[1] if m.group(2) == 'swapped'
                else Lp[0] + Lp[1] + Lp[3] + Lp[4])
        if m.group(3) != want:
            return f"order check EQ{k}: claimed string false"
    if has_verify:
        vl, cl = {}, {}
        for ln in lines:
            m = _VERIFY_RE.match(ln)
            if m:
                vl[int(m.group(1))] = m
                continue
            m = _CONCAT_RE.match(ln)
            if m:
                cl[int(m.group(1))] = m
        for k, (Lp, Rp) in enumerate(eqs, 1):
            if k == skip_eq:
                continue        # r6 bail: corrupt example checked by lint_r6
            if k in vl:
                m = vl[k]
                if m.group(2) != Rp:
                    return f"EQ{k}: rendered RHS != prompt RHS"
                opch = Lp[2]
                sign = bool(m.group(3))
                mag = Rp
                if sign:
                    if 'leading' in m.group(3):
                        if Rp[0] != opch:
                            return f"EQ{k}: leading-sign note wrong"
                        mag = Rp[1:]
                    else:
                        if Rp[-1] != opch:
                            return f"EQ{k}: trailing-sign note wrong"
                        mag = Rp[:-1]
                elif len(Rp) > 1 and (Rp[0] == opch or Rp[-1] == opch):
                    return f"EQ{k}: sign glyph present but unnoted"
                try:
                    tru = [mapping[c] for c in mag]
                except KeyError:
                    return f"EQ{k}: RHS symbol outside solver map"
                if any(t is None for t in tru):
                    return f"EQ{k}: RHS symbol unmapped in solver map"
                dd = [int(x) for x in m.group(4).split()]
                if dd != tru:
                    return f"EQ{k}: decoded digits != true RHS decode"
                if bool(m.group(5)) != bool(rev):
                    return f"EQ{k}: reversed flag mismatch"
                msb = dd[::-1] if rev else dd
                expv = int(''.join(map(str, msb))) * (-1 if sign else 1)
                if int(m.group(6)) != expv:
                    return f"EQ{k}: stated expected != decoded value"
                a, opw, b = int(m.group(7)), m.group(8), int(m.group(9))
                gotv = int(m.group(10))
                if gotv != expv:
                    return f"EQ{k}: computed != expected"
                op = WORD_OP.get(opw)
                if op is None:
                    return f"EQ{k}: unknown op word {opw!r}"
                v = C2.OPS[op](a, b)
                if v is None:
                    return f"EQ{k}: op undefined on operands"
                if op in C2.NEGPRE:
                    v = -abs(v)
                if v != gotv:
                    return f"EQ{k}: arithmetic wrong ({a} {opw} {b} != {gotv})"
                l0, l1, r0, r1 = ((Lp[1], Lp[0], Lp[4], Lp[3]) if rev
                                  else (Lp[0], Lp[1], Lp[3], Lp[4]))
                ta = mapping[l0] * 10 + mapping[l1]
                tb = mapping[r0] * 10 + mapping[r1]
                if a != ta or b != tb:
                    return f"EQ{k}: operand decode mismatch"
            elif k in cl:
                m = cl[k]
                if m.group(2) != Rp:
                    return f"EQ{k}: concat RHS != prompt RHS"
                pat = (Lp[0] + Lp[1] + Lp[3] + Lp[4] if m.group(3) == 'in order'
                       else Lp[3] + Lp[4] + Lp[0] + Lp[1])
                if Rp != pat:
                    return f"EQ{k}: concat structure claim false"
            else:
                return f"EQ{k} missing from verify block"
    # ---- final decode: per-digit pairs -> answer (skip for pure-concat answers)
    qop = meta.get('qop')
    if qop in C2.CONCATS:
        # structural answer: symbols taken straight from the query
        if not any(ln.rstrip().endswith(f"-> {answer}") for ln in lines):
            return "concat: no direct symbol-take line ending in the answer"
        return None
    dms = [m for ln in lines for m in [_DECODE_RE.search(ln)] if m]
    if not dms:
        return "no per-digit decode line"
    m = dms[-1]
    dpairs = [(int(p.split('=', 1)[0]), p.split('=', 1)[1]) for p in m.group(1).split()]
    core = ''.join(s for _, s in dpairs)
    if core != m.group(2):
        return "decode pairs != decoded string"
    for d, s in dpairs:
        if mapping.get(s) != d:
            return f"decode pair {d}={s} contradicts solver map"
    if core != answer:
        g = qL[2]
        if answer not in (g + core, core + g):
            return f"decoded string {core!r} not the answer {answer!r}"
        if not any(ln.rstrip().endswith(f"-> {answer}") for ln in lines):
            return "signed answer line missing"
    return None

# ---------------------------------------------------------------- r4 lint additions
_TRYANY_RE = _re.compile(r"^\s*(?:try|Try) ")
_UNPIN_RE = _re.compile(r"un-pin ([A-J]) = (\d)\. tried \1: ([\d,]+)\.")
_TRYDIG_RE = _re.compile(r"^try ([A-J]) = (\d):")
_TRSET_RE = _re.compile(r"tried ([A-J]): ([\d,]+)\.")
_OPEXH_RE = _re.compile(
    r"EQ(\d+): (\d+) ([xyzw]) (\d+) must give (\S+?): (.+?) - none fits")
_REFUTE_RE = _re.compile(
    r"^Try ([xyzw]) = (.+?) on EQ(\d+): .*\. no -> mistake, backtrack")
_LOCKEDRULES_RE = _re.compile(r"^Locked rules: (.*)$")

def lint_r4(cot, answer, prompt=None, meta=None, skip_eq=None):
    """r4 lint = full r3 truthfulness lint + search-state discipline checks:
    (i)   no two identical try-lines anywhere in the trace;
    (ii)  after 'un-pin L = d. tried L: ...' the next try of L uses a NEW digit;
    (ii') tried-sets per letter grow consistently within a contiguous run;
    (iii) no op refuted on an equation (after the map is locked) may be verified
          against that same equation later ('refuted-then-locked' fabrication);
    (iv)  op-exhaustion lines: every listed op value recomputed; 'none fits'
          checked against the stated expected value when numeric."""
    err = lint(cot, answer, prompt=prompt, meta=meta, skip_eq=skip_eq)
    if err:
        return err
    lines = cot.splitlines()
    # (i) duplicate try-lines = the greedy-loop signature
    seen = set()
    for ln in lines:
        if _TRYANY_RE.match(ln):
            key = ln.strip()
            if key in seen:
                return f"duplicate try-line: {key[:64]}"
            seen.add(key)
    # scope boundaries: a level-up revision / exhaustion / mode switch closes the
    # scope, after which a letter may legitimately be re-tried from scratch
    def _boundary(ln, l):
        if 'LITTLE-ENDIAN' in ln or ln.startswith('Locked map:'):
            return True
        if 'all passes exhausted' in ln:
            return True     # r6 bail: the leave-one-out re-solve opens a new scope
                            # (phrase verified absent from all r3/r4/r5 corpora)
        if f"all digits for {l} fail" in ln:
            return True
        m = _re.search(r"un-pin ([A-J]) = \d", ln)
        return bool(m and m.group(1) != l)
    # (ii) post-un-pin retry must differ from every tried candidate at that scope
    for idx, ln in enumerate(lines):
        m = _UNPIN_RE.search(ln)
        if not m:
            continue
        l, d, tr = m.group(1), m.group(2), m.group(3).split(',')
        if d not in tr:
            return f"un-pin {l}={d} not recorded in its tried-set"
        for ln2 in lines[idx + 1:]:
            if _boundary(ln2, l):
                break
            m2 = _TRYDIG_RE.match(ln2.strip())
            if m2 and m2.group(1) == l:
                if m2.group(2) in tr:
                    return f"retry of already-tried digit {l} = {m2.group(2)}"
                break
    # (ii') tried-set growth within a contiguous (same-scope) run per letter
    runs = {}
    for ln in lines:
        for l in list(runs):
            if _boundary(ln, l):
                runs.pop(l)
        for m in _TRSET_RE.finditer(ln):
            l, cur = m.group(1), m.group(2).split(',')
            prev = runs.get(l)
            if prev is not None and cur == prev:
                return f"tried-set for {l} did not grow: {','.join(cur)}"
            runs[l] = cur
    # (iii) refuted-then-verified (state-aware: only after the map is locked,
    # where no digit revision can re-validate an op on that equation; keyed by
    # GLYPH + op + equation — another glyph's lock of the same op word is fine)
    lockidx = next((i for i, ln in enumerate(lines)
                    if ln.startswith('Locked map:')), None)
    if lockidx is not None and prompt is not None:
        pr = C2.parse(prompt)
        if pr is None:
            return "prompt unparseable (r4)"
        eqs4, qL4 = pr
        olet4 = {}
        for L4, _R4 in eqs4:
            if L4[2] not in olet4:
                olet4[L4[2]] = OPLETTERS[min(len(olet4), 3)]
        if qL4[2] not in olet4:
            olet4[qL4[2]] = OPLETTERS[min(len(olet4), 3)]
        eq_glyph = {k: olet4[L4[2]] for k, (L4, _R4) in enumerate(eqs4, 1)}
        refuted = set()
        for ln in lines[lockidx:]:
            m = _REFUTE_RE.match(ln)
            if m:
                refuted.add((m.group(1), m.group(2), int(m.group(3))))
        for ln in lines[lockidx:]:
            m = _VERIFY_RE.match(ln)
            if m:
                k = int(m.group(1))
                if (eq_glyph.get(k), m.group(8), k) in refuted:
                    return f"op {m.group(8)!r} refuted on EQ{k} then verified"
        lr = next((ln for ln in lines[lockidx:] if _LOCKEDRULES_RE.match(ln)), None)
        if lr:
            pairs = _re.findall(r"([xyzw]) = ([^,]+)", _LOCKEDRULES_RE.match(lr).group(1))
            for g, opw in pairs:
                opw = opw.strip()
                eqs_ref = {k for gg, w, k in refuted if gg == g and w == opw}
                if eqs_ref:
                    # locking an op this glyph refuted under the final state = fabrication
                    return f"locked rule {g} = {opw} was refuted on EQ{sorted(eqs_ref)[0]}"
    # (iv-pre) meta-free arithmetic re-derivation of every verify / pin-try line:
    # an invented "compute a op b = X" can never pass, even without the solver map
    for ln in lines:
        for mm in (_VERIFY_RE.match(ln), _PINTRY_RE.match(ln)):
            if not mm:
                continue
            if mm.re is _VERIFY_RE:
                dd = [int(x) for x in mm.group(4).split()]
                msb = dd[::-1] if mm.group(5) else dd
                expv = int(''.join(map(str, msb))) * (-1 if mm.group(3) else 1)
                if int(mm.group(6)) != expv:
                    return f"verify decode value wrong: {ln.strip()[:48]}"
                a, opw, b, gotv = (int(mm.group(7)), mm.group(8),
                                   int(mm.group(9)), int(mm.group(10)))
                if gotv != expv:
                    return f"verify mismatch stamped match: {ln.strip()[:48]}"
            else:
                a, opw, b, gotv = (int(mm.group(4)), mm.group(2),
                                   int(mm.group(6)), int(mm.group(7)))
            op = WORD_OP.get(opw)
            if op is None:
                return f"unknown op word {opw!r}"
            v = C2.OPS[op](a, b)
            if v is not None and op in C2.NEGPRE:
                v = -abs(v)
            if v is None or v != gotv:
                return f"arithmetic invented: {a} {opw} {b} != {gotv}"
    # (iv) op-exhaustion arithmetic
    for ln in lines:
        m = _OPEXH_RE.search(ln)
        if not m:
            continue
        a, b = int(m.group(2)), int(m.group(4))
        exp = m.group(5)
        body = m.group(6)
        if body.endswith(')') and '(+' in body:
            body = body[:body.rindex('(+')].rstrip().rstrip(',')
        for item in body.split(', '):
            if ' ' not in item:
                return f"opexh item unparseable: {item!r}"
            opw, val = item.rsplit(' ', 1)
            op = WORD_OP.get(opw)
            if op is None:
                return f"opexh unknown op {opw!r}"
            got = _got_str(op, a, b)
            if val == '?':
                if got is not None:
                    return f"opexh: {opw} marked undefined but = {got}"
                continue
            if got != val:
                return f"opexh arithmetic wrong: {a} {opw} {b} != {val}"
            try:
                if int(exp) == int(val):
                    return f"opexh claims none fits but {opw} = {val} = expected"
            except ValueError:
                pass
    return None

# ---------------------------------------------------------------- pipelines
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(ROOT, 'competition_dataset', 'tokenizer.json'))

def load_real(cat):
    return [r for r in csv.DictReader(open(f"{ROOT}/competition_dataset/train_categorized.csv"))
            if r['category'] == cat]

def val_ids():
    ids = set()
    for line in open(f"{ROOT}/pipeline/data/val.jsonl"):
        d = json.loads(line)
        ids.add(d.get('id'))
    return ids

def _emit(kept, drop, tl, r, prompt, ans, meta, tk, rid, cat):
    ep_rng = random.Random(zlib.crc32(str(rid).encode('utf-8')))
    cot, tier, info = render_trace(prompt, meta, ans, ep_rng=ep_rng)
    if cot is None:
        key = f'render-t{tier}: {str(info)[:48]}'
        drop[key] = drop.get(key, 0) + 1
        return False
    err = lint(cot, ans, prompt=prompt, meta=meta)
    if err:
        drop[f'lint: {err[:48]}'] = drop.get(f'lint: {err[:48]}', 0) + 1
        return False
    ntok = len(tk.encode(cot).ids)
    if ntok >= 5000:                      # hard per-trace budget (max < 5000)
        drop['too-long'] = drop.get('too-long', 0) + 1
        return False
    tl[tier].append(ntok)
    kept.append({'id': rid, 'category': cat, 'prompt': prompt,
                 'cot': cot, 'final': ans, 'tier': tier, 'ntok': ntok})
    return True

def render_real(out_path, deadline=10.0, limit=None):
    tk = tokenizer()
    vids = val_ids()
    rows = load_real('cryptarithm_deduce')
    if limit:
        rows = rows[:limit]
    kept, drop = [], {}
    tl = {1: [], 2: [], 3: []}
    for r in rows:
        if r['id'] in vids:
            continue
        res = C2.solve(r['prompt'], deadline_s=deadline)
        if res is None:
            drop['solver-none'] = drop.get('solver-none', 0) + 1
            continue
        ans, meta = res
        if ans != r['answer'].strip():
            drop['solver-wrong'] = drop.get('solver-wrong', 0) + 1
            continue
        _emit(kept, drop, tl, r, r['prompt'], ans, meta, tk, r['id'], r['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl

def render_synth(out_path, n, seed=15):
    tk = tokenizer()
    rng = random.Random(seed)
    kept, drop = [], {}
    tl = {1: [], 2: [], 3: []}
    tries = 0
    while len(kept) < n and tries < n * 40:
        tries += 1
        p = gen_puzzle(rng)
        if p is None:
            continue
        res = C2.solve(p['prompt'], deadline_s=6.0)
        if res is None or res[0] != p['answer']:
            drop['solver-mismatch'] = drop.get('solver-mismatch', 0) + 1
            continue
        ans, meta = res
        _emit(kept, drop, tl, p, p['prompt'], ans, meta, tk,
              f'synth-s{seed}-{len(kept):05d}', p['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl

def stats(tl):
    import statistics
    out = {}
    for t, v in tl.items():
        if not v:
            continue
        v = sorted(v)
        out[t] = dict(n=len(v), med=int(statistics.median(v)),
                      p95=v[int(0.95 * len(v)) - 1] if len(v) > 1 else v[0], mx=v[-1])
    return out

# ---------------------------------------------------------------- r4 pipelines
def synth_meta(p):
    """lint/cross-check meta straight from the generator truth (no solver call)."""
    return {'mapping': {s: d for d, s in p['map'].items()}, 'rev': p['rev'],
            'ops': p['ops'], 'qop': p['qop'], 'sign_end': p['sign_end']}

def _emit4(kept, drop, tl, infos, prompt, ans, meta, tk, rid, cat):
    ep_rng = random.Random(zlib.crc32(str(rid).encode('utf-8')))
    cot, tier, info = render_trace_r4(prompt, ans, ep_rng=ep_rng, meta=meta)
    if cot is None:
        key = f'render-t{tier}: {str(info)[:48]}'
        drop[key] = drop.get(key, 0) + 1
        return False
    err = lint_r4(cot, ans, prompt=prompt, meta=meta)
    if err:
        drop[f'lint: {err[:48]}'] = drop.get(f'lint: {err[:48]}', 0) + 1
        return False
    ntok = len(tk.encode(cot).ids)
    if ntok >= 5000:
        drop['too-long'] = drop.get('too-long', 0) + 1
        return False
    tl[tier].append(ntok)
    infos.append(info)
    kept.append({'id': rid, 'category': cat, 'prompt': prompt,
                 'cot': cot, 'final': ans, 'tier': tier, 'ntok': ntok})
    return True

def episode_stats(infos):
    n = len(infos) or 1
    has_exh = sum(1 for i in infos
                  if i.get('n_deep', 0) + i.get('n_opexh', 0) + i.get('mode_switch', 0)
                  + i.get('n_exh', 0) > 0)
    return {'n': len(infos),
            'exh_episode_frac': round(has_exh / n, 4),
            'deep_backtracks': sum(i.get('n_deep', 0) for i in infos),
            'op_exhaustions': sum(i.get('n_opexh', 0) for i in infos),
            'mode_switches': sum(i.get('mode_switch', 0) for i in infos),
            'collides': sum(i.get('n_collide', 0) for i in infos)}

def render_real4(out_path, deadline=10.0, limit=None):
    tk = tokenizer()
    vids = val_ids()
    rows = load_real('cryptarithm_deduce')
    if limit:
        rows = rows[:limit]
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    for r in rows:
        if r['id'] in vids:
            continue
        ans = r['answer'].strip()
        res = C2.solve(r['prompt'], gold=ans, deadline_s=deadline)
        if res is None:
            drop['gold-unsolvable'] = drop.get('gold-unsolvable', 0) + 1
            continue
        _, meta = res
        _emit4(kept, drop, tl, infos, r['prompt'], ans, meta, tk, r['id'], r['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl, infos

def render_synth4(out_path, n, seed=15):
    tk = tokenizer()
    rng = random.Random(seed)
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    tries = 0
    while len(kept) < n and tries < n * 40:
        tries += 1
        p = gen_puzzle(rng)
        if p is None:
            continue
        _emit4(kept, drop, tl, infos, p['prompt'], p['answer'], synth_meta(p), tk,
               f'synth4-s{seed}-{len(kept):05d}', p['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl, infos

# ---- policy coverage (the r4 gate): NO gold conditioning anywhere; gold is only
# compared against the final boxed answer. Run on ALL real deduce rows.
_COV = {}

def _cov_init():
    _COV['tk'] = tokenizer()

def _cov_work(r):
    ep_rng = random.Random(zlib.crc32(str(r['id']).encode('utf-8')))
    ans = r['answer'].strip()
    t0 = time.time()
    try:
        cot, tier, info = render_trace_r4(r['prompt'], ans, ep_rng=ep_rng, meta=None)
    except Exception as e:
        return {'id': r['id'], 'ok': False, 'why': f'EXC {type(e).__name__}: {e}'[:80],
                'ntok': None, 'dt': time.time() - t0}
    if cot is None:
        return {'id': r['id'], 'ok': False, 'why': str(info)[:80], 'ntok': None,
                'dt': time.time() - t0}
    ntok = len(_COV['tk'].encode(cot).ids)
    lint_err = lint_r4(cot, ans, prompt=r['prompt'])
    return {'id': r['id'], 'ok': ntok <= 3500, 'why': 'solved' if ntok <= 3500 else 'over-3500',
            'ntok': ntok, 'lint': lint_err, 'info': info, 'tier': tier,
            'dt': time.time() - t0}

def coverage(procs=8, out=None):
    rows = load_real('cryptarithm_deduce')
    import multiprocessing as mp
    with mp.Pool(procs, initializer=_cov_init) as pool:
        res = pool.map(_cov_work, rows, chunksize=8)
    ok = sum(r['ok'] for r in res)
    print(f"POLICY COVERAGE (boxed==gold AND ntok<=3500): {ok}/{len(rows)}"
          f" = {ok/len(rows):.4f}")
    fails = {}
    for r in res:
        if not r['ok']:
            fails[r['why']] = fails.get(r['why'], 0) + 1
    for w, c in sorted(fails.items(), key=lambda t: -t[1]):
        print(f"  FAIL {c:4d}  {w}")
    toks = sorted(r['ntok'] for r in res if r['ntok'])
    if toks:
        print(f"tokens (solved-any): med {toks[len(toks)//2]}"
              f" p95 {toks[int(len(toks)*.95)]} max {toks[-1]}")
    solved = [r for r in res if r['ok']]
    lint_bad = sum(1 for r in solved if r.get('lint'))
    print(f"lint failures among covered: {lint_bad}")
    print('episodes:', episode_stats([r['info'] for r in solved if r.get('info')]))
    if out:
        with open(out, 'w') as f:
            for r in res:
                r.pop('info', None)
                f.write(json.dumps(r) + '\n')
    return ok / len(rows)

# ================================================================ r5: two-regime glyph model
# (analysis/reports/cryptarithm_structure_hunt.md, confirmed 2026-06-12)
# The generator flips a per-ROW coin (pA ~ 0.60):
#   regime A "canonical": every operator glyph IS the op's own arithmetic char —
#     '+' carries an add-family op, '*' mul-family, '-' sub-family (concat renders
#     under '+'/'*'; rsub ~0 under '-'); concat may DUPLICATE across '+' and '*'.
#   regime B "scrambled": glyphs drawn without replacement from the 26-char pool.
# Plus: operands NEVER have a leading zero (examples AND query; 0/803 gold configs
# killed by the hard prune). r5 renderer = r4 policy + per-row regime detection
# (family search tiers on canonical rows, ~60%), leading-zero domain pruning, and
# the family-aware guess bet (canonical hit .593 vs r4 exclusion .304).
CANON_CHARS = frozenset('+-*')
EM_PRIORS_PATH = os.path.join(ROOT, 'pipeline', 'data', 'em_priors_lo.json')
FALLBACK_MARK = "treat the operator glyphs as arbitrary symbols"

# family search tiers, ordered by EM prior (em_priors_lo.json famA):
FAM_PLAIN5 = {'+': ['add'], '*': ['mul'],
              '-': ['sub_signed', 'absdiff', 'neg_absdiff']}
FAM_FULL5 = {'+': ['add', 'add_m1', 'add_p1'],
             '*': ['mul', 'mul_m1', 'mul_p1'],
             '-': ['sub_signed', 'absdiff', 'neg_absdiff', 'rmod', 'mod']}
# lint families = full family membership (not search tiers)
LINT_FAM5 = {'+': ADD_FAM | set(C2.CONCATS),
             '*': MUL_FAM | set(C2.CONCATS),
             '-': SUB_FAM | {'mod', 'rmod'}}
# double-check pool for fully-forced canonical traces: the search tiers
# (FAM_FULL5) may have truthfully ELIMinated every alternative, so the injected
# double-check draws from the WHOLE family incl. the +-2 offsets the search
# never tries (in-family -> lint_r5-clean; always leaves a refutable candidate)
DBL_FAM5 = {'+': ['add', 'add_m1', 'add_p1', 'add_p2', 'add_m2'],
            '*': ['mul', 'mul_m1', 'mul_p1', 'mul_p2', 'mul_m2'],
            '-': ['sub_signed', 'absdiff', 'neg_absdiff', 'rmod', 'mod',
                  'absdiff_p1', 'absdiff_m1', 'absdiff_p2', 'absdiff_m2']}
FAM_WORD5 = {'+': 'add', '*': 'times', '-': 'minus'}

_EM5 = None

def em_priors():
    global _EM5
    if _EM5 is None:
        _EM5 = json.load(open(EM_PRIORS_PATH))
    return _EM5

def famA_order(ch):
    fa = em_priors()['famA_p'].get(ch, {})
    return sorted(fa, key=lambda o: -fa[o])

def opB_order():
    ob = em_priors()['opB_p']
    return sorted(ob, key=lambda o: -ob[o])

def _gen_dists5():
    """generator distributions from the EM priors, truncated to real mass
    (famA >= 0.02 keeps exactly the measured family vocab incl. NO rsub under '-';
    opB >= 0.004 keeps exactly the 13-op scrambled tail)."""
    em = em_priors()
    famA = {ch: {o: p for o, p in d.items() if p >= 0.02}
            for ch, d in em['famA_p'].items()}
    opB = {o: p for o, p in em['opB_p'].items() if p >= 0.004}
    return famA, opB

P_A5 = 0.599                      # em_priors_lo pA
P_REV_A5 = math.exp(-0.7786901448638808)   # modeA[rev]  ~ .459
P_REV_B5 = math.exp(-1.052813058659372)    # modeB[rev]  ~ .349

BANNER5_CANON = [
    "Assume standard digit order (left digit = tens) and the plain family rules"
    " first ('+': a+b; '*': a*b; '-': a-b, |a-b|, -|a-b|); family offset variants"
    " only if these fail.",
    "No assignment works: standard reading with plain family rules is impossible."
    " Revise one assumption at a time, digit order first -> the puzzle may write"
    " numbers LITTLE-ENDIAN (digits reversed). Re-read every number MSB-first by"
    " reversing:",
    "Little-endian with plain family rules fails too -> allow the family offset"
    " variants ('+': a+b-1, a+b+1; '*': a*b-1, a*b+1; '-': b mod a, a mod b), back"
    " to standard digit order.",
    "Standard reading still fails with family offsets -> re-read LITTLE-ENDIAN"
    " (family offsets allowed):",
    "Every family reading fails -> the canonical-layout assumption is wrong: "
    + FALLBACK_MARK + " (any rule can hide behind any glyph). Back to standard"
    " digit order, full rule set.",
    "Last resort: LITTLE-ENDIAN with the full rule set (glyphs still arbitrary"
    " symbols):",
]
BANNER5_SCRAM = [
    "Assume standard digit order (left digit = tens) and the plain rules"
    " first (a+b, a*b, a-b, b-a, |a-b|, -|a-b|); offsets only if these fail.",
    "No assignment works: standard reading with plain rules is impossible."
    " Revise one assumption at a time, digit order first -> the puzzle may"
    " write numbers LITTLE-ENDIAN (digits reversed). Re-read every number"
    " MSB-first by reversing:",
    "Little-endian with plain rules fails too -> the rules are not all plain."
    " Allow the +1/-1/+2/-2 offset variants (and mod), back to standard"
    " digit order.",
    "Standard reading still fails with offsets -> re-read LITTLE-ENDIAN"
    " (offsets allowed):",
]

def prune_leading(eng, vrows, qleads):
    """r5 no-leading-zero: operand tens letters (examples + query) cannot be 0.
    Returns the letters actually pruned (for the rendered constraint line)."""
    leads = set()
    for r in vrows:
        leads.add(r.a0); leads.add(r.b0)
    leads |= {l for l in qleads if l in eng.dom}
    out = sorted(l for l in leads if 0 in eng.dom[l])
    for l in out:
        eng.dom[l] = eng.dom[l] - {0}
    return out

def gen_puzzle_r5(rng, force_guess=False):
    """two-regime synthetic generator (synthetic == measured real distribution).
    Regime A: glyphs ARE the canonical chars, op ~ famA[char] (concat may dup
    across '+'/'*', value ops distinct by family construction). Regime B: r4
    behavior with the EM opB tail. Operands always in [10,99] (no leading zero)."""
    famA_gen, opB_gen = _gen_dists5()
    canonical = rng.random() < P_A5
    rev = rng.random() < (P_REV_A5 if canonical else P_REV_B5)
    sign_end = 'suf' if (rev and rng.random() < P_SUF_GIVEN_REV_SIGN) else 'pre'
    nex = w_choice(rng, N_EX_W)
    ng = min(w_choice(rng, N_GLYPH_W), nex)
    if canonical:
        ng = min(ng, 3 - (1 if force_guess else 0))
    ntot = ng + (1 if force_guess else 0)
    digs = rng.sample(DIGIT_POOL, 10)
    smap = {d: digs[d] for d in range(10)}
    if canonical:
        glyphs = rng.sample(['+', '-', '*'], ntot)
        gops = {gl: w_choice(rng, famA_gen[gl]) for gl in glyphs}
    else:
        pool = [c for c in DIGIT_POOL + OP_EXTRA if c not in digs]
        glyphs = rng.sample(pool, ntot)
        pool_ops = dict(opB_gen)
        gops = {}
        for gl in glyphs:
            op = w_choice(rng, pool_ops)
            gops[gl] = op
            pool_ops.pop(op)   # measured: ops drawn WITHOUT replacement
    qglyph = glyphs[-1] if force_guess else rng.choice(glyphs)
    exglyphs = glyphs[:-1] if force_guess else glyphs
    asg = list(exglyphs) + [rng.choice(exglyphs) for _ in range(nex - len(exglyphs))]
    rng.shuffle(asg)
    lines = []
    for gl in asg:
        op = gops[gl]
        for _ in range(40):
            a = rng.randint(10, 99); b = rng.randint(10, 99)
            if op in ('concat_fwd', 'concat_rev'):
                da = [a // 10, a % 10]; db = [b // 10, b % 10]
                if rev:
                    da = da[::-1]; db = db[::-1]
                lhs = smap[da[0]] + smap[da[1]] + gl + smap[db[0]] + smap[db[1]]
                rhs = (lhs[0] + lhs[1] + lhs[3] + lhs[4]) if op == 'concat_fwd' else \
                      (lhs[3] + lhs[4] + lhs[0] + lhs[1])
                lines.append((lhs, rhs)); break
            r = render_eq(a, b, op, smap, gl, rev, sign_end)
            if r is not None:
                lines.append(r); break
        else:
            return None
    qop = gops[qglyph]
    for _ in range(60):
        a = rng.randint(10, 99); b = rng.randint(10, 99)
        if qop in ('concat_fwd', 'concat_rev'):
            da = [a // 10, a % 10]; db = [b // 10, b % 10]
            if rev:
                da = da[::-1]; db = db[::-1]
            qL = smap[da[0]] + smap[da[1]] + qglyph + smap[db[0]] + smap[db[1]]
            ans = (qL[0] + qL[1] + qL[3] + qL[4]) if qop == 'concat_fwd' else \
                  (qL[3] + qL[4] + qL[0] + qL[1])
            break
        r = render_eq(a, b, qop, smap, qglyph, rev, sign_end)
        if r is not None:
            qL, ans = r
            break
    else:
        return None
    prompt = HDR + "\n" + "\n".join(f"{l} = {r}" for l, r in lines) + "\n" + QSTR + qL
    return {'prompt': prompt, 'answer': ans, 'rev': rev, 'ops': dict(gops),
            'map': dict(smap), 'sign_end': sign_end, 'qglyph': qglyph, 'qop': qop,
            'regime': 'A' if canonical else 'B',
            'category': 'cryptarithm_guess' if force_guess else 'cryptarithm_deduce'}

def render_trace_r5(prompt, answer, ep_rng=None, meta=None):
    """r5 policy renderer = r4 + two-regime detection (family tiers on canonical
    rows, scrambled fallback STATED), no-leading-zero pruning, concat-dup
    exemption, family-aware guess bet. Truth-free; `answer` (gold) only gates
    the final boxed string. returns (cot_text, tier, info) or (None, tier, reason)."""
    rng = ep_rng if ep_rng is not None else random.Random(12345)
    if C2.parse(prompt) is None:
        return None, 0, "unparseable"
    rows, let, olet, eqs, qL = letterize(prompt, False)
    qop_ch = qL[2]
    qg = olet[qop_ch]
    opch_set = {L0[2] for L0, _ in eqs} | {qop_ch}
    canonical = all(ch in CANON_CHARS for ch in opch_set)
    dup_ok = frozenset(C2.CONCATS) if canonical else frozenset()
    inv_let = {v: k for k, v in let.items()}
    is_guess = all(r.gl != qg for r in rows)
    info = {'n_exh': 0, 'n_deep': 0, 'n_opexh': 0, 'mode_switch': 0,
            'canonical': int(canonical), 'fallback': 0}

    L = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("")
    L.append("First, assign letters to each symbol:")
    for s, a in let.items():
        L.append(f"{s} -> {a}")
    L.append("Operators:")
    for ch, a in olet.items():
        L.append(f"{ch} -> {a}")
    L.append("")
    L.append("Examples in letter form (each left side: two digit-symbols, operator, two digit-symbols):")
    for k, (Lr, Rr) in enumerate(eqs, 1):
        opch = Lr[2]
        mag = Rr; sign_note = ''
        if len(Rr) > 1 and Rr[0] == opch:
            mag = Rr[1:]; sign_note = 'minus sign in front'
        elif len(Rr) > 1 and Rr[-1] == opch:
            mag = Rr[:-1]; sign_note = 'minus sign at the end'
        ll = ''.join(let[c] for c in (Lr[0], Lr[1])) + f" {olet[opch]} " + \
             ''.join(let[c] for c in (Lr[3], Lr[4]))
        rl = ''.join(let[c] for c in mag)
        L.append(f"EQ{k}: {Lr} = {Rr} -> {ll} = {rl}" + (f" ({sign_note})" if sign_note else ""))
    qlets = f"{let[qL[0]]}{let[qL[1]]} {qg} {let[qL[3]]}{let[qL[4]]}"
    L.append(f"Query: {qL} -> {qlets}")
    L.append("")
    byg, cands, conc = build_cands(rows)
    L.append("Classify each operator from cues:")
    for g, rs in byg.items():
        L.append(fam_cue(g, rs, cands[g]))
    if is_guess:
        L.append(f"{qg}: appears only in the query -> rule unknown, decide at the end.")
    if canonical:
        # the regime inference, stated once (shrinks every search below)
        L.append("The operator glyphs are the literal "
                 + ", ".join(f"'{ch}'" for ch in sorted(opch_set))
                 + " characters -> each glyph should follow its own family:"
                 " '+' add-family, '*' times-family, '-' minus-family."
                 " Try family rules first.")
    L.append("")

    # ---- Tier 1 short-circuit (structural concat; mode-free)
    if not is_guess and qg in conc:
        qtrue = conc[qg]
        tier = 1
        L.append(f"The question operator is {qg}, which is {OP_WORD[qtrue]}.")
        chk = None
        for r in byg.get(qg, []):
            Lr, Rr = r.raw
            fwd = Lr[0] + Lr[1] + Lr[3] + Lr[4]
            rv = Lr[3] + Lr[4] + Lr[0] + Lr[1]
            if fwd != rv:
                chk = (r.k, Rr, fwd, rv)
                break
        if chk is None:
            return None, tier, "concat: no order-discriminating example"
        k_, Rr_, fwd_, rv_ = chk
        wrongs, rights = (rv_, fwd_) if qtrue == 'concat_fwd' else (fwd_, rv_)
        wname, rname = (('swapped', 'in order') if qtrue == 'concat_fwd'
                        else ('in order', 'swapped'))
        if rights != Rr_:
            return None, tier, "concat order check inconsistent"
        L.append(f"Check the order on EQ{k_}: {wname} would give {wrongs}."
                 f" {wrongs} vs {Rr_}. no -> mistake, backtrack; {rname} gives"
                 f" {rights}. {rights} vs {Rr_}. match.")
        A2, B2 = qlets.split()[0], qlets.split()[2]
        s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
        if qtrue == 'concat_fwd':
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {A2} || {B2} = {A2}{B2}")
            syms = s1 + s2
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s1} then {s2} -> {syms}")
        else:
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {B2} || {A2} = {B2}{A2}")
            syms = s2 + s1
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s2} then {s1} -> {syms}")
        L.append("")
        L.append(f"\\boxed{{{syms}}}")
        if syms != answer:
            return None, tier, f"concat mismatch {syms!r} != {answer!r}"
        return "\n".join(L), tier, info

    # ---- assumption-revision pass schedule
    # canonical: family-plain -> family-offsets (std/rev each), then the STATED
    # scrambled fallback (the ~1% of all-canonical rows that are secretly regime B);
    # scrambled: the r4 joint-prior schedule.
    sign_end = infer_sign_end(eqs, opch_set)
    eng = None
    rev = False
    ep = {'rng': rng, 'p': P_DIST_EP, 'dist_left': 1, 'n_collide': 0,
          'wrong_left': PIN_WRONG_CAP, 'n_elim2': 0, 'let': let, 'rev': False}
    if canonical:
        fam_plain = {olet[ch]: FAM_PLAIN5[ch] for ch in olet if ch in FAM_PLAIN5}
        fam_full = {olet[ch]: FAM_FULL5[ch] for ch in olet if ch in FAM_FULL5}
        PASSES = [(False, fam_plain), (True, fam_plain), (False, fam_full),
                  (True, fam_full), (False, RVOCAB), (True, RVOCAB)]
        BANNERS = BANNER5_CANON
        fallback_from = 4
    else:
        PASSES = [(False, PLAIN_VOCAB), (True, PLAIN_VOCAB),
                  (False, RVOCAB), (True, RVOCAB)]
        BANNERS = BANNER5_SCRAM
        fallback_from = 99
    for pidx, (mode_try, vocab) in enumerate(PASSES):
        mrows, mlet, molet, _, _ = letterize(prompt, mode_try)
        mbyg, mcands, mconc = build_cands(mrows, vocab=vocab)
        vrows = [r for r in mrows if r.gl not in mconc]
        if not vrows:
            return None, 2, "no value rows but non-concat query"
        if any(not mcands[g] for g in mcands):
            g0 = next(g for g in mcands if not mcands[g])
            L.append(BANNERS[pidx])
            word = "family rule in this tier" if (canonical and pidx < fallback_from) \
                else "rule in this tier"
            L.append(f"{g0}: no {word} fits its result shapes -> this pass"
                     f" cannot work, move on.")
            info['n_exh'] += 1
            if pidx == len(PASSES) - 1:
                return None, 2, "all policy passes refuted"
            continue
        need = set()
        for r in vrows:
            need |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
        e2 = PolicyP2(vrows, {g: c for g, c in mcands.items() if g not in mconc},
                      truth=None)
        e2.dup_ok = dup_ok
        ep['rev'] = mode_try
        L.append(BANNERS[pidx])
        if pidx >= fallback_from:
            info['fallback'] = 1
        if mode_try:
            for r in mrows:
                L.append(f"EQ{r.k}: {r.lhs()} = {r.rstr()}")
            L.append(f"Query reads: {let[qL[1]]}{let[qL[0]]} {qg} {let[qL[4]]}{let[qL[3]]}")
            L.append("")
        # r5: no-leading-zero domain prune (example AND query operands)
        qleads = ({let[qL[1]], let[qL[4]]} if mode_try
                  else {let[qL[0]], let[qL[3]]})
        pruned = prune_leading(e2, vrows, qleads)
        if pruned:
            L.append("No operand starts with 0 -> "
                     + ", ".join(pruned) + " != 0 (leading digits).")
        L.append("Deduce the digits:")
        budget = [POLICY_NODE_CAP]
        try:
            ok = e2.policy_branch(need, budget)
        except Budget as b:
            return None, 2, f"policy budget ({b}) pass {pidx} rev={mode_try}"
        except Contra as e:
            ok = False
            e2.steps.append(('ROOTCONTRA', e))
        if ok:
            rows, conc = mrows, mconc
            cands = mcands
            byg = mbyg
            rev = mode_try
            eng = e2
            info['mode_switch'] = 1 if rev else 0
            info['pass'] = pidx
            render_steps4(e2.steps, L, ep=ep)
            break
        st = e2.steps
        if st and st[-1][0] == 'ROOTCONTRA':
            render_steps4(st[:-1][:4], L, ep=None)
            L.append(f"Contradiction: {st[-1][1].reason}.")
        else:
            if st and st[-1][0] == 'PEXH':
                render_steps4(st[:-1], L, ep=None)
                _, l_, tried_ = st[-1]
                L.append(f"all digits for {l_} fail ({_trset(l_, tried_)}) -> no pin"
                         f" above {l_} to revise inside this reading -> an assumption"
                         f" is wrong.")
            else:
                render_steps4(st, L, ep=None)
        info['n_exh'] += 1
        if pidx == len(PASSES) - 1:
            return None, 2, "all policy passes refuted"
    if eng is None:
        return None, 2, "policy: no pass solved"
    def _count(steps):
        for s in steps:
            if s[0] == 'PTRY':
                if getattr(s[3], 'opexh', None) is not None:
                    info['n_opexh'] += 1
            elif s[0] == 'PDEEP':
                info['n_deep'] += 1
                _count(s[3])
            elif s[0] == 'PENTER':
                _count(s[4])
    _count(eng.steps)
    M = eng.assigned()
    if meta is not None:
        tm = {let[s]: d for s, d in meta['mapping'].items()
              if d is not None and s in let}
        for l, d in tm.items():
            if l in M and M[l] != d:
                return None, 2, f"policy map disagrees with solver at {l}"
        if bool(meta.get('rev')) != rev:
            return None, 2, "policy mode disagrees with solver"
    # ---- op pins: surviving ops at the final map; distinct assignment by prior
    vrows = [r for r in rows if r.gl not in conc]
    survs = {}
    for g in eng.cands:
        surv = []
        for op in eng.cands[g]:
            okop = True
            for r in [r for r in vrows if r.gl == g]:
                a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
                fv = eng._fit(op, r, a, b)
                if fv is None:
                    okop = False; break
                v, digs = fv
                if any(M[r.res[k]] != digs[k] for k in range(r.rl)):
                    okop = False; break
            if okop:
                surv.append(op)
        if not surv:
            return None, 2, f"no op fits glyph {g} at final map"
        survs[g] = [op for op in C2.PRIORITY if op in surv]
    for g, rs in byg.items():
        if g in conc:
            pats = set.intersection(*[set(r.conc) for r in rs])
            survs[g] = [op for op in C2.PRIORITY if op in pats]
    glyph_order = sorted(survs)
    ba = C2._best_distinct([survs[g] for g in glyph_order], set(), dup_ok)
    if ba is not None:
        pins = dict(zip(glyph_order, ba[1]))
    else:
        vord = [g for g in glyph_order if g not in conc]
        ba2 = C2._best_distinct([survs[g] for g in vord], set(), dup_ok)
        if ba2 is None:
            return None, 2, "no distinct op assignment"
        pins = dict(zip(vord, ba2[1]))
        for g in glyph_order:
            if g in conc:
                pins[g] = survs[g][0]
    tie_lines = []
    taken = {op: g for g, op in pins.items()}
    for g in glyph_order:
        if g in conc or len(survs[g]) <= 1:
            continue
        top = survs[g][0]
        if pins[g] == top:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in survs[g])} on"
                             f" every example; keep the most common: {g} = {OP_WORD[top]}.")
        else:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in survs[g])};"
                             f" {OP_WORD[top]} is taken by {taken.get(top, '?')} and the"
                             f" rules are distinct -> {g} = {OP_WORD[pins[g]]}.")
    tier = 1 if (not is_guess and pins.get(qg) in ('concat_fwd', 'concat_rev')) else \
           (3 if (not is_guess and pins.get(qg) in MUL_FAM | SMALL_FAM
                  | {'a2_plus_b', 'a_plus_b2', 'lcm', 'gcd'}) else 2)
    L.append("")
    L.append("Locked map: " + ", ".join(f"{l}={d}" for l, d in sorted(M.items())))
    if len(set(M.values())) != len(M):
        return None, tier, "injectivity violated in locked map"
    L.append("Distinct check: " + " ".join(str(d) for _, d in sorted(M.items()))
             + " - all different. ok")
    # ---- rule-pin episodes (truthful failed tries with tried-set state)
    gone_r = {}
    narrative_live = {g: list(c) for g, c in cands.items() if g not in conc}
    for st in eng.steps:
        if st[0] == 'ELIM':
            _, r0, _a0, _b0, tries0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(
                op for op, _v in tries0 if op not in surv0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
        elif st[0] == 'ELIM2':
            _, r0, _a0, _b0, vals0, opk0, _M0 = st
            gone_r.setdefault(r0.gl, set()).update(op for op, _v in vals0)
            narrative_live[r0.gl] = [opk0]
        elif st[0] == 'OPSHRINK':
            _, r0, gone0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(gone0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
    byg2 = {}
    for r in vrows:
        byg2.setdefault(r.gl, []).append(r)
    # r5: on canonical rows the injected double-check must stay INSIDE the family
    dbl5 = None
    if canonical:
        inv_olet5 = {l: ch for ch, l in olet.items()}
        dbl5 = {g: DBL_FAM5.get(inv_olet5.get(g), ())
                for g in narrative_live}
    pin_lines, n_no = pin_episodes_r4(byg2, conc, narrative_live, pins,
                                      {g: set(s) for g, s in survs.items()},
                                      M, let, rev, gone_r,
                                      cap=ep['wrong_left'], have_no=ep['n_elim2'],
                                      dbl_cands=dbl5)
    if n_no + ep['n_elim2'] == 0:
        return None, tier, "no truthful negative episode renderable"
    L.extend(pin_lines)
    L.extend(tie_lines)
    L.append("Locked rules: " + ", ".join(f"{g} = {OP_WORD[pins[g]]}" for g in sorted(pins)))
    L.append("")
    # ---- verify: decode the RHS through the locked map FIRST, then compute
    L.append("Verify every example (decode RHS first, then compute):")
    for r in rows:
        op = pins[r.gl]
        Lr, Rr = r.raw
        opch_row = Lr[2]
        if op in ('concat_fwd', 'concat_rev'):
            pat = (Lr[0] + Lr[1] + Lr[3] + Lr[4] if op == 'concat_fwd'
                   else Lr[3] + Lr[4] + Lr[0] + Lr[1])
            if Rr != pat:
                return None, tier, f"concat verify failed EQ{r.k}"
            order = 'in order' if op == 'concat_fwd' else 'swapped'
            L.append(f"EQ{r.k}: RHS {Rr} = operand symbols {order}"
                     f" ({Lr[0]}{Lr[1]} and {Lr[3]}{Lr[4]}). match")
            continue
        sign = False; side = None
        mag = Rr
        if len(Rr) > 1 and Rr[0] == opch_row:
            sign = True; side = 'leading'; mag = Rr[1:]
        elif len(Rr) > 1 and Rr[-1] == opch_row:
            sign = True; side = 'trailing'; mag = Rr[:-1]
        dd = [M[let[c]] for c in mag]
        msb = dd[::-1] if rev else dd
        exp_num = int(''.join(map(str, msb)))
        exp_str = ('-' if sign else '') + str(exp_num)
        a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
        v = C2.OPS[op](a, b)
        if v is None:
            return None, tier, f"verify undefined EQ{r.k}"
        got = ('-' if (v < 0 or op in C2.NEGPRE) else '') + str(abs(v))
        if got != exp_str:
            return None, tier, f"verify failed EQ{r.k}: {got} != {exp_str}"
        note = f" ({side} {opch_row} = minus)" if sign else ""
        dec = " ".join(str(d) for d in dd)
        revw = "reversed " if rev else ""
        L.append(f"EQ{r.k}: RHS {Rr}{note} -> {dec} -> {revw}{exp_str};"
                 f" compute {a} {OP_WORD[op]} {b} = {got}. match")
    L.append("")
    # ---- apply to the query (r5: no-leading-zero on free query letters too)
    qa_, qb_, qc_, qd_ = let[qL[0]], let[qL[1]], let[qL[3]], let[qL[4]]
    qlead = {qb_, qd_} if rev else {qa_, qc_}
    missing = [x for x in (qa_, qb_, qc_, qd_) if x not in M]
    if missing:
        free_digs = sorted(set(range(10)) - set(M.values()))
        ml = sorted(set(missing))
        if len(ml) == 1 and len(free_digs) == 1:
            if ml[0] in qlead and free_digs[0] == 0:
                return None, tier, "query leading letter forced to 0"
            M[ml[0]] = free_digs[0]
            L.append(f"{ml[0]} appears only in the query; the one unused digit is"
                     f" {free_digs[0]}, so {ml[0]} = {free_digs[0]}.")
        elif len(ml) <= len(free_digs) and len(ml) <= 3:
            avail = list(free_digs)
            asg = {}
            lead_noted = []
            for l in ml:
                pick = next((d for d in avail
                             if not (l in qlead and d == 0)), None)
                if pick is None:
                    return None, tier, "query leading letter forced to 0"
                if l in qlead and 0 in avail:
                    lead_noted.append(l)
                asg[l] = pick
                avail.remove(pick)
            for l, d in asg.items():
                M[l] = d
            note = ""
            if lead_noted:
                note = (" (" + ", ".join(f"{l} != 0, leading digit"
                                         for l in lead_noted) + ")")
            L.append("Letters " + ", ".join(ml) + " appear only in the query;"
                     f" unused digits are {{{','.join(map(str, free_digs))}}};"
                     " no example constrains them - take them in order" + note + ": "
                     + ", ".join(f"{l} = {asg[l]}" for l in ml) + ".")
        else:
            return None, tier, "query letters not deducible by policy"
    if is_guess:
        used_ops = set(pins.values())
        usedw = ", ".join(OP_WORD[o] for o in pins.values())
        if canonical:
            # r5 family-aware bet: the unseen glyph's CHAR reveals its family
            order = [o for o in famA_order(qop_ch) if o not in used_ops]
            if not order:
                return None, tier, "guess: family exhausted"
            qapply = order[0]
            famword = FAM_WORD5[qop_ch]
            L.append(f"The query operator {qg} never appears in the examples, but"
                     f" its glyph is '{qop_ch}' itself -> a {famword}-family rule."
                     f" The rules in one puzzle are all different ({usedw} taken)"
                     f" -> bet {qg} = {OP_WORD[qapply]}.")
        else:
            qapply = next((o for o in opB_order()
                           if o not in used_ops and o in RVOCAB + ['concat_fwd']),
                          None)
            if qapply is None:
                return None, tier, "guess: no candidate op"
            L.append(f"The query operator {qg} never appears in the examples. The rules"
                     f" in one puzzle are all different, so {qg} is not"
                     f" {usedw}. The most common remaining rule is"
                     f" {OP_WORD[qapply]} -> bet {qg} = {OP_WORD[qapply]}.")
        if qapply in ('concat_fwd', 'concat_rev'):
            s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
            syms = s1 + s2 if qapply == 'concat_fwd' else s2 + s1
            L.append(f"Applying concatenation: take the query operand symbols:"
                     f" -> {syms}")
            L.append("")
            L.append(f"\\boxed{{{syms}}}")
            if syms != answer:
                return None, tier, "guess policy miss"
            return "\n".join(L), tier, info
    else:
        qapply = pins[qg]
    if rev:
        A = M[qb_] * 10 + M[qa_]; B = M[qd_] * 10 + M[qc_]
        L.append(f"Apply to the query (little-endian): {qb_}{qa_} {qg} {qd_}{qc_}"
                 f" = {A} {OP_WORD[qapply]} {B}")
    else:
        A = M[qa_] * 10 + M[qb_]; B = M[qc_] * 10 + M[qd_]
        L.append(f"Apply to the query: {qa_}{qb_} {qg} {qc_}{qd_} = {A} {OP_WORD[qapply]} {B}")
    if A < 10 or B < 10:
        return None, tier, "query operand leading zero under locked map"
    v = C2.OPS[qapply](A, B)
    if v is None:
        return None, tier, "query value undefined"
    sign_needed = (qapply in C2.NEGPRE) or (qapply in C2.SIGNED and v < 0)
    if v < 0 and not sign_needed:
        return None, tier, "negative query value for unsigned op"
    mag = abs(v)
    digs = [int(c) for c in str(mag)]
    inv_d = {d: l for l, d in M.items()}
    miss_d = sorted({d for d in digs if d not in inv_d})
    if miss_d:
        unas = sorted(set(let.values()) - set(M))
        if len(miss_d) == 1 and len(unas) == 1:
            M[unas[0]] = miss_d[0]
            inv_d[miss_d[0]] = unas[0]
            L.append(f"Digit {miss_d[0]} never appeared in a value example; the only"
                     f" letter without a digit is {unas[0]} -> {unas[0]} = {miss_d[0]}.")
        elif len(miss_d) <= len(unas) and len(miss_d) <= 2:
            for d, l in zip(miss_d, unas):
                M[l] = d
                inv_d[d] = l
            L.append("Digits " + ",".join(map(str, miss_d)) + " never appeared in a"
                     " value example; letters without digits: " + ",".join(unas)
                     + " - take them in order: "
                     + ", ".join(f"{l} = {d}" for d, l in zip(miss_d, unas)) + ".")
        else:
            return None, tier, f"answer digit unmapped: {mag}"
    dig2sym = {d: inv_let[inv_d[d]] for d in set(digs)}
    L.append(f"= {'-' if (v < 0 or qapply in C2.NEGPRE) else ''}{mag}")
    disp = digs[::-1] if rev else digs
    pairs = " ".join(f"{d}={dig2sym[d]}" for d in disp)
    core = ''.join(dig2sym[d] for d in disp)
    if rev:
        L.append(f"Little-endian, write digits reversed: {mag} ->"
                 f" {''.join(map(str, disp))}; decode digit by digit: {pairs} -> {core}")
    else:
        L.append(f"Decode digit by digit: {pairs} -> {core}")
    syms = core
    if sign_needed:
        if sign_end == 'suf':
            syms = core + qop_ch
            L.append(f"Negative -> append the operator glyph {qop_ch} as the sign"
                     f" (as in the examples) -> {syms}")
        else:
            syms = qop_ch + core
            L.append(f"Negative -> prefix the operator glyph {qop_ch} as the sign"
                     f" -> {syms}")
    L.append("")
    L.append(f"\\boxed{{{syms}}}")
    if syms != answer:
        return None, tier, f"final mismatch {syms!r} != {answer!r}"
    info['branches'] = eng.nbranch
    info['n_no'] = n_no + ep['n_elim2']
    info['n_collide'] = ep['n_collide']
    return "\n".join(L), tier, info

# ---------------------------------------------------------------- r5 lint
_BET_RE = _re.compile(r"bet ([xyzw]) = (\S+)")
_COULDBE_RE = _re.compile(r"^([xyzw]) could still be (.+?)\.(?: |$)")

def lint_r5(cot, answer, prompt=None, meta=None, skip_eq=None):
    """r5 lint = full lint_r4 + canonical-family discipline: on a canonical row
    (every operator glyph literally '+', '-' or '*'), no rendered op try /
    candidate list / locked rule / guess bet may use an op outside the glyph's
    own family UNLESS the trace has already stated the regime fallback
    (FALLBACK_MARK). Scrambled rows: no extra constraint (r4 behavior)."""
    err = lint_r4(cot, answer, prompt=prompt, meta=meta, skip_eq=skip_eq)
    if err:
        return err
    if prompt is None:
        return None
    pr = C2.parse(prompt)
    if pr is None:
        return "prompt unparseable (r5)"
    eqs, qL = pr
    olet = {}
    for L0, _R0 in eqs:
        if L0[2] not in olet:
            olet[L0[2]] = OPLETTERS[min(len(olet), 3)]
    if qL[2] not in olet:
        olet[qL[2]] = OPLETTERS[min(len(olet), 3)]
    if not all(ch in CANON_CHARS for ch in olet):
        return None
    let_char = {l: ch for ch, l in olet.items()}
    eq_glyph = {k: olet[L0[2]] for k, (L0, _R0) in enumerate(eqs, 1)}

    def _bad(g, opw):
        op = WORD_OP.get(opw.strip())
        if op is None:
            return False
        ch = let_char.get(g)
        return bool(ch) and op not in LINT_FAM5[ch]

    fallback = False
    for ln in cot.splitlines():
        if FALLBACK_MARK in ln:
            fallback = True
        if fallback:
            continue
        m = _PINTRY_RE.match(ln)
        if m and _bad(m.group(1), m.group(2)):
            return (f"canonical row: {m.group(1)} ('{let_char.get(m.group(1))}')"
                    f" tried out-of-family {m.group(2)} without regime fallback")
        m = _COULDBE_RE.match(ln)
        if m:
            for opw in m.group(2).split(' / '):
                if _bad(m.group(1), opw):
                    return (f"canonical row: {m.group(1)} candidate list has"
                            f" out-of-family {opw} without regime fallback")
        m = _LOCKEDRULES_RE.match(ln)
        if m:
            for g, opw in _re.findall(r"([xyzw]) = ([^,]+)", m.group(1)):
                if _bad(g, opw):
                    return (f"canonical row: locked rule {g} = {opw.strip()}"
                            f" out-of-family without regime fallback")
        m = _BET_RE.search(ln)
        if m and _bad(m.group(1), m.group(2).rstrip('.')):
            return (f"canonical row: guess bet {m.group(2)} out-of-family for"
                    f" '{let_char.get(m.group(1))}' without regime fallback")
        m = _OPEXH_RE.search(ln)
        if m:
            g = m.group(3)
            body = m.group(6)
            if body.endswith(')') and '(+' in body:
                body = body[:body.rindex('(+')].rstrip().rstrip(',')
            for item in body.split(', '):
                if ' ' not in item:
                    continue
                opw = item.rsplit(' ', 1)[0]
                if _bad(g, opw):
                    return (f"canonical row: op-exhaustion lists out-of-family"
                            f" {opw} for {g} without regime fallback")
    return None

# ---------------------------------------------------------------- r5 pipelines
def _emit5(kept, drop, tl, infos, prompt, ans, meta, tk, rid, cat):
    ep_rng = random.Random(zlib.crc32(str(rid).encode('utf-8')))
    cot, tier, info = render_trace_r5(prompt, ans, ep_rng=ep_rng, meta=meta)
    if cot is None:
        key = f'render-t{tier}: {str(info)[:48]}'
        drop[key] = drop.get(key, 0) + 1
        return False
    err = lint_r5(cot, ans, prompt=prompt, meta=meta)
    if err:
        drop[f'lint: {err[:48]}'] = drop.get(f'lint: {err[:48]}', 0) + 1
        return False
    ntok = len(tk.encode(cot).ids)
    if ntok >= 5000:
        drop['too-long'] = drop.get('too-long', 0) + 1
        return False
    tl[tier].append(ntok)
    infos.append(info)
    kept.append({'id': rid, 'category': cat, 'prompt': prompt,
                 'cot': cot, 'final': ans, 'tier': tier, 'ntok': ntok})
    return True

def render_real5(out_path, deadline=10.0, limit=None):
    C2.HARD_OPND_LO = True
    reg = C2.load_regime(EM_PRIORS_PATH)
    tk = tokenizer()
    vids = val_ids()
    rows = load_real('cryptarithm_deduce')
    if limit:
        rows = rows[:limit]
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    for r in rows:
        if r['id'] in vids:
            continue
        ans = r['answer'].strip()
        res = C2.solve(r['prompt'], gold=ans, deadline_s=deadline, regime=reg)
        if res is None:
            drop['gold-unsolvable'] = drop.get('gold-unsolvable', 0) + 1
            continue
        _, meta = res
        _emit5(kept, drop, tl, infos, r['prompt'], ans, meta, tk, r['id'], r['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl, infos

def render_synth5(out_path, n, seed=171):
    tk = tokenizer()
    rng = random.Random(seed)
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    tries = 0
    while len(kept) < n and tries < n * 40:
        tries += 1
        p = gen_puzzle_r5(rng)
        if p is None:
            continue
        _emit5(kept, drop, tl, infos, p['prompt'], p['answer'], synth_meta(p), tk,
               f'synth5-s{seed}-{len(kept):05d}', p['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl, infos

# ---- r5 policy coverage (the gate): truth-free, gold only gates the boxed string
def _cov5_work(r):
    ep_rng = random.Random(zlib.crc32(str(r['id']).encode('utf-8')))
    ans = r['answer'].strip()
    t0 = time.time()
    try:
        cot, tier, info = render_trace_r5(r['prompt'], ans, ep_rng=ep_rng, meta=None)
    except Exception as e:
        return {'id': r['id'], 'ok': False, 'why': f'EXC {type(e).__name__}: {e}'[:80],
                'ntok': None, 'dt': time.time() - t0}
    if cot is None:
        return {'id': r['id'], 'ok': False, 'why': str(info)[:80], 'ntok': None,
                'dt': time.time() - t0}
    ntok = len(_COV['tk'].encode(cot).ids)
    lint_err = lint_r5(cot, ans, prompt=r['prompt'])
    return {'id': r['id'], 'ok': ntok <= 3500, 'why': 'solved' if ntok <= 3500 else 'over-3500',
            'ntok': ntok, 'lint': lint_err, 'info': info, 'tier': tier,
            'dt': time.time() - t0}

def coverage5(procs=8, out=None):
    rows = load_real('cryptarithm_deduce')
    import multiprocessing as mp
    with mp.Pool(procs, initializer=_cov_init) as pool:
        res = pool.map(_cov5_work, rows, chunksize=8)
    ok = sum(r['ok'] for r in res)
    print(f"R5 POLICY COVERAGE (boxed==gold AND ntok<=3500): {ok}/{len(rows)}"
          f" = {ok/len(rows):.4f}")
    fails = {}
    for r in res:
        if not r['ok']:
            fails[r['why']] = fails.get(r['why'], 0) + 1
    for w, c in sorted(fails.items(), key=lambda t: -t[1]):
        print(f"  FAIL {c:4d}  {w}")
    toks = sorted(r['ntok'] for r in res if r['ntok'])
    if toks:
        print(f"tokens (solved-any): med {toks[len(toks)//2]}"
              f" p95 {toks[int(len(toks)*.95)]} max {toks[-1]}")
    solved = [r for r in res if r['ok']]
    lint_bad = sum(1 for r in solved if r.get('lint'))
    print(f"lint failures among covered: {lint_bad}")
    can = [r for r in solved if r.get('info', {}).get('canonical')]
    print(f"covered canonical: {len(can)} | covered scrambled: {len(solved) - len(can)}")
    print('episodes:', episode_stats([r['info'] for r in solved if r.get('info')]))
    if out:
        with open(out, 'w') as f:
            for r in res:
                r.pop('info', None)
                f.write(json.dumps(r) + '\n')
    return ok / len(rows)

# ================================================================ r6: exposure-bias hardening
# r5 trained to NLL 0.055 yet scored .04 greedy (64% truncation): the failures were
# the MODEL'S OWN generation errors, never seen in data. r6 fixes them IN THE DATA
# (all r5 policy/structure/lint kept; nothing about policy coverage changes):
#   FIX A  symbol-table CONSTRUCTION: the table is derived from rendered per-position
#          scans of every LHS (pos 1,2,4,5 = digit symbols, pos 3 = operator) + RHS
#          symbol collection + union + op/digit disjointness check. Never stated
#          from nowhere -> no ghost entries / invented operator sets.
#   FIX B  MONOTONE TRY-COUNTERS: every try line carries a strictly-increasing
#          global counter t1, t2, ... (lint: +1 steps over the whole trace) ->
#          a verbatim loop can no longer be a fixed-point of greedy decoding.
#   FIX C  BAIL-OUT ENDINGS (~10% of synth traces): puzzles built with ONE corrupted
#          example so the whole pass schedule truly exhausts; the trace then renders
#          a truthful terminal move - "all passes exhausted (tN) -> one example must
#          be corrupt -> drop one example at a time, retry std+plain -> the drop
#          that turns consistent IS the corrupt line -> solve the rest, decode" -
#          and boxed == gold BY CONSTRUCTION (the corrupted row is EXTRA: dropping
#          it restores the original solvable puzzle whose decode is gold).
BAIL_MARK = "all passes exhausted"
TABLE6_MARK = "positions 1,2,4,5 are digit symbols, position 3 is the operator"
DISJ6_MARK = "never appear in a digit position -> disjoint. ok"
P_BAIL6 = 0.10                  # bail-out fraction of the synth corpus

_T_RE = _re.compile(r"^t(\d+): (.*)$")
# countable = lines that are a search TRY event (digit try, op try, collide try,
# drop-retry). Counter prefixes are added at render time and validated+stripped
# by lint_r6 before delegating to lint_r5 (whose grammars stay unchanged).
_CTBL_RE = _re.compile(
    r"^(?:try [A-J] = \d|Try [xyzw] = |[A-J] in \{[\d,]+\}: try |drop EQ\d+:)")
_BAILREF_RE = _re.compile(r"all passes exhausted \(t(\d+)\)")
_SCAN6_RE = _re.compile(
    r"^EQ(\d+) LHS (\S{5}) -> digits (\S) (\S) (\S) (\S) , op (\S) ; RHS (\S+) -> (.+)$")
_QSCAN6_RE = _re.compile(
    r"^Query LHS (\S{5}) -> digits (\S) (\S) (\S) (\S) , op (\S)$")
_RHSDESC6_SIGN_RE = _re.compile(
    r"^(leading|trailing) (\S) is the op glyph \(sign\), digits ((?:\S )*\S)$")
_RHSDESC6_RE = _re.compile(r"^digits ((?:\S )*\S)$")
_TABLED6_RE = _re.compile(r"^(\S) -> ([A-J])$")
_TABLEO6_RE = _re.compile(r"^(\S) -> ([xyzw])$")
_DROP6_FAIL_RE = _re.compile(r"^drop EQ(\d+): still impossible\.$")
_DROP6_OK_RE = _re.compile(
    r"^drop EQ(\d+): consistent - EQ(\d+) is the corrupt line, exclude it\.$")
_CORRUPTV6_RE = _re.compile(
    r"^EQ(\d+): RHS (\S+) -> ((?:\d )*\d) -> (-?\d+); compute (\d+) (.+?) (\d+) ="
    r" (-?\d+)\. (-?\d+) vs (-?\d+)\. no - corrupt example, excluded\.$")

def add_counters6(L):
    """prefix every countable line with a strictly-increasing global counter and
    fill the bail line's (t{T}) placeholder with the count reached at that point."""
    out = []
    n = 0
    for ln in L:
        if '(t{T})' in ln:
            ln = ln.replace('(t{T})', f'(t{n})')
        if _CTBL_RE.match(ln):
            n += 1
            ln = f"t{n}: {ln}"
        out.append(ln)
    return out

def _finish6(L):
    return "\n".join(add_counters6(L))

def strip_counters6(cot):
    out = []
    for ln in cot.splitlines():
        m = _T_RE.match(ln)
        out.append(m.group(2) if m else ln)
    return "\n".join(out)

def table_lines6(eqs, qL, let, olet, canonical):
    """FIX A: positional symbol-table construction. Every symbol in the union
    table is COPIED from a rendered per-position scan line; the op/digit
    disjointness is checked explicitly (truthful: the generator's digit pool
    never contains an operator char)."""
    L = []
    L.append("Build the symbol table positionally. Every example LHS is 5 chars:"
             " " + TABLE6_MARK + ".")
    for k, (Lr, Rr) in enumerate(eqs, 1):
        opch = Lr[2]
        sign = None
        mag = Rr
        if len(Rr) > 1 and Rr[0] == opch:
            sign = 'leading'; mag = Rr[1:]
        elif len(Rr) > 1 and Rr[-1] == opch:
            sign = 'trailing'; mag = Rr[:-1]
        seg = (f"EQ{k} LHS {Lr} -> digits {Lr[0]} {Lr[1]} {Lr[3]} {Lr[4]} , op {opch}"
               f" ; RHS {Rr} -> ")
        if sign:
            seg += f"{sign} {opch} is the op glyph (sign), digits " + " ".join(mag)
        else:
            seg += "digits " + " ".join(mag)
        L.append(seg)
    L.append(f"Query LHS {qL} -> digits {qL[0]} {qL[1]} {qL[3]} {qL[4]} , op {qL[2]}")
    L.append("Digit symbols in order of first appearance:")
    for s, a in let.items():           # insertion order == first appearance
        L.append(f"{s} -> {a}")
    L.append("Operators:")
    for ch, a in olet.items():
        L.append(f"{ch} -> {a}")
    ops_txt = " ".join(olet)
    if canonical:
        L.append(f"Operator chars {ops_txt} are the literal arithmetic signs and "
                 + DISJ6_MARK)
    else:
        L.append(f"Operator chars {ops_txt} " + DISJ6_MARK)
    return L

def render_trace_r6(prompt, answer, ep_rng=None, meta=None, bail=False):
    """r6 renderer = r5 (two-regime policy, family tiers, no-leading-zero,
    family-aware guess) + FIX A (positional table construction) + FIX B
    (monotone try-counters) + FIX C (bail=True: truthful leave-one-out terminal
    move when every pass exhausts; bail=False keeps exact r5 refusal behavior).
    returns (cot_text, tier, info) or (None, tier, reason)."""
    rng = ep_rng if ep_rng is not None else random.Random(12345)
    if C2.parse(prompt) is None:
        return None, 0, "unparseable"
    rows, let, olet, eqs, qL = letterize(prompt, False)
    qop_ch = qL[2]
    qg = olet[qop_ch]
    opch_set = {L0[2] for L0, _ in eqs} | {qop_ch}
    canonical = all(ch in CANON_CHARS for ch in opch_set)
    dup_ok = frozenset(C2.CONCATS) if canonical else frozenset()
    inv_let = {v: k for k, v in let.items()}
    is_guess = all(r.gl != qg for r in rows)
    info = {'n_exh': 0, 'n_deep': 0, 'n_opexh': 0, 'mode_switch': 0,
            'canonical': int(canonical), 'fallback': 0, 'bail': 0}

    # the positional scans below must be truthful: no op char may sit in a digit slot
    for L0, R0 in eqs:
        if any(c in opch_set for c in (L0[0], L0[1], L0[3], L0[4])):
            return None, 0, "op char in digit position"
        mag0 = R0
        if len(R0) > 1 and R0[0] == L0[2]:
            mag0 = R0[1:]
        elif len(R0) > 1 and R0[-1] == L0[2]:
            mag0 = R0[:-1]
        if any(c in opch_set for c in mag0):
            return None, 0, "op char inside RHS"
    if any(c in opch_set for c in (qL[0], qL[1], qL[3], qL[4])):
        return None, 0, "op char in query digit position"

    L = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("")
    L.extend(table_lines6(eqs, qL, let, olet, canonical))
    L.append("")
    L.append("Examples in letter form (each left side: two digit-symbols, operator, two digit-symbols):")
    for k, (Lr, Rr) in enumerate(eqs, 1):
        opch = Lr[2]
        mag = Rr; sign_note = ''
        if len(Rr) > 1 and Rr[0] == opch:
            mag = Rr[1:]; sign_note = 'minus sign in front'
        elif len(Rr) > 1 and Rr[-1] == opch:
            mag = Rr[:-1]; sign_note = 'minus sign at the end'
        ll = ''.join(let[c] for c in (Lr[0], Lr[1])) + f" {olet[opch]} " + \
             ''.join(let[c] for c in (Lr[3], Lr[4]))
        rl = ''.join(let[c] for c in mag)
        L.append(f"EQ{k}: {Lr} = {Rr} -> {ll} = {rl}" + (f" ({sign_note})" if sign_note else ""))
    qlets = f"{let[qL[0]]}{let[qL[1]]} {qg} {let[qL[3]]}{let[qL[4]]}"
    L.append(f"Query: {qL} -> {qlets}")
    L.append("")
    byg, cands, conc = build_cands(rows)
    L.append("Classify each operator from cues:")
    for g, rs in byg.items():
        L.append(fam_cue(g, rs, cands[g]))
    if is_guess:
        L.append(f"{qg}: appears only in the query -> rule unknown, decide at the end.")
    if canonical:
        L.append("The operator glyphs are the literal "
                 + ", ".join(f"'{ch}'" for ch in sorted(opch_set))
                 + " characters -> each glyph should follow its own family:"
                 " '+' add-family, '*' times-family, '-' minus-family."
                 " Try family rules first.")
    L.append("")

    # ---- Tier 1 short-circuit (structural concat; mode-free)
    if not is_guess and qg in conc:
        qtrue = conc[qg]
        tier = 1
        L.append(f"The question operator is {qg}, which is {OP_WORD[qtrue]}.")
        chk = None
        for r in byg.get(qg, []):
            Lr, Rr = r.raw
            fwd = Lr[0] + Lr[1] + Lr[3] + Lr[4]
            rv = Lr[3] + Lr[4] + Lr[0] + Lr[1]
            if fwd != rv:
                chk = (r.k, Rr, fwd, rv)
                break
        if chk is None:
            return None, tier, "concat: no order-discriminating example"
        k_, Rr_, fwd_, rv_ = chk
        wrongs, rights = (rv_, fwd_) if qtrue == 'concat_fwd' else (fwd_, rv_)
        wname, rname = (('swapped', 'in order') if qtrue == 'concat_fwd'
                        else ('in order', 'swapped'))
        if rights != Rr_:
            return None, tier, "concat order check inconsistent"
        L.append(f"Check the order on EQ{k_}: {wname} would give {wrongs}."
                 f" {wrongs} vs {Rr_}. no -> mistake, backtrack; {rname} gives"
                 f" {rights}. {rights} vs {Rr_}. match.")
        A2, B2 = qlets.split()[0], qlets.split()[2]
        s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
        if qtrue == 'concat_fwd':
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {A2} || {B2} = {A2}{B2}")
            syms = s1 + s2
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s1} then {s2} -> {syms}")
        else:
            L.append(f"Applying: {OP_WORD[qtrue]}({A2}, {B2}) = {B2} || {A2} = {B2}{A2}")
            syms = s2 + s1
            L.append(f"No digits involved: take the query operand symbols directly:"
                     f" {s2} then {s1} -> {syms}")
        L.append("")
        L.append(f"\\boxed{{{syms}}}")
        if syms != answer:
            return None, tier, f"concat mismatch {syms!r} != {answer!r}"
        return _finish6(L), tier, info

    # ---- assumption-revision pass schedule (identical to r5)
    sign_end = infer_sign_end(eqs, opch_set)
    eng = None
    rev = False
    corrupt_k = None
    rows_solve = None
    ep = {'rng': rng, 'p': P_DIST_EP, 'dist_left': 1, 'n_collide': 0,
          'wrong_left': PIN_WRONG_CAP, 'n_elim2': 0, 'let': let, 'rev': False}
    if canonical:
        fam_plain = {olet[ch]: FAM_PLAIN5[ch] for ch in olet if ch in FAM_PLAIN5}
        fam_full = {olet[ch]: FAM_FULL5[ch] for ch in olet if ch in FAM_FULL5}
        PASSES = [(False, fam_plain), (True, fam_plain), (False, fam_full),
                  (True, fam_full), (False, RVOCAB), (True, RVOCAB)]
        BANNERS = BANNER5_CANON
        fallback_from = 4
        vocab0 = fam_plain
        rulesw = "the plain family rules"
    else:
        PASSES = [(False, PLAIN_VOCAB), (True, PLAIN_VOCAB),
                  (False, RVOCAB), (True, RVOCAB)]
        BANNERS = BANNER5_SCRAM
        fallback_from = 99
        vocab0 = PLAIN_VOCAB
        rulesw = "the plain rules"
    exhausted = False
    for pidx, (mode_try, vocab) in enumerate(PASSES):
        mrows, mlet, molet, _, _ = letterize(prompt, mode_try)
        mbyg, mcands, mconc = build_cands(mrows, vocab=vocab)
        vrows = [r for r in mrows if r.gl not in mconc]
        if not vrows:
            return None, 2, "no value rows but non-concat query"
        if any(not mcands[g] for g in mcands):
            g0 = next(g for g in mcands if not mcands[g])
            L.append(BANNERS[pidx])
            word = "family rule in this tier" if (canonical and pidx < fallback_from) \
                else "rule in this tier"
            L.append(f"{g0}: no {word} fits its result shapes -> this pass"
                     f" cannot work, move on.")
            info['n_exh'] += 1
            if pidx == len(PASSES) - 1:
                exhausted = True
            continue
        need = set()
        for r in vrows:
            need |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
        e2 = PolicyP2(vrows, {g: c for g, c in mcands.items() if g not in mconc},
                      truth=None)
        e2.dup_ok = dup_ok
        ep['rev'] = mode_try
        L.append(BANNERS[pidx])
        if pidx >= fallback_from:
            info['fallback'] = 1
        if mode_try:
            for r in mrows:
                L.append(f"EQ{r.k}: {r.lhs()} = {r.rstr()}")
            L.append(f"Query reads: {let[qL[1]]}{let[qL[0]]} {qg} {let[qL[4]]}{let[qL[3]]}")
            L.append("")
        qleads = ({let[qL[1]], let[qL[4]]} if mode_try
                  else {let[qL[0]], let[qL[3]]})
        pruned = prune_leading(e2, vrows, qleads)
        if pruned:
            L.append("No operand starts with 0 -> "
                     + ", ".join(pruned) + " != 0 (leading digits).")
        L.append("Deduce the digits:")
        budget = [POLICY_NODE_CAP]
        try:
            ok = e2.policy_branch(need, budget)
        except Budget as b:
            return None, 2, f"policy budget ({b}) pass {pidx} rev={mode_try}"
        except Contra as e:
            ok = False
            e2.steps.append(('ROOTCONTRA', e))
        if ok:
            if bail:
                return None, 2, "bail: puzzle not exhausting (a pass solved it)"
            rows, conc = mrows, mconc
            cands = mcands
            byg = mbyg
            rev = mode_try
            eng = e2
            rows_solve = rows
            info['mode_switch'] = 1 if rev else 0
            info['pass'] = pidx
            render_steps4(e2.steps, L, ep=ep)
            break
        st = e2.steps
        if st and st[-1][0] == 'ROOTCONTRA':
            render_steps4(st[:-1][:4], L, ep=None)
            L.append(f"Contradiction: {st[-1][1].reason}.")
        else:
            if st and st[-1][0] == 'PEXH':
                render_steps4(st[:-1], L, ep=None)
                _, l_, tried_ = st[-1]
                L.append(f"all digits for {l_} fail ({_trset(l_, tried_)}) -> no pin"
                         f" above {l_} to revise inside this reading -> an assumption"
                         f" is wrong.")
            else:
                render_steps4(st, L, ep=None)
        info['n_exh'] += 1
        if pidx == len(PASSES) - 1:
            exhausted = True
    if eng is None and not (bail and exhausted):
        return None, 2, "all policy passes refuted"
    if eng is None:
        # ---- FIX C: truthful terminal move for the despair state.
        # Every pass truly exhausted. One example line must be corrupt (the
        # generator never emits an inconsistent puzzle). Take the best consistent
        # partial reading: drop one example at a time (only ops with another
        # example, else that rule would be unknowable) and retry the highest-prior
        # pass; the first drop that turns consistent IS the corrupt line.
        if is_guess:
            return None, 2, "bail: guess rows not bailed"
        mrows, mlet, molet, _, _ = letterize(prompt, False)
        gcount = {}
        for r in mrows:
            gcount[r.gl] = gcount.get(r.gl, 0) + 1
        L.append("all passes exhausted (t{T}): no reading fits every example ->"
                 " one example line must be corrupt.")
        L.append("Terminate the search: take the best consistent partial reading -"
                 " drop one example at a time (only equations whose operator appears"
                 f" in another example) and retry standard digit order with {rulesw}:")
        for r0 in mrows:
            if gcount[r0.gl] < 2:
                continue
            sub_rows = [r for r in mrows if r.k != r0.k]
            sbyg, scands, sconc = build_cands(sub_rows, vocab=vocab0)
            svrows = [r for r in sub_rows if r.gl not in sconc]
            if not svrows or any(not scands[g] for g in scands):
                L.append(f"drop EQ{r0.k}: still impossible.")
                continue
            sneed = set()
            for r in svrows:
                sneed |= {r.a0, r.a1, r.b0, r.b1} | set(r.res)
            e6 = PolicyP2(svrows,
                          {g: c for g, c in scands.items() if g not in sconc},
                          truth=None)
            e6.dup_ok = dup_ok
            qleads6 = {let[qL[0]], let[qL[3]]}
            pruned6 = prune_leading(e6, svrows, qleads6)
            budget = [POLICY_NODE_CAP]
            try:
                ok6 = e6.policy_branch(sneed, budget)
            except (Budget, Contra):
                ok6 = False
            if not ok6:
                L.append(f"drop EQ{r0.k}: still impossible.")
                continue
            corrupt_k = r0.k
            L.append(f"drop EQ{r0.k}: consistent - EQ{r0.k} is the corrupt line,"
                     f" exclude it.")
            if pruned6:
                L.append("No operand starts with 0 -> "
                         + ", ".join(pruned6) + " != 0 (leading digits).")
            L.append("Deduce the digits:")
            ep['rev'] = False
            render_steps4(e6.steps, L, ep=ep)
            eng = e6
            rows = mrows
            rows_solve = sub_rows
            conc = sconc
            cands = scands
            byg = sbyg
            rev = False
            info['bail'] = 1
            break
        if eng is None:
            return None, 2, "bail: no single-drop recovery"
    def _count(steps):
        for s in steps:
            if s[0] == 'PTRY':
                if getattr(s[3], 'opexh', None) is not None:
                    info['n_opexh'] += 1
            elif s[0] == 'PDEEP':
                info['n_deep'] += 1
                _count(s[3])
            elif s[0] == 'PENTER':
                _count(s[4])
    _count(eng.steps)
    M = eng.assigned()
    if meta is not None:
        tm = {let[s]: d for s, d in meta['mapping'].items()
              if d is not None and s in let}
        for l, d in tm.items():
            if l in M and M[l] != d:
                return None, 2, f"policy map disagrees with solver at {l}"
        if bool(meta.get('rev')) != rev:
            return None, 2, "policy mode disagrees with solver"
    # ---- op pins: surviving ops at the final map (live rows only)
    vrows = [r for r in rows_solve if r.gl not in conc]
    survs = {}
    for g in eng.cands:
        surv = []
        for op in eng.cands[g]:
            okop = True
            for r in [r for r in vrows if r.gl == g]:
                a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
                fv = eng._fit(op, r, a, b)
                if fv is None:
                    okop = False; break
                v, digs = fv
                if any(M[r.res[k]] != digs[k] for k in range(r.rl)):
                    okop = False; break
            if okop:
                surv.append(op)
        if not surv:
            return None, 2, f"no op fits glyph {g} at final map"
        survs[g] = [op for op in C2.PRIORITY if op in surv]
    for g, rs in byg.items():
        if g in conc:
            pats = set.intersection(*[set(r.conc) for r in rs])
            survs[g] = [op for op in C2.PRIORITY if op in pats]
    glyph_order = sorted(survs)
    ba = C2._best_distinct([survs[g] for g in glyph_order], set(), dup_ok)
    if ba is not None:
        pins = dict(zip(glyph_order, ba[1]))
    else:
        vord = [g for g in glyph_order if g not in conc]
        ba2 = C2._best_distinct([survs[g] for g in vord], set(), dup_ok)
        if ba2 is None:
            return None, 2, "no distinct op assignment"
        pins = dict(zip(vord, ba2[1]))
        for g in glyph_order:
            if g in conc:
                pins[g] = survs[g][0]
    tie_lines = []
    taken = {op: g for g, op in pins.items()}
    for g in glyph_order:
        if g in conc or len(survs[g]) <= 1:
            continue
        top = survs[g][0]
        if pins[g] == top:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in survs[g])} on"
                             f" every example; keep the most common: {g} = {OP_WORD[top]}.")
        else:
            tie_lines.append(f"{g} fits {' and '.join(OP_WORD[o] for o in survs[g])};"
                             f" {OP_WORD[top]} is taken by {taken.get(top, '?')} and the"
                             f" rules are distinct -> {g} = {OP_WORD[pins[g]]}.")
    tier = 1 if (not is_guess and pins.get(qg) in ('concat_fwd', 'concat_rev')) else \
           (3 if (not is_guess and pins.get(qg) in MUL_FAM | SMALL_FAM
                  | {'a2_plus_b', 'a_plus_b2', 'lcm', 'gcd'}) else 2)
    L.append("")
    L.append("Locked map: " + ", ".join(f"{l}={d}" for l, d in sorted(M.items())))
    if len(set(M.values())) != len(M):
        return None, tier, "injectivity violated in locked map"
    L.append("Distinct check: " + " ".join(str(d) for _, d in sorted(M.items()))
             + " - all different. ok")
    # ---- rule-pin episodes (live rows only)
    gone_r = {}
    narrative_live = {g: list(c) for g, c in cands.items() if g not in conc}
    for st in eng.steps:
        if st[0] == 'ELIM':
            _, r0, _a0, _b0, tries0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(
                op for op, _v in tries0 if op not in surv0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
        elif st[0] == 'ELIM2':
            _, r0, _a0, _b0, vals0, opk0, _M0 = st
            gone_r.setdefault(r0.gl, set()).update(op for op, _v in vals0)
            narrative_live[r0.gl] = [opk0]
        elif st[0] == 'OPSHRINK':
            _, r0, gone0, surv0 = st
            gone_r.setdefault(r0.gl, set()).update(gone0)
            narrative_live[r0.gl] = [op for op in narrative_live[r0.gl] if op in surv0]
    byg2 = {}
    for r in vrows:
        byg2.setdefault(r.gl, []).append(r)
    dbl5 = None
    if canonical:
        inv_olet5 = {l: ch for ch, l in olet.items()}
        dbl5 = {g: DBL_FAM5.get(inv_olet5.get(g), ())
                for g in narrative_live}
    pin_lines, n_no = pin_episodes_r4(byg2, conc, narrative_live, pins,
                                      {g: set(s) for g, s in survs.items()},
                                      M, let, rev, gone_r,
                                      cap=ep['wrong_left'], have_no=ep['n_elim2'],
                                      dbl_cands=dbl5)
    if n_no + ep['n_elim2'] == 0 and corrupt_k is None:
        return None, tier, "no truthful negative episode renderable"
    L.extend(pin_lines)
    L.extend(tie_lines)
    L.append("Locked rules: " + ", ".join(f"{g} = {OP_WORD[pins[g]]}" for g in sorted(pins)))
    L.append("")
    # ---- verify: decode the RHS through the locked map FIRST, then compute
    L.append("Verify every example (decode RHS first, then compute):")
    for r in rows:
        op = pins.get(r.gl)
        Lr, Rr = r.raw
        opch_row = Lr[2]
        if corrupt_k is not None and r.k == corrupt_k:
            # truthful mismatch line for the diagnosed-corrupt example
            if op is None:
                return None, tier, "bail: corrupt glyph has no pinned rule"
            if any(c not in let or let[c] not in M for c in Rr):
                return None, tier, "bail: corrupt RHS symbol unmapped"
            if any(let[c] not in M for c in (Lr[0], Lr[1], Lr[3], Lr[4])):
                return None, tier, "bail: corrupt LHS symbol unmapped"
            dd = [M[let[c]] for c in Rr]
            exp_str = str(int(''.join(map(str, dd))))      # std order, no sign
            a = M[let[Lr[0]]] * 10 + M[let[Lr[1]]]
            b = M[let[Lr[3]]] * 10 + M[let[Lr[4]]]
            got = _got_str(op, a, b)
            if got is None or _cmp_eq(got, exp_str):
                return None, tier, "bail: corrupt example not refutable at final map"
            dec = " ".join(str(d) for d in dd)
            L.append(f"EQ{r.k}: RHS {Rr} -> {dec} -> {exp_str};"
                     f" compute {a} {OP_WORD[op]} {b} = {got}. {got} vs {exp_str}."
                     f" no - corrupt example, excluded.")
            continue
        if op in ('concat_fwd', 'concat_rev'):
            pat = (Lr[0] + Lr[1] + Lr[3] + Lr[4] if op == 'concat_fwd'
                   else Lr[3] + Lr[4] + Lr[0] + Lr[1])
            if Rr != pat:
                return None, tier, f"concat verify failed EQ{r.k}"
            order = 'in order' if op == 'concat_fwd' else 'swapped'
            L.append(f"EQ{r.k}: RHS {Rr} = operand symbols {order}"
                     f" ({Lr[0]}{Lr[1]} and {Lr[3]}{Lr[4]}). match")
            continue
        sign = False; side = None
        mag = Rr
        if len(Rr) > 1 and Rr[0] == opch_row:
            sign = True; side = 'leading'; mag = Rr[1:]
        elif len(Rr) > 1 and Rr[-1] == opch_row:
            sign = True; side = 'trailing'; mag = Rr[:-1]
        dd = [M[let[c]] for c in mag]
        msb = dd[::-1] if rev else dd
        exp_num = int(''.join(map(str, msb)))
        exp_str = ('-' if sign else '') + str(exp_num)
        a = M[r.a0] * 10 + M[r.a1]; b = M[r.b0] * 10 + M[r.b1]
        v = C2.OPS[op](a, b)
        if v is None:
            return None, tier, f"verify undefined EQ{r.k}"
        got = ('-' if (v < 0 or op in C2.NEGPRE) else '') + str(abs(v))
        if got != exp_str:
            return None, tier, f"verify failed EQ{r.k}: {got} != {exp_str}"
        note = f" ({side} {opch_row} = minus)" if sign else ""
        dec = " ".join(str(d) for d in dd)
        revw = "reversed " if rev else ""
        L.append(f"EQ{r.k}: RHS {Rr}{note} -> {dec} -> {revw}{exp_str};"
                 f" compute {a} {OP_WORD[op]} {b} = {got}. match")
    L.append("")
    # ---- apply to the query (no-leading-zero on free query letters too)
    qa_, qb_, qc_, qd_ = let[qL[0]], let[qL[1]], let[qL[3]], let[qL[4]]
    qlead = {qb_, qd_} if rev else {qa_, qc_}
    missing = [x for x in (qa_, qb_, qc_, qd_) if x not in M]
    if missing:
        free_digs = sorted(set(range(10)) - set(M.values()))
        ml = sorted(set(missing))
        if len(ml) == 1 and len(free_digs) == 1:
            if ml[0] in qlead and free_digs[0] == 0:
                return None, tier, "query leading letter forced to 0"
            M[ml[0]] = free_digs[0]
            L.append(f"{ml[0]} appears only in the query; the one unused digit is"
                     f" {free_digs[0]}, so {ml[0]} = {free_digs[0]}.")
        elif len(ml) <= len(free_digs) and len(ml) <= 3:
            avail = list(free_digs)
            asg = {}
            lead_noted = []
            for l in ml:
                pick = next((d for d in avail
                             if not (l in qlead and d == 0)), None)
                if pick is None:
                    return None, tier, "query leading letter forced to 0"
                if l in qlead and 0 in avail:
                    lead_noted.append(l)
                asg[l] = pick
                avail.remove(pick)
            for l, d in asg.items():
                M[l] = d
            note = ""
            if lead_noted:
                note = (" (" + ", ".join(f"{l} != 0, leading digit"
                                         for l in lead_noted) + ")")
            L.append("Letters " + ", ".join(ml) + " appear only in the query;"
                     f" unused digits are {{{','.join(map(str, free_digs))}}};"
                     " no example constrains them - take them in order" + note + ": "
                     + ", ".join(f"{l} = {asg[l]}" for l in ml) + ".")
        else:
            return None, tier, "query letters not deducible by policy"
    if is_guess:
        used_ops = set(pins.values())
        usedw = ", ".join(OP_WORD[o] for o in pins.values())
        if canonical:
            order = [o for o in famA_order(qop_ch) if o not in used_ops]
            if not order:
                return None, tier, "guess: family exhausted"
            qapply = order[0]
            famword = FAM_WORD5[qop_ch]
            L.append(f"The query operator {qg} never appears in the examples, but"
                     f" its glyph is '{qop_ch}' itself -> a {famword}-family rule."
                     f" The rules in one puzzle are all different ({usedw} taken)"
                     f" -> bet {qg} = {OP_WORD[qapply]}.")
        else:
            qapply = next((o for o in opB_order()
                           if o not in used_ops and o in RVOCAB + ['concat_fwd']),
                          None)
            if qapply is None:
                return None, tier, "guess: no candidate op"
            L.append(f"The query operator {qg} never appears in the examples. The rules"
                     f" in one puzzle are all different, so {qg} is not"
                     f" {usedw}. The most common remaining rule is"
                     f" {OP_WORD[qapply]} -> bet {qg} = {OP_WORD[qapply]}.")
        if qapply in ('concat_fwd', 'concat_rev'):
            s1, s2 = qL[0] + qL[1], qL[3] + qL[4]
            syms = s1 + s2 if qapply == 'concat_fwd' else s2 + s1
            L.append(f"Applying concatenation: take the query operand symbols:"
                     f" -> {syms}")
            L.append("")
            L.append(f"\\boxed{{{syms}}}")
            if syms != answer:
                return None, tier, "guess policy miss"
            return _finish6(L), tier, info
    else:
        qapply = pins[qg]
    if rev:
        A = M[qb_] * 10 + M[qa_]; B = M[qd_] * 10 + M[qc_]
        L.append(f"Apply to the query (little-endian): {qb_}{qa_} {qg} {qd_}{qc_}"
                 f" = {A} {OP_WORD[qapply]} {B}")
    else:
        A = M[qa_] * 10 + M[qb_]; B = M[qc_] * 10 + M[qd_]
        L.append(f"Apply to the query: {qa_}{qb_} {qg} {qc_}{qd_} = {A} {OP_WORD[qapply]} {B}")
    if A < 10 or B < 10:
        return None, tier, "query operand leading zero under locked map"
    v = C2.OPS[qapply](A, B)
    if v is None:
        return None, tier, "query value undefined"
    sign_needed = (qapply in C2.NEGPRE) or (qapply in C2.SIGNED and v < 0)
    if v < 0 and not sign_needed:
        return None, tier, "negative query value for unsigned op"
    mag = abs(v)
    digs = [int(c) for c in str(mag)]
    inv_d = {d: l for l, d in M.items()}
    miss_d = sorted({d for d in digs if d not in inv_d})
    if miss_d:
        unas = sorted(set(let.values()) - set(M))
        if len(miss_d) == 1 and len(unas) == 1:
            M[unas[0]] = miss_d[0]
            inv_d[miss_d[0]] = unas[0]
            L.append(f"Digit {miss_d[0]} never appeared in a value example; the only"
                     f" letter without a digit is {unas[0]} -> {unas[0]} = {miss_d[0]}.")
        elif len(miss_d) <= len(unas) and len(miss_d) <= 2:
            for d, l in zip(miss_d, unas):
                M[l] = d
                inv_d[d] = l
            L.append("Digits " + ",".join(map(str, miss_d)) + " never appeared in a"
                     " value example; letters without digits: " + ",".join(unas)
                     + " - take them in order: "
                     + ", ".join(f"{l} = {d}" for d, l in zip(miss_d, unas)) + ".")
        else:
            return None, tier, f"answer digit unmapped: {mag}"
    dig2sym = {d: inv_let[inv_d[d]] for d in set(digs)}
    L.append(f"= {'-' if (v < 0 or qapply in C2.NEGPRE) else ''}{mag}")
    disp = digs[::-1] if rev else digs
    pairs = " ".join(f"{d}={dig2sym[d]}" for d in disp)
    core = ''.join(dig2sym[d] for d in disp)
    if rev:
        L.append(f"Little-endian, write digits reversed: {mag} ->"
                 f" {''.join(map(str, disp))}; decode digit by digit: {pairs} -> {core}")
    else:
        L.append(f"Decode digit by digit: {pairs} -> {core}")
    syms = core
    if sign_needed:
        if sign_end == 'suf':
            syms = core + qop_ch
            L.append(f"Negative -> append the operator glyph {qop_ch} as the sign"
                     f" (as in the examples) -> {syms}")
        else:
            syms = qop_ch + core
            L.append(f"Negative -> prefix the operator glyph {qop_ch} as the sign"
                     f" -> {syms}")
    L.append("")
    L.append(f"\\boxed{{{syms}}}")
    if syms != answer:
        return None, tier, f"final mismatch {syms!r} != {answer!r}"
    info['branches'] = eng.nbranch
    info['n_no'] = n_no + ep['n_elim2']
    info['n_collide'] = ep['n_collide']
    return _finish6(L), tier, info

# ---------------------------------------------------------------- r6 lint
def _lint_table6(lines, prompt):
    """FIX A lint: the symbol table must be DERIVED. Requires (1) one truthful
    positional scan line per example + query, (2) the union table exactly equal
    to the first-appearance walk of the rendered scans (ghost entries / invented
    operator sets impossible), (3) a truthful op/digit disjointness line."""
    pr = C2.parse(prompt)
    if pr is None:
        return "prompt unparseable (r6)"
    eqs, qL = pr
    mi = next((i for i, ln in enumerate(lines) if TABLE6_MARK in ln), None)
    if mi is None:
        return "table construction block missing"
    idx = mi + 1
    digit_walk = []      # symbols in rendered scan order
    op_walk = []
    for k, (L0, R0) in enumerate(eqs, 1):
        if idx >= len(lines):
            return f"table scan line missing for EQ{k}"
        m = _SCAN6_RE.match(lines[idx])
        if not m or int(m.group(1)) != k:
            return f"table scan line missing for EQ{k}"
        if m.group(2) != L0:
            return f"table scan EQ{k}: LHS != prompt LHS"
        if (m.group(3), m.group(4), m.group(5), m.group(6)) != (L0[0], L0[1], L0[3], L0[4]):
            return f"table scan EQ{k}: digit positions wrong"
        if m.group(7) != L0[2]:
            return f"table scan EQ{k}: op position wrong"
        if m.group(8) != R0:
            return f"table scan EQ{k}: RHS != prompt RHS"
        opch = L0[2]
        side = None
        mag = R0
        if len(R0) > 1 and R0[0] == opch:
            side = 'leading'; mag = R0[1:]
        elif len(R0) > 1 and R0[-1] == opch:
            side = 'trailing'; mag = R0[:-1]
        desc = m.group(9)
        if side:
            md = _RHSDESC6_SIGN_RE.match(desc)
            if not md or md.group(1) != side or md.group(2) != opch:
                return f"table scan EQ{k}: sign annotation wrong"
            if md.group(3).split(' ') != list(mag):
                return f"table scan EQ{k}: RHS digit list wrong"
        else:
            md = _RHSDESC6_RE.match(desc)
            if not md or md.group(1).split(' ') != list(mag):
                return f"table scan EQ{k}: RHS digit list wrong"
        digit_walk += [L0[0], L0[1], L0[3], L0[4]] + list(mag)
        if opch not in op_walk:
            op_walk.append(opch)
        idx += 1
    m = _QSCAN6_RE.match(lines[idx]) if idx < len(lines) else None
    if not m:
        return "table scan line missing for the query"
    if m.group(1) != qL:
        return "query scan: LHS != prompt query"
    if (m.group(2), m.group(3), m.group(4), m.group(5)) != (qL[0], qL[1], qL[3], qL[4]):
        return "query scan: digit positions wrong"
    if m.group(6) != qL[2]:
        return "query scan: op position wrong"
    digit_walk += [qL[0], qL[1], qL[3], qL[4]]
    if qL[2] not in op_walk:
        op_walk.append(qL[2])
    idx += 1
    if idx >= len(lines) or lines[idx] != "Digit symbols in order of first appearance:":
        return "digit-table header missing"
    idx += 1
    pairs = []
    while idx < len(lines):
        m = _TABLED6_RE.match(lines[idx])
        if not m:
            break
        pairs.append((m.group(1), m.group(2)))
        idx += 1
    if idx >= len(lines) or lines[idx] != "Operators:":
        return "operator-table header missing"
    idx += 1
    opairs = []
    while idx < len(lines):
        m = _TABLEO6_RE.match(lines[idx])
        if not m:
            break
        opairs.append((m.group(1), m.group(2)))
        idx += 1
    # expected union: first-appearance walk over the rendered (= prompt) scans
    exp_digits = []
    for s in digit_walk:
        if s not in exp_digits:
            exp_digits.append(s)
    exp_pairs = [(s, LETTERS[i]) for i, s in enumerate(exp_digits)]
    if pairs != exp_pairs:
        return ("digit table does not match the positional scan walk"
                f" (got {len(pairs)} entries)")
    exp_opairs = [(ch, OPLETTERS[min(i, 3)]) for i, ch in enumerate(op_walk)]
    if opairs != exp_opairs:
        return "operator table does not match the scanned operator set"
    if idx >= len(lines) or DISJ6_MARK not in lines[idx]:
        return "disjointness check line missing"
    if not lines[idx].startswith("Operator chars " + " ".join(op_walk)):
        return "disjointness line lists the wrong operator chars"
    if set(op_walk) & set(exp_digits):
        return "disjointness line untruthful (op char in a digit position)"
    return None

def _lint_bail6(lines, prompt, meta):
    """FIX C lint: bail structure + the truthful corrupt-example mismatch line.
    Returns (err, skip_eq)."""
    pr = C2.parse(prompt) if prompt else None
    eqs = pr[0] if pr else None
    bi = next((i for i, ln in enumerate(lines) if BAIL_MARK in ln), None)
    if bi is None:
        return "bail mark missing", None
    drops = []
    for ln in lines[bi:]:
        m = _DROP6_FAIL_RE.match(ln)
        if m:
            drops.append((int(m.group(1)), False))
            continue
        m = _DROP6_OK_RE.match(ln)
        if m:
            if m.group(1) != m.group(2):
                return "bail: corrupt-line indices disagree", None
            drops.append((int(m.group(1)), True))
    if not drops:
        return "bail: no drop-retry lines", None
    ks = [k for k, _ in drops]
    if ks != sorted(ks) or len(set(ks)) != len(ks):
        return "bail: drop order not increasing", None
    if any(ok for k, ok in drops[:-1]) or not drops[-1][1]:
        return "bail: consistent drop must be exactly the last", None
    skip_eq = drops[-1][0]
    if eqs is not None and not (1 <= skip_eq <= len(eqs)):
        return "bail: corrupt EQ index out of range", None
    # the truthful mismatch line for the corrupt example
    cm = None
    for ln in lines:
        m = _CORRUPTV6_RE.match(ln)
        if m:
            if cm is not None:
                return "bail: multiple corrupt verify lines", None
            cm = m
    if cm is None:
        return "bail: corrupt verify line missing", None
    if int(cm.group(1)) != skip_eq:
        return "bail: corrupt verify line indexes the wrong EQ", None
    dd = [int(x) for x in cm.group(3).split()]
    if int(cm.group(4)) != int(''.join(map(str, dd))):
        return "bail: corrupt decode value wrong", None
    a, opw, b = int(cm.group(5)), cm.group(6), int(cm.group(7))
    gotv = cm.group(8)
    op = WORD_OP.get(opw)
    if op is None:
        return f"bail: unknown op word {opw!r}", None
    v = C2.OPS[op](a, b)
    if v is not None and op in C2.NEGPRE:
        v = -abs(v)
    if v is None or str(v) != gotv:
        return f"bail: corrupt-line arithmetic invented ({a} {opw} {b} != {gotv})", None
    if cm.group(9) != gotv or cm.group(10) != cm.group(4):
        return "bail: corrupt comparator values not derived", None
    if _cmp_eq(cm.group(9), cm.group(10)):
        return "bail: corrupt line claims 'no' on equal values", None
    if eqs is not None:
        Lp, Rp = eqs[skip_eq - 1]
        if cm.group(2) != Rp:
            return "bail: corrupt RHS != prompt RHS", None
        if meta is not None:
            mapping = meta['mapping']
            try:
                tru = [mapping[c] for c in Rp]
            except KeyError:
                return "bail: corrupt RHS symbol outside solver map", None
            if any(t is None for t in tru) or dd != tru:
                return "bail: corrupt decode != solver-map decode", None
            ta = mapping.get(Lp[0], None), mapping.get(Lp[1], None)
            tb = mapping.get(Lp[3], None), mapping.get(Lp[4], None)
            if None in ta or None in tb or a != ta[0] * 10 + ta[1] or b != tb[0] * 10 + tb[1]:
                return "bail: corrupt operand decode mismatch", None
    return None, skip_eq

def lint_r6(cot, answer, prompt=None, meta=None):
    """r6 lint = counter monotonicity (FIX B) + table-construction presence/
    truthfulness (FIX A) + bail structure (FIX C, with the corrupt example
    exempted from verify completeness) + full lint_r5 on the counter-stripped
    text (every r3/r4/r5 check preserved verbatim)."""
    lines = cot.splitlines()
    seen_n = 0
    for ln in lines:
        m = _T_RE.match(ln)
        if m:
            k = int(m.group(1))
            if k != seen_n + 1:
                return f"counter not monotone: t{k} after t{seen_n}"
            seen_n = k
            if not _CTBL_RE.match(m.group(2)):
                return f"counter on a non-try line: {ln[:48]}"
        elif _CTBL_RE.match(ln):
            return f"try line without counter: {ln[:48]}"
        bm = _BAILREF_RE.search(ln)
        if bm and int(bm.group(1)) != seen_n:
            return f"bail counter t{bm.group(1)} != tries so far ({seen_n})"
    stripped = strip_counters6(cot)
    slines = stripped.splitlines()
    if prompt is not None:
        err = _lint_table6(slines, prompt)
        if err:
            return err
    skip_eq = None
    if BAIL_MARK in cot:
        err, skip_eq = _lint_bail6(slines, prompt, meta)
        if err:
            return err
    return lint_r5(stripped, answer, prompt=prompt, meta=meta, skip_eq=skip_eq)

# ---------------------------------------------------------------- r6 bail generator
def gen_puzzle_r6_bail(rng):
    """FIX C construction (one attempt; returns None to resample). Start from a
    solvable two-regime puzzle P that the r6 policy solves at pass 0 (std+plain).
    Append ONE corrupted example (existing value glyph, operands/RHS drawn from
    symbols P already uses, RHS a random non-concat digit string) at a random
    position. The full pass schedule on P' then has to exhaust (verified by
    execution inside render_trace_r6); dropping the corrupted line restores P,
    so the bail's best-consistent-partial-map decode IS gold by construction."""
    p = gen_puzzle_r5(rng)
    if (p is None or p['category'] != 'cryptarithm_deduce' or p['rev']
            or p['qop'] in C2.CONCATS):
        return None
    pr = C2.parse(p['prompt'])
    if pr is None:
        return None
    eqs, qL = pr
    cot0, _t0, info0 = render_trace_r6(p['prompt'], p['answer'],
                                       ep_rng=random.Random(0), meta=None)
    if cot0 is None or not isinstance(info0, dict) or info0.get('pass') != 0:
        return None
    inv = {s: d for d, s in p['map'].items()}
    ex_syms = set()
    for L0, R0 in eqs:
        ex_syms |= {L0[0], L0[1], L0[3], L0[4]}
        mag = R0
        if len(R0) > 1 and R0[0] == L0[2]:
            mag = R0[1:]
        elif len(R0) > 1 and R0[-1] == L0[2]:
            mag = R0[:-1]
        ex_syms |= set(mag)
    S = sorted(inv[s] for s in ex_syms if s in inv)
    Sl = [d for d in S if d != 0]
    if len(S) < 3 or not Sl:
        return None
    valg = [g for g in sorted({L0[2] for L0, _ in eqs})
            if p['ops'].get(g) not in C2.CONCATS]
    if not valg:
        return None
    g0 = rng.choice(valg)
    smap = p['map']
    lhs = rhs = None
    for _ in range(30):
        a0, a1 = rng.choice(Sl), rng.choice(S)
        b0, b1 = rng.choice(Sl), rng.choice(S)
        cl = smap[a0] + smap[a1] + g0 + smap[b0] + smap[b1]
        rl = rng.choice((2, 3))
        rd = [rng.choice(Sl)] + [rng.choice(S) for _ in range(rl - 1)]
        cr = ''.join(smap[d] for d in rd)
        if cr in (cl[0] + cl[1] + cl[3] + cl[4], cl[3] + cl[4] + cl[0] + cl[1]):
            continue
        true_v = C2.OPS[p['ops'][g0]](a0 * 10 + a1, b0 * 10 + b1)
        if true_v is not None and str(abs(true_v)) == ''.join(map(str, rd)):
            continue        # accidentally consistent with the glyph's own rule
        lhs, rhs = cl, cr
        break
    if lhs is None:
        return None
    k0 = rng.randint(1, len(eqs) + 1)
    exlines = [f"{L0} = {R0}" for L0, R0 in eqs]
    exlines.insert(k0 - 1, f"{lhs} = {rhs}")
    prompt = HDR + "\n" + "\n".join(exlines) + "\n" + QSTR + qL
    return {'prompt': prompt, 'answer': p['answer'], 'meta': synth_meta(p),
            'corrupt_k': k0, 'regime': p['regime'],
            'category': 'cryptarithm_deduce'}

# ---------------------------------------------------------------- r6 pipelines
def _emit6(kept, drop, tl, infos, prompt, ans, meta, tk, rid, cat, bail=False):
    ep_rng = random.Random(zlib.crc32(str(rid).encode('utf-8')))
    cot, tier, info = render_trace_r6(prompt, ans, ep_rng=ep_rng, meta=meta, bail=bail)
    if cot is None:
        key = f'render-t{tier}: {str(info)[:48]}'
        drop[key] = drop.get(key, 0) + 1
        return False
    err = lint_r6(cot, ans, prompt=prompt, meta=meta)
    if err:
        drop[f'lint: {err[:48]}'] = drop.get(f'lint: {err[:48]}', 0) + 1
        return False
    ntok = len(tk.encode(cot).ids)
    if ntok >= 5000:
        drop['too-long'] = drop.get('too-long', 0) + 1
        return False
    tl[tier].append(ntok)
    infos.append(info)
    kept.append({'id': rid, 'category': cat, 'prompt': prompt,
                 'cot': cot, 'final': ans, 'tier': tier, 'ntok': ntok})
    return True

def render_real6(out_path, deadline=10.0, limit=None):
    C2.HARD_OPND_LO = True
    reg = C2.load_regime(EM_PRIORS_PATH)
    tk = tokenizer()
    vids = val_ids()
    rows = load_real('cryptarithm_deduce')
    if limit:
        rows = rows[:limit]
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    for r in rows:
        if r['id'] in vids:
            continue
        ans = r['answer'].strip()
        res = C2.solve(r['prompt'], gold=ans, deadline_s=deadline, regime=reg)
        if res is None:
            drop['gold-unsolvable'] = drop.get('gold-unsolvable', 0) + 1
            continue
        _, meta = res
        _emit6(kept, drop, tl, infos, r['prompt'], ans, meta, tk, r['id'], r['category'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl, infos

def render_synth6(out_path, n, seed=181, bail_frac=P_BAIL6):
    """normal two-regime rows (1 - bail_frac) + bail-out rows (bail_frac)."""
    tk = tokenizer()
    rng = random.Random(seed)
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    n_bail = int(round(n * bail_frac))
    n_norm = n - n_bail
    tries = 0
    while len(kept) < n_norm and tries < n * 40:
        tries += 1
        p = gen_puzzle_r5(rng)
        if p is None:
            continue
        _emit6(kept, drop, tl, infos, p['prompt'], p['answer'], synth_meta(p), tk,
               f'synth6-s{seed}-{len(kept):05d}', p['category'])
    nb = 0
    tries = 0
    while nb < n_bail and tries < n_bail * 400:
        tries += 1
        bp = gen_puzzle_r6_bail(rng)
        if bp is None:
            continue
        if _emit6(kept, drop, tl, infos, bp['prompt'], bp['answer'], bp['meta'], tk,
                  f'synth6b-s{seed}-{nb:04d}', bp['category'], bail=True):
            nb += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    return kept, drop, tl, infos

# ---- r6 policy coverage (same protocol as r4/r5: truth-free, no bail, gold
# only gates the boxed string; success = boxed == gold AND <= 3500 tokens)
def _cov6_work(r):
    ep_rng = random.Random(zlib.crc32(str(r['id']).encode('utf-8')))
    ans = r['answer'].strip()
    t0 = time.time()
    try:
        cot, tier, info = render_trace_r6(r['prompt'], ans, ep_rng=ep_rng, meta=None)
    except Exception as e:
        return {'id': r['id'], 'ok': False, 'why': f'EXC {type(e).__name__}: {e}'[:80],
                'ntok': None, 'dt': time.time() - t0}
    if cot is None:
        return {'id': r['id'], 'ok': False, 'why': str(info)[:80], 'ntok': None,
                'dt': time.time() - t0}
    ntok = len(_COV['tk'].encode(cot).ids)
    lint_err = lint_r6(cot, ans, prompt=r['prompt'])
    return {'id': r['id'], 'ok': ntok <= 3500, 'why': 'solved' if ntok <= 3500 else 'over-3500',
            'ntok': ntok, 'lint': lint_err, 'info': info, 'tier': tier,
            'dt': time.time() - t0}

def coverage6(procs=8, out=None):
    rows = load_real('cryptarithm_deduce')
    import multiprocessing as mp
    with mp.Pool(procs, initializer=_cov_init) as pool:
        res = pool.map(_cov6_work, rows, chunksize=8)
    ok = sum(r['ok'] for r in res)
    print(f"R6 POLICY COVERAGE (boxed==gold AND ntok<=3500): {ok}/{len(rows)}"
          f" = {ok/len(rows):.4f}")
    fails = {}
    for r in res:
        if not r['ok']:
            fails[r['why']] = fails.get(r['why'], 0) + 1
    for w, c in sorted(fails.items(), key=lambda t: -t[1]):
        print(f"  FAIL {c:4d}  {w}")
    toks = sorted(r['ntok'] for r in res if r['ntok'])
    if toks:
        print(f"tokens (solved-any): med {toks[len(toks)//2]}"
              f" p95 {toks[int(len(toks)*.95)]} max {toks[-1]}")
    solved = [r for r in res if r['ok']]
    lint_bad = sum(1 for r in solved if r.get('lint'))
    print(f"lint failures among covered: {lint_bad}")
    can = [r for r in solved if r.get('info', {}).get('canonical')]
    print(f"covered canonical: {len(can)} | covered scrambled: {len(solved) - len(can)}")
    print('episodes:', episode_stats([r['info'] for r in solved if r.get('info')]))
    if out:
        with open(out, 'w') as f:
            for r in res:
                r.pop('info', None)
                f.write(json.dumps(r) + '\n')
    return ok / len(rows)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stats'
    if cmd == 'render-real':
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        kept, drop, tl = render_real(f"{ROOT}/pipeline/data/crypt_r3/crypt_deduce_real.jsonl", limit=lim)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
    elif cmd == 'render-synth':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        out = sys.argv[4] if len(sys.argv) > 4 else \
            f"{ROOT}/pipeline/data/crypt_r3/crypt_deduce_synth.jsonl"
        kept, drop, tl = render_synth(out, n, seed=seed)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
    elif cmd == 'gen-test':
        rng = random.Random(7)
        for _ in range(3):
            p = gen_puzzle(rng)
            if p:
                print(p['prompt'])
                print('ANSWER:', p['answer'], '| ops:', p['ops'], 'rev:', p['rev'])
                print('---')
    elif cmd == 'render-real4':
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        kept, drop, tl, infos = render_real4(
            f"{ROOT}/pipeline/data/crypt_r4/crypt_deduce_real.jsonl", limit=lim)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
        print('episodes:', episode_stats(infos))
    elif cmd == 'render-synth4':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 750
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 151
        out = sys.argv[4] if len(sys.argv) > 4 else \
            f"{ROOT}/pipeline/data/crypt_r4/shards/synth_s{seed}.jsonl"
        kept, drop, tl, infos = render_synth4(out, n, seed=seed)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
        print('episodes:', episode_stats(infos))
    elif cmd == 'coverage':
        procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        out = sys.argv[3] if len(sys.argv) > 3 else None
        coverage(procs=procs, out=out)
    elif cmd == 'render-real5':
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        kept, drop, tl, infos = render_real5(
            f"{ROOT}/pipeline/data/crypt_r5/crypt_deduce_real.jsonl", limit=lim)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
        print('episodes:', episode_stats(infos))
        print('canonical kept:', sum(i.get('canonical', 0) for i in infos))
    elif cmd == 'render-synth5':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 750
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 171
        out = sys.argv[4] if len(sys.argv) > 4 else \
            f"{ROOT}/pipeline/data/crypt_r5/shards/synth_s{seed}.jsonl"
        kept, drop, tl, infos = render_synth5(out, n, seed=seed)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
        print('episodes:', episode_stats(infos))
        print('canonical kept:', sum(i.get('canonical', 0) for i in infos))
    elif cmd == 'coverage5':
        procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        out = sys.argv[3] if len(sys.argv) > 3 else None
        coverage5(procs=procs, out=out)
    elif cmd == 'render-real6':
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        kept, drop, tl, infos = render_real6(
            f"{ROOT}/pipeline/data/crypt_r6/crypt_deduce_real.jsonl", limit=lim)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
        print('episodes:', episode_stats(infos))
        print('canonical kept:', sum(i.get('canonical', 0) for i in infos))
    elif cmd == 'render-synth6':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 750
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 181
        out = sys.argv[4] if len(sys.argv) > 4 else \
            f"{ROOT}/pipeline/data/crypt_r6/shards/synth_s{seed}.jsonl"
        kept, drop, tl, infos = render_synth6(out, n, seed=seed)
        print('kept', len(kept), 'drop', json.dumps(drop, indent=0))
        print('tokens per tier:', stats(tl))
        print('episodes:', episode_stats(infos))
        print('canonical kept:', sum(i.get('canonical', 0) for i in infos))
        print('bail kept:', sum(i.get('bail', 0) for i in infos))
    elif cmd == 'coverage6':
        procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        out = sys.argv[3] if len(sys.argv) > 3 else None
        coverage6(procs=procs, out=out)
    elif cmd == 'gen-test6':
        rng = random.Random(7)
        got = 0
        while got < 2:
            bp = gen_puzzle_r6_bail(rng)
            if not bp:
                continue
            cot, t, info = render_trace_r6(bp['prompt'], bp['answer'],
                                           ep_rng=random.Random(1),
                                           meta=bp['meta'], bail=True)
            if cot is None:
                continue
            got += 1
            print(bp['prompt'])
            print('--- corrupt_k:', bp['corrupt_k'], '| regime:', bp['regime'])
            print(cot)
            print('=====')
    elif cmd == 'gen-test5':
        rng = random.Random(7)
        for _ in range(4):
            p = gen_puzzle_r5(rng)
            if p:
                print(p['prompt'])
                print('ANSWER:', p['answer'], '| regime:', p['regime'],
                      '| ops:', p['ops'], 'rev:', p['rev'])
                print('---')
