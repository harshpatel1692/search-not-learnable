"""cryptarithm2 -- gold-free cryptarithm solver (deduce + guess).

Puzzle spec (verified on all 823 train rows):
  * each example line: `s0 s1 OP s2 s3 = RHS` (LHS exactly 5 chars), operator = char index 2.
  * per-row injective digit->symbol cipher over radix digits; op glyphs never collide with digit symbols.
  * per-glyph operation from a wide vocab (add/mul/sub/absdiff/mod/gcd/lcm/fdiv/a2_plus_b/concat...).
  * global mode per row: standard | little_endian (operand digit strings AND result string reversed).
  * negative results render the row's own op glyph as sign: prefix in standard, suffix after reversal.

Algorithm:
  1. parse + (per mode hypothesis) normalize each equation to MSB-first index form.
  2. per-glyph candidate ops: structural concat detection, sign-pattern filter,
     magnitude length/value-range filter (operands are 2-digit, leading digit nonzero).
  3. DFS over digit assignment of constrained symbols (operand symbols first, most-shared first)
     with two sound prunes: full-operand forcing (operands assigned => result digits forced)
     and units-digit congruence. ENUMERATES all solutions (up to cap).
  4. answers voted across solutions by measured generator prior (op/mode marginals).

Gold-conditioning (data-gen only): pass gold= to append the query equation as a constraint;
gold symbols unseen elsewhere join the search as fresh symbols (injectivity still applies).
"""
import math, time

# ---------------------------------------------------------------- value ops
def _fd(a, b): return a // b if b else None
def _rd(a, b): return b // a if a else None
def _mo(a, b): return a % b if b else None
def _rm(a, b): return b % a if a else None
def _lcm(a, b): return a * b // math.gcd(a, b) if (a and b) else 0

def _off(fn, d):
    def g(a, b):
        v = fn(a, b)
        return None if v is None else v + d
    return g

_add = lambda a, b: a + b
_mul = lambda a, b: a * b
_ad  = lambda a, b: abs(a - b)

OPS = {
    'add': _add, 'add_p1': _off(_add, 1), 'add_m1': _off(_add, -1),
    'add_p2': _off(_add, 2), 'add_m2': _off(_add, -2),
    'mul': _mul, 'mul_p1': _off(_mul, 1), 'mul_m1': _off(_mul, -1),
    'mul_p2': _off(_mul, 2), 'mul_m2': _off(_mul, -2),
    'sub_signed': lambda a, b: a - b, 'rsub_signed': lambda a, b: b - a,
    'absdiff': _ad, 'absdiff_p1': _off(_ad, 1), 'absdiff_m1': _off(_ad, -1),
    'absdiff_p2': _off(_ad, 2), 'absdiff_m2': _off(_ad, -2),
    'neg_absdiff': _ad,
    'a2_plus_b': lambda a, b: a * a + b, 'a_plus_b2': lambda a, b: a + b * b,
    'mod': _mo, 'rmod': _rm, 'gcd': math.gcd, 'lcm': _lcm,
    'fdiv': _fd, 'rdiv': _rd,
}
SIGNED = {'sub_signed', 'rsub_signed'}     # sign rendered iff value < 0
NEGPRE = {'neg_absdiff'}                   # sign always rendered
CONCATS = ('concat_fwd', 'concat_rev')

# measured generator prior: gold-conditioned solve of all 823 cryptarithm train rows
# (798 solved; pipeline/data/cryptarithm_gold_meta.jsonl). Per-glyph op counts,
# de-biased by EM: unambiguous fits counted directly, ambiguous fit-sets
# (e.g. {lcm,mul} x160, {neg_absdiff,sub_signed} x144) distributed by base rate.
MEASURED_OP_COUNTS = {
    'mul': 377.6, 'add': 343.0, 'sub_signed': 316.0, 'concat_fwd': 158.0,
    'absdiff': 101.3, 'neg_absdiff': 89.9, 'add_m1': 83.0, 'mul_m1': 81.0,
    'add_p1': 81.0, 'mul_p1': 78.0, 'rsub_signed': 51.4, 'concat_rev': 39.0,
    'add_p2': 15.0, 'absdiff_m2': 14.0, 'mod': 13.2, 'rmod': 11.2,
    'absdiff_p2': 6.0, 'a2_plus_b': 6.0, 'mul_p2': 5.0, 'mul_m2': 4.0,
    'a_plus_b2': 4.0, 'absdiff_m1': 4.0, 'lcm': 3.4, 'gcd': 2.0, 'rdiv': 2.0,
    'absdiff_p1': 2.0, 'add_m2': 2.0, 'fdiv': 0.5,   # fdiv unobserved; keep reachable
}
MEASURED_MODE_COUNTS = {False: 493, True: 305}
MEASURED_RADIX_COUNTS = {10: 798, 11: 0}
OP_LOGP = {}
MODE_LOGP = {False: 0.0, True: -1.5}
RADIX_LOGP = {10: 0.0, 11: -5.0}
_PRIOR_FLOOR = -11.0

def op_logp(op):
    return OP_LOGP.get(op, _PRIOR_FLOOR)

# priority = descending prior (rebuilt by set_prior)
PRIORITY = list(OPS) + list(CONCATS)

def set_prior(op_counts, mode_counts=None, radix_counts=None):
    """install measured counts as log-prior + rebuild PRIORITY."""
    global OP_LOGP, MODE_LOGP, RADIX_LOGP, PRIORITY
    tot = sum(op_counts.values()) or 1
    OP_LOGP = {o: math.log((c + 0.5) / tot) for o, c in op_counts.items()}
    PRIORITY = sorted(set(list(OPS) + list(CONCATS)),
                      key=lambda o: (-OP_LOGP.get(o, _PRIOR_FLOOR), o))
    if mode_counts:
        t = sum(mode_counts.values())
        MODE_LOGP = {k: math.log((mode_counts.get(k, 0) + 0.5) / t) for k in (False, True)}
    if radix_counts:
        t = sum(radix_counts.values())
        RADIX_LOGP = {k: math.log((radix_counts.get(k, 0) + 0.5) / t) for k in (10, 11)}

set_prior(MEASURED_OP_COUNTS, MEASURED_MODE_COUNTS, MEASURED_RADIX_COUNTS)

# ---------------------------------------------------------------- parsing
def parse(prompt):
    body = prompt.split('examples:\n', 1)[1]
    body, q = body.split('\nNow, determine the result for:', 1)
    eqs = []
    for line in body.splitlines():
        line = line.strip()
        if ' = ' not in line:
            continue
        L, R = line.split(' = ', 1)
        if len(L) != 5:
            return None
        eqs.append((L, R))
    qL = q.strip()
    if len(qL) != 5:
        return None
    return eqs, qL

def normalize(L, R, rev, opch):
    """-> (l0,l1,r0,r1, res_syms MSB-first, sign, rl, sign_end) in symbol space, or None.
    Sign (the row's own op glyph) may sit at EITHER end of the RHS regardless of mode
    (train data: 442 prefix / 27 suffix lines, both modes observed with prefix)."""
    a0, a1, op, b0, b1 = L
    sign = False
    end = ''
    mag = R
    if len(R) > 1 and R[0] == opch:
        sign = True; end = 'pre'; mag = R[1:]
    elif len(R) > 1 and R[-1] == opch:
        sign = True; end = 'suf'; mag = R[:-1]
    if rev:
        mag = mag[::-1]
        l0, l1, r0, r1 = a1, a0, b1, b0
    else:
        l0, l1, r0, r1 = a0, a1, b0, b1
    if not mag:
        return None
    return (l0, l1, r0, r1, tuple(mag), sign, len(mag), end)

# ---------------------------------------------------------------- candidate filtering
def op_bounds(op, lo, hi):
    """(magmin, magmax, can_sign, must_sign) over operands a,b in [lo,hi]."""
    d = {'_p1': 1, '_m1': -1, '_p2': 2, '_m2': -2}.get(op[-3:], 0)
    if op.startswith('add'):
        return (2 * lo + d, 2 * hi + d, False, False)
    if op.startswith('mul'):
        return (lo * lo + d, hi * hi + d, False, False)
    if op in SIGNED:
        return (0, hi - lo, True, False)
    if op == 'neg_absdiff':
        return (0, hi - lo, True, True)
    if op.startswith('absdiff'):
        return (max(d, 0), hi - lo + d, False, False)
    if op in ('a2_plus_b', 'a_plus_b2'):
        return (lo * lo + lo, hi * hi + hi, False, False)
    if op in ('mod', 'rmod'):
        return (0, hi - 1, False, False)
    if op == 'gcd':
        return (1, hi, False, False)
    if op == 'lcm':
        return (lo, hi * hi, False, False)
    if op in ('fdiv', 'rdiv'):
        return (0, hi // lo if lo else hi, False, False)
    return (0, hi * hi, True, False)

def concat_patterns(L, R, sign_any):
    """structural concat patterns this raw line is consistent with (mode-independent)."""
    if len(R) != 4 or sign_any:
        return set()
    out = set()
    if R == L[0] + L[1] + L[3] + L[4]:
        out.add('concat_fwd')
    if R == L[3] + L[4] + L[0] + L[1]:
        out.add('concat_rev')
    return out

def glyph_candidates(raw_rows, norm_rows, radix, opvocab, operand_lo=None):
    """candidate ops for one glyph given all its rows. concat is decided structurally and,
    when it matches every row, returned alone (coincidental arithmetic fits are ~impossible)."""
    cpat = None
    for (L, R), nr in zip(raw_rows, norm_rows):
        p = concat_patterns(L, R, nr[5])
        cpat = p if cpat is None else (cpat & p)
        if not cpat:
            break
    if cpat:
        return [c for c in PRIORITY if c in cpat], True
    lo = radix if operand_lo is None else operand_lo
    hi = radix * radix - 1                     # operands 2-digit
    cands = []
    for op in opvocab:
        ok = True
        for nr in norm_rows:
            sign, rl = nr[5], nr[6]
            mn, mx, can_s, must_s = op_bounds(op, lo, hi)
            if sign and not can_s:
                ok = False; break
            if not sign and must_s:
                ok = False; break
            vlo = 0 if rl == 1 else radix ** (rl - 1)
            vhi = radix ** rl - 1
            if mx < vlo or mn > vhi:
                ok = False; break
        if ok:
            cands.append(op)
    return [c for c in PRIORITY if c in cands], False

def units_set(op, lu, ru, base, sign):
    """sound units-digit constraint (None = not derivable)."""
    b = base
    if op.startswith('add'):
        d = {'_p1': 1, '_m1': -1, '_p2': 2, '_m2': -2}.get(op[-3:], 0)
        return {(lu + ru + d) % b}
    if op.startswith('mul'):
        d = {'_p1': 1, '_m1': -1, '_p2': 2, '_m2': -2}.get(op[-3:], 0)
        return {(lu * ru + d) % b}
    if op == 'a2_plus_b':
        return {(lu * lu + ru) % b}
    if op == 'a_plus_b2':
        return {(lu + ru * ru) % b}
    if op == 'sub_signed':
        return {((ru - lu) if sign else (lu - ru)) % b}
    if op == 'rsub_signed':
        return {((lu - ru) if sign else (ru - lu)) % b}
    if op in ('absdiff', 'neg_absdiff'):
        return {(lu - ru) % b, (ru - lu) % b}
    if op.startswith('absdiff'):
        d = {'_p1': 1, '_m1': -1, '_p2': 2, '_m2': -2}[op[-3:]]
        return {(lu - ru + d) % b, (ru - lu + d) % b}
    return None

# ---------------------------------------------------------------- core search
def _to_digits(v, base, rl):
    digs = []
    for _ in range(rl):
        digs.append(v % base); v //= base
    return digs[::-1]

# r5 (structure hunt, 2026-06-12): operands NEVER have a leading zero — both example
# and query operands are 2-digit values in [10,99]. Hard prune confirmed zero-risk
# (kills 0/803 gold configs; free vote 457 -> 465). Module flag so the r4 entry
# points keep their exact behavior; r5 entry points set it True.
HARD_OPND_LO = False

def _row_ok(op, eq, val, used_by, radix):
    """partial-assignment check. val[s]=digit or -1. used_by[d]=symbol idx or -1."""
    l0, l1, r0, r1, res, sign, rl = eq
    if HARD_OPND_LO and (val[l0] == 0 or val[r0] == 0):
        return False
    if val[l0] >= 0 and val[l1] >= 0 and val[r0] >= 0 and val[r1] >= 0:
        a = val[l0] * radix + val[l1]
        b = val[r0] * radix + val[r1]
        v = OPS[op](a, b)
        if v is None:
            return False
        if op in SIGNED:
            if (v < 0) != sign:
                return False
            v = abs(v)
        elif op in NEGPRE:
            if not sign:
                return False
            v = abs(v)
        else:
            if sign or v < 0:
                return False
        if v == 0:
            if rl != 1:
                return False
        elif not (radix ** (rl - 1) <= v < radix ** rl):
            return False
        digs = _to_digits(v, radix, rl)
        need = {}
        for k in range(rl):
            s = res[k]; d = digs[k]
            if val[s] >= 0:
                if val[s] != d:
                    return False
            else:
                pd = need.get(s)
                if pd is None:
                    u = used_by[d]
                    if u >= 0 and u != s:
                        return False
                    need[s] = d
                elif pd != d:
                    return False
        if len(set(need.values())) != len(need):
            return False
        return True
    # partial: units congruence
    ur = res[rl - 1]
    if val[l1] >= 0 and val[r1] >= 0 and val[ur] >= 0:
        al = units_set(op, val[l1], val[r1], radix, sign)
        if al is not None and val[ur] not in al:
            return False
    return True

def _search(glyphs, n, radix, deadline, max_sols, node_budget=2_000_000, order=None, trace=None,
            rng=None):
    """glyphs: list of (opchar, cand_ops, [eqs index-form]). enumerate digit assignments.
    returns list of (val tuple, {opchar: [valid ops]}). order: optional fixed variable order.
    trace: optional list collecting (depth, sym, digit, ok) decision events (for CoT replay).
    rng: when given, digits are tried in per-depth random order so that hitting max_sols
    yields an approximately unbiased SAMPLE of the solution space (not a lexicographic prefix)."""
    val = [-1] * n
    used_by = [-1] * radix
    if rng is not None:
        dig_orders = []
        for _ in range(n):
            o = list(range(radix)); rng.shuffle(o)
            dig_orders.append(o)
    else:
        dig_orders = [list(range(radix))] * n
    if order is None:
        freq = [0] * n
        is_opnd = [False] * n
        for _, cands, eqs in glyphs:
            for l0, l1, r0, r1, res, sign, rl in eqs:
                for s in (l0, l1, r0, r1):
                    freq[s] += 2; is_opnd[s] = True
                for s in res:
                    freq[s] += 1
        order = sorted(range(n), key=lambda i: (not is_opnd[i], -freq[i], i))
    sols = []
    nodes = [0]

    def glyph_ok(g):
        _, cands, eqs = g
        for c in cands:
            ok = True
            for eq in eqs:
                if not _row_ok(c, eq, val, used_by, radix):
                    ok = False; break
            if ok:
                return True
        return False

    def dfs(depth):
        nodes[0] += 1
        if nodes[0] & 1023 == 0 and time.time() > deadline:
            return False
        if nodes[0] > node_budget:
            return False
        if depth == n:
            ov = {}
            for opch, cands, eqs in glyphs:
                valid = [c for c in cands
                         if all(_row_ok(c, eq, val, used_by, radix) for eq in eqs)]
                if not valid:
                    return True
                ov[opch] = valid
            sols.append((tuple(val), ov))
            return len(sols) < max_sols
        si = order[depth]
        for d in dig_orders[depth]:
            if used_by[d] >= 0:
                continue
            val[si] = d; used_by[d] = si
            ok = all(glyph_ok(g) for g in glyphs)
            if trace is not None:
                trace.append((depth, si, d, ok))
            if ok:
                if not dfs(depth + 1):
                    val[si] = -1; used_by[d] = -1
                    return False
            val[si] = -1; used_by[d] = -1
        return True

    dfs(0)
    return sols, nodes[0], order

# ---------------------------------------------------------------- distinct-ops (r4)
# Measured on gold meta (2026-06-12): glyph ops are drawn WITHOUT REPLACEMENT —
# 10/798 rows show duplicate attributions (all explainable as ambiguous fit-sets,
# iid draws would predict ~150); guess rows: qop duplicates an example op 2/161.
# The gold config's true assignment is always distinct and every true op is in its
# glyph's valid set, so hard-enforcing distinctness never kills a gold config.
def _best_distinct(other_sets, banned, dup_ok=frozenset()):
    """max-sum-logp assignment of one op per set, all distinct, avoiding `banned`.
    other_sets: lists of prior-ordered op names. -> (sum_logp, choice tuple) or None.
    dup_ok (r5): ops exempt from the distinct/banned constraint (canonical-regime
    concat may duplicate across '+' and '*' — independent family draws)."""
    best = None
    n = len(other_sets)
    def rec(i, used, acc, choice):
        nonlocal best
        if best is not None and acc <= best[0]:
            return                      # logp <= 0: acc can only decrease
        if i == n:
            best = (acc, tuple(choice))
            return
        for op in other_sets[i][:12]:
            if op not in dup_ok and (op in used or op in banned):
                continue
            rec(i + 1, used | {op}, acc + op_logp(op), choice + [op])
    rec(0, frozenset(), 0.0, [])
    return best

# ---------------------------------------------------------------- regime mixture (r5)
# Two-regime glyph model (analysis/reports/cryptarithm_structure_hunt.md): the
# generator flips a per-ROW coin (pA ~ 0.60). Regime A "canonical": every operator
# glyph IS the op's own arithmetic character ('+' add-family, '*' mul-family,
# '-' sub-family; concat renders under '+'/'*'; rsub ~0 under '-'). Regime B
# "scrambled": glyphs drawn without replacement from the 26-char pool, op ~ opB.
# A candidate config scores logsumexp of the two regime terms; the -log P(26,k)
# glyph-assignment factor on the B term is ESSENTIAL (without it regime A never
# dominates on canonical rows and the mixture LOSES — measured -5 rows net).
# Held-out validated: vote 457/659 -> 469/659 with the hard operand prune.
CANON = frozenset('+-*')

def load_regime(path):
    """load EM priors (analysis/crypt_struct/em_priors_lo.json schema)."""
    import json as _json
    pr = _json.load(open(path))
    return {
        'lpA': math.log(pr['pA']), 'lpB': math.log(1.0 - pr['pA']),
        'famA': pr['famA'], 'opB': pr['opB'],
        'modeA': {False: pr['modeA']['false'], True: pr['modeA']['true']},
        'modeB': {False: pr['modeB']['false'], True: pr['modeB']['true']},
    }

def _bd_generic(other_sets, banned, lp_of, dup_ok=frozenset()):
    """max-sum assignment over [(glyph, [ops])], one op each, distinct except dup_ok;
    per-op logp from lp_of(glyph, op) (None = op impossible). -> (sum, choices) or None."""
    best = None
    n = len(other_sets)
    def rec(i, used, acc, choice):
        nonlocal best
        if best is not None and acc <= best[0]:
            return
        if i == n:
            best = (acc, tuple(choice))
            return
        g, vops = other_sets[i]
        for op in vops[:12]:
            if op not in dup_ok and (op in used or op in banned):
                continue
            lp = lp_of(g, op)
            if lp is None:
                continue
            rec(i + 1, used | {op}, acc + lp, choice + [op])
    rec(0, frozenset(), 0.0, [])
    return best

def regime_score(regime, gops, qop, qt, rev, radix, comp_lp):
    """mixture log-score of one candidate config -> (score, ops_assignment) or None.
    gops: {glyph_char: [valid ops]} (qop included on deduce rows)."""
    others = [(g, v) for g, v in gops.items() if g != qop]
    k = len(gops) + (0 if qop in gops else 1)
    rev = bool(rev)
    terms = []
    best_assign = None
    # regime A: all glyphs canonical chars AND every op in its glyph's family
    if qop in CANON and all(g in CANON for g in gops):
        famA = regime['famA']
        qa = famA.get(qop, {}).get(qt)
        if qa is not None:
            oa = _bd_generic(others, {qt}, lambda g, op: famA.get(g, {}).get(op),
                             dup_ok=frozenset(CONCATS))
            if oa is not None:
                terms.append(regime['lpA'] + regime['modeA'][rev] + qa + oa[0]
                             + comp_lp + RADIX_LOGP[radix])
                best_assign = oa[1]
    # regime B: k glyphs drawn without replacement from the 26-char pool
    glyph_lp = -sum(math.log(26 - i) for i in range(k))
    opB = regime['opB']
    ob = _bd_generic(others, {qt}, lambda g, op: opB.get(op, -11.0))
    if ob is not None:
        tB = (regime['lpB'] + glyph_lp + regime['modeB'][rev]
              + opB.get(qt, -11.0) + ob[0] + comp_lp + RADIX_LOGP[radix])
        if not terms or tB >= terms[0]:
            best_assign = ob[1]
        terms.append(tB)
    if not terms:
        return None
    M = max(terms)
    sc = M + math.log(sum(math.exp(t - M) for t in terms))
    return sc, dict(zip([g for g, _ in others], best_assign)) if best_assign else {}

# ---------------------------------------------------------------- top level
def _render(v, sign_needed, opch, rev, radix, inv, sign_end='pre'):
    """value -> display string using digit->symbol inv map; None if some digit unmapped."""
    digs = []
    x = abs(v)
    if x == 0:
        digs = [0]
    while x:
        digs.append(x % radix); x //= radix
    digs = digs[::-1]
    if any(inv[d] is None for d in digs):
        return None
    s = ''.join(inv[d] for d in digs)
    if rev:
        s = s[::-1]
    if sign_needed:
        s = (s + opch) if sign_end == 'suf' else (opch + s)
    return s

def solve(prompt, gold=None, deadline_s=10.0, max_sols=3000, opvocab=None, want_trace=False,
          guess_policy=None, operand_lo=None, distinct_ops=True, regime=None):
    """returns (answer, meta) or None.
    meta: mapping/ops/rev/radix/score/n_solutions/alternatives/...
    gold: data-gen only -- conditions the query equation on the gold answer.
    guess_policy: ordered op list for guess rows (query glyph unseen in examples).
    distinct_ops: enforce the measured without-replacement glyph-op draw (r4);
    False restores the r3-and-earlier iid scoring.
    regime (r5): load_regime() dict -> two-regime mixture scoring (requires
    distinct_ops); combine with HARD_OPND_LO=True for the full r5 model."""
    pr = parse(prompt)
    if pr is None:
        return None
    eqs, qL = pr
    qa, qb, qop, qc, qd = qL
    opchars = {L[2] for L, R in eqs} | {qop}
    if opvocab is None:
        opvocab = list(OPS)
    t_end = time.time() + deadline_s
    work = list(eqs)
    if gold is not None:
        work = work + [(qL, gold.strip())]

    cand_answers = {}   # answer -> best (score, meta)
    passes = [(False, 10), (True, 10), (False, 11), (True, 11)]
    for rev, radix in passes:
            if radix == 11 and cand_answers:
                continue    # radix 11 only as a fallback (never observed in train)
            if time.time() > t_end:
                break
            # normalize
            norm = []
            ok = True
            for L, R in work:
                nr = normalize(L, R, rev, L[2])
                if nr is None or any(c in opchars for c in nr[4]):
                    ok = False; break   # sign glyph inside magnitude => wrong mode hypothesis
                norm.append(nr)
            if not ok:
                continue
            # symbol table
            syms = set()
            for (L, R), nr in zip(work, norm):
                syms |= {nr[0], nr[1], nr[2], nr[3]} | set(nr[4])
            qsy = [c for c in (qa, qb, qc, qd)]
            syms |= set(qsy)
            syms -= opchars
            syms = sorted(syms)
            if len(syms) > radix:
                continue
            idx = {c: i for i, c in enumerate(syms)}
            n = len(syms)
            # group rows per glyph
            byg = {}
            for (L, R), nr in zip(work, norm):
                byg.setdefault(L[2], ([], []))
                byg[L[2]][0].append((L, R))
                byg[L[2]][1].append(nr)
            glyphs = []
            feasible = True
            conc_only = {}
            for opch, (raws, nrs) in byg.items():
                cands, is_conc = glyph_candidates(raws, nrs, radix, opvocab, operand_lo)
                if not cands:
                    feasible = False; break
                conc_only[opch] = is_conc
                ieqs = [(idx[a], idx[b], idx[c], idx[d],
                         tuple(idx[s] for s in res), sign, rl)
                        for (a, b, c, d, res, sign, rl, _e) in nrs]
                glyphs.append((opch, cands, ieqs))
            if not feasible:
                continue
            # constrained symbols = those in non-concat glyph rows
            constrained = set()
            for opch, cands, ieqs in glyphs:
                if conc_only[opch]:
                    continue
                for l0, l1, r0, r1, res, sign, rl in ieqs:
                    constrained |= {l0, l1, r0, r1} | set(res)
            cons = sorted(constrained)
            cmap = {s: i for i, s in enumerate(cons)}
            cglyphs = []
            for opch, cands, ieqs in glyphs:
                if conc_only[opch]:
                    continue
                cglyphs.append((opch, cands, [
                    (cmap[l0], cmap[l1], cmap[r0], cmap[r1],
                     tuple(cmap[s] for s in res), sign, rl)
                    for (l0, l1, r0, r1, res, sign, rl) in ieqs]))
            deadline = min(t_end, time.time() + deadline_s / 2)
            trace = [] if want_trace else None
            if cglyphs:
                import random as _rnd
                sols, nodes, order = _search(cglyphs, len(cons), radix, deadline,
                                             max_sols, trace=trace,
                                             rng=_rnd.Random(0xC0DE + radix + rev))
            else:
                sols, nodes, order = [((), {})], 0, []
            for valt, ov in sols:
                # rebuild full symbol -> digit map (concat-only symbols unassigned)
                smap = {s: None for s in syms}
                for s in syms:
                    gi = idx[s]
                    if gi in cmap:
                        smap[s] = valt[cmap[gi]]
                # per-glyph chosen op = highest-prior valid; concat glyphs from candidates
                gops = {}
                for opch, cands, ieqs in glyphs:
                    gops[opch] = ov.get(opch, cands)
                others_ch = [opch for opch in gops if opch != qop]
                others_sets = [gops[opch] for opch in others_ch]
                score = MODE_LOGP[rev] + RADIX_LOGP[radix]
                if not distinct_ops:
                    for opch, vops in gops.items():
                        if opch == qop and gold is None:
                            continue
                        score += op_logp(vops[0]) if vops else _PRIOR_FLOOR
                # ---- compute the query answer(s) ----
                qcands = gops.get(qop)
                if qcands is None or not qcands:
                    if guess_policy:
                        qcands = list(guess_policy)
                    elif regime is not None and distinct_ops:
                        # r5 family-aware guess: canonical rows -> the query glyph's
                        # CHAR reveals its family (famA order); scrambled -> opB order
                        if qop in CANON and all(g in CANON for g in gops):
                            fa = regime['famA'].get(qop, {})
                            qcands = sorted(fa, key=lambda o: -fa[o])[:8]
                        else:
                            ob = regime['opB']
                            qcands = sorted(ob, key=lambda o: -ob[o])[:10]
                    elif distinct_ops:
                        qcands = PRIORITY[:10]   # exclusion picks the top non-used op
                    else:
                        qcands = [PRIORITY[0]]
                used = set(d for d in smap.values() if d is not None)
                fs = sorted({s for s in qsy if smap[s] is None})
                free_digs = [d for d in range(radix) if d not in used]
                # completions of unconstrained query symbols (uniform over arrangements)
                if fs and len(fs) <= len(free_digs):
                    import itertools as _it
                    perms = list(_it.permutations(free_digs, len(fs)))[:60]
                else:
                    perms = [()]
                comp_lp = -math.log(len(perms)) if perms else 0.0
                for pm in perms:
                    cmap2 = dict(smap)
                    for s, d in zip(fs, pm):
                        cmap2[s] = d
                    inv = [None] * radix
                    for s, d in cmap2.items():
                        if d is not None:
                            inv[d] = s
                    ends = [nr[7] for nr in norm if nr[5]]
                    s_end = ('suf' if ends.count('suf') > ends.count('pre')
                             else 'pre') if ends else 'pre'
                    for qt in qcands:
                        if qt == 'concat_fwd':
                            ans = qa + qb + qc + qd
                        elif qt == 'concat_rev':
                            ans = qc + qd + qa + qb
                        else:
                            if any(cmap2[s] is None for s in qsy):
                                continue
                            if rev:
                                A = cmap2[qb] * radix + cmap2[qa]
                                B = cmap2[qd] * radix + cmap2[qc]
                            else:
                                A = cmap2[qa] * radix + cmap2[qb]
                                B = cmap2[qc] * radix + cmap2[qd]
                            if HARD_OPND_LO and (A < radix or B < radix):
                                continue   # r5: query operands are 2-digit, no leading 0
                            v = OPS[qt](A, B)
                            if v is None:
                                continue
                            sign_needed = (qt in NEGPRE) or (qt in SIGNED and v < 0)
                            if v < 0 and qt not in SIGNED and qt not in NEGPRE:
                                continue
                            ans = _render(v, sign_needed, qop, rev, radix, inv, s_end)
                            if ans is None:
                                continue
                        if regime is not None and distinct_ops:
                            rs = regime_score(regime, gops, qop, qt, rev, radix, comp_lp)
                            if rs is None:
                                continue    # impossible under both regimes
                            sc, ops_assign = rs
                            ops_meta = dict(ops_assign)
                            ops_meta[qop] = qt
                        elif distinct_ops:
                            ba = _best_distinct(others_sets, {qt})
                            if ba is None:
                                continue    # no distinct assignment with query = qt
                            sc = score + ba[0] + op_logp(qt) + comp_lp
                            ops_meta = dict(zip(others_ch, ba[1]))
                            ops_meta[qop] = qt
                        else:
                            sc = score + op_logp(qt) + comp_lp
                            ops_meta = {o: v[0] if v else None for o, v in gops.items()}
                        if gold is not None and ans != gold.strip():
                            continue
                        meta = {'radix': radix, 'rev': rev, 'mapping': dict(cmap2),
                                'ops': ops_meta,
                                'opsets': gops, 'qop': qt, 'score': sc, 'nodes': nodes,
                                'order': [syms[cons[i]] for i in order] if cglyphs else [],
                                'trace': trace, 'syms': syms, 'norm': norm, 'work': work,
                                'concat_only': conc_only, 'sign_end': s_end}
                        cand_answers.setdefault(ans, []).append((sc, meta))
    if not cand_answers:
        return None
    # posterior mass per answer = sum of exp(score) over its candidate configs
    M = max(sc for v in cand_answers.values() for sc, _ in v)
    mass = {a: sum(math.exp(sc - M) for sc, _ in v) for a, v in cand_answers.items()}
    key = lambda a: (round(mass[a], 9), max(sc for sc, _ in cand_answers[a]), a)
    ans = max(mass, key=key)
    sc, meta = max(cand_answers[ans], key=lambda t: t[0])
    meta['n_answers'] = len(cand_answers)
    tot = sum(mass.values())
    meta['posterior'] = mass[ans] / tot
    meta['alternatives'] = [(a, round(mass[a] / tot, 4))
                            for a in sorted(mass, key=key, reverse=True)[:5]]
    return ans, meta

# ---------------------------------------------------------------- gold measurement / eval CLI
_W = {}

def _work(r):
    t0 = time.time()
    try:
        res = solve(r['prompt'], gold=(r['answer'].strip() if _W['gold'] else None),
                    deadline_s=_W['deadline'], guess_policy=_W.get('policy'),
                    regime=_W.get('regime'))
    except Exception:
        res = None
    dt = time.time() - t0
    if res is None:
        return {'id': r['id'], 'cat': r['category'], 'gold': r['answer'].strip(),
                'pred': None, 'ok': False, 'dt': dt}
    ans, meta = res
    return {'id': r['id'], 'cat': r['category'], 'gold': r['answer'].strip(),
            'pred': ans, 'ok': ans == r['answer'].strip(), 'dt': dt,
            'radix': meta['radix'], 'rev': meta['rev'],
            'ops': meta['ops'], 'qop': meta['qop'],
            'mapping': {k: v for k, v in meta['mapping'].items()},
            'n_answers': meta['n_answers'], 'nodes': meta['nodes'],
            'alternatives': meta['alternatives']}

def _init(w):
    _W.update(w)
    if w.get('hard_lo'):
        global HARD_OPND_LO
        HARD_OPND_LO = True

def _eval_main():
    import csv, json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cat', default='cryptarithm_deduce')
    ap.add_argument('--n', type=int, default=10**9)
    ap.add_argument('--gold', action='store_true')
    ap.add_argument('--deadline', type=float, default=10.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--prior', default=None, help='json file with measured op/mode/radix counts')
    ap.add_argument('--regime', default=None,
                    help='EM priors json (em_priors_lo.json) -> r5 two-regime mixture scoring')
    ap.add_argument('--hard-lo', action='store_true',
                    help='r5 hard no-leading-zero operand prune (HARD_OPND_LO)')
    args = ap.parse_args()
    if args.prior:
        pr = json.load(open(args.prior))
        set_prior(pr['op'], {k == 'True': v for k, v in pr['mode'].items()},
                  {int(k): v for k, v in pr['radix'].items()})
    rows = [r for r in csv.DictReader(open('competition_dataset/train_categorized.csv'))
            if r['category'] == args.cat or args.cat == 'all'][:args.n]
    rows = [r for r in rows if r['category'].startswith('cryptarithm')]
    out = open(args.out, 'w') if args.out else None
    w = {'gold': args.gold, 'deadline': args.deadline,
         'regime': load_regime(args.regime) if args.regime else None,
         'hard_lo': args.hard_lo}
    t0 = time.time()
    if args.procs > 1:
        import multiprocessing as mp
        with mp.Pool(args.procs, initializer=_init, initargs=(w,)) as pool:
            results = pool.map(_work, rows, chunksize=4)
    else:
        _init(w)
        results = [_work(r) for r in rows]
    ok = sum(r['ok'] for r in results)
    wrong = sum((not r['ok']) and r['pred'] is not None for r in results)
    none = sum(r['pred'] is None for r in results)
    dts = sorted(r['dt'] for r in results)
    import statistics
    print(f"{args.cat} gold={args.gold}: CORRECT {ok} ({100*ok/len(rows):.1f}%) | "
          f"WRONG {wrong} | NONE {none} | n={len(rows)} | "
          f"med {statistics.median(dts):.2f}s p95 {dts[int(0.95*len(dts))-1]:.2f}s "
          f"max {dts[-1]:.2f}s | wall {time.time()-t0:.0f}s")
    if out:
        for r in results:
            out.write(json.dumps(r) + '\n')
        out.close()

if __name__ == '__main__':
    _eval_main()
