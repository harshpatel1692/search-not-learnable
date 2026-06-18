"""cryptarithm r7 — ONE lazy self-routing FORCED-CHAIN grammar (no DFS anywhere).

Design (approved 2026-06-12, LOG.md): every trace follows one policy whose branch
points are conditioned on OBSERVABLE trace state (truthful-conditional lesson):
  1. symbol-table CONSTRUCTION block = r6 FIX A byte-for-byte (table_lines6 reused;
     lint re-derives every scan line from the prompt, table == first-appearance walk).
  2. query-op family ROUTING check — one truthful observable test: the result digits
     are the operand digits juxtaposed (concat) or not (value). Real: 59 concat /
     600 value — both branches abundant.
  3. forced-chain propagation lines — the propagation engine's inference steps
     (analysis/crypt_struct/propagation.py = the solving core), ONE deduction per
     numbered line, each line has exactly one surviving candidate (nothing to loop on).
  4. on stall: the trace NAMES the stall level (mode-unresolved / rule-ambiguous /
     digit-unpinned), opens ONE binary case split (task-A decision: depth capped at 1
     — depth-2 measured +4/659 rows only), kills the dead case with the concrete
     contradiction the engine hit, continues the live case to the answer.
  5. on exhaustion: truthful bail — the trace provably exhausts (no split resolves,
     verified by execution), then commits prior-ordered GUESSES (each labeled
     "(guess)"), refutes truthfully if a guess contradicts, and always boxes a
     best-guess answer (never fabricated verification).

Everything is DETERMINISTIC given the prompt (the engine's iteration order is pinned)
so lint_r7's final layer can re-execute the whole policy and demand equality, on top
of independent local re-derivation of the table / routing / arithmetic / encode lines.
ASCII-only.  Step counters s1,s2,... are strictly monotone (lint-checked).

Usage:
  python3 pipeline/synth/cryptarithm_r7.py measure7 [procs]     # coverage + stats on 659 real rows
  python3 pipeline/synth/cryptarithm_r7.py render-real7         # -> data/crypt_r7/crypt_deduce_real.jsonl
  python3 pipeline/synth/cryptarithm_r7.py render-synth7 N SEED [OUT]
  python3 pipeline/synth/cryptarithm_r7.py gen-test7            # sample traces of each kind
"""
import json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
from solvers import cryptarithm2 as C2
import propagation as PE
from synth import cryptarithm_cot as CC

SPLIT_DEPTH = 1        # task-A decision input: 2-split = .496 < .58 gate -> capped at 1
MAX_COMMITS = 5        # bail guess-descent budget (commits + refutes)
MODE = {False: 'standard', True: 'little-endian'}
OP_WORD = CC.OP_WORD
LETTERS = CC.LETTERS
OPLETTERS = CC.OPLETTERS

CHAIN_HDR = ("Deduce the digit map; every numbered line below is forced"
             " (exactly one candidate survives).")
NOSPLIT_LINE = ("no single case split settles it: every binary choice leaves the"
                " answer undecided.")

STALL_TXT = {
    'mode-unresolved': "both digit orders are still possible (mode undecided).",
    'query-op-ambiguous': None,                  # built with the live candidates
    'query-op-ambiguous(concat order)': "both concatenation orders fit every example.",
    'query-digit-unpinned': None,                # built with the unpinned letters
    'answer-digit-unmapped': "the result needs a digit that no symbol is pinned to.",
    'query-operand-leading-zero': "the pinned reading gives a query operand below 10.",
    'query-undefined': "the pinned rule is undefined on the query operands.",
}


# ---------------------------------------------------------------- letters
def _lets(eqs, qL):
    """letter tables by first appearance (same walk as table_lines6 / letterize)."""
    opchars = {L[2] for L, R in eqs} | {qL[2]}
    let, olet = {}, {}
    for L, R in eqs:
        for c in L + R:
            if c not in opchars and c not in let:
                let[c] = LETTERS[len(let)]
        if L[2] not in olet:
            olet[L[2]] = OPLETTERS[min(len(olet), 3)]
    for c in qL:
        if c not in opchars and c not in let:
            let[c] = LETTERS[len(let)]
    if qL[2] not in olet:
        olet[qL[2]] = OPLETTERS[min(len(olet), 3)]
    return let, olet, opchars


# ---------------------------------------------------------------- engine helpers
def _prior_ops(eng):
    """deterministic prior order over the query glyph's surviving value rules."""
    if eng.canonical and eng.qop in PE.FAMA:
        base = [o for o in PE.FAMA[eng.qop] if o not in PE.CONCATS]
    else:
        base = [o for o in PE.OPB if o not in PE.CONCATS]
    base += [o for o in PE.FULL if o not in base and o not in PE.CONCATS]
    return [o for o in base if o in eng.opdom[eng.qop]]


def _s_end(eng, rev):
    ends = []
    for _, _, nb in eng.lines:
        nr = nb[rev]
        if nr and nr[5]:
            ends.append(nr[7])
    return 'suf' if ends.count('suf') > ends.count('pre') else 'pre'


def answer_detail(eng):
    """replicate Eng.answer() but return every intermediate needed for rendering,
    or None when the state is not fully forced (mirrors answer()'s stall cases)."""
    qod = eng.opdom[eng.qop]
    if qod <= PE.CONCATS:
        return None                                  # routing intercepts concat queries
    if len(eng.modes) > 1 or len(qod) > 1:
        return None
    qsy = (eng.qa, eng.qb, eng.qc, eng.qd)
    if any(len(eng.dom[s]) > 1 for s in qsy):
        return None
    rev = next(iter(eng.modes))
    val = {s: next(iter(eng.dom[s])) for s in qsy}
    if rev:
        A = val[eng.qb] * 10 + val[eng.qa]
        B = val[eng.qd] * 10 + val[eng.qc]
    else:
        A = val[eng.qa] * 10 + val[eng.qb]
        B = val[eng.qc] * 10 + val[eng.qd]
    if A < 10 or B < 10:
        return None
    op = next(iter(qod))
    v = C2.OPS[op](A, B)
    if v is None:
        return None
    sign_needed = (op in C2.NEGPRE) or (op in C2.SIGNED and v < 0)
    if v < 0 and op not in C2.SIGNED and op not in C2.NEGPRE:
        return None
    inv = [None] * 10
    for s in eng.syms:
        if len(eng.dom[s]) == 1:
            inv[next(iter(eng.dom[s]))] = s
    digs = []
    x = abs(v)
    if x == 0:
        digs = [0]
    while x:
        digs.append(x % 10)
        x //= 10
    digs = digs[::-1]
    if any(inv[d] is None for d in digs):
        return None
    s_end = _s_end(eng, rev)
    ans = C2._render(v, sign_needed, eng.qop, rev, 10, inv, s_end)
    if ans is None:
        return None
    return {'rev': rev, 'op': op, 'A': A, 'B': B, 'v': v, 'sign': sign_needed,
            's_end': s_end, 'val': val, 'enc': [(d, inv[d]) for d in digs],
            'ans': ans}


def _stall_why(eng):
    return eng.answer()[1]


# ---------------------------------------------------------------- split exploration
def explore_split(eng):
    """one_split semantics with event capture. Returns
    {'point': (kind, who, vals), 'branches': [b0, b1]} or None; each branch =
    {'val','sev','out'} with out = ('ans', det) | ('dead', contra) | ('stall', why)."""
    base = eng.snapshot()
    for kind, who, vals in eng.binary_points()[:12]:
        branches = []
        for v in vals:
            eng.restore(base)
            i0 = len(eng.sev)
            try:
                eng._local_fix(eng._apply_choice(kind, who, v))
            except PE.Contra as e:
                branches.append({'val': v, 'sev': list(eng.sev[i0:]), 'out': ('dead', e)})
                continue
            det = answer_detail(eng)
            if det is not None:
                branches.append({'val': v, 'sev': list(eng.sev[i0:]), 'out': ('ans', det)})
            else:
                branches.append({'val': v, 'sev': list(eng.sev[i0:]),
                                 'out': ('stall', _stall_why(eng))})
        kinds = [b['out'][0] for b in branches]
        resolved = (kinds.count('ans') == 1 and kinds.count('dead') == 1) or \
                   (kinds == ['ans', 'ans']
                    and branches[0]['out'][1]['ans'] == branches[1]['out'][1]['ans'])
        if resolved:
            eng.restore(base)
            return {'point': (kind, who, list(vals)), 'branches': branches}
    eng.restore(base)
    return None


# ---------------------------------------------------------------- bail guess descent
def _pick_commit(eng, why):
    """deterministic prior-ordered guess at the named stall level, or None."""
    if why == 'mode-unresolved':
        v = False if False in eng.modes else True
        return ('mode', None, v,
                f"{MODE[v]} order is the more common convention")
    if why and why.startswith('query-op-ambiguous'):
        ops = _prior_ops(eng)
        if not ops:
            return None
        return ('op', eng.qop, ops[0],
                f"{OP_WORD[ops[0]]} is the most common rule still possible")
    if why == 'query-digit-unpinned':
        qsy = []
        for s in (eng.qa, eng.qb, eng.qc, eng.qd):
            if s not in qsy:
                qsy.append(s)
        open_ = [s for s in qsy if len(eng.dom[s]) > 1]
        if not open_:
            return None
        s = min(open_, key=lambda t: (len(eng.dom[t]), qsy.index(t)))
        return ('sym', s, min(eng.dom[s]),
                "take the smallest candidate of the most constrained query symbol")
    if why == 'answer-digit-unmapped':
        # find the first result digit no symbol is pinned to; guess its holder
        rev = next(iter(eng.modes)) if len(eng.modes) == 1 else False
        # recompute v from pinned query digits (this stall implies they are pinned)
        try:
            val = {s: next(iter(eng.dom[s]))
                   for s in (eng.qa, eng.qb, eng.qc, eng.qd)}
        except StopIteration:
            return None
        if rev:
            A = val[eng.qb] * 10 + val[eng.qa]
            B = val[eng.qd] * 10 + val[eng.qc]
        else:
            A = val[eng.qa] * 10 + val[eng.qb]
            B = val[eng.qc] * 10 + val[eng.qd]
        ops = _prior_ops(eng)
        if not ops:
            return None
        v = C2.OPS[ops[0]](A, B)
        if v is None:
            return None
        pinned = {next(iter(eng.dom[s])) for s in eng.syms if len(eng.dom[s]) == 1}
        x = abs(v)
        digs = [0] if x == 0 else []
        while x:
            digs.append(x % 10)
            x //= 10
        for d in digs[::-1]:
            if d in pinned:
                continue
            cands = [s for s in eng.syms if d in eng.dom[s] and len(eng.dom[s]) > 1]
            if not cands:
                return None
            s = min(cands, key=lambda t: (len(eng.dom[t]), t))
            return ('sym', s, d,
                    f"digit {d} of the result must be written by some symbol;"
                    f" take the most constrained candidate")
        return None
    return None


def _assemble(eng):
    """no commit possible / budget spent: assemble the most constrained reading
    directly (greedy, deterministic). Returns (det, picks) or None."""
    rev = False if False in eng.modes else True
    qsy_o = []
    for s in (eng.qa, eng.qb, eng.qc, eng.qd):
        if s not in qsy_o:
            qsy_o.append(s)
    for op in (_prior_ops(eng) or []):
        pick = {}
        ok = True
        for s in qsy_o:
            if len(eng.dom[s]) == 1:
                pick[s] = next(iter(eng.dom[s]))
                continue
            cands = sorted(eng.dom[s] - set(pick.values()))
            if not cands:
                ok = False
                break
            pick[s] = cands[0]
        if not ok:
            continue
        val = {s: pick[s] for s in qsy_o}
        if rev:
            A = val[eng.qb] * 10 + val[eng.qa]
            B = val[eng.qd] * 10 + val[eng.qc]
        else:
            A = val[eng.qa] * 10 + val[eng.qb]
            B = val[eng.qc] * 10 + val[eng.qd]
        if A < 10 or B < 10:
            continue
        v = C2.OPS[op](A, B)
        if v is None:
            continue
        sign_needed = (op in C2.NEGPRE) or (op in C2.SIGNED and v < 0)
        if v < 0 and op not in C2.SIGNED and op not in C2.NEGPRE:
            continue
        inv = [None] * 10
        for s in eng.syms:
            if len(eng.dom[s]) == 1:
                inv[next(iter(eng.dom[s]))] = s
        for s, d in pick.items():
            inv[d] = s
        x = abs(v)
        digs = [0] if x == 0 else []
        while x:
            digs.append(x % 10)
            x //= 10
        digs = digs[::-1]
        if any(inv[d] is None for d in digs):
            # extend greedily: give unmapped digits to free symbols that allow them
            ext = dict()
            ok2 = True
            taken = {d for d, s in enumerate(inv) if s is not None}
            for d in digs:
                if inv[d] is not None or d in ext:
                    continue
                cands = [s for s in eng.syms if d in eng.dom[s]
                         and len(eng.dom[s]) > 1 and s not in pick
                         and s not in ext.values()]
                if not cands:
                    ok2 = False
                    break
                ext[d] = min(cands, key=lambda t: (len(eng.dom[t]), t))
            if not ok2:
                continue
            for d, s in ext.items():
                inv[d] = s
                pick[s] = d
        s_end = _s_end(eng, rev)
        ans = C2._render(v, sign_needed, eng.qop, rev, 10, inv, s_end)
        if ans is None:
            continue
        det = {'rev': rev, 'op': op, 'A': A, 'B': B, 'v': v, 'sign': sign_needed,
               's_end': s_end, 'val': {s: pick.get(s, next(iter(eng.dom[s])))
                                       for s in (eng.qa, eng.qb, eng.qc, eng.qd)},
               'enc': [(d, inv[d]) for d in digs], 'ans': ans}
        return det, pick
    return None


def guess_descent(eng, why0):
    """commit prior-ordered guesses until an answer falls out (or assemble).
    Truthful refutation: a guess that contradicts is rendered (commit + its
    propagation + the contradiction), excluded, and if its level empties under
    an earlier mode guess the mode guess is flipped (once). Returns
    {'steps': [...], 'det': det or None} where steps =
    ('stall', why, state) | ('commit', kind, who, v, rationale, sev) |
    ('contra', contra, kind, who, v) | ('forced', kind, who, v, sev) |
    ('flip', v) | ('assemble', det, guessed)."""
    steps = []
    root = eng.snapshot()
    first_mode = None
    switched = False
    budget = MAX_COMMITS
    first_iter = True

    def _dead_level():
        """deepest guess level exhausted: flip the first mode guess once."""
        nonlocal switched, first_mode
        if first_mode is None or switched:
            return False
        eng.restore(root)
        other = not first_mode
        if other not in eng.modes:
            return False
        steps.append(('flip', other))
        switched = True
        first_mode = None
        try:
            i1 = len(eng.sev)
            eng._local_fix(eng._apply_choice('mode', None, other))
            steps[-1] = ('flip', other)
            steps.append(('forced', 'mode', None, other, list(eng.sev[i1:])))
        except PE.Contra:
            return False
        return True

    while budget > 0:
        det = answer_detail(eng)
        if det is not None:
            return {'steps': steps, 'det': det}
        if first_iter:
            why = why0
            first_iter = False
        else:
            why = _stall_why(eng)
            steps.append(('stall', why, _stall_state(eng, why)))
        ch = _pick_commit(eng, why)
        if ch is None:
            break
        kind, who, v, rat = ch
        snap = eng.snapshot()
        i0 = len(eng.sev)
        budget -= 1
        try:
            eng._local_fix(eng._apply_choice(kind, who, v))
            steps.append(('commit', kind, who, v, rat, list(eng.sev[i0:])))
            if kind == 'mode' and first_mode is None and not switched:
                first_mode = v
        except PE.Contra as e:
            sev = list(eng.sev[i0:])
            eng.restore(snap)
            steps.append(('commit', kind, who, v, rat, sev))
            steps.append(('contra', e, kind, who, v))
            dead = False
            forced_v = None
            if kind == 'mode':
                eng.modes.discard(v)
                dead = not eng.modes
                if len(eng.modes) == 1:
                    forced_v = next(iter(eng.modes))
            elif kind == 'op':
                eng.opdom[who].discard(v)
                dead = not eng.opdom[who]
                if len(eng.opdom[who]) == 1:
                    forced_v = next(iter(eng.opdom[who]))
            else:
                eng.dom[who].discard(v)
                dead = not eng.dom[who]
                if len(eng.dom[who]) == 1:
                    forced_v = next(iter(eng.dom[who]))
            if not dead and forced_v is not None:
                try:
                    i1 = len(eng.sev)
                    eng._local_fix(eng._apply_choice(kind, who, forced_v))
                    steps.append(('forced', kind, who, forced_v, list(eng.sev[i1:])))
                except PE.Contra:
                    dead = True
            if dead and not _dead_level():
                break
    det = answer_detail(eng)
    if det is not None:
        return {'steps': steps, 'det': det}
    asm = _assemble(eng)
    if asm is not None:
        steps.append(('assemble', asm[0], asm[1]))
        return {'steps': steps, 'det': asm[0], 'assembled': True}
    return {'steps': steps, 'det': None}


# ---------------------------------------------------------------- driver
def drive(prompt, deadline_s=30.0):
    """deterministic truth-free r7 policy. Returns a plan dict or None (unrenderable)."""
    pr = C2.parse(prompt)
    if pr is None:
        return None
    eqs, qL = pr
    let, olet, opchars = _lets(eqs, qL)
    # table truthfulness guards (same as r6)
    for L0, R0 in eqs:
        if any(c in opchars for c in (L0[0], L0[1], L0[3], L0[4])):
            return None
        mag0 = R0
        if len(R0) > 1 and R0[0] == L0[2]:
            mag0 = R0[1:]
        elif len(R0) > 1 and R0[-1] == L0[2]:
            mag0 = R0[:-1]
        if any(c in opchars for c in mag0):
            return None
    if any(c in opchars for c in (qL[0], qL[1], qL[3], qL[4])):
        return None
    if len(let) > 10:
        return None
    canonical = opchars <= set('+-*')
    qlines = [(k, L, R) for k, (L, R) in enumerate(eqs, 1) if L[2] == qL[2]]
    if not qlines:
        return None                                  # guess row: not this grammar
    pats = []
    inter = None
    for k, L, R in qlines:
        sign = len(R) > 1 and (R[0] == L[2] or R[-1] == L[2])
        p = C2.concat_patterns(L, R, sign)
        pats.append((k, L, R, p))
        inter = p if inter is None else (inter & p)
    plan = {'eqs': eqs, 'qL': qL, 'let': let, 'olet': olet, 'canonical': canonical,
            'pats': pats, 'inter': inter}
    if len(inter) == 1:
        op = next(iter(inter))
        ans = (qL[0] + qL[1] + qL[3] + qL[4]) if op == 'concat_fwd' \
            else (qL[3] + qL[4] + qL[0] + qL[1])
        plan.update(kind='concat', qop_op=op, ans=ans)
        return plan
    if len(inter) == 2:
        # both orders fit every example (1 real row): truthful bail on the order
        ans = qL[0] + qL[1] + qL[3] + qL[4]          # fwd = the more common draw
        plan.update(kind='concat2', qop_op='concat_fwd', ans=ans)
        return plan
    # ---- value rule -> propagation engine (two vocab stages, as the probe) ----
    deadline = time.time() + deadline_s
    glyphs = set(opchars)
    if canonical:
        stages = [({g: PE.FAMA[g] for g in glyphs}, 'famA'),
                  ({g: PE.FULL for g in glyphs}, 'FULL')]
    else:
        stages = [({g: PE.OPB for g in glyphs}, 'opB'),
                  ({g: PE.FULL for g in glyphs}, 'FULL')]
    fallbacks = []
    for vocab, stage in stages:
        eng = None
        try:
            eng = PE.Eng(eqs, qL, vocab, canonical, deadline)
            eng.propagate()
        except PE.Contra as e:
            fallbacks.append({'stage': stage, 'sev': list(eng.sev) if eng else [],
                              'contra': e})
            continue
        except PE.Timeout:
            return None
        plan.update(stage=stage, fallbacks=fallbacks, eng=eng,
                    chain_sev=list(eng.sev))
        det = answer_detail(eng)
        if det is not None:
            plan.update(kind='chain', det=det)
            return plan
        why = _stall_why(eng)
        plan['stall1'] = why
        plan['stall_state'] = _stall_state(eng, why)
        sp = explore_split(eng) if SPLIT_DEPTH >= 1 else None
        if sp is not None:
            plan.update(kind='split', split=sp)
            return plan
        try:
            bail = guess_descent(eng, why)
        except PE.Timeout:
            return None
        plan.update(kind='bail', bail=bail)
        if bail['det'] is None:
            plan.update(kind='lastresort',
                        ans=qL[0] + qL[1] + qL[3] + qL[4])
        return plan
    plan.update(kind='lastresort', fallbacks=fallbacks, stage=None,
                chain_sev=[], ans=qL[0] + qL[1] + qL[3] + qL[4])
    return plan


def _stall_state(eng, why):
    """observable facts for the stall line (live candidates)."""
    st = {}
    if why and why.startswith('query-op-ambiguous'):
        st['ops'] = sorted(eng.opdom[eng.qop])
    if why == 'query-digit-unpinned':
        qsy = []
        for s in (eng.qa, eng.qb, eng.qc, eng.qd):
            if s not in qsy:
                qsy.append(s)
        st['open'] = [s for s in qsy if len(eng.dom[s]) > 1]
    return st


# ---------------------------------------------------------------- serializer
class _Num:
    """strictly monotone step counter."""
    def __init__(self):
        self.n = 0

    def __call__(self, txt):
        self.n += 1
        return f"s{self.n}: {txt}"


def _ev_line(ev, let, olet, stage):
    """one rendered line per structured engine event; None = not rendered."""
    k = ev[0]
    if k == 'mode_parse':
        return (f"only the {MODE[ev[1]]} digit order parses every example (the other"
                f" would put a sign glyph inside a number) -> {MODE[ev[1]]} order.")
    if k == 'concat_pin':
        g = olet[ev[1]]
        ops = ev[2]
        if len(ops) == 1:
            order = 'in order' if next(iter(ops)) == 'concat_fwd' else 'swapped'
            return (f"{g}: every {g} example RHS is its operand symbols juxtaposed"
                    f" ({order}) -> {g} is concatenation; it pins no digits.")
        return (f"{g}: every {g} example RHS is its operand symbols juxtaposed"
                f" -> {g} is concatenation (order still open); it pins no digits.")
    if k == 'regime':
        if stage == 'FULL':
            return ("allow the FULL rule set: any glyph may hide any rule"
                    " (concatenation only where the RHS is a juxtaposition).")
        if ev[1]:
            return ("the operator glyphs are the literal + - * signs -> each follows"
                    " its own family: '+' add-family, '*' times-family,"
                    " '-' minus-family.")
        return ("the operator glyphs are arbitrary symbols -> use the observed rule"
                " set for each.")
    if k == 'lead0':
        return "no number starts with 0 -> a symbol in a leading position is not 0."
    if k == 'pin':
        s, d, why = ev[1], ev[2], ev[3]
        lt = let[s]
        if isinstance(why, tuple) and why and why[0] == 'fit':
            return f"{lt} = {d}: the only digit that fits every {olet[why[1]]} example."
        if isinstance(why, tuple) and why and why[0] == 'only_left':
            return f"{lt} = {d}: the only digit left for it (all others are taken)."
        if isinstance(why, tuple) and why and why[0] == 'bij':
            return (f"{lt} = {d}: all ten digits are used and no other symbol"
                    f" can take {d}.")
        if isinstance(why, tuple) and why and why[0] == 'split':
            return None                              # the case header states it
        return f"{lt} = {d}."
    if k == 'worp':
        return (f"rules are drawn without replacement ->"
                f" {olet[ev[1]]} is not {OP_WORD[ev[2]]}.")
    if k == 'opkill':
        ops = ", ".join(OP_WORD[o] for o in ev[2])
        return (f"{olet[ev[1]]} can not be {ops}: no digit assignment fits its"
                f" examples -> drop.")
    if k == 'modekill':
        m, g = ev[1], ev[2]
        return (f"{MODE[m]} order fails for every remaining {olet[g]} rule"
                f" -> {MODE[not m]} order.")
    if k == 'tighten':
        return None        # domain prunes: sound but verbose; pins carry the chain
                           # (token-budget decision, r7 gates: med <= ~900)
    if k == 'split_case':
        return None                                  # the case header states it
    raise AssertionError(f"unrenderable event {ev!r}")


def _contra_line(e, let, olet):
    ev = e.ev
    k = ev[0]
    if k == 'c_inj':
        return f"contradiction: {let[ev[1]]} loses every digit (all are taken)."
    if k == 'c_wipe':
        return (f"contradiction: no digit fits {let[ev[1]]} on the"
                f" {olet[ev[2]]} examples.")
    if k == 'c_allops':
        return f"contradiction: every rule for {olet[ev[1]]} is eliminated."
    if k == 'c_modes':
        return "contradiction: both digit orders die."
    if k == 'c_lead0':
        return (f"contradiction: {let[ev[1]]} must be 0 but it leads a number.")
    if k == 'c_worp':
        return f"contradiction: no rule is left for {olet[ev[1]]}."
    if k == 'c_concat_voc':
        return (f"contradiction: {olet[ev[1]]} looks like concatenation but the"
                f" assumed rule set has none.")
    if k == 'c_voc':
        return f"contradiction: no candidate rule for {olet[ev[1]]}."
    if k == 'c_nomode':
        return "contradiction: neither digit order parses the examples."
    if k == 'c_sym10':
        return "contradiction: more than 10 digit symbols."
    return f"contradiction: {e}."


def _stall_line(why, st, let, olet, qg):
    if why and why.startswith('query-op-ambiguous') and st.get('ops'):
        ops = " or ".join(OP_WORD[o] for o in st['ops'])
        return f"forced deduction stalls: {qg} could still be {ops}."
    if why == 'query-digit-unpinned' and st.get('open'):
        ls = " ".join(let[s] for s in st['open'])
        return f"forced deduction stalls: query symbol(s) {ls} are not pinned."
    txt = STALL_TXT.get(why) or f"{why}."
    return f"forced deduction stalls: {txt}"


def _case_desc(kind, who, v, let, olet):
    if kind == 'mode':
        return f"{MODE[v]} order"
    if kind == 'op':
        return f"{olet[who]} = {OP_WORD[v]}"
    return f"{let[who]} = {v}"


def _split_point_line(kind, who, vals, let, olet):
    a = _case_desc(kind, who, vals[0], let, olet)
    b = _case_desc(kind, who, vals[1], let, olet)
    return (f"try a case split: both {a} and {b} are still possible"
            f" -> case A = {a}, case B = {b}.")


def _final_block(det, qL, let, olet, head):
    """Resolved/Best-guess reading -> operands -> compute -> encode lines (no box)."""
    L = []
    qg = olet[qL[2]]
    qsy = []
    for s in (qL[0], qL[1], qL[3], qL[4]):
        if s not in qsy:
            qsy.append(s)
    asg = " ".join(f"{let[s]}={det['val'][s]}" for s in qsy)
    L.append(f"{head}: {MODE[det['rev']]} order; {qg} = {OP_WORD[det['op']]}; {asg}.")
    if det['rev']:
        L.append(f"Read the query operands little-endian (reverse the digits):"
                 f" {qL[0]}{qL[1]} -> {det['A']}, {qL[3]}{qL[4]} -> {det['B']}.")
    else:
        L.append(f"Query operands: {qL[0]}{qL[1]} -> {det['A']},"
                 f" {qL[3]}{qL[4]} -> {det['B']}.")
    L.append(f"Compute: {det['A']} {OP_WORD[det['op']]} {det['B']} = {det['v']}.")
    enc = " ".join(f"{d}={s}" for d, s in det['enc'])
    body = ''.join(s for _, s in det['enc'])
    tail = ""
    if det['rev']:
        body = body[::-1]
        tail += f"; little-endian -> reverse to {body}"
    if det['sign']:
        if det['s_end'] == 'suf':
            body = body + qL[2]
            tail += f"; negative -> sign glyph {qL[2]} at the end: {body}"
        else:
            body = qL[2] + body
            tail += f"; negative -> sign glyph {qL[2]} in front: {body}"
    assert body == det['ans'], (body, det['ans'])
    L.append(f"Encode digit by digit: {enc} -> {''.join(s for _, s in det['enc'])}"
             f"{tail}.")
    return L


def serialize(plan):
    """plan -> full ASCII CoT text (deterministic)."""
    eqs, qL = plan['eqs'], plan['qL']
    let, olet = plan['let'], plan['olet']
    qg = olet[qL[2]]
    L = ["We need to infer the transformation rule from the examples.", ""]
    L.extend(CC.table_lines6(eqs, qL, let, olet, plan['canonical']))
    L.append("")
    # letter-form recap dropped (token-budget decision, r7 gates: every letter is
    # already defined by the table; only the query is restated in letters):
    qlets = f"{let[qL[0]]}{let[qL[1]]} {qg} {let[qL[3]]}{let[qL[4]]}"
    L.append(f"Query: {qL} -> {qlets}")
    L.append("")
    # ---- routing check ----
    L.append(f"Routing check on the query operator {qg}: is each {qg} example RHS"
             f" just the operand symbols written next to each other?")
    pats = plan['pats']
    if plan['kind'] in ('concat', 'concat2'):
        for k, Le, Re, p in pats:
            ino, sw = Le[0] + Le[1] + Le[3] + Le[4], Le[3] + Le[4] + Le[0] + Le[1]
            if p == {'concat_fwd', 'concat_rev'}:
                verdict = 'matches both ways'
            elif p == {'concat_fwd'}:
                verdict = 'matches in order'
            else:
                verdict = 'matches swapped'
            L.append(f"EQ{k} RHS {Re} vs {ino} (in order) / {sw} (swapped)"
                     f" -> {verdict}.")
        if plan['kind'] == 'concat':
            order = 'in order' if plan['qop_op'] == 'concat_fwd' else 'swapped'
            L.append(f"Every {qg} example matches {order} -> {qg} is concatenation"
                     f" ({order}).")
            a, b = qL[0] + qL[1], qL[3] + qL[4]
            first, second = (a, b) if plan['qop_op'] == 'concat_fwd' else (b, a)
            L.append(f"Apply to the query: operands {a} and {b} -> {first}{second}.")
        else:
            L.append(f"Both orders match every {qg} example -> the order can not be"
                     f" deduced; best guess: in order, the more common draw (guess).")
            L.append(f"Apply to the query: operands {qL[0]}{qL[1]} and"
                     f" {qL[3]}{qL[4]} -> {plan['ans']}.")
        L.append(f"\\boxed{{{plan['ans']}}}")
        return "\n".join(L)
    # value verdict: render lines up to the structural refutation
    run = None
    shown = []
    for k, Le, Re, p in pats:
        ino, sw = Le[0] + Le[1] + Le[3] + Le[4], Le[3] + Le[4] + Le[0] + Le[1]
        if len(Re) != 4:
            shown.append(f"EQ{k} RHS {Re} has {len(Re)} symbols, the operands have 4"
                         f" -> not a juxtaposition.")
            run = set()
        elif not p:
            shown.append(f"EQ{k} RHS {Re} vs {ino} (in order) / {sw} (swapped)"
                         f" -> neither.")
            run = set()
        else:
            verdict = ('matches both ways' if len(p) == 2 else
                       'matches in order' if 'concat_fwd' in p else 'matches swapped')
            shown.append(f"EQ{k} RHS {Re} vs {ino} (in order) / {sw} (swapped)"
                         f" -> {verdict}.")
            run = p if run is None else (run & p)
        if run is not None and not run:
            break
    L.extend(shown)
    L.append(f"No single juxtaposition order survives -> {qg} is a value rule.")
    L.append(CHAIN_HDR)
    num = _Num()
    # ---- fallback stages ----
    for fb in plan.get('fallbacks', []):
        for ev in fb['sev']:
            t = _ev_line(ev, let, olet, fb['stage'])
            if t:
                L.append(num(t))
        L.append(num(_contra_line(fb['contra'], let, olet)))
        L.append(num("the assumed rule set fails -> allow the FULL rule set"
                     " (any glyph may hide any rule)."))
    if plan['kind'] == 'lastresort' and plan.get('stage') is None:
        L.append(num("no rule set yields a consistent reading -> fall back to the"
                     " simplest transformation: copy the operand symbols (guess)."))
        L.append(f"Best guess: the operand symbols juxtaposed -> {plan['ans']}.")
        L.append(f"\\boxed{{{plan['ans']}}}")
        return "\n".join(L)
    # ---- main chain ----
    for ev in plan['chain_sev']:
        t = _ev_line(ev, let, olet, plan['stage'])
        if t:
            L.append(num(t))
    if plan['kind'] == 'chain':
        L.extend(_final_block(plan['det'], qL, let, olet, "Resolved"))
        L.append(f"\\boxed{{{plan['det']['ans']}}}")
        return "\n".join(L)
    # ---- stall ----
    L.append(num(_stall_line(plan['stall1'], plan.get('stall_state', {}),
                             let, olet, qg)))
    if plan['kind'] == 'split':
        sp = plan['split']
        kind, who, vals = sp['point']
        L.append(num(_split_point_line(kind, who, vals, let, olet)))
        descs = [_case_desc(kind, who, v, let, olet) for v in vals]
        outs = [b['out'][0] for b in sp['branches']]
        for ci, br in enumerate(sp['branches']):
            L.append(f"Case {'AB'[ci]} ({descs[ci]}):")
            for ev in br['sev']:
                t = _ev_line(ev, let, olet, plan['stage'])
                if t:
                    L.append(num(t))
            if br['out'][0] == 'dead':
                L.append(num(_contra_line(br['out'][1], let, olet)
                             + f" -> case {'AB'[ci]} dies."))
            elif outs == ['ans', 'ans']:
                L.extend(_final_block(br['out'][1], qL, let, olet,
                                      f"Case {'AB'[ci]} resolved"))
        if outs == ['ans', 'ans']:
            ans = sp['branches'][0]['out'][1]['ans']
            L.append(f"Both cases give the same result {ans} -> the answer does not"
                     f" depend on the choice.")
            L.append(f"\\boxed{{{ans}}}")
            return "\n".join(L)
        live = next(i for i, o in enumerate(outs) if o == 'ans')
        det = sp['branches'][live]['out'][1]
        L.append(f"Only case {'AB'[live]} survives and it is fully forced.")
        L.extend(_final_block(det, qL, let, olet, "Resolved"))
        L.append(f"\\boxed{{{det['ans']}}}")
        return "\n".join(L)
    # ---- bail ----
    assert plan['kind'] in ('bail', 'lastresort')
    bail = plan['bail']
    L.append(num(NOSPLIT_LINE))
    for st in bail['steps']:
        if st[0] == 'stall':
            L.append(num(_stall_line(st[1], {}, let, olet, qg)))
        elif st[0] == 'commit':
            _, kind, who, v, rat, sev = st
            word = {'mode': 'order', 'op': 'rule', 'sym': 'digit'}[kind]
            L.append(num(f"best guess at the {word} level:"
                         f" {rat} -> assume {_case_desc(kind, who, v, let, olet)}"
                         f" (guess)."))
            for ev in sev:
                t = _ev_line(ev, let, olet, plan['stage'])
                if t:
                    L.append(num(t))
        elif st[0] == 'contra':
            _, e, kind, who, v = st
            L.append(num(_contra_line(e, let, olet)
                         + f" -> the {_case_desc(kind, who, v, let, olet)} guess"
                         f" dies; exclude it."))
        elif st[0] == 'forced':
            _, kind, who, v, sev = st
            L.append(num(f"only {_case_desc(kind, who, v, let, olet)} remains"
                         f" -> forced."))
            for ev in sev:
                t = _ev_line(ev, let, olet, plan['stage'])
                if t:
                    L.append(num(t))
        elif st[0] == 'assemble':
            _, det, picks = st
            parts = [f"{olet[qL[2]]} = {OP_WORD[det['op']]} (highest prior remaining)"]
            parts += [f"{let[s]} = {d} (smallest remaining candidate)"
                      for s, d in sorted(picks.items(), key=lambda t: let[t[0]])
                      ]
            L.append(num("refinement exhausted -> assemble the most constrained"
                         " reading: " + "; ".join(parts) + " (guess)."))
    if plan['kind'] == 'lastresort':
        L.append(num("no consistent reading can be assembled -> fall back to the"
                     " simplest transformation: copy the operand symbols (guess)."))
        L.append(f"Best guess: the operand symbols juxtaposed -> {plan['ans']}.")
        L.append(f"\\boxed{{{plan['ans']}}}")
        return "\n".join(L)
    det = bail['det']
    L.extend(_final_block(det, qL, let, olet, "Best guess"))
    L.append(f"\\boxed{{{det['ans']}}}")
    return "\n".join(L)


def render_trace_r7(prompt, deadline_s=30.0):
    """deterministic r7 render. returns (cot, kind, info) or (None, None, reason)."""
    plan = drive(prompt, deadline_s=deadline_s)
    if plan is None:
        return None, None, "undrivable (guess row / malformed / timeout)"
    try:
        cot = serialize(plan)
    except AssertionError as e:
        return None, plan.get('kind'), f"serialize-assert {e}"
    info = {'kind': plan['kind'], 'canonical': int(plan['canonical']),
            'stage': plan.get('stage'), 'stall': plan.get('stall1'),
            'n_commits': sum(1 for s in plan.get('bail', {}).get('steps', [])
                             if s[0] in ('commit', 'contra'))
            if plan['kind'] in ('bail', 'lastresort') else 0}
    if plan['kind'] == 'split':
        info['split_point'] = plan['split']['point'][0]
    return cot, plan['kind'], info


# ================================================================ lint_r7
# INVERTED lint (r4-r6 convention): returns None iff the trace is TRUTHFUL.
# Layers (local checks first -> targeted messages; engine re-execution last):
#   1. ASCII-only.
#   2. exactly one \boxed{} and it is the final line.
#   3. step counters s1,s2,... strictly monotone +1.
#   4. symbol-table block re-derived from the prompt (r6 FIX-A lint, reused:
#      per-position scans truthful, union table == first-appearance walk).
#   5. routing-check conclusion re-derived from the prompt: the claimed verdict
#      (value rule / concat in-order / concat swapped / both orders) must equal
#      the TRUE intersection of juxtaposition patterns over the query-glyph
#      examples; for concat kinds, boxed == the derived juxtaposition.
#   6. final-block arithmetic re-derived locally: the Resolved/Best-guess
#      assignment -> operands per stated mode -> Compute recomputed via the op
#      table -> Encode digits == digits of the computed value -> reversal/sign
#      chain -> boxed == the final encoded string.
#   7. engine re-execution: re-run the WHOLE deterministic r7 policy (drive())
#      on the prompt and demand line equality.  This makes every deduction line
#      re-derivable (a true forced inference at that exact position), proves a
#      split is REAL (the engine itself found both cases alive and killed one
#      with that exact contradiction), proves a bail truly exhausts, and ties
#      the boxed answer to the engine answer (chain/split) or the engine's
#      principled guess (bail/lastresort).
_S_RE = re.compile(r"^s(\d+): ")
_OPW_ALT = "|".join(sorted((re.escape(w) for w in OP_WORD.values()),
                           key=len, reverse=True))
_HEAD_RE = re.compile(
    r"^(Resolved|Best guess|Case [AB] resolved): (standard|little-endian) order;"
    r" [xyzw] = (" + _OPW_ALT + r"); ((?:[A-J]=\d ?)+)\.$")
_COMP_RE = re.compile(
    r"^Compute: (-?\d+) (" + _OPW_ALT + r") (-?\d+) = (-?\d+)\.$")
_ENC_RE = re.compile(r"^Encode digit by digit: ((?:\d=\S )*\d=\S) -> (.+)\.$")
_LASTR_RE = re.compile(r"^Best guess: the operand symbols juxtaposed -> (\S+)\.$")
ROUTE_HDR = "Routing check on the query operator"
WORD_OP = {v: k for k, v in OP_WORD.items()}


def _query_inter(eqs, qL):
    """TRUE intersection of juxtaposition patterns over query-glyph examples."""
    inter = None
    for L, R in eqs:
        if L[2] != qL[2]:
            continue
        sign = len(R) > 1 and (R[0] == L[2] or R[-1] == L[2])
        p = C2.concat_patterns(L, R, sign)
        inter = p if inter is None else (inter & p)
    return inter


def _lint_final_block(lines, i, m, let, qL, boxed):
    """re-derive one Resolved/Best-guess block locally. returns (err, final_str)."""
    rev = (m.group(2) == 'little-endian')
    op = WORD_OP[m.group(3)]
    asg = {}
    for tok in m.group(4).split():
        lt, d = tok.split('=')
        asg[lt] = int(d)
    try:
        val = {c: asg[let[c]] for c in (qL[0], qL[1], qL[3], qL[4])}
    except KeyError as e:
        return f"final block: query letter {e} missing from the assignment", None
    if rev:
        A = val[qL[1]] * 10 + val[qL[0]]
        B = val[qL[4]] * 10 + val[qL[3]]
    else:
        A = val[qL[0]] * 10 + val[qL[1]]
        B = val[qL[3]] * 10 + val[qL[4]]
    cm = None
    for j in range(i + 1, min(i + 4, len(lines))):
        cm = _COMP_RE.match(lines[j])
        if cm:
            ci = j
            break
    if cm is None:
        return "final block: Compute line missing", None
    if int(cm.group(1)) != A or int(cm.group(3)) != B:
        return (f"final block: operands {cm.group(1)},{cm.group(3)} differ from the"
                f" assignment reading ({A},{B})"), None
    if WORD_OP[cm.group(2)] != op:
        return "final block: Compute op differs from the resolved rule", None
    v = C2.OPS[op](A, B)
    if v is None or int(cm.group(4)) != v:
        return (f"final block: arithmetic invented ({A} {cm.group(2)} {B}"
                f" != {cm.group(4)})"), None
    em = _ENC_RE.match(lines[ci + 1]) if ci + 1 < len(lines) else None
    if em is None:
        return "final block: Encode line missing", None
    pairs = [t.split('=') for t in em.group(1).split()]
    digs = []
    x = abs(v)
    if x == 0:
        digs = [0]
    while x:
        digs.append(x % 10)
        x //= 10
    digs = digs[::-1]
    if [int(d) for d, _ in pairs] != digs:
        return "final block: encoded digits != digits of the computed value", None
    body0 = ''.join(s for _, s in pairs)
    rest = em.group(2)
    if not rest.startswith(body0):
        return "final block: encoded string != the digit-by-digit symbols", None
    cur, rest = body0, rest[len(body0):]
    if rev:
        want = "; little-endian -> reverse to " + body0[::-1]
        if not rest.startswith(want):
            return "final block: little-endian reversal wrong/missing", None
        cur, rest = body0[::-1], rest[len(want):]
    sign_needed = (op in C2.NEGPRE) or (op in C2.SIGNED and v < 0)
    if v < 0 and not sign_needed:
        return "final block: negative value under a sign-free rule", None
    if sign_needed:
        suf = f"; negative -> sign glyph {qL[2]} at the end: {cur}{qL[2]}"
        pre = f"; negative -> sign glyph {qL[2]} in front: {qL[2]}{cur}"
        if rest == suf:
            cur = cur + qL[2]
        elif rest == pre:
            cur = qL[2] + cur
        else:
            return "final block: sign-glyph placement wrong/missing", None
        rest = ""
    if rest:
        return "final block: unexplained tail on the Encode line", None
    return None, cur


def lint_r7(cot, prompt, deadline_s=60.0):
    """returns None iff truthful (every layer documented above passes)."""
    for ch in cot:
        if ord(ch) > 126 or (ord(ch) < 32 and ch != '\n'):
            return f"non-ASCII char {ch!r}"
    lines = cot.splitlines()
    if not lines or not (lines[-1].startswith("\\boxed{") and lines[-1].endswith("}")):
        return "boxed answer is not the final line"
    if sum(1 for ln in lines if "\\boxed{" in ln) != 1:
        return "multiple boxed answers"
    boxed = lines[-1][len("\\boxed{"):-1]
    n = 0
    for ln in lines:
        sm = _S_RE.match(ln)
        if sm:
            k = int(sm.group(1))
            if k != n + 1:
                return f"counter not monotone: s{k} after s{n}"
            n = k
    pr = C2.parse(prompt)
    if pr is None:
        return "prompt unparseable"
    eqs, qL = pr
    err = CC._lint_table6(lines, prompt)
    if err:
        return "table: " + err
    # ---- routing conclusion re-derived from the prompt ----
    if not any(ROUTE_HDR in ln for ln in lines):
        return "routing check missing"
    inter = _query_inter(eqs, qL)
    if inter is None:
        return "query glyph has no examples (guess row: not this grammar)"
    is_value = any(ln.endswith("is a value rule.") for ln in lines)
    is_c1 = any("is concatenation (in order)." in ln
                or "is concatenation (swapped)." in ln for ln in lines)
    is_c2 = any("Both orders match every" in ln for ln in lines)
    if sum([is_value, is_c1, is_c2]) != 1:
        return "routing conclusion missing or contradictory"
    if is_value and len(inter) != 0:
        return "routing conclusion wrong: a juxtaposition order DOES survive"
    if is_c1:
        if len(inter) != 1:
            return "routing conclusion wrong: concat does not survive uniquely"
        op0 = next(iter(inter))
        order = 'in order' if op0 == 'concat_fwd' else 'swapped'
        if not any(f"is concatenation ({order})." in ln for ln in lines):
            return "routing conclusion wrong: concat order mislabeled"
        exp = (qL[0] + qL[1] + qL[3] + qL[4]) if op0 == 'concat_fwd' \
            else (qL[3] + qL[4] + qL[0] + qL[1])
        if boxed != exp:
            return "boxed != the concatenation the routing check derived"
    if is_c2:
        if len(inter) != 2:
            return "routing conclusion wrong: both orders do NOT fit every example"
        if boxed != qL[0] + qL[1] + qL[3] + qL[4]:
            return "boxed != the stated in-order guess"
    # ---- final-block local re-derivation (value rows) ----
    if is_value:
        let, olet, _ = _lets(eqs, qL)
        heads = [(i, m) for i, m in ((i, _HEAD_RE.match(ln))
                                     for i, ln in enumerate(lines)) if m]
        lasts = [m.group(1) for ln in lines if (m := _LASTR_RE.match(ln))]
        if lasts:
            exp = qL[0] + qL[1] + qL[3] + qL[4]
            if lasts[-1] != exp or boxed != exp:
                return "lastresort guess != the operand symbols juxtaposed"
        elif not heads:
            return "no final derivation block before the boxed answer"
        finals = []
        for i, m in heads:
            err, fin = _lint_final_block(lines, i, m, let, qL, boxed)
            if err:
                return err
            finals.append(fin)
        if heads:
            if any("Both cases give the same result" in ln for ln in lines):
                if len(set(finals)) != 1 or boxed != finals[0]:
                    return "boxed != the (claimed-equal) case results"
            elif boxed != finals[-1]:
                return "teleported answer: boxed != the derived encoded string"
    # ---- engine re-execution: the whole policy, line by line ----
    plan2 = drive(prompt, deadline_s=deadline_s)
    if plan2 is None:
        return "engine cannot drive this prompt"
    if not is_value:
        tk = 'concat2' if is_c2 else 'concat'
    elif any("copy the operand symbols (guess)" in ln for ln in lines):
        tk = 'lastresort'
    elif any("try a case split" in ln for ln in lines):
        tk = 'split'
    elif any(NOSPLIT_LINE in ln for ln in lines):
        tk = 'bail'
    else:
        tk = 'chain'
    ek = plan2['kind']
    if tk != ek:
        if tk == 'split' and ek == 'chain':
            return "unjustified split: the forced chain resolves without it"
        if tk in ('bail', 'lastresort') and ek in ('chain', 'split'):
            return f"premature bail: the engine still resolves (kind={ek})"
        if tk == 'chain' and ek in ('split', 'bail', 'lastresort'):
            return f"teleported answer: engine kind {ek} but trace is a pure chain"
        return f"trace kind {tk} != engine kind {ek}"
    try:
        exp = serialize(plan2)
    except AssertionError as e:
        return f"engine re-render failed: {e}"
    if exp != cot:
        el = exp.splitlines()
        for j, (a, b) in enumerate(zip(lines, el)):
            if a != b:
                return (f"line {j + 1} not engine-derivable: got {a[:64]!r}"
                        f" expected {b[:64]!r}")
        return f"length mismatch vs engine re-render ({len(lines)} vs {len(el)} lines)"
    return None


# ================================================================ corpus drivers
OUT_DIR = os.path.join(ROOT, 'pipeline', 'data', 'crypt_r7')
CHAIN_KINDS = frozenset(('chain', 'concat'))          # zero-branch resolutions
SPLIT_KINDS = frozenset(('split',))
BAIL_KINDS = frozenset(('bail', 'lastresort', 'concat2'))


def _boxed_of(cot):
    return cot.splitlines()[-1][len("\\boxed{"):-1]


def _real_work(r):
    cot, kind, info = render_trace_r7(r['prompt'], deadline_s=45.0)
    if cot is None:
        return {'id': r['id'], 'drop': str(info)}
    gold = r['answer'].strip()
    final = _boxed_of(cot)
    if kind in CHAIN_KINDS | SPLIT_KINDS and final != gold:
        return {'id': r['id'], 'drop': f'sound-but-wrong ({kind})'}
    return {'id': r['id'], 'category': 'cryptarithm_deduce', 'prompt': r['prompt'],
            'cot': cot, 'final': final, 'kind': kind, 'hit': int(final == gold),
            'canonical': info['canonical']}


def render_real7(procs=10):
    import multiprocessing as mp
    os.makedirs(OUT_DIR, exist_ok=True)
    vids = CC.val_ids()
    rows = [r for r in CC.load_real('cryptarithm_deduce') if r['id'] not in vids]
    n_excl = len(CC.load_real('cryptarithm_deduce')) - len(rows)
    print(f"real deduce rows: {len(rows)} after excluding {n_excl} val ids")
    with mp.Pool(procs) as pool:
        res = pool.map(_real_work, rows, chunksize=4)
    kept = [r for r in res if 'drop' not in r]
    drops = [r for r in res if 'drop' in r]
    tok = CC.tokenizer()
    out = os.path.join(OUT_DIR, 'crypt_deduce_real.jsonl')
    with open(out, 'w') as f:
        for r in kept:
            r['ntok'] = len(tok.encode(r['cot']).ids)
            f.write(json.dumps(r) + '\n')
    from collections import Counter
    print(f"kept {len(kept)} -> {out}")
    print("kinds:", Counter(r['kind'] for r in kept))
    print("drops:", Counter(d['drop'].split(' (')[0] for d in drops))
    bail = [r for r in kept if r['kind'] in BAIL_KINDS]
    if bail:
        print(f"bail-guess hit rate (real): {sum(r['hit'] for r in bail)}/{len(bail)}"
              f" = {sum(r['hit'] for r in bail) / len(bail):.3f}")


def _synth_work(i_seed):
    i, seed = i_seed
    rng = random.Random(seed * 1_000_000 + i)
    p = CC.gen_puzzle_r5(rng)
    if p is None:
        return None
    cot, kind, info = render_trace_r7(p['prompt'], deadline_s=30.0)
    if cot is None:
        return None
    final = _boxed_of(cot)
    if kind in CHAIN_KINDS | SPLIT_KINDS and final != p['answer']:
        return None                                  # solver-verified strata only
    return {'i': i, 'category': 'cryptarithm_deduce', 'prompt': p['prompt'],
            'cot': cot, 'final': final, 'kind': kind,
            'hit': int(final == p['answer']), 'canonical': info['canonical']}


def render_synth7(n=3000, seed=200, procs=10,
                  frac=(0.55, 0.30, 0.15), canon_frac=0.59):
    """strata REBALANCED vs the real residual (51% would teach premature bailing):
    ~55% chain / 30% split / 15% bail; canonical sub-quota per stratum ~= the
    real canonical fraction .58-.60. Solver-verified: chain/split boxed == gold
    by construction; bail boxed = the engine's principled guess (hit recorded)."""
    import multiprocessing as mp
    from collections import Counter
    os.makedirs(OUT_DIR, exist_ok=True)
    quota = {}
    for st, fr in zip(('chain', 'split', 'bail'), frac):
        q = round(n * fr)
        qc = round(q * canon_frac)
        quota[(st, 1)] = qc
        quota[(st, 0)] = q - qc
    got = {k: [] for k in quota}
    seen_prompts = set()
    i0, batch = 0, 4000
    with mp.Pool(procs) as pool:
        while any(len(got[k]) < quota[k] for k in quota) and i0 < 120000:
            res = pool.map(_synth_work, [(i, seed) for i in range(i0, i0 + batch)],
                           chunksize=16)
            i0 += batch
            for r in res:
                if r is None or r['prompt'] in seen_prompts:
                    continue
                st = ('chain' if r['kind'] in CHAIN_KINDS else
                      'split' if r['kind'] in SPLIT_KINDS else 'bail')
                k = (st, r['canonical'])
                if len(got[k]) < quota[k]:
                    got[k].append(r)
                    seen_prompts.add(r['prompt'])
            print(f"  scanned {i0}: " + " ".join(
                f"{k[0]}/{'c' if k[1] else 's'} {len(v)}/{quota[k]}"
                for k, v in sorted(got.items())))
    short = {k: quota[k] - len(got[k]) for k in quota if len(got[k]) < quota[k]}
    if short:
        print(f"WARNING: unfilled sub-quotas {short}")
    rows = [r for k in sorted(got) for r in got[k]]
    rng = random.Random(seed)
    rng.shuffle(rows)
    tok = CC.tokenizer()
    out = os.path.join(OUT_DIR, 'crypt_deduce_synth.jsonl')
    with open(out, 'w') as f:
        for j, r in enumerate(rows):
            r['id'] = f"synr7-s{seed}-{j:05d}"
            r.pop('i', None)
            r['ntok'] = len(tok.encode(r['cot']).ids)
            f.write(json.dumps(r) + '\n')
    print(f"kept {len(rows)} -> {out}")
    print("kinds:", Counter(r['kind'] for r in rows))
    print(f"canonical fraction: {sum(r['canonical'] for r in rows) / len(rows):.3f}")
    bail = [r for r in rows if r['kind'] in BAIL_KINDS]
    if bail:
        print(f"bail-guess hit rate (synth): {sum(r['hit'] for r in bail)}/{len(bail)}"
              f" = {sum(r['hit'] for r in bail) / len(bail):.3f}")


def _lint_work(d):
    if d['category'] == 'cryptarithm_guess':
        err = CC.lint_r6(d['cot'], d['final'], prompt=d['prompt'])
    else:
        err = lint_r7(d['cot'], d['prompt'])
    return (d['id'], err)


def gate7(procs=10):
    """hard corpus gates: independent re-lint of EVERY row, 0 val leaks,
    ASCII sweep, token distribution."""
    import multiprocessing as mp
    from collections import Counter
    vids = CC.val_ids()
    val_prompts = {json.loads(l)['prompt']
                   for l in open(os.path.join(ROOT, 'pipeline', 'data', 'val.jsonl'))}
    allrows = []
    for fn in ('crypt_deduce_real.jsonl', 'crypt_deduce_synth.jsonl', 'crypt_guess.jsonl'):
        path = os.path.join(OUT_DIR, fn)
        rows = [json.loads(l) for l in open(path)]
        allrows += rows
        print(f"{fn}: {len(rows)} rows")
    leaks = [d['id'] for d in allrows if d['id'] in vids]
    pleaks = [d['id'] for d in allrows if d['prompt'] in val_prompts]
    print(f"val leaks: id {len(leaks)} prompt {len(pleaks)}")
    nonascii = [d['id'] for d in allrows
                if any(ord(c) > 126 or (ord(c) < 32 and c != '\n')
                       for c in d['cot'] + str(d['final']))]
    print(f"non-ascii rows: {len(nonascii)} {nonascii[:5]}")
    nt = sorted(d['ntok'] for d in allrows)
    med, p95, mx = nt[len(nt) // 2], nt[int(len(nt) * .95)], nt[-1]
    print(f"ntok: med {med} p95 {p95} max {mx}"
          f"  (gates: med<=~900 p95<=3200 max<5000)")
    with mp.Pool(procs) as pool:
        res = pool.map(_lint_work, allrows, chunksize=8)
    fails = [(i, e) for i, e in res if e]
    print(f"independent re-lint: {len(fails)} fails / {len(allrows)} rows")
    for i, e in fails[:10]:
        print(f"  LINT-FAIL {i}: {e}")
    ok = (not leaks and not pleaks and not nonascii and not fails
          and med <= 900 and p95 <= 3200 and mx < 5000)
    print("GATES " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'gen-test7'
    if cmd == 'render-real7':
        render_real7(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == 'render-synth7':
        render_synth7(n=int(sys.argv[2]) if len(sys.argv) > 2 else 3000,
                      seed=int(sys.argv[3]) if len(sys.argv) > 3 else 200,
                      procs=int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    elif cmd == 'gate7':
        sys.exit(gate7(int(sys.argv[2]) if len(sys.argv) > 2 else 10))
    elif cmd == 'gen-test7':
        rng = random.Random(7)
        seen = set()
        while len(seen) < 4:
            p = CC.gen_puzzle_r5(rng)
            if p is None:
                continue
            cot, kind, info = render_trace_r7(p['prompt'])
            if cot is None or kind in seen:
                continue
            seen.add(kind)
            err = lint_r7(cot, p['prompt'])
            print(f"=== kind={kind} lint={err!r} gold={p['answer']!r}"
                  f" boxed={_boxed_of(cot)!r}\n{cot}\n")
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
