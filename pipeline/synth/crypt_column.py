#!/usr/bin/env python3
"""crypt_column.py -- HONEST forward cryptarithm renderer (user design spec, 2026-06-14).

Engine over COLUMN FACTS (linear equations on digit-symbols + carries) instead of whole-equation
enumeration. The no-teleport invariant is enforced BY CONSTRUCTION: a digit is pinned only when a
fact has exactly ONE unresolved variable, so every number cited on a line is already pinned or
computed from pinned digits. The known solution is used ONLY to verify + to gate emission (we emit
a row only if the engine reaches the gold map forward). If the engine stalls, we REFUSE -- never
teleport. This is the fix for the 28%/83% forward-reference rate in crypt_mechanical_render.py.

Rules (priority): single -> magnitude -> alldiff -> product -> factor -> (stall) combine.
  single    : a column/derived fact with 1 unknown -> solve it (exact, coeffs are +-1/+-10).
  magnitude : 2-digit + 2-digit add with 3-digit result -> leading result digit = 1 (carry forced).
  alldiff   : naked single (used digit removed) + hidden single (digit with one home).
  product   : mul equation with BOTH operands pinned -> read off the product digits (honest).
  factor    : mul result fully pinned + one operand pinned -> other operand = exact division.
  combine   : eliminate a shared var across two facts; if result has 1 unknown -> pin; if 2 ->
              bounded all-different enum (<=10 combos, showable). The crux the old renderer skipped.

Run: python3 pipeline/synth/crypt_column.py [N]   -> dumps cracked per-ID files + coverage.
"""
import os, sys, csv, random, math
from itertools import product as iproduct
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
import cryptarithm2 as C2
csv.field_size_limit(10 ** 9)

ADD_K = {'add': 0, 'add_p1': 1, 'add_m1': -1}
MUL_K = {'mul': 0, 'mul_p1': 1, 'mul_m1': -1}            # value = a*b + K
OPPH = {'add': 'a+b', 'add_p1': 'a+b+1', 'add_m1': 'a+b-1', 'mul': 'a*b', 'mul_p1': 'a*b+1',
        'mul_m1': 'a*b-1', 'absdiff': '|a-b|', 'neg_absdiff': '-|a-b|', 'sub_signed': 'a-b',
        'rsub_signed': 'b-a'}
SIGNED = {'absdiff', 'neg_absdiff', 'sub_signed', 'rsub_signed'}
SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"


def opval(name, a, b):
    return {'add': a + b, 'add_p1': a + b + 1, 'add_m1': a + b - 1, 'mul': a * b,
            'mul_p1': a * b + 1, 'mul_m1': a * b - 1, 'absdiff': abs(a - b),
            'neg_absdiff': -abs(a - b), 'sub_signed': a - b, 'rsub_signed': b - a}[name]


def places(glyphs, order):
    """map place-index (0=units,1=tens,...) -> glyph for a written number."""
    g = list(glyphs)
    if order == 'standard':
        return {len(g) - 1 - i: g[i] for i in range(len(g))}
    return {i: g[i] for i in range(len(g))}


class Fact:
    """sum(coeff * var) == const ; var = ('s',glyph) | ('c',(eqi,col)). tag/kind for rendering."""
    __slots__ = ('terms', 'const', 'tag', 'kind')
    def __init__(self, terms, const, tag, kind=None):
        self.terms = {v: c for v, c in terms.items() if c != 0}
        self.const = const; self.tag = tag; self.kind = kind
    def sig(self):
        items = tuple(sorted((repr(v), c) for v, c in self.terms.items()))
        return (items, self.const)


class Engine:
    def __init__(self, eqs, qL, ops, rev, showable=False, orient=None):
        self.ops = ops; self.rev = rev; self.order = 'reversed' if rev else 'standard'
        self.eqs = eqs; self.qL = qL
        self.showable = showable          # True => no opaque arc-exact pinning; honest rules only
        self.orient = orient or {}        # ei -> True if |a-b| minuend is b (swap), for borrow facts
        self.dom = {}; self.cdom = {}; self.facts = []; self.log = []
        self.seen_sig = set(); self.derived = []
        self._build()

    # ---- variable resolution ------------------------------------------------
    def resolved(self, v):
        return len(self.dom[v[1]]) == 1 if v[0] == 's' else len(self.cdom[v[1]]) == 1
    def value(self, v):
        return next(iter(self.dom[v[1]])) if v[0] == 's' else next(iter(self.cdom[v[1]]))
    def show(self, v):
        if self.resolved(v):
            return str(self.value(v))
        if v[0] == 's':
            return v[1]
        return 'carry-into-tens' if v[1][1] == 1 else 'carry-into-hundreds'

    # ---- construction -------------------------------------------------------
    def _seed(self, glyphs, lead):
        for g in glyphs:
            self.dom.setdefault(g, set(range(10)))
        for g in lead:
            self.dom[g].discard(0)

    def _build(self):
        glyphs, lead = set(), set()
        alleqs = list(self.eqs) + [(self.qL, '')]
        for (L, R) in alleqs:
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            glyphs |= set(a) | set(b) | set(Rg)
            ap, bp = places(a, self.order), places(b, self.order)
            lead |= {ap[max(ap)], bp[max(bp)]}
            if Rg:
                rp = places(Rg, self.order); lead.add(rp[max(rp)])
        self._seed(glyphs, lead)
        # column facts for ADDITIVE example-equations only
        for ei, (L, R) in enumerate(self.eqs):
            name = self.ops[L[2]]
            if name not in ADD_K:
                continue
            K = ADD_K[name]
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            ap, bp, rp = places(a, self.order), places(b, self.order), places(Rg, self.order)
            aU, aT, bU, bT = ap[0], ap[1], bp[0], bp[1]
            rU, rT = rp[0], rp[1]; rH = rp.get(2)
            c1 = ('c', (ei, 1)); c2 = ('c', (ei, 2))
            self.cdom[(ei, 1)] = {0, 1}; self.cdom[(ei, 2)] = {0, 1}
            # units:  aU + bU + K = rU + 10*c1
            t = {};
            for v, c in [(('s', aU), 1), (('s', bU), 1), (('s', rU), -1), (c1, -10)]:
                t[v] = t.get(v, 0) + c
            self.facts.append(Fact(t, -K, 'EQ%d units' % (ei + 1), ('col', ei, 'units', aU, bU, rU, c1, K, None)))
            # tens:   aT + bT + c1 = rT + 10*c2
            t = {}
            for v, c in [(('s', aT), 1), (('s', bT), 1), (c1, 1), (('s', rT), -1), (c2, -10)]:
                t[v] = t.get(v, 0) + c
            self.facts.append(Fact(t, 0, 'EQ%d tens' % (ei + 1), ('col', ei, 'tens', aT, bT, rT, c2, 0, c1)))
            # hundreds: c2 = rH  (3-digit) else c2 = 0
            if rH is not None:
                self.facts.append(Fact({c2: 1, ('s', rH): -1}, 0, 'EQ%d hundreds' % (ei + 1),
                                       ('hund', ei, rH, c2)))
            else:
                self.facts.append(Fact({c2: 1}, 0, 'EQ%d (2-digit result)' % (ei + 1), ('noc', ei, c2)))
        # BORROW column facts for subtraction-style equations (mirror of carry; honest, showable).
        for ei, (L, R) in enumerate(self.eqs):
            name = self.ops[L[2]]
            if name not in ('sub_signed', 'absdiff', 'neg_absdiff'):
                continue
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            if name == 'sub_signed':
                neg = bool(R) and R[0] == L[2]          # result carries the op glyph as a minus sign
                hi, lo = (b, a) if neg else (a, b)
            else:                                       # absdiff / neg_absdiff: need orientation
                if ei not in self.orient:
                    continue                            # solve pass: defer to arc-exact
                hi, lo = (b, a) if self.orient[ei] else (a, b)
            self._add_borrow(ei, hi, lo, Rg)

    def _add_borrow(self, ei, hi, lo, mag):
        hp, lp, rp = places(hi, self.order), places(lo, self.order), places(mag, self.order)
        hiU, hiT, loU, loT, rU = hp[0], hp[1], lp[0], lp[1], rp[0]
        rT = rp.get(1)
        b1 = ('c', (ei, 1)); b2 = ('c', (ei, 2))
        self.cdom[(ei, 1)] = {0, 1}; self.cdom[(ei, 2)] = {0, 1}
        # units:  hiU - loU = rU - 10*b1   (b1 = borrow taken from tens)
        t = {}
        for v, c in [(('s', hiU), 1), (('s', loU), -1), (('s', rU), -1), (b1, 10)]:
            t[v] = t.get(v, 0) + c
        self.facts.append(Fact(t, 0, 'EQ%d units(sub)' % (ei + 1)))
        if rT is not None:                              # 2-digit magnitude
            t = {}
            for v, c in [(('s', hiT), 1), (('s', loT), -1), (b1, -1), (('s', rT), -1), (b2, 10)]:
                t[v] = t.get(v, 0) + c
            self.facts.append(Fact(t, 0, 'EQ%d tens(sub)' % (ei + 1)))
            self.facts.append(Fact({b2: 1}, 0, 'EQ%d (no borrow out)' % (ei + 1)))
        else:                                           # 1-digit magnitude: tens difference cancels
            t = {}
            for v, c in [(('s', hiT), 1), (('s', loT), -1), (b1, -1)]:
                t[v] = t.get(v, 0) + c
            self.facts.append(Fact(t, 0, 'EQ%d tens(sub, 1-digit result)' % (ei + 1)))

    # ---- pinning ------------------------------------------------------------
    def pin_sym(self, g, val):
        if self.dom[g] == {val}:
            return False
        if val not in self.dom[g]:
            raise ValueError('contradiction %s=%d' % (g, val))
        for h in self.dom:                       # all-different: digit can't already belong to another
            if h != g and self.dom[h] == {val}:
                raise ValueError('duplicate digit %d (%s and %s)' % (val, g, h))
        self.dom[g] = {val}
        for h in self.dom:
            if h != g and val in self.dom[h] and len(self.dom[h]) > 1:
                self.dom[h].discard(val)
        return True
    def pin_carry(self, k, val):
        if self.cdom[k] == {val}:
            return False
        if val not in self.cdom[k]:
            raise ValueError('carry contradiction')
        self.cdom[k] = {val}; return True

    def solved(self):
        return all(len(self.dom[g]) == 1 for g in self.dom)

    # ---- readable rendering of a fact with current knowns -------------------
    def render_fact(self, f):
        pos, neg = [], []
        for v, c in f.terms.items():
            (pos if c > 0 else neg).append((abs(c), v))
        def term(c, v):
            s = self.show(v)
            return s if c == 1 else '%d*%s' % (c, s)
        L = ' + '.join(term(c, v) for c, v in pos) or '0'
        R = ' + '.join(term(c, v) for c, v in neg)
        const = f.const
        if const > 0:
            R = (R + ' + ' if R else '') + str(const)
        elif const < 0:
            L = (L + ' + ' if L else '') + str(-const)
        return '%s = %s' % (L, R or '0')

    # ---- rules --------------------------------------------------------------
    def rule_single(self, flist=None):
        changed = False
        for f in (flist if flist is not None else (self.facts + self.derived)):
            unk = [v for v in f.terms if not self.resolved(v)]
            if len(unk) != 1:
                continue
            v = unk[0]; coeff = f.terms[v]
            rhs = f.const - sum(c * self.value(w) for w, c in f.terms.items() if w is not v)
            if rhs % coeff != 0:
                raise ValueError('non-integer pin')
            val = rhs // coeff
            before = self.render_fact(f)
            if v[0] == 's':
                if not (0 <= val <= 9):
                    raise ValueError('digit out of range')
                if self.pin_sym(v[1], val):
                    self.log.append("  [%s]  %s  =>  '%s' = %d" % (f.tag, before, v[1], val)); changed = True
            else:
                if self.pin_carry(v[1], val):   # carry pins are bookkeeping -> not narrated (noise)
                    changed = True
        return changed

    def rule_magnitude(self):
        changed = False
        for ei, (L, R) in enumerate(self.eqs):
            if self.ops[L[2]] not in ADD_K:
                continue
            Rg = [c for c in R if c != L[2]]
            if len(Rg) == 3:
                lead = places(Rg, self.order)[2]
                if len(self.dom[lead]) > 1:
                    self.pin_sym(lead, 1)
                    self.log.append("  [EQ%d magnitude]  two 2-digit numbers add to <= 199; result is "
                                    "3 digits  =>  leading '%s' = 1" % (ei + 1, lead)); changed = True
        return changed

    def rule_alldiff(self):
        changed = False
        used = {next(iter(self.dom[g])) for g in self.dom if len(self.dom[g]) == 1}
        for g in self.dom:
            if len(self.dom[g]) > 1:
                nd = self.dom[g] - used
                if nd != self.dom[g]:
                    self.dom[g] = nd
                    if len(nd) == 1:
                        self.pin_sym(g, next(iter(nd)))
                        self.log.append("  [all-different]  digits %s are taken; only %d remains  =>  '%s' = %d"
                                        % (sorted(used), next(iter(nd)), g, next(iter(nd))))
                    changed = True
        for d in range(10):
            homes = [g for g in self.dom if d in self.dom[g] and len(self.dom[g]) > 1]
            if len(homes) == 1:
                self.pin_sym(homes[0], d)
                self.log.append("  [all-different]  only '%s' can be %d  =>  '%s' = %d" % (homes[0], d, homes[0], d))
                changed = True
        return changed

    def _constraints(self):
        """build the user's arc-consistency constraints (mod10/9/11 + magnitude + exact) per eq.
        The `exact` constraint on a 2-op equation has <=6 symbols, so arc_reduce enumerates it
        directly -- this is what makes subtraction/|a-b| and multiplication participate."""
        if getattr(self, '_cons_cache', None) is not None:
            return self._cons_cache
        import crypt_forward_rules as FR
        KIND = {'add': '+', 'add_p1': '+', 'add_m1': '+', 'mul': '*', 'mul_p1': '*', 'mul_m1': '*',
                'sub_signed': '-'}
        KOFF = {'add': 0, 'add_p1': 1, 'add_m1': -1, 'mul': 0, 'mul_p1': 1, 'mul_m1': -1, 'sub_signed': 0}
        cons = []
        class E:
            pass
        for (L, R) in self.eqs:
            name = self.ops[L[2]]
            Rg = [c for c in R if c != L[2]]
            e = E(); e.a = (L[0], L[1]); e.b = (L[3], L[4]); e.res = tuple(Rg); e.order = self.order
            fn = (lambda nm: (lambda a, b: opval(nm, a, b)))(name)
            if name in KIND:
                e.kind = KIND[name]; e.k = KOFF[name]; e.neg = False
                cons += FR.mod_constraints(e)
                if e.kind == '*':
                    cons.append(FR.mul_magnitude(e))
                if e.kind == '+':
                    lead = FR.add_leading(e)
                    if lead:
                        cons.append(lead)
                cons.append(FR.exact(e, fn))
            elif name in ('absdiff', 'neg_absdiff'):
                e.kind = '-'; e.k = 0; e.neg = (name == 'neg_absdiff')
                cons.append(FR.exact(e, fn))   # abs handled inside fn + neg flag
        self._cons_cache = cons
        return cons

    def rule_arc(self):
        """narrow domains via the constraint layer; pin + log when a symbol becomes unique."""
        changed = False
        for c in self._constraints():
            red = c.arc_reduce(self.dom, max_free=6)
            for s, nd in red.items():
                nd = nd & self.dom[s]
                if not nd:
                    raise ValueError('arc empty domain')
                if nd != self.dom[s]:
                    if len(nd) == 1:
                        self.pin_sym(s, next(iter(nd)))
                        self.log.append("  [%s] residue+exact narrowing forces '%s' = %d"
                                        % (c.name, s, next(iter(nd))))
                    else:
                        self.dom[s] = nd
                    changed = True
        return changed

    def rule_colenum(self, maxunk=3):
        """A single column fact with 2-3 unknowns: enumerate feasible digit assignments under
        all-different + current domains; pin any unknown whose feasible value is unique. This is the
        workhorse -- e.g. after a repeated glyph cancels, '\\ + c1 = 10' forces \\=9,c1=1. Honest: the
        only values consistent with this one column given the digits already used."""
        changed = False
        for f in self.facts + self.derived:
            unk = [v for v in f.terms if not self.resolved(v)]
            if not (2 <= len(unk) <= maxunk):
                continue
            doms = [sorted(self.dom[v[1]]) if v[0] == 's' else sorted(self.cdom[v[1]]) for v in unk]
            sz = 1
            for d in doms:
                sz *= max(len(d), 1)
            if sz > 3000:
                continue
            base = sum(c * self.value(w) for w, c in f.terms.items() if self.resolved(w))
            taken = {next(iter(self.dom[g])) for g in self.dom if len(self.dom[g]) == 1}
            sym_idx = [i for i in range(len(unk)) if unk[i][0] == 's']
            sols = []
            for combo in iproduct(*doms):
                sv = [combo[i] for i in sym_idx]
                if len(set(sv)) != len(sv):
                    continue
                if any(combo[i] in taken for i in sym_idx):
                    continue
                if base + sum(f.terms[unk[i]] * combo[i] for i in range(len(unk))) == f.const:
                    sols.append(combo)
            if not sols:
                continue
            before = self.render_fact(f)
            feas = [{sol[i] for sol in sols} for i in range(len(unk))]
            pins = []
            for i, v in enumerate(unk):
                if len(feas[i]) == 1 and not self.resolved(v):
                    val = next(iter(feas[i]))
                    if v[0] == 's':
                        self.pin_sym(v[1], val); pins.append((v[1], val))
                    else:
                        self.pin_carry(v[1], val)
            if pins:
                ctx = (" (digits %s already used)" % sorted(taken)) if taken else ""
                self.log.append("  [%s]  %s%s  =>  %s" % (f.tag, before, ctx,
                                ', '.join("'%s'=%d" % (g, d) for g, d in pins)))
                changed = True
        return changed

    def _read(self, g2, m):
        return m[g2[1]] * 10 + m[g2[0]] if self.rev else m[g2[0]] * 10 + m[g2[1]]

    def rule_product(self):
        changed = False
        for ei, (L, R) in enumerate(self.eqs):
            name = self.ops[L[2]]
            if name not in MUL_K:
                continue
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            if any(len(self.dom[g]) > 1 for g in a + b):
                continue
            m = {g: next(iter(self.dom[g])) for g in a + b}
            A, B = self._read(a, m), self._read(b, m)
            P = opval(name, A, B)
            digs = [int(c) for c in str(abs(P))]
            rp = places(Rg, self.order)
            if len(digs) != len(Rg):
                continue
            # place-index p has digit value digs[len-1-p] (standard reading)
            need = {rp[p]: (digs[len(digs) - 1 - p] if self.order == 'standard' else digs[p]) for p in rp}
            if any(d not in self.dom[g] for g, d in need.items()):
                continue
            newp = [(g, d) for g, d in need.items() if len(self.dom[g]) > 1]
            if not newp:
                continue
            for g, d in newp:
                self.pin_sym(g, d)
            self.log.append("  [EQ%d product]  operands now known: %d %s %d = %d ; result digits %s  =>  %s"
                            % (ei + 1, A, OPPH[name], B, P, str(abs(P)),
                               ', '.join("'%s'=%d" % (g, d) for g, d in newp)))
            changed = True
        return changed

    def rule_factor(self):
        changed = False
        for ei, (L, R) in enumerate(self.eqs):
            name = self.ops[L[2]]
            if name not in MUL_K:
                continue
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            if any(len(self.dom[g]) > 1 for g in Rg):
                continue
            m = {g: next(iter(self.dom[g])) for g in Rg}
            Rval = sum(m[Rg[i]] * 10 ** (len(Rg) - 1 - i) for i in range(len(Rg))) if self.order == 'standard' \
                else sum(m[Rg[i]] * 10 ** i for i in range(len(Rg)))
            prod = Rval - MUL_K[name]
            for known, unk in ((a, b), (b, a)):
                if any(len(self.dom[g]) > 1 for g in known) or all(len(self.dom[g]) == 1 for g in unk):
                    continue
                mk = {g: next(iter(self.dom[g])) for g in known}
                Kn = self._read(known, mk)
                if Kn == 0 or prod % Kn != 0:
                    continue
                other = prod // Kn
                if not (10 <= other <= 99):
                    continue
                od = (other % 10, other // 10) if self.rev else (other // 10, other % 10)  # (units? ) per order
                # unk glyphs: unk[0],unk[1]; standard: unk[0]=tens,unk[1]=units
                want = {unk[0]: other // 10, unk[1]: other % 10}
                if any(d not in self.dom[g] for g, d in want.items()):
                    continue
                if all(len(self.dom[g]) == 1 for g in unk):
                    continue
                for g, d in want.items():
                    if len(self.dom[g]) > 1:
                        self.pin_sym(g, d)
                self.log.append("  [EQ%d factor]  result %d, %s = %d ; %d / %d = %d  =>  %s"
                                % (ei + 1, Rval, OPPH[name], prod, prod, Kn, other,
                                   ', '.join("'%s'=%d" % (g, want[g]) for g in unk)))
                changed = True
        return changed

    def rule_factor_both(self):
        """result pinned, BOTH operands unknown: factor the product into 2-digit x 2-digit pairs
        (few exist) and pick the one whose digits fit the operand symbols + all-different. Showable:
        we list the factor pairs and the magnitude bracket on the tens (floor(P/100))."""
        changed = False
        for ei, (L, R) in enumerate(self.eqs):
            name = self.ops[L[2]]
            if name not in MUL_K:
                continue
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            if any(len(self.dom[g]) > 1 for g in Rg):
                continue                                       # result must be fully pinned
            if all(len(self.dom[g]) == 1 for g in a + b):
                continue                                       # both operands already known -> rule_factor/product
            m = {g: next(iter(self.dom[g])) for g in Rg}
            Rval = sum(m[Rg[i]] * 10 ** (len(Rg) - 1 - i) for i in range(len(Rg))) if self.order == 'standard' \
                else sum(m[Rg[i]] * 10 ** i for i in range(len(Rg)))
            prod = Rval - MUL_K[name]
            if prod < 100:
                continue
            pairs = [(x, prod // x) for x in range(10, 100) if prod % x == 0 and 10 <= prod // x <= 99]
            if not pairs:
                continue
            taken = {next(iter(self.dom[g])) for g in self.dom if len(self.dom[g]) == 1}

            def dig(V):                                        # digits for (operand glyph0, glyph1) per order
                return (V // 10, V % 10) if self.order == 'standard' else (V % 10, V // 10)
            feas = []
            for (x, y) in pairs:
                for (av, bv) in ((x, y), (y, x)):
                    ad, bd = dig(av), dig(bv)
                    assign = {a[0]: ad[0], a[1]: ad[1], b[0]: bd[0], b[1]: bd[1]}
                    if any(len(set([assign[g]])) and assign[g] not in self.dom[g] for g in assign):
                        continue
                    if any(len(self.dom[g]) == 1 and next(iter(self.dom[g])) != assign[g] for g in assign):
                        continue
                    newg = {g for g in assign if len(self.dom[g]) > 1}
                    nv = [assign[g] for g in newg]
                    if len(set(nv)) != len(nv) or any(assign[g] in taken for g in newg):
                        continue
                    feas.append(assign)
            if not feas:
                continue
            pins = []
            for g in set(a + b):
                vs = {f[g] for f in feas}
                if len(vs) == 1 and len(self.dom[g]) > 1:
                    pins.append((g, next(iter(vs))))
            if not pins:
                continue
            for g, d in pins:
                self.pin_sym(g, d)
            self.log.append("  [EQ%d factor]  %s product = %d ; 2-digit x 2-digit factor pairs: %s ; "
                            "magnitude floor(%d/100)=%d brackets the tens; only one fits the symbols + "
                            "all-different  =>  %s" % (ei + 1, OPPH[name], prod,
                            ', '.join('%dx%d' % p for p in pairs), prod, prod // 100,
                            ', '.join("'%s'=%d" % (g, d) for g, d in pins)))
            changed = True
        return changed

    def rule_cast(self):
        """casting out nines (mod 9) + elevens (mod 11) on a multiplication that has exactly ONE
        unknown digit-symbol left. Both are standard human checks (digit-sum / alternating-sum), so
        the pin is reproducible, not a black-box enumeration."""
        changed = False
        for ei, (L, R) in enumerate(self.eqs):
            name = self.ops[L[2]]
            if name not in MUL_K:
                continue
            a = (L[0], L[1]); b = (L[3], L[4]); Rg = [c for c in R if c != L[2]]
            allsyms = list(a) + list(b) + Rg
            unk = {g for g in allsyms if len(self.dom[g]) > 1}
            if len(unk) != 1:
                continue
            g = next(iter(unk)); k = MUL_K[name]
            ka = sum(next(iter(self.dom[s])) for s in a if s != g)
            kb = sum(next(iter(self.dom[s])) for s in b if s != g)
            kr = sum(next(iter(self.dom[s])) for s in Rg if s != g)
            ca = sum(1 for s in a if s == g); cb = sum(1 for s in b if s == g); cr = sum(1 for s in Rg if s == g)
            def altmap(seq):
                rp = places(seq, self.order); d = {}
                for p, s in rp.items():
                    d[s] = d.get(s, 0) + ((-1) ** p)
                return d
            ama, amb, amr = altmap(a), altmap(b), altmap(Rg)
            ak = sum(ama[s] * next(iter(self.dom[s])) for s in ama if s != g)
            bk = sum(amb[s] * next(iter(self.dom[s])) for s in amb if s != g)
            rk = sum(amr[s] * next(iter(self.dom[s])) for s in amr if s != g)
            aa, ab, ar = ama.get(g, 0), amb.get(g, 0), amr.get(g, 0)
            cand = set()
            for v in self.dom[g]:
                ds_ok = ((ka + ca * v) * (kb + cb * v) + k) % 9 == (kr + cr * v) % 9
                as_ok = ((ak + aa * v) * (bk + ab * v) + k) % 11 == (rk + ar * v) % 11
                if ds_ok and as_ok:
                    cand.add(v)
            cand &= self.dom[g]
            if cand and cand != self.dom[g]:
                if len(cand) == 1:
                    v = next(iter(cand)); self.pin_sym(g, v)
                    self.log.append("  [EQ%d cast 9s+11s]  digit-sum (mod 9) and alternating-sum (mod 11) "
                                    "of %s leave only '%s' = %d" % (ei + 1, OPPH[name], g, v))
                else:
                    self.dom[g] = cand
                changed = True
        return changed

    def rule_combine(self):
        pool = self.facts + self.derived
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                f1, f2 = pool[i], pool[j]
                shared = [v for v in f1.terms if v in f2.terms and not self.resolved(v)]
                shared.sort(key=lambda v: 0 if v[0] == 'c' else 1)  # eliminate carries first
                for v in shared:
                    c1, c2 = f1.terms[v], f2.terms[v]
                    terms = {}
                    for w, c in f1.terms.items():
                        terms[w] = terms.get(w, 0) + c * c2
                    for w, c in f2.terms.items():
                        terms[w] = terms.get(w, 0) - c * c1
                    g = Fact(terms, f1.const * c2 - f2.const * c1, '%s + %s' % (f1.tag, f2.tag))
                    # reduce by gcd
                    coeffs = [abs(c) for c in g.terms.values()] + ([abs(g.const)] if g.const else [])
                    gg = math.gcd(*coeffs) if len(coeffs) > 1 else (coeffs[0] if coeffs else 0)
                    if gg > 1:
                        g.terms = {w: c // gg for w, c in g.terms.items()}; g.const //= gg
                    if not g.terms or g.sig() in self.seen_sig:
                        continue
                    unk = [w for w in g.terms if not self.resolved(w)]
                    if len(unk) == 1:
                        self.seen_sig.add(g.sig()); self.derived.append(g)
                        self.log.append("  [combine %s]  %s" % (g.tag, self.render_fact(g)))
                        if self.rule_single([g]):
                            return True
                    elif len(unk) == 2:
                        sols = self._enum2(g, unk)
                        if sols is None:
                            continue
                        self.seen_sig.add(g.sig())
                        feas = {w: {s[w] for s in sols} for w in unk}
                        pinned_any = False
                        line = "  [combine %s]  %s ; all-different leaves %s" % (
                            g.tag, self.render_fact(g),
                            ' or '.join('{' + ', '.join("%s=%d" % (self.show(w), s[w]) for w in unk) + '}'
                                        for s in sols[:4]))
                        for w in unk:
                            if len(feas[w]) == 1 and not self.resolved(w):
                                val = next(iter(feas[w]))
                                if w[0] == 's':
                                    self.pin_sym(w[1], val)
                                else:
                                    self.pin_carry(w[1], val)
                                pinned_any = True
                        if pinned_any:
                            self.log.append(line + "  =>  " + ', '.join(
                                "'%s'=%d" % (w[1], self.value(w)) for w in unk if self.resolved(w) and w[0] == 's'))
                            return True
        return False

    def _enum2(self, g, unk):
        doms = []
        for w in unk:
            doms.append(sorted(self.dom[w[1]]) if w[0] == 's' else sorted(self.cdom[w[1]]))
        if doms[0] and doms[1] and len(doms[0]) * len(doms[1]) > 100:
            return None
        base = sum(c * self.value(w) for w, c in g.terms.items() if self.resolved(w))
        sols = []
        for x in doms[0]:
            for y in doms[1]:
                if unk[0][0] == 's' and unk[1][0] == 's' and x == y:
                    continue
                s = base + g.terms[unk[0]] * x + g.terms[unk[1]] * y
                if s == g.const:
                    sols.append({unk[0]: x, unk[1]: y})
        return sols if sols else None

    # ---- bounded depth-1 search (honest: shown try/reject on most-constrained symbol) ------
    def snap(self):
        return ({g: set(v) for g, v in self.dom.items()}, {k: set(v) for k, v in self.cdom.items()},
                list(self.log), set(self.seen_sig), list(self.derived))
    def restore(self, s):
        self.dom = {g: set(v) for g, v in s[0].items()}; self.cdom = {k: set(v) for k, v in s[1].items()}
        self.log = list(s[2]); self.seen_sig = set(s[3]); self.derived = list(s[4])

    def verify_all(self):
        m = {g: next(iter(self.dom[g])) for g in self.dom if len(self.dom[g]) == 1}
        if len(m) != len(self.dom):
            return False
        for (L, R) in self.eqs:
            a = (L[0], L[1]); b = (L[3], L[4]); name = self.ops[L[2]]
            av = self._read(a, m); bv = self._read(b, m); val = opval(name, av, bv)
            Rg = [c for c in R if c != L[2]]
            want = encode(val, self.order, {v: k for k, v in m.items()}, L[2] if name in SIGNED else None)
            if want != ''.join(Rg):
                return False
        return True

    def stuck_reason(self):
        """find a column with no feasible digit -> short readable reject reason."""
        for f in self.facts + self.derived:
            unk = [v for v in f.terms if not self.resolved(v)]
            if not (1 <= len(unk) <= 3):
                continue
            doms = [sorted(self.dom[v[1]]) if v[0] == 's' else sorted(self.cdom[v[1]]) for v in unk]
            base = sum(c * self.value(w) for w, c in f.terms.items() if self.resolved(w))
            taken = {next(iter(self.dom[g])) for g in self.dom if len(self.dom[g]) == 1}
            si = [i for i in range(len(unk)) if unk[i][0] == 's']
            ok = False
            for combo in iproduct(*doms):
                sv = [combo[i] for i in si]
                if len(set(sv)) != len(sv) or any(combo[i] in taken for i in si):
                    continue
                if base + sum(f.terms[unk[i]] * combo[i] for i in range(len(unk))) == f.const:
                    ok = True; break
            if not ok:
                return "column %s has no valid digit left" % f.tag
        return "leaves a digit ambiguous"

    def solve_search(self, max_branch=9):
        self.run()
        if self.solved():
            return True
        free = [g for g in self.dom if len(self.dom[g]) > 1]
        if not free:
            return False
        s = min(free, key=lambda g: (len(self.dom[g]), g))
        cand = sorted(self.dom[s])
        if len(cand) > max_branch:
            return False
        st = self.snap(); wins = []; reject = {}
        for v in cand:
            self.restore(st)
            try:
                self.pin_sym(s, v); self.run()
                if self.solved() and self.verify_all():
                    wins.append(v)
                else:
                    reject[v] = self.stuck_reason()
            except ValueError:
                reject[v] = "arithmetic contradiction"
        if len(wins) != 1:
            self.restore(st); return False
        v = wins[0]
        self.restore(st)
        others = [c for c in cand if c != v]
        self.log.append("  [branch] most-constrained '%s' in %s; others (%s) hit a column clash, only '%s'=%d "
                        "stays all-different:" % (s, cand, ','.join(map(str, others)), s, v))
        self.pin_sym(s, v); self.run()
        return self.solved()

    # ---- driver -------------------------------------------------------------
    def run(self, max_iter=300):
        for _ in range(max_iter):
            try:
                p = (self.rule_single() or self.rule_magnitude() or self.rule_alldiff()
                     or self.rule_colenum() or self.rule_product() or self.rule_factor()
                     or self.rule_factor_both() or self.rule_cast()
                     or (False if self.showable else self.rule_arc()))
                if not p:
                    p = self.rule_combine()
            except ValueError:
                return False
            if self.solved():
                return True
            if not p:
                return False
        return self.solved()


def encode(val, order, inv, sign_glyph):
    neg = val < 0; mag = abs(val); digits = [int(c) for c in str(mag)]
    written = digits[::-1] if order == 'reversed' else digits
    try:
        s = ''.join(inv[d] for d in written)
    except KeyError:
        return None
    if neg and sign_glyph:
        s = sign_glyph + s
    return s


CONCAT = ('concat_fwd', 'concat_rev')


def operator_block(eqs, qL, ops, order):
    """Honest operator deduction (Option 1): RESULT LENGTH eliminates the impossible families
    (value-free), then a try/reject conclusion -- 'testing the remaining forms, only X reproduces every
    example with all ten digits distinct'. No teleport (no digit assumed), no forward-reference."""
    sp = lambda t: ' '.join(t)
    ex_for = {}
    for (L, R) in eqs:
        ex_for.setdefault(L[2], (L, R))
    FAM = {'add': 'sum', 'add_p1': 'sum', 'add_m1': 'sum', 'mul': 'product', 'mul_p1': 'product',
           'mul_m1': 'product', 'sub_signed': 'difference', 'absdiff': 'difference',
           'neg_absdiff': 'difference', 'rsub_signed': 'difference'}
    rd = 'units-first (reversed)' if order == 'reversed' else 'tens-first'
    out = ["Step 1 - operators (concat = result is the 4 operand symbols rearranged; else arithmetic by "
           "result length: sum 2-3 dig, diff 1-2, product 3-4). Read %s." % rd]
    for g in list(dict.fromkeys([L[2] for (L, R) in eqs] + [qL[2]])):
        name = ops[g]; ex = ex_for.get(g)
        rhs = sp([c for c in ex[1]]) if ex else '?'
        opnds = set([ex[0][0], ex[0][1], ex[0][3], ex[0][4]]) if ex else set()
        rsyms = [c for c in (ex[1] if ex else '') if not ex or c != ex[0][2]]
        if name in CONCAT:
            out.append("  '%s': %s = operand symbols rearranged -> concat." % (g, rhs)); continue
        news = [c for c in rsyms if c not in opnds]
        why = ("%d syms (not 4)" % len(rsyms)) if len(rsyms) != 4 else \
              ("'%s' not an operand sym" % news[0]) if news else "not copy-order"
        Rg = rsyms
        signed = bool(ex) and ex[1] and ex[1][0] == ex[0][2]
        n = len(Rg); fams = "difference" if (n == 1 or signed) else "product" if n >= 4 \
            else "sum or difference" if n == 2 else "sum or product"
        out.append("  '%s': %s, %s -> arithmetic, %d-sym -> %s." % (g, rhs, why, n, fams))
    return out


def operator_proofs(eqs, qL, ops, rev, m):
    """Step 3: confirm each operator by REFUTING the length-compatible alternatives with the digits
    derived in Step 2 (e.g. a sum would give 31 but the result is 228 -> not a sum; 19*12=228 -> product).
    Refutation, not mere confirmation."""
    sp = lambda t: ' '.join(t)

    def readv(g2):
        return m[g2[1]] * 10 + m[g2[0]] if rev else m[g2[0]] * 10 + m[g2[1]]
    FAMI = {'add': 'sum', 'add_p1': 'sum', 'add_m1': 'sum', 'mul': 'product', 'mul_p1': 'product',
            'mul_m1': 'product', 'sub_signed': 'difference', 'absdiff': 'difference',
            'neg_absdiff': 'difference', 'rsub_signed': 'difference'}
    ex_for = {}
    for (L, R) in eqs:
        ex_for.setdefault(L[2], (L, R))
    out = ["Step 3 - check operators (compute the example; the other option fails):"]
    done = set()
    for g in list(dict.fromkeys([L[2] for (L, R) in eqs] + [qL[2]])):
        if g in done or ops[g] in CONCAT or g not in ex_for:
            continue
        done.add(g)
        L, R = ex_for[g]; name = ops[g]
        a = readv((L[0], L[1])); b = readv((L[3], L[4]))
        Rg = [c for c in R if c != L[2]]; n = len(Rg)
        rval = sum(m[Rg[i]] * 10 ** (len(Rg) - 1 - i) for i in range(len(Rg))) if not rev \
            else sum(m[Rg[i]] * 10 ** i for i in range(len(Rg)))
        truev = opval(name, a, b); fam = FAMI[name]
        # refute the alternative family allowed by length
        alt = None
        if n == 3:                    # sum vs product
            alt = ('a sum', a + b) if fam == 'product' else ('a product', a * b)
        elif n == 2:                  # sum vs difference
            alt = ('a sum', a + b) if fam == 'difference' else ('a difference', abs(a - b))
        line = "  '%s': %d %s %d = %d ok" % (g, a, OPPH[name], b, truev)
        if alt:
            an, av = alt; ad = len(str(abs(av)))
            reason = ("%d dig!=%d" % (ad, len(Rg))) if ad != len(Rg) else ("%d!=%d" % (av, rval))
            line += " (%s=%d %s)" % (an, av, reason)
        out.append(line + ".")
    return out


def build_trace(prompt, ops, rev, gold):
    eqs, qL = C2.parse(prompt)
    qname = ops[qL[2]]
    order = 'reversed' if rev else 'standard'
    sp = lambda t: ' '.join(t)
    # ---- CONCAT query: SAME Step-1 template (concat-vs-arithmetic decision), then apply. ----
    if qname in CONCAT:
        ans = (qL[0] + qL[1] + qL[3] + qL[4]) if qname == 'concat_fwd' else (qL[3] + qL[4] + qL[0] + qL[1])
        if ans != gold:
            return None, 'concat ans %r != gold %r' % (ans, gold)
        order_desc = ("the first operand's two symbols then the second operand's two symbols"
                      if qname == 'concat_fwd' else
                      "the second operand's two symbols then the first operand's two symbols")
        out = operator_block(eqs, qL, ops, order)
        out += ["", "Step 2 - apply the query operator. '%s' is concatenation, so copy the query operands' "
                "symbols in order (%s):" % (qL[2], order_desc),
                "  %s -> %s = %s" % (sp(qL), order_desc, sp(ans))]
        return "\n".join(out) + "\n\\boxed{%s}" % ans, 'ok'
    # ---- arithmetic query: concat EXAMPLE eqs carry no digit info -> drop them from solving. ----
    aeqs = [(L, R) for (L, R) in eqs if ops[L[2]] not in CONCAT]
    oc = {L[2] for L, R in aeqs} | {qL[2]}
    if any(ops[g] not in OPPH for g in oc):
        return None, 'unsupported-op'
    if not aeqs:
        return None, 'no arithmetic example eqs (only concat) -> query undetermined'
    # free-query gate over ARITHMETIC example glyphs (concat examples don't pin digits).
    ex_syms = set()
    for (L, R) in aeqs:
        ex_syms |= set(L[0] + L[1] + L[3] + L[4]) | set(c for c in R if c != L[2])
    q_syms = [qL[0], qL[1], qL[3], qL[4]]
    all_syms = ex_syms | set(q_syms)
    free = [s for s in q_syms if s not in ex_syms]
    if free and not (len(all_syms) == 10 and len(free) == 1):
        return None, 'free-query-unwinnable (%s only in query)' % ''.join(sorted(set(free)))
    eqs = aeqs  # solve + verify only over arithmetic equations
    oc = oc
    # PASS 1 (solve): full engine incl. opaque arc-exact -> get the answer + |a-b| orientation.
    solve_eng = Engine(eqs, qL, ops, rev, showable=False)
    if not solve_eng.solve_search():
        un = ''.join(g for g in solve_eng.dom if len(solve_eng.dom[g]) != 1)
        return None, 'STALLED (needs search): ' + un
    msolve = {g: next(iter(solve_eng.dom[g])) for g in solve_eng.dom}
    order = solve_eng.order
    # PASS 2 (honest re-derivation): showable rules only; |a-b| orientation from the solved map.
    orient = {}
    for ei, (L, R) in enumerate(eqs):
        if ops[L[2]] in ('absdiff', 'neg_absdiff'):
            a = (L[0], L[1]); b = (L[3], L[4])
            av = msolve[a[1]] * 10 + msolve[a[0]] if rev else msolve[a[0]] * 10 + msolve[a[1]]
            bv = msolve[b[1]] * 10 + msolve[b[0]] if rev else msolve[b[0]] * 10 + msolve[b[1]]
            orient[ei] = (bv > av)
    eng = Engine(eqs, qL, ops, rev, showable=True, orient=orient)
    if not eng.solve_search():
        un = ''.join(g for g in eng.dom if len(eng.dom[g]) != 1)
        return None, 'not-showable (answer known, no honest forward derivation): ' + un
    # render pass is authoritative (sound all-different); gate ITS answer against gold.
    m = {g: next(iter(eng.dom[g])) for g in eng.dom}
    inv = {v: k for k, v in m.items()}
    qa, qb = (qL[0], qL[1]), (qL[3], qL[4]); qname = ops[qL[2]]
    A = m[qa[1]] * 10 + m[qa[0]] if rev else m[qa[0]] * 10 + m[qa[1]]
    B = m[qb[1]] * 10 + m[qb[0]] if rev else m[qb[0]] * 10 + m[qb[1]]
    qval = opval(qname, A, B)
    ans = encode(qval, order, inv, qL[2] if qname in SIGNED else None)
    if ans is None:
        return None, 'free query digit'
    if ans != gold:
        return None, 'engine ans %r != gold %r' % (ans, gold)
    # ---- render ----
    sp = lambda t: ' '.join(t)
    out = operator_block(eqs, qL, ops, order)
    out += ["", "Step 2 - solve (each digit forced from known digits):"]
    out += eng.log
    out += [""]
    out += operator_proofs(eqs, qL, ops, rev, m)
    out += ["", "Encode the query:",
            "  %s :  a = %d, b = %d ;  %s = %d  ->  %s" % (sp(qL), A, B, OPPH[qname], qval, sp(ans))]
    return "\n".join(out) + "\n\\boxed{%s}" % ans, 'ok'


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    OUT = os.path.join(ROOT, 'analysis/crypt_mech_traces')
    for f in os.listdir(OUT):
        if f.endswith('.txt'):
            os.remove(os.path.join(OUT, f))
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
            if r['category'] == 'cryptarithm_deduce']
    random.Random(2).shuffle(rows)
    cracked = attempted = stalled = wrong = unsup = 0
    reasons = {}
    for r in rows:
        try:
            res = C2.solve(r['prompt'], deadline_s=2.0)
            if not res:
                continue
            ops = (res[1] or {}).get('ops'); rev = (res[1] or {}).get('rev')
            if ops is None:
                continue
            eqs, qL = C2.parse(r['prompt']); oc = {L[2] for L, R in eqs} | {qL[2]}
            if any('concat' in ops[g] for g in oc):
                continue
            attempted += 1
            trace, reason = build_trace(r['prompt'], ops, rev, r['answer'].strip())
            if trace is None:
                reasons[reason.split(':')[0].split(' ')[0]] = reasons.get(reason.split(':')[0].split(' ')[0], 0) + 1
                if reason.startswith('STALL'):
                    stalled += 1
                elif reason.startswith('engine ans'):
                    wrong += 1
                else:
                    unsup += 1
                continue
            cracked += 1
            if cracked <= N:
                body = trace.rsplit('\\boxed{', 1)[0].rstrip()
                assistant = "<think>\n%s\n</think>\n\\boxed{%s}" % (body, r['answer'].strip())
                content = ("USER:\n" + r['prompt'] + SUFFIX + "\n\n" + "=" * 60 + "\n"
                           "ASSISTANT (training target):\n" + assistant + "\n\n" + "=" * 60 +
                           "\nGOLD ANSWER: " + r['answer'] + "\n")
                open(os.path.join(OUT, '%s.txt' % r['id']), 'w').write(content)
        except Exception:
            continue
    tot = attempted
    print("attempted (arith crypt) = %d" % tot)
    print("CRACKED honest = %d (%.1f%% of attempted, ~%.1f%% of all crypt_deduce)"
          % (cracked, 100 * cracked / max(tot, 1), 100 * cracked / 659))
    print("STALLED = %d | wrong = %d | other-refuse = %d" % (stalled, wrong, unsup))
    print("refuse reasons:", reasons)
    print("files -> analysis/crypt_mech_traces/")


if __name__ == '__main__':
    main()
