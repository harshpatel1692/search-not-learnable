#!/usr/bin/env python3
"""
crypt_forward_rules.py  —  rules to ADD to the forward engine (user-provided, 2026-06-14).

The new power (all forward, no branching, table-level arithmetic):
  * three MODULI per equation: mod 10 (units / times-table), mod 9 (cast nines),
    mod 11 (cast elevens). These linearize multiplication into residue constraints,
    so a mul equation can finally participate in propagation instead of being an
    opaque "factor the 4-digit number" wall.
  * MAGNITUDE bracket on the tens-product for multiplication (top-down squeeze).
  * EXACT arithmetic constraint (ground truth, final pruning).
  * CARRY (add) / BORROW (sub) column facts -> a linear subsystem with combine.
  * GLOBAL: digit-sum = 45 (all 10 symbols) + all-different.
  * FREE-QUERY-DIGIT detection (unwinnable -> drop, never search).
"""
from itertools import product

# ============================================================
# reading-order-aware accessors
# ============================================================
def operand_roles(sym2, order):
    """(tens_sym, units_sym) for a 2-char operand."""
    return (sym2[0], sym2[1]) if order == "standard" else (sym2[1], sym2[0])

def result_units_first(res_syms, order):
    """result MAGNITUDE symbols ordered least-significant-digit first."""
    return list(res_syms[::-1]) if order == "standard" else list(res_syms)

def value_of(res_syms, order, m):
    return sum(m[s] * (10 ** i) for i, s in enumerate(result_units_first(res_syms, order)))

def digitsum(res_syms, m):                       # order-independent
    return sum(m[s] for s in res_syms)

def altsum(res_syms, order, m):                  # units-first alternating sum (== value mod 11)
    return sum(((-1) ** i) * m[s] for i, s in enumerate(result_units_first(res_syms, order)))


# ============================================================
# the THREE moduli   (kind = '+' | '*' | '-' ; op = kind plus offset k, with R = raw + k)
# ============================================================
def lhs_mod(kind, aT, aU, bT, bU, mod):
    if mod == 9:    A, B = aU + aT, bU + bT       # n ≡ digitsum (mod 9)
    elif mod == 11: A, B = aU - aT, bU - bT       # n ≡ altsum   (mod 11)
    else:           A, B = aU, bU                 # mod 10: units only
    v = A + B if kind == '+' else A * B if kind == '*' else A - B
    return v % mod

def rhs_mod(res_syms, order, k, mod, m):
    if mod == 9:    Rm = digitsum(res_syms, m) % 9
    elif mod == 11: Rm = altsum(res_syms, order, m) % 11
    else:           Rm = m[result_units_first(res_syms, order)[0]] % 10
    return (Rm - k) % mod                         # raw = R - k


# ============================================================
# generic Constraint: narrows domains by enumerating ONLY the free scope vars
# ============================================================
class Constraint:
    def __init__(self, scope, test, name):
        self.scope = list(dict.fromkeys(scope))
        self.test = test
        self.name = name

    def arc_reduce(self, dom, max_free=6):
        free = [s for s in self.scope if len(dom[s]) > 1]
        if not free or len(free) > max_free:
            return {}
        fixed = {s: next(iter(dom[s])) for s in self.scope if len(dom[s]) == 1}
        feas = {s: set() for s in free}
        for combo in product(*[sorted(dom[s]) for s in free]):
            a = dict(fixed); a.update(dict(zip(free, combo)))
            vals = list(a.values())
            if len(set(vals)) != len(vals):        # local all-different
                continue
            if self.test(a):
                for s in free:
                    feas[s].add(a[s])
        return {s: (feas[s] & dom[s]) for s in free if feas[s] != dom[s]}


# ============================================================
# constraint builders per equation
# ============================================================
def mod_constraints(eq):
    aT, aU = operand_roles(eq.a, eq.order)
    bT, bU = operand_roles(eq.b, eq.order)
    cs = []
    for mod in (10, 9, 11):
        def test(m, mod=mod):
            return lhs_mod(eq.kind, m[aT], m[aU], m[bT], m[bU], mod) == \
                   rhs_mod(eq.res, eq.order, eq.k, mod, m)
        scope = [aT, aU, bT, bU] + list(eq.res)
        cs.append(Constraint(scope, test, f"{eq.kind}.mod{mod}"))
    return cs

def mul_magnitude(eq):
    aT, _ = operand_roles(eq.a, eq.order)
    bT, _ = operand_roles(eq.b, eq.order)
    def test(m):
        raw = value_of(eq.res, eq.order, m) - eq.k          # = a*b
        return 100 * m[aT] * m[bT] <= raw < 100 * (m[aT] + 1) * (m[bT] + 1)
    return Constraint([aT, bT] + list(eq.res), test, "mul.magnitude")

def add_leading(eq):
    if len(eq.res) < 3:
        return None
    lead = result_units_first(eq.res, eq.order)[-1]
    def test(m):
        return m[lead] == 1
    return Constraint([lead], test, "add.magnitude(lead=1)")

def exact(eq, fn):
    aT, aU = operand_roles(eq.a, eq.order)
    bT, bU = operand_roles(eq.b, eq.order)
    def test(m):
        a, b = 10 * m[aT] + m[aU], 10 * m[bT] + m[bU]
        target = -value_of(eq.res, eq.order, m) if eq.neg else value_of(eq.res, eq.order, m)
        return fn(a, b) == target
    return Constraint([aT, aU, bT, bU] + list(eq.res), test, "exact")

def build_constraints(eq, fn):
    cs = list(mod_constraints(eq))
    if eq.kind == '*':
        cs.append(mul_magnitude(eq))
    if eq.kind == '+':
        lead = add_leading(eq)
        if lead: cs.append(lead)
    cs.append(exact(eq, fn))
    return cs


def alldiff_singles(dom, syms, log):
    changed = False
    placed = {next(iter(dom[s])) for s in syms if len(dom[s]) == 1}
    for s in syms:
        if len(dom[s]) > 1 and (dom[s] - placed) != dom[s]:
            dom[s] -= placed; changed = True
    for v in range(10):
        homes = [s for s in syms if v in dom[s]]
        if len(homes) == 1 and len(dom[homes[0]]) > 1:
            dom[homes[0]] = {v}; changed = True
            log.append(f"[all-different] digit {v} has only one home  =>  {homes[0]} = {v}")
    return changed


def detect_free_query_digit(query_syms, example_syms, n_symbols):
    free = [s for s in query_syms if s not in example_syms]
    if not free:
        return None
    if n_symbols == 10 and len(free) == 1:
        return None
    return free


def solve(constraints, syms, dom, log):
    while True:
        progress = False
        for c in constraints:
            red = c.arc_reduce(dom)
            for s, nd in red.items():
                if nd != dom[s]:
                    dom[s] = nd; progress = True
                    if len(nd) == 1:
                        log.append(f"[{c.name}]  =>  {s} = {next(iter(nd))}")
        progress = alldiff_singles(dom, syms, log) or progress
        if not progress:
            break
    return all(len(dom[s]) == 1 for s in syms)


if __name__ == "__main__":
    class EQ: pass
    eq = EQ(); eq.a, eq.b, eq.res = "'{", "?>", "%\\`?"
    eq.kind, eq.k, eq.neg, eq.order = '*', -1, False, "standard"
    m = {"'": 5, "{": 2, "?": 9, ">": 0, "%": 4, "\\": 6, "`": 7}
    aT, aU = operand_roles(eq.a, eq.order); bT, bU = operand_roles(eq.b, eq.order)
    a, b = 10*m[aT]+m[aU], 10*m[bT]+m[bU]
    print(f"a={a} b={b}  a*b={a*b}  result value={value_of(eq.res, eq.order, m)}  raw={value_of(eq.res, eq.order, m)-eq.k}")
    for mod in (10, 9, 11):
        L = lhs_mod(eq.kind, m[aT], m[aU], m[bT], m[bU], mod)
        R = rhs_mod(eq.res, eq.order, eq.k, mod, m)
        print(f"  mod{mod:>2}:  lhs={L}  rhs={R}  {'OK' if L == R else 'MISMATCH'}")
    dom = {s: {m[s]} for s in m}
    dom["?"] = set(range(1, 10))
    print("\nbefore:  '?' (b tens) in", sorted(dom["?"]))
    for c in build_constraints(eq, lambda a, b: a*b - 1):
        red = c.arc_reduce(dom)
        if "?" in red:
            dom["?"] &= red["?"]
            print(f"  [{c.name}] narrows '?' -> {sorted(dom['?'])}")
    print("after:   '?' (b tens) in", sorted(dom["?"]), "(gold = 9)")
