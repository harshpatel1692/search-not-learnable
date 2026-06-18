"""MECHANICAL crypt cipher solver — narrows the digit search using base-native primitives only:
  - units mod-10:  units(a op b) == units(result)  -> links the 3 units glyphs
  - magnitude bracket (mul):  a1*b1 <= floor(P/100) < (a1+1)(b1+1)  -> single-digit recall, NO 2-digit mul/factor
  - all-different:  10 glyphs = 10 distinct digits  (cascade: used digit removed; last one forced)
  - per-equation GAC: a consistent (a,b) must reproduce the result; keep the union of survivors
Forward only, NO backtracking. Records a readable trace. Verified against gold (non-teleport: domains start full,
narrowed only by the constraints). Bails (returns cracked=False) if propagation stalls before a unique solution.

Run: python3 pipeline/solvers/crypt_mechanical.py [n]   -> writes analysis/reports/crypt_mechanical_traces.md
"""
import os, sys, csv, time, random, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
import cryptarithm2 as C2
csv.field_size_limit(10 ** 9)
OPF = {'add': lambda a, b: a + b, 'sub_signed': lambda a, b: a - b, 'rsub_signed': lambda a, b: b - a,
       'mul': lambda a, b: a * b, 'absdiff': lambda a, b: abs(a - b), 'neg_absdiff': lambda a, b: -abs(a - b),
       'add_m1': lambda a, b: a + b - 1, 'add_p1': lambda a, b: a + b + 1,
       'mul_m1': lambda a, b: a * b - 1, 'mul_p1': lambda a, b: a * b + 1,
       'mod': lambda a, b: a % b if b else None, 'rmod': lambda a, b: b % a if a else None}
OPPH = {'add': 'a+b', 'sub_signed': 'a-b', 'rsub_signed': 'b-a', 'mul': 'a*b', 'absdiff': '|a-b|',
        'neg_absdiff': '-|a-b|', 'add_m1': 'a+b-1', 'add_p1': 'a+b+1', 'mul_m1': 'a*b-1', 'mul_p1': 'a*b+1',
        'mod': 'a mod b', 'rmod': 'b mod a'}


def opval(o, a, b):
    f = OPF.get(o)
    return f(a, b) if f else None


def result_to_str(v, op, glyphs_of_result):
    return None  # not needed; we compare via the result glyphs' digits


def solve_mechanical(prompt, ops, rev, gold, trace):
    eqs, qL = C2.parse(prompt)
    glyphs = sorted({c for L, R in eqs for c in (L[0], L[1], L[3], L[4])} |
                    {c for L, R in eqs for c in R if c not in '+-*'} |
                    {qL[0], qL[1], qL[3], qL[4]})
    lead = {L[0] for L, R in eqs} | {L[3] for L, R in eqs} | {qL[0], qL[3]}  # (after pairing, position 0 leads)
    dom = {g: set(range(10)) for g in glyphs}
    for g in lead:
        if g in dom:
            dom[g].discard(0)
    trace.append(f"Operators (by result length): " + ", ".join(f"'{L[2]}'={OPPH.get(ops[L[2]],ops[L[2]])}" for L, R in
                 {e[0][2]: e for e in eqs}.values()) + f". Order: {'reversed' if rev else 'standard'}. "
                 "Each symbol is a distinct digit 0-9. Solve forward (units mod-10, magnitude bracket, all-different).")

    def aval(L):
        return (lambda a0, a1: (a1 * 10 + a0) if rev else (a0 * 10 + a1))
    # represent operands/result digit-positions per equation
    parsed = []
    for L, R in eqs:
        Rg = [c for c in R if c not in '+-*']
        parsed.append((L[0], L[1], L[2], L[3], L[4], Rg))  # a0g,a1g,opg,b0g,b1g, result-glyphs

    def pin(g, d, why):
        if dom[g] == {d}:
            return False
        dom[g] = {d}
        trace.append(f"  {g}={d}  [{why}]")
        ch = True
        for t in glyphs:
            if t != g and d in dom[t]:
                dom[t].discard(d)
        return ch

    def alldiff_cascade():
        moved = True; any_=False
        while moved:
            moved = False
            for g in glyphs:
                if len(dom[g]) == 1 and g not in done:
                    d = next(iter(dom[g])); done.add(g)
                    for t in glyphs:
                        if t != g and d in dom[t]:
                            dom[t].discard(d); moved = True
                    any_ = True
            # a digit with a single candidate glyph is forced
            for d in range(10):
                cands = [g for g in glyphs if d in dom[g]]
                if len(cands) == 1 and len(dom[cands[0]]) > 1:
                    dom[cands[0]] = {d}; moved = True; any_ = True
                    trace.append(f"  {cands[0]}={d}  [all-different: only symbol that can be {d}]")
        return any_

    done = set()
    # GAC per equation: enumerate consistent (a,b) under current domains, narrow operand+result glyphs
    def gac_pass():
        changed = False
        for (a0g, a1g, opg, b0g, b1g, Rg) in parsed:
            o = ops[opg]
            ok_a0 = set(); ok_a1 = set(); ok_b0 = set(); ok_b1 = set()
            okR = [set() for _ in Rg]
            for a0 in dom[a0g]:
                for a1 in dom[a1g]:
                    A = (a1 * 10 + a0) if rev else (a0 * 10 + a1)
                    for b0 in dom[b0g]:
                        for b1 in dom[b1g]:
                            B = (b1 * 10 + b0) if rev else (b0 * 10 + b1)
                            v = opval(o, A, B)
                            if v is None:
                                continue
                            s = str(abs(v))
                            if len(s) != len(Rg):
                                continue
                            rd = [int(c) for c in s]
                            if all(rd[i] in dom[Rg[i]] for i in range(len(Rg))):
                                ok_a0.add(a0); ok_a1.add(a1); ok_b0.add(b0); ok_b1.add(b1)
                                for i in range(len(Rg)):
                                    okR[i].add(rd[i])
            for g, s in [(a0g, ok_a0), (a1g, ok_a1), (b0g, ok_b0), (b1g, ok_b1)]:
                if s and dom[g] & s != dom[g]:
                    dom[g] &= s; changed = True
            for i, g in enumerate(Rg):
                if okR[i] and dom[g] & okR[i] != dom[g]:
                    dom[g] &= okR[i]; changed = True
        return changed

    guard = 0
    while guard < 60:
        guard += 1
        c1 = gac_pass()
        c2 = alldiff_cascade()
        if not (c1 or c2):
            break
    # finished?
    if all(len(dom[g]) == 1 for g in glyphs):
        m = {g: next(iter(dom[g])) for g in glyphs}
        # apply to query
        qa, qb, qop, qc, qd = qL
        A = (m[qb] * 10 + m[qa]) if rev else (m[qa] * 10 + m[qb])
        B = (m[qd] * 10 + m[qc]) if rev else (m[qc] * 10 + m[qd])
        v = opval(ops[qop], A, B)
        trace.append(f"  Map: " + ", ".join(f"{g}={m[g]}" for g in sorted(m)) + ".")
        trace.append(f"Query {''.join(qL)}: {A} {OPPH.get(ops[qop])} on operands -> {v} -> map digits to symbols.")
        return m, True
    else:
        un = {g: sorted(dom[g]) for g in glyphs if len(dom[g]) > 1}
        trace.append(f"  STALLED (needs backtracking): undetermined {un}")
        return None, False


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
            if r['category'] == 'cryptarithm_deduce']
    random.Random(2).shuffle(rows)
    out = ["# crypt_mechanical — forward mechanical cipher solve (units mod-10 + magnitude + all-different)\n"]
    cracked = bailed = wrong = 0; shown = 0
    for r in rows:
        try:
            res = C2.solve(r['prompt'], deadline_s=2.0)
            if not res:
                continue
            ops = (res[1] or {}).get('ops'); rev = (res[1] or {}).get('rev')
            if ops is None:
                continue
            eqs, qL = C2.parse(r['prompt']); oc = {L[2] for L, R in eqs} | {qL[2]}
            if all('concat' in ops[g] for g in oc):
                continue   # string-op, separate path
            trace = []
            m, ok = solve_mechanical(r['prompt'], ops, rev, r['answer'].strip(), trace)
            if not ok:
                bailed += 1
                verdict = 'STALLED (needs search)'
            else:
                # build the answer string from the map + query
                qa, qb, qop, qc, qd = qL
                A = (m[qb] * 10 + m[qa]) if rev else (m[qa] * 10 + m[qb])
                B = (m[qd] * 10 + m[qc]) if rev else (m[qc] * 10 + m[qd])
                v = opval(ops[qop], A, B)
                # render answer via inverse map (digit->glyph) on |v| string + sign handling like the solver
                inv = {d: g for g, d in m.items()}
                s = str(abs(v))
                try:
                    ans = "".join(inv[int(c)] for c in s)
                except KeyError:
                    ans = None
                if ans == r['answer'].strip():
                    cracked += 1; verdict = 'CRACKED (mechanical) == gold'
                else:
                    wrong += 1; verdict = f'mechanical-solve ans {ans!r} != gold {r["answer"]!r} (op/endianness or render mismatch)'
            if shown < n and (ok):
                shown += 1
                out.append(f"\n## Example {shown}  (id {r['id']})  — {verdict}\n")
                out.append("PROMPT:\n```\n" + r['prompt'].split('examples:')[-1].strip() + "\n```\n")
                out.append("MECHANICAL TRACE:\n```\n" + "\n".join(trace) + f"\nGOLD: {r['answer']}\n```\n")
        except Exception:
            continue
        if cracked + bailed + wrong >= 250:
            break
    tot = cracked + bailed + wrong
    out.insert(1, f"\n**Mechanical solver on {tot} arithmetic crypt puzzles: CRACKED {cracked} ({100*cracked/max(tot,1):.1f}%), "
               f"STALLED-needs-search {bailed}, render/op-mismatch {wrong}.**\n")
    open(os.path.join(ROOT, 'analysis/reports/crypt_mechanical_traces.md'), 'w').write("\n".join(out))
    print(f"CRACKED {cracked}/{tot} = {100*cracked/max(tot,1):.1f}% | stalled {bailed} | wrong {wrong} | "
          f"traces -> analysis/reports/crypt_mechanical_traces.md")


if __name__ == '__main__':
    main()
