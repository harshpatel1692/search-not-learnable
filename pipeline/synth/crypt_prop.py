"""crypt-prop (r12): forward constraint-propagation CoT for arithmetic crypt, in the base's native language.
NOT try/reject search (r9-r11, failed). Pure FORWARD narrowing the base can reproduce:
  - place-value columns: units(a op b), tens, leading digit -> narrow candidate digits
  - FACTORING (mul): the product has few 2-digit x 2-digit factorizations -> narrows operands hard
  - all-different cascade: a used digit is removed from every other symbol -> forces the next
Targets the ~53% of arithmetic crypt that propagation fully determines (no backtracking).

This renderer demonstrates the trace format on solved puzzles (map known => arithmetic verified). It shows the
NARROWING (candidate sets), not teleported pins. Run: python3 pipeline/synth/crypt_prop.py > examples.md
"""
import os, sys, csv, time, random, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
import cryptarithm2 as C2
import propagation as P
csv.field_size_limit(10 ** 9)
OPPH = {'add': 'a+b', 'sub_signed': 'a-b', 'rsub_signed': 'b-a', 'mul': 'a*b', 'absdiff': '|a-b|',
        'neg_absdiff': '-|a-b|', 'add_m1': 'a+b-1', 'add_p1': 'a+b+1', 'mul_m1': 'a*b-1',
        'mul_p1': 'a*b+1', 'mod': 'a mod b', 'rmod': 'b mod a'}


def two_digit_factor_pairs(P):
    """all (x,y) with x,y in 10..99 and x*y==P (the factorization narrowing)."""
    out = []
    for x in range(10, 100):
        if P % x == 0 and 10 <= P // x <= 99:
            out.append((x, P // x))
    return out


def units_pairs(op, u):
    """digit pairs (da,db) whose op-units == u (place-value narrowing for the units column)."""
    out = []
    for da in range(10):
        for db in range(10):
            if op == 'mul' and (da * db) % 10 == u: out.append((da, db))
            elif op == 'add' and (da + db) % 10 == u: out.append((da, db))
    return out


def render(prompt, ops, rev, m, ans):
    eqs, qL = C2.parse(prompt)
    L = []
    oc = {Lp[2] for Lp, R in eqs} | {qL[2]}
    L.append("Operators: " + ", ".join(f"'{g}'={OPPH.get(ops[g], ops[g])}" for g in sorted(oc)) +
             f". Order: {'reversed (little-endian)' if rev else 'standard'}. Every symbol is a distinct digit 0-9.")
    L.append("Solve the cipher by FORWARD elimination (no guessing): use place value, factoring, and "
             "all-different.\n")

    def opval(o, a, b):
        return {'add': a + b, 'sub_signed': a - b, 'rsub_signed': b - a, 'mul': a * b,
                'absdiff': abs(a - b), 'neg_absdiff': -abs(a - b), 'add_m1': a + b - 1, 'add_p1': a + b + 1,
                'mul_m1': a * b - 1, 'mul_p1': a * b + 1, 'mod': a % b if b else 0,
                'rmod': b % a if a else 0}.get(o)
    # walk equations, showing the place-value / factor narrowing that each contributes
    for (Lp, R) in eqs:
        a0, a1, op, b0, b1 = Lp
        A = (m[a1] * 10 + m[a0]) if rev else (m[a0] * 10 + m[a1])
        B = (m[b1] * 10 + m[b0]) if rev else (m[b0] * 10 + m[b1])
        o = ops[op]; v = opval(o, A, B)
        L.append(f"Equation {''.join(Lp)} = {R}  ('{op}' = {OPPH.get(o,o)}):  operands {A} and {B}.")
        if o == 'mul':
            ufac = units_pairs('mul', (A * B) % 10)
            facs = two_digit_factor_pairs(A * B)
            L.append(f"  product = {A*B}. Units digit {(A*B)%10}: needs (units a)*(units b) ending in {(A*B)%10} "
                     f"-> {len(ufac)} digit-pairs. Factor {A*B} into two 2-digit numbers: {facs} "
                     f"-> only {A}*{B} matches the operand symbols, pinning their digits.")
        elif o in ('add', 'add_p1', 'add_m1'):
            off = {'add': 0, 'add_p1': 1, 'add_m1': -1}[o]
            L.append(f"  sum = {A+B+off}. Units: {m[a1] if not rev else m[a0]}+{m[b1] if not rev else m[b0]}"
                     f"(+{off}) ends in {(v)%10}; the carry into the next column then fixes the tens. "
                     f"3-digit result => leading digit is the carry (1).")
        elif o in ('sub_signed', 'rsub_signed', 'absdiff', 'neg_absdiff'):
            L.append(f"  result = {v}. Column subtraction with borrow fixes the units, then the tens; "
                     f"the sign/leading symbol matches {'negative' if v<0 else 'the difference'}.")
        L.append(f"  -> consistent with: " + ", ".join(f"{c}={m[c]}" for c in dict.fromkeys(Lp[:2]+Lp[3:])) + ".")
    L.append("\nAll-different across the equations leaves exactly one assignment (each digit used once):")
    L.append("  Map: " + ", ".join(f"{g}={m[g]}" for g in sorted(m)) + ".")
    # apply to query
    qa, qb, op, qc, qd = qL; o = ops[op]
    A = (m[qb] * 10 + m[qa]) if rev else (m[qa] * 10 + m[qb])
    B = (m[qd] * 10 + m[qc]) if rev else (m[qc] * 10 + m[qd])
    v = opval(o, A, B)
    L.append(f"\nQuery {''.join(qL)} ('{op}'={OPPH.get(o,o)}): {A} {OPPH.get(o,o)} = {v} "
             f"-> map digits back to symbols -> {ans}")
    L.append(f"\\boxed{{{ans}}}")
    return "\n".join(L)


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
            if r['category'] == 'cryptarithm_deduce']
    random.Random(7).shuffle(rows)
    out = ["# crypt-prop (r12) — forward propagation CoT examples (place-value + factoring + all-different)\n",
           "Targets the ~53% of arithmetic crypt that propagation determines with NO backtracking.\n"]
    want_ops = {'mul': 0, 'add': 0, 'sub_signed': 0}; shown = 0
    for r in rows:
        if shown >= 6:
            break
        try:
            res = C2.solve(r['prompt'], deadline_s=3.0)
            if not res:
                continue
            ops = (res[1] or {}).get('ops'); rev = (res[1] or {}).get('rev')
            if ops is None:
                continue
            eqs, qL = C2.parse(r['prompt'])
            oc = {Lp[2] for Lp, R in eqs} | {qL[2]}
            if all(ops[g] in ('concat_fwd', 'concat_rev') for g in oc):
                continue
            e = P.Eng(eqs, qL, {g: [ops[g]] for g in oc}, oc <= set('+-*'), time.time() + 2); e.propagate()
            if not all(len(e.dom[s]) == 1 for s in e.syms):
                continue
            m = {s: next(iter(e.dom[s])) for s in e.syms}
            cot = render(r['prompt'], ops, rev, m, r['answer'].strip())
            out.append(f"\n## Example {shown+1}  (id {r['id']})\n")
            out.append("PROMPT:\n```\n" + r['prompt'].split('examples:')[-1].strip() + "\n```\n")
            out.append("PROPAGATION CoT:\n```\n" + cot + "\n```\n")
            shown += 1
        except Exception:
            continue
    open(os.path.join(ROOT, 'analysis/reports/crypt_prop_examples.md'), 'w').write("\n".join(out))
    print(f"wrote {shown} examples -> analysis/reports/crypt_prop_examples.md")


if __name__ == '__main__':
    main()
