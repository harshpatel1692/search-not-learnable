"""Base probe: can the 30B solve the dominant 3-tap form XNOR(X, NOT_A_AND_B(Y,Z)) (0.56 in bit-star-r2)
via PEEL -> 2-tap, reasoning in decimal/hex? Peeling the XOR-attached head X leaves a clean 2-tap residual
R = out XOR X = (Y) OR (NOT Z). So: find head -> solve residual 2-tap -> recombine out = head XOR R.

Compares base accuracy WITH the peel method (instruction + 1 worked example) vs WITHOUT (baseline).
  python3 pipeline/probe_peel.py [n_test]
"""
import os, sys, json, re, random
sys.path.insert(0, os.path.dirname(__file__))
import nvidia_api as NV
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = 'nvidia/nemotron-3-nano-30b-a3b'
FORM = 'XNOR(X, NOT_A_AND_B(Y, Z))'

def bit(x, p): return (x >> (7 - p)) & 1
def b8(x): return format(x, '08b')
def src_pos(j, t):
    op, k = t
    if op == 'rot': return (j + k) % 8
    if op == 'shl':
        s = j + k; return s if s < 8 else None
    s = j - k; return s if s >= 0 else None
def tapval(x, t):
    v = 0
    for j in range(8):
        p = src_pos(j, t)
        if p is not None and bit(x, p): v |= 1 << (7 - j)
    return v
def tname(t):
    op, k = t; return {'rot': 'ROTL', 'shl': 'SHL', 'shr': 'SHR'}[op] + str(k)
def shift_expl(t):
    op, k = t
    if op == 'rot': return f"{tname(t)} = rotate-left {k} bits"
    if op == 'shl': return f"{tname(t)} = (x*{2**k}) mod 256"
    return f"{tname(t)} = x >> {k}"

def parse(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    m = re.search(r'output for:\s*([01]{8})', prompt)
    return [(int(a, 2), int(b, 2)) for a, b in pairs], int(m.group(1), 2)

def render_worked(prompt, td):
    """A correct, decimal/hex peel->2-tap worked trace (head known = the true X tap)."""
    ex, q = parse(prompt)
    tX, tY, tZ = tuple(td['{A}']), tuple(td['{B}']), tuple(td['{C}'])
    L = []
    L.append("Method: a 3-shift rule of this kind has the form  output = HEAD XOR rest, where HEAD is one "
             "shifted copy and 'rest' is a 2-copy rule. Find HEAD by testing copies: the right HEAD makes "
             "R = output XOR HEAD collapse to a clean 2-copy rule. Then solve that 2-copy rule and recombine.")
    L.append("Shifts (decimal): SHLk=(x*2^k)mod256, SHRk=x>>k, ROTLk=rotate-left k.")
    L.append("Examples in hex:")
    for a, b in ex: L.append(f"  0x{a:02X} -> 0x{b:02X}")
    # pick a wrong head to demonstrate the search fails, then the true head
    allt = [(o, k) for o in ('rot', 'shl', 'shr') for k in range(1, 8)]
    wrong = next(t for t in allt if t not in (tX, tY, tZ))
    L.append(f"Try HEAD={tname(wrong)} ({shift_expl(wrong)}). R = out XOR HEAD:")
    rs = []
    for a, b in ex[:4]:
        h = tapval(a, wrong); r = b ^ h; rs.append(f"0x{b:02X}^0x{h:02X}=0x{r:02X}")
    L.append("  " + ", ".join(rs) + " -- no single 2-copy rule fits these, reject.")
    L.append(f"Try HEAD={tname(tX)} ({shift_expl(tX)}). R = out XOR HEAD:")
    rs = []
    for a, b in ex[:4]:
        h = tapval(a, tX); r = b ^ h; rs.append(f"0x{b:02X}^0x{h:02X}=0x{r:02X}")
    L.append("  " + ", ".join(rs) + " -- these DO fit a 2-copy rule.")
    L.append(f"Solve the residual R: R = {tname(tY)} OR (NOT {tname(tZ)}).")
    a0, b0 = ex[0]
    h0 = tapval(a0, tX); y0 = tapval(a0, tY); z0 = tapval(a0, tZ); r0 = y0 | ((~z0) & 0xFF)
    L.append(f"  verify on 0x{a0:02X}: {tname(tY)}=0x{y0:02X}, {tname(tZ)}=0x{z0:02X}, "
             f"R={tname(tY)}|(NOT {tname(tZ)})=0x{r0:02X}; HEAD XOR R = 0x{h0:02X}^0x{r0:02X}=0x{h0 ^ r0:02X}=0x{b0:02X} ok.")
    hq = tapval(q, tX); yq = tapval(q, tY); zq = tapval(q, tZ); rq = yq | ((~zq) & 0xFF); outq = hq ^ rq
    L.append(f"Apply to query 0x{q:02X}: HEAD={tname(tX)}=0x{hq:02X}; "
             f"{tname(tY)}=0x{yq:02X},{tname(tZ)}=0x{zq:02X}, R=0x{rq:02X}; output=0x{hq:02X}^0x{rq:02X}=0x{outq:02X}={b8(outq)}.")
    L.append(f"\\boxed{{{b8(outq)}}}")
    return "\n".join(L)

BOX = "Put the final 8-bit output string inside \\boxed{}, e.g. \\boxed{01101100}."
METHOD = ("Solve the bit rule. It combines 3 shifted copies of the 8-bit input. Use the PEEL method: "
          "output = HEAD XOR (a 2-copy rule). Reason in decimal/hex. " + BOX)
PLAIN = ("Find the hidden bit-transformation rule from the examples and apply it to the query. " + BOX)

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    NV.set_experiment('peel_probe')
    solved = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(ROOT, 'pipeline/data/bitmanip_solved.jsonl'))}
    import csv; csv.field_size_limit(10 ** 9)
    tr = {r['id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
          if r['category'] == 'bit_manipulation'}
    def form(e):
        o = []
        for t in re.findall(r'\{[ABC]\}', e):
            if t not in o: o.append(t)
        g = e
        for i, t in enumerate(o): g = g.replace(t, 'XYZ'[i])
        return g
    rows = [i for i in solved if solved[i].get('correct') and i in tr and form(solved[i]['expr']) == FORM]
    rng = random.Random(3); rng.shuffle(rows)
    worked = render_worked(tr[rows[0]]['prompt'], solved[rows[0]]['td'])
    test = rows[1:1 + n]
    print(f"=== PEEL probe: {len(test)} held-out XNOR-form rows (base bit-star-r2 gets this form 0.56) ===")
    print("--- worked example given to the model ---\n" + worked + "\n" + "=" * 60)
    res = {'method': 0, 'plain': 0, 'm_trunc': 0, 'p_trunc': 0}
    for rid in test:
        prompt = tr[rid]['prompt']; gold = tr[rid]['answer'].strip()
        # WITH method + worked example
        pm = METHOD + "\n\nWorked example:\n" + worked + "\n\nNow solve:\n" + prompt
        om = NV.ask(pm, model=MODEL, max_tokens=6000, temperature=0.0, add_box=False, meta={'id': rid, 'arm': 'peel'})
        # baseline
        op = NV.ask(PLAIN + "\n\n" + prompt, model=MODEL, max_tokens=6000, temperature=0.0, add_box=False,
                    meta={'id': rid, 'arm': 'plain'})
        for tag, o in (('method', om), ('plain', op)):
            if not isinstance(o, dict) or 'error' in o: continue
            pred = (o.get('answer') or '').strip()
            res[tag] += (pred == gold)
            res[tag[0] + '_trunc'] += (o.get('finish') == 'length')
        print(f"  {rid}: gold {gold} | peel {(om.get('answer') if isinstance(om,dict) else 'ERR')} | "
              f"plain {(op.get('answer') if isinstance(op,dict) else 'ERR')}", flush=True)
    print(f"\nRESULT on {len(test)} XNOR-form rows:")
    print(f"  PEEL method:  {res['method']}/{len(test)} correct  (truncated {res['m_trunc']})")
    print(f"  PLAIN base:   {res['plain']}/{len(test)} correct  (truncated {res['p_trunc']})")

if __name__ == '__main__':
    main()
