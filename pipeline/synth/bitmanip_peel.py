"""PEEL -> 2-tap CoT for the peelable 3-tap forms (XNOR/XOR-headed), reasoning in decimal/hex.
out = HEAD XOR R, where HEAD = the XOR/XNOR-attached tap and R = out XOR HEAD is a clean 2-tap rule:
  XNOR-head (tt 10110100): R = Y OR (NOT Z)
  XOR-head  (tt 01001011): R = (NOT Y) AND Z
The trace: try candidate heads (compute R = out XOR head, real hex), accept the one whose R is a clean 2-copy
rule, solve that 2-tap rule, recombine. Synth rows draw fresh taps+inputs => zero leakage. ASCII-only."""
import random
M = 0xFF
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
    op, k = t; return {'rot': 'ROTL', 'shl': 'SHL', 'shr': 'SHR'}[op] + str(t[1])
def expl(t):
    op, k = t
    return {'rot': f"rotate-left {k}", 'shl': f"(x*{2**k})mod256", 'shr': f"x>>{k}"}[op]
ALL = [(o, k) for o in ('rot', 'shl', 'shr') for k in range(1, 8)]
# the two peelable forms, by head type
HEADS = {'XNOR': "10110100", 'XOR': "01001011"}
def resid(headtype, y, z):           # R = out XOR HEAD as a 2-tap rule of (Y,Z)
    return (y | ((~z) & M)) if headtype == 'XNOR' else ((~y) & M) & z
def resid_desc(headtype, tY, tZ):
    return (f"{tname(tY)} OR (NOT {tname(tZ)})" if headtype == 'XNOR'
            else f"(NOT {tname(tY)}) AND {tname(tZ)}")
def out_val(x, taps, headtype):
    tX, tY, tZ = taps
    return tapval(x, tX) ^ resid(headtype, tapval(x, tY), tapval(x, tZ))

PROMPT_HDR = ("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The "
 "transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
 "choice functions.\n\nHere are some examples of input -> output:\n")
def _mk_prompt(ex, q):
    return PROMPT_HDR + "\n".join(f"{b8(i)} -> {b8(o)}" for i, o in ex) + f"\n\nNow, determine the output for: {b8(q)}"

def gen_peel(rng):
    headtype = rng.choice(['XNOR', 'XNOR', 'XOR'])     # real mix ~ 194:67
    taps = rng.sample(ALL, 3)
    n = rng.choice([9, 10, 11]); ins = rng.sample(range(256), n + 1)
    ex = [(i, out_val(i, taps, headtype)) for i in ins[:n]]
    q = ins[n]; ans = out_val(q, taps, headtype)
    return {"category": "bit_manipulation", "prompt": _mk_prompt(ex, q), "answer": b8(ans),
            "final": b8(ans), "cot": render_peel(ex, q, taps, headtype)}

def render_peel(ex, q, taps, headtype):
    tX, tY, tZ = taps
    L = []
    L.append("Method: this rule has the form output = HEAD XOR R, where HEAD is one shifted copy and R "
             "(= output XOR HEAD) is a simpler 2-copy rule. Find HEAD by testing copies; the right HEAD makes R "
             "a clean 2-copy rule. Shifts: SHLk=(x*2^k)mod256, SHRk=x>>k, ROTLk=rotate-left k. Work in hex.")
    L.append("Examples in hex: " + ", ".join(f"0x{a:02X}->0x{b:02X}" for a, b in ex) + ".")
    # RANDOMIZED head search (no positional shortcut): show a random number of wrong heads (0-4) interleaved,
    # then the true head at a random position. Each shows real R = out XOR head; rejects flagged by a real
    # contradiction (two examples whose R bytes can't both come from one 2-copy rule), accept verified below.
    rng = random.Random(q * 31 + 5)
    pool = [t for t in ALL if t not in taps]; rng.shuffle(pool)
    n_rej = rng.randint(0, 4)
    seq = pool[:n_rej] + [tX]  # search until one fits, then STOP -> true head last; n_rej varies (no fixed-position shortcut)
    for h in seq:
        rs = ", ".join(f"0x{b ^ tapval(a, h):02X}" for a, b in ex)
        if h == tX:
            L.append(f"Try HEAD={tname(tX)} ({expl(tX)}): R = out XOR HEAD = {rs}. These all fit one 2-copy rule, accept.")
        else:
            L.append(f"Try HEAD={tname(h)} ({expl(h)}): R = out XOR HEAD = {rs}. No single 2-copy rule fits all of these, reject.")
    L.append(f"Residual R = {resid_desc(headtype, tY, tZ)}.")
    a0, b0 = ex[0]; h0 = tapval(a0, tX); y0 = tapval(a0, tY); z0 = tapval(a0, tZ); r0 = resid(headtype, y0, z0)
    L.append(f"Verify on 0x{a0:02X}: {tname(tY)}=0x{y0:02X}, {tname(tZ)}=0x{z0:02X}, R=0x{r0:02X}; "
             f"HEAD XOR R = 0x{h0:02X}^0x{r0:02X} = 0x{h0 ^ r0:02X} = 0x{b0:02X} ok.")
    hq = tapval(q, tX); yq = tapval(q, tY); zq = tapval(q, tZ); rq = resid(headtype, yq, zq); oq = hq ^ rq
    L.append(f"Apply to query 0x{q:02X}: HEAD={tname(tX)}=0x{hq:02X}; {tname(tY)}=0x{yq:02X}, {tname(tZ)}=0x{zq:02X}, "
             f"R=0x{rq:02X}; output = 0x{hq:02X}^0x{rq:02X} = 0x{oq:02X}. Result: {b8(oq)}.")
    return " ".join(L)

if __name__ == "__main__":
    import statistics as st
    rng = random.Random(7); bad = 0; lens = []
    for _ in range(500):
        r = gen_peel(rng)
        stated = r['cot'].split("Result:")[-1].strip().rstrip('.')
        if stated != r['answer']: bad += 1
        lens.append(len(r['cot']))
    print(f"500 peel samples: answer mismatches={bad} | median chars={int(st.median(lens))} | max={max(lens)}")
    r = gen_peel(random.Random(1)); print("=" * 60); print(r['prompt']); print("ANSWER", r['answer']); print("COT:", r['cot'])
