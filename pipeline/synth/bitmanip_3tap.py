# DEAD END (2026-06-14): hand-written 3-tap grammar -> bit 0.062 (REGRESSION).
# Teleports the tap-search (model hallucinates rejects + locks wrong rule) and grammar-mixing
# misroutes the tap-count. 3-tap was NEVER 0.07 (that was a confound) -- it's 0.503 in bit-star-r2.
# Bit lever is STaR self-harvest (run_bit_star_r3.sh), NOT a hand grammar. See LOG.md 2026-06-14.
"""GENUINE 3-input (3-tap) bit_manipulation CoT — the tail the existing perbit renderer SKIPS.

bit_global.py (98.88%) models bit_manip as: build up to 3 transformed copies (taps = ROTL/SHL/SHR k) of the
8-bit input, then combine them with ONE 3-input Boolean gate applied bitwise. The 3-tap subset (34% of bit) uses a
CLOSED set of exactly 11 distinct 3-input gates (verified in bitmanip_solved.jsonl). The model already solves 1-2
tap ~100% but fails 3-tap at ~0.07 because it has never seen a genuine 3-input trace.

This renderer teaches the FAITHFUL bit_global procedure, made learnable by DERIVING the gate (read its 3-input
truth table off the example columns) instead of searching 256 gates:
  1. it's a 3-input rule (2-input gates can't fit bit 0) -> witnessed rejects
  2. find the 3 source shifts (taps) -> guided witnessed search (test triples for a consistent gate)
  3. read the 3-input truth table abc->out off the columns (uniform; no per-form enumeration)
  4. verify on a column
  5. apply to the query: build the 3 taps, look up each column -> output, box

ASCII-only (ops as words). Whole values shown MSB-first. Synth rows draw fresh taps+inputs => zero train leakage."""
import random

M = 0xFF
def b8(x): return format(x, '08b')
def bit(x, p): return (x >> (7 - p)) & 1          # MSB-first position p

# ---- the 11 real 3-input gates: truth table over abc=000,001,...,111, with real-frequency weights ----
GATES = [
    ("10110100", 194), ("01001011", 67), ("00000111", 61), ("10110110", 53),
    ("11010111", 50), ("00110101", 48), ("00010111", 48), ("01001111", 11),
    ("00010110", 5), ("01000101", 3), ("10110000", 2),
]
def tt_out(tt, a, b, c): return int(tt[(a << 2) | (b << 1) | c])

# ---- taps: a transformed copy of the input ----
def src_pos(j, t):
    """input position feeding output position j for tap t=(op,k); None if a shift pushes a 0 in at the edge."""
    op, k = t
    if op == 'rot': return (j + k) % 8
    if op == 'shl':
        s = j + k; return s if s < 8 else None
    s = j - k; return s if s >= 0 else None            # shr
def tname(t):
    op, k = t; return {'rot': 'ROTL', 'shl': 'SHL', 'shr': 'SHR'}[op] + str(k)
def tapval(x, t):
    """the whole 8-bit transformed copy as an int."""
    v = 0
    for j in range(8):
        p = src_pos(j, t)
        if p is not None and bit(x, p): v |= 1 << (7 - j)
    return v
ALL_TAPS = [(op, k) for op in ('rot', 'shl', 'shr') for k in range(1, 8)]   # 21 taps (rot0 identity dropped)

def apply_rule(x, taps, tt):
    tA, tB, tC = taps
    out = 0
    for j in range(8):
        a = bit(tapval(x, tA), j); b = bit(tapval(x, tB), j); c = bit(tapval(x, tC), j)
        if tt_out(tt, a, b, c): out |= 1 << (7 - j)
    return out

# ---- 2-input gates, used only to WITNESS that bit 0 is not a 2-input rule ----
G2 = {'XOR': lambda a, b: a ^ b, 'OR': lambda a, b: a | b, 'AND': lambda a, b: a & b,
      'XNOR': lambda a, b: 1 - (a ^ b)}

PROMPT_HDR = ("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The "
 "transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
 "choice functions.\n\nHere are some examples of input -> output:\n")
def _mk_prompt(ex, q):
    return PROMPT_HDR + "\n".join(f"{b8(i)} -> {b8(o)}" for i, o in ex) + f"\n\nNow, determine the output for: {b8(q)}"

def _wchoice(rng, wpairs):
    tot = sum(w for _, w in wpairs); r = rng.uniform(0, tot); a = 0
    for v, w in wpairs:
        a += w
        if r <= a: return v
    return wpairs[-1][0]


def gen3tap(rng):
    """Fresh genuine-3-input row: pick one of the 11 gates (real weights), 3 distinct taps, random inputs."""
    tt = _wchoice(rng, GATES)
    taps = rng.sample(ALL_TAPS, 3)
    n = rng.choice([8, 9, 10, 11]); ins = rng.sample(range(256), n + 1)
    ex = [(i, apply_rule(i, taps, tt)) for i in ins[:n]]
    q = ins[n]; ans = apply_rule(q, taps, tt)
    return {"category": "bit_manipulation", "prompt": _mk_prompt(ex, q), "answer": b8(ans),
            "final": b8(ans), "cot": render(ex, q, ans, taps, tt)}


def _colstr(c): return "".join(map(str, c))

def _two_input_rejects(rng, ex, col0):
    """Show that no 1-or-2-input gate reproduces output column 0 -> it must be a 3-input rule. A few witnessed misses."""
    L = []
    tried = 0
    cands = [(op, pa, pb) for op in ('XOR', 'OR', 'AND', 'XNOR') for pa in range(8) for pb in range(pa, 8)]
    rng.shuffle(cands)
    for op, pa, pb in cands:
        c = [G2[op](bit(i, pa), bit(i, pb)) for i, _ in ex]
        if c != col0:
            L.append(f"  {op}(pos{pa},pos{pb}) -> {_colstr(c)} != {_colstr(col0)}, no.")
            tried += 1
        if tried >= 3: break
    return L

def _tap_search(rng, ex, taps, tt):
    """Witnessed, guided search for the 3 source shifts: test candidate triples, reject the ones with no consistent
    3-input gate, accept the triple that does. Keeps it short (a few near-miss rejects + the hit)."""
    L = []
    tA, tB, tC = taps
    # build a couple of plausible near-miss triples (perturb one tap) that FAIL the consistency test
    def consistent(tr):
        """does some single 3-input gate fit every example column under taps tr?"""
        table = {}
        for i, o in ex:
            va, vb, vc = tapval(i, tr[0]), tapval(i, tr[1]), tapval(i, tr[2])
            for j in range(8):
                key = (bit(va, j), bit(vb, j), bit(vc, j)); ob = bit(o, j)
                if key in table and table[key] != ob: return False
                table[key] = ob
        return True
    rejects = []
    pool = [t for t in ALL_TAPS if t not in taps]
    rng.shuffle(pool)
    for repl_idx in (0, 1, 2):
        for nt in pool:
            tr = list(taps); tr[repl_idx] = nt
            if nt in tr[:repl_idx] + tr[repl_idx + 1:]: continue
            if not consistent(tr):
                rejects.append(tuple(tr)); break
        if len(rejects) >= 2: break
    for tr in rejects:
        L.append(f"  taps {tname(tr[0])},{tname(tr[1])},{tname(tr[2])} -> no single gate fits all columns, no.")
    L.append(f"  taps {tname(tA)},{tname(tB)},{tname(tC)} -> one gate fits every column, MATCH.")
    return L


def render(ex, q, ans, taps, tt):
    tA, tB, tC = taps
    L = []
    L.append("Read each 8-bit value MSB-first, positions 0..7 left-to-right. By the hint this rule builds a few "
             "shifted/rotated copies of the input and combines them with one Boolean gate applied to each column. "
             "Each output bit is one gate of input bits at fixed source positions that shift +1 per output bit.")
    col0 = [bit(o, 0) for _, o in ex]
    # 1) rule out 1/2-input
    L.append(f"Bit 0 column = {_colstr(col0)}. First check 1- and 2-input gates -- none fit:")
    L += _two_input_rejects(random.Random(q * 131 + 7), ex, col0)
    L.append("So bit 0 needs THREE source bits: this is a 3-input rule over three shifted copies (taps).")
    # 2) find the 3 taps
    L.append("Find the three source shifts (each ROTL/SHL/SHR by 1..7; shifts read 0 past the edge). Test triples; "
             "keep the one where a single 3-input gate is consistent across all columns:")
    L += _tap_search(random.Random(q * 977 + 3), ex, taps, tt)
    L.append(f"Taps: A={tname(tA)}, B={tname(tB)}, C={tname(tC)}.")
    # 3) read the 3-input truth table off the columns
    L.append("Read the gate's truth table abc->out from the example columns (a,b,c = bits of copies A,B,C):")
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                rows.append(f"{a}{b}{c}->{tt_out(tt, a, b, c)}")
    L.append("  " + ", ".join(rows) + ".")
    # 4) verify on one example column
    i0, o0 = ex[0]
    vA, vB, vC = tapval(i0, tA), tapval(i0, tB), tapval(i0, tC)
    pred0 = "".join(str(tt_out(tt, bit(vA, j), bit(vB, j), bit(vC, j))) for j in range(8))
    L.append(f"Verify on {b8(i0)}: A={b8(vA)}, B={b8(vB)}, C={b8(vC)}; apply the table per column -> {pred0} = {b8(o0)} ok.")
    # 5) apply to the query, column by column
    qA, qB, qC = tapval(q, tA), tapval(q, tB), tapval(q, tC)
    steps = []
    qb = []
    for j in range(8):
        a, b, c = bit(qA, j), bit(qB, j), bit(qC, j); r = tt_out(tt, a, b, c)
        steps.append(f"b{j}:{a}{b}{c}->{r}"); qb.append(str(r))
    L.append(f"Apply to query {b8(q)}: A={b8(qA)}, B={b8(qB)}, C={b8(qC)}. " + "; ".join(steps) +
             f". Result: {''.join(qb)}.")
    return " ".join(L)


if __name__ == "__main__":
    import statistics as st
    rng = random.Random(7)
    bad = 0; lens = []
    for _ in range(500):
        r = gen3tap(rng)
        stated = r['cot'].split("Result:")[-1].strip().rstrip('.')
        if stated != r['answer']: bad += 1
        lens.append(len(r['cot']))
    print(f"500 samples: CoT-answer mismatches={bad} | median chars={int(st.median(lens))} | max={max(lens)}")
    rng = random.Random(1)
    for _ in range(2):
        r = gen3tap(rng)
        print("=" * 70); print("PROMPT:\n" + r['prompt']); print("ANSWER:", r['answer']); print("\nCOT:", r['cot'])
