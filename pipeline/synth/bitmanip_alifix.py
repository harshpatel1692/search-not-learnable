"""bitmanip_alifix: render corrected bit_manipulation CoT in Ali's grammar (v16).

GOAL
  Ali (LB 0.86 reference adapter) solves bit_manipulation at 76.7% with a rigid
  per-output-bit pair+stride script.  Forensics on her 374 failed bit rows:
  53.5% are PUNTS ("default 1" for bits whose fitting rule existed in her own
  tables), 22.5% are stride-EXTRAPOLATIONS ("?ab" placeholders applied unverified,
  contradicting her tables).  This emitter re-renders her failed rows in HER
  grammar but with correct tables and a "Recheck pending" step that CONTINUES
  THE SEARCH at exactly the decision points where she punts/extrapolates.

ALI'S GRAMMAR (learned from ~25 traces in analysis/ali_validation/{passed,failed}/
bit_manipulation/*.txt; bit positions are MSB-first 0..7):

  1. Preamble: "We need to deduce the transformation by matching the example outputs."
  2. Per-example bit expansion: "Output N: <byte>" then 8 lines "p v"; same for
     inputs.  (Dropped in our rendering - see TOKEN COMPRESSION below.)
  3. Column tables.  Line shape:  "<label> <column> <hash>[ match j[ k..]]"
     - column = the rule's value on examples 1..N, as an N-digit bit string;
     - hash   = "a" if the column is constant (all 0s or all 1s) else the popcount
       (her decimal sums are wrong ~5% of the time - one of her failure roots;
       ours are exact);
     - "match j" suffix marks equality with output column j (multi: "match 2 6").
     Families, in fixed order:
       Identity   lines "a <col> <hash>"             rule name Ia
       NOT        lines "a <col> <hash>"             rule name NOTa
       Constant   lines "c <col> a"                  rule name Cc
       AND OR XOR symmetric pairs, lines "ab ba <col> <hash>", rule ANDab etc.;
                  enumerated in stride-distance groups d=1,2,3 (a=0..7, b=(a+d)%8)
                  then d=4 (a=0..3), blank line between groups
       AND-NOT OR-NOT XOR-NOT  asymmetric (X-NOT ab = X(in_a, NOT in_b)), lines
                  "ab <col> <hash>", groups d=1..7, a=0..7
  4. "Matching output with <Family>": 8 lines "j <entries|absent>" - the inverted
     index of the match suffixes (entries in table order; symmetric pairs listed
     both ways: "35 53 57 75").
  5. Per family "Left"/"Right" anchored stride runs:
       Left  = runs anchored at output bit 0 extending rightward with +1 stride;
       Right = anchored at bit 7 extending leftward with -1 stride.
     One line per anchor showing the run and the failing extension marked "x":
       "06 17x"  /  "6 7 0 1 2x"  (pairs / unary; constant anchors show the bare
       value, e.g. "1").  Then "Best: <rule names>: <len>" or "none\nBest: none".
     NOTE: she only ever anchors at bits 0 and 7 - interior runs are invisible to
     her, another failure root (the Recheck step below recovers them).
  6. "Selecting" block:
       "Lefts"/"Rights": per family "<Family> <best run names>: <len>" or "none"
       "Left longest: n" / "Right longest: n"
       "<Side> winner: Identity yes, NOT no, ..." (the longer side is printed
        first; tie -> Left first)
       "Best left/right: <names>: n", "Truncated left/right: ..." (if the two runs
        would overlap, the weaker one is cut to 8-len(stronger))
       "Tentative from right" (bits 7..0, right run filled, rest "pending"),
       "Tentative" (0..7, both runs), "Preferred from <winning side>" (pending
       bits get stride placeholders continuing the winning run: "?ab" for pairs,
       "?a?" for unary, "?Cc" for constant), "Preferred" (0..7).
  7. "Matching": per pending bit, the placeholder is tested against every family
     AT THE PLACEHOLDER POSITIONS ONLY (both orders for pairs):
       "4 ?53 - Identity absent, NOT absent, Constant absent, AND absent, OR
        absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent"
     hits are written as the rule name ("NOT5", "AND-NOT57", "Constant C0").
     "Perfect match": per family yes/no (yes = hit at EVERY pending bit).
     "Matched": pending bits filled from the perfect family, else per-bit first
     hit, else "none".
  8. HER ENDING (the bug): "Selected" copies Matched with "none" -> "default 1"
     (punt) or keeps unverified "?ab" placeholders (extrapolation), then applies.
     OUR ENDING (the fix): a "Recheck pending" section between Matched and
     Selected - for every unresolved bit, the family match tables (already in
     the trace) are scanned in family order and the first fitting rule is taken:
       "4 Identity absent, NOT absent, Constant absent, AND absent, OR 25 52,
        XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent -> OR25"
     "Selected" then contains 8 real verified rules - no "default", no "?".
  9. "Applying to <query>": "Input" + 8 lines "p v", then "Output" lines:
       "0 I6 = 0" / "1 NOT2 = NOT(1) = 0" / "7 C1 = 1"
       "4 AND35 = AND(1,0) = 0" / "5 AND-NOT17 = AND(0,NOT(1)) = 0"
       (OR-NOT -> "OR(x,NOT(y))", XOR-NOT -> "XOR(x,NOT(y))")
     Closing: "I will now return the answer in \\boxed{}" /
              "The answer in \\boxed is" / "\\boxed{<answer>}".

TOKEN COMPRESSION vs her verbatim grammar (hers runs 6.5-7.5k tokens against the
7680 budget; 3 of her 374 fails are pure truncations, and our Recheck ADDS
lines, so verbatim would not fit).  Changes, each keeping every retained line
shape identical (verbatim ~7.0k -> ours median 3.8k, max <=4.5k hard gate):
  - per-example 8-line "Output N:"/"Input N:" bit expansions dropped (columns
    are transposed straight from the prompt's example block; ~900 tokens);
  - a new 8-line "Output bit columns complemented" table; NOT / OR-NOT / XOR-NOT
    columns are NOT re-enumerated - they are complement-derived: a "cmatch j"
    suffix on Identity / AND-NOT / XOR lines marks equality with COMPLEMENTED
    output column j (NOTa = comp(Ia); OR-NOTab = comp(AND-NOTba); XOR-NOTab =
    comp(XORab)), and only those three derived families keep a
    "Matching output with X" block (saves ~1900 tokens);
  - "Matching output with X" blocks for Identity/Constant/AND/OR/XOR/AND-NOT
    dropped (fully redundant with the "match j" suffixes on their table lines);
  - "Lefts"/"Rights" family summaries, "<Side> winner:" lines, "Tentative from
    right" and the duplicated "Preferred" table dropped (each redundant with an
    adjacent retained block: per-family "Best:", "Best <side>:", "Tentative",
    "Preferred from <side>");
  - all-absent placeholder Matching lines compressed to "- all absent".

RULE CHOICE among equally-fitting candidates (under-determined bits): every
choice point (best-anchor-run tie, Matched per-bit fill, Recheck pick) prefers
the rule bit_stride's longest-run cover assigns to that bit when it is among
the candidates.  The generator's truth is one global <=3-tap rule whose per-bit
projections form stride runs, so this prior is both the empirically best guess
(84.6% standalone) and a bias SFT can absorb; it is solver-derived, never
gold-derived.  Rows where the rendered procedure still misses gold are dropped.

SELF-CONSISTENCY (cod-emit discipline), enforced by the built-in linter on an
independent re-parse of the emitted text:
  - every Selected rule's column equals the output column on EVERY example
    (verification covers every example pair);
  - apply lines use exactly the Selected rules and are arithmetically correct;
  - boxed answer == the byte assembled from the apply lines == row 'final';
  - no "default" anywhere, no "?" in Selected/Output blocks, ASCII-only;
  - the train.csv answer is never copied in: the boxed value is what the
    procedure computes; rows where that differs from gold are DROPPED.

ELIGIBILITY: Ali-failed ids, not in val.jsonl, bit_stride coverage == 8 (a
<=2-input rule exists for every output bit -> the Recheck step can always
finish).  Rows whose rendered answer != train.csv answer are dropped (per-bit
rules under-determined by the examples disagreeing on the query).

Output: pipeline/data/v16/bit_alifix.jsonl  rows
  {id, category, prompt, cot, final, tier, src}
  tier = alifix_punt | alifix_extrap | alifix_punt_extrap | alifix_other
  (from HER trace's Selected block: default lines / ? entries / both / neither)
Report: analysis/reports/bit_alifix_build.md

PASSED MODE (python3 bitmanip_alifix.py passed): re-renders Ali's 1228 PASSED
bit rows in the SAME compressed dialect (v16 must not mix two dialects of the
bit grammar - the v15 format-interference lesson).  Identical pipeline and
gates; the only difference is the preference prior at choice points: her own
trace's Selected rule is parsed out and, when it verifies on every example
pair, takes priority over bit_stride's rule (warm start: prefer what she
already knows); bits where her rule is a punt/placeholder or fails an example
fall back to the stride prior.  Output: pipeline/data/v16/bit_alipass.jsonl
rows {id, category, prompt, cot, final, src='alipass', n_tokens}.
"""
import csv, json, os, re, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
import bit_stride as bs

FAMS = ['Identity', 'NOT', 'Constant', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']
SYM = {'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b}
ASYM = {'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b),
        'XOR-NOT': lambda a, b: a ^ (1 - b)}

def bitc(x, p): return (x >> (7 - p)) & 1
def comp(s): return ''.join('1' if c == '0' else '0' for c in s)
def hashv(col): return 'a' if len(set(col)) == 1 else str(col.count('1'))

def sym_pairs():
    groups = []
    for d in (1, 2, 3):
        groups.append([(a, (a + d) % 8) for a in range(8)])
    groups.append([(a, a + 4) for a in range(4)])
    return groups

def asym_pairs():
    return [[(a, (a + d) % 8) for a in range(8)] for d in range(1, 8)]

# ---------------- rule objects: ('I',a) ('NOT',a) ('C',c) (fam,a,b) ----------------
def rname(r):
    if r[0] == 'I': return 'I%d' % r[1]
    if r[0] == 'NOT': return 'NOT%d' % r[1]
    if r[0] == 'C': return 'C%d' % r[1]
    return '%s%d%d' % (r[0], r[1], r[2])

def reval(r, x):
    if r[0] == 'I': return bitc(x, r[1])
    if r[0] == 'NOT': return 1 - bitc(x, r[1])
    if r[0] == 'C': return r[1]
    f = SYM.get(r[0]) or ASYM[r[0]]
    return f(bitc(x, r[1]), bitc(x, r[2]))

def shift(r, k):
    """stride-shift a rule by k output positions (cyclic)."""
    if r[0] in ('I', 'NOT'): return (r[0], (r[1] + k) % 8)
    if r[0] == 'C': return r
    return (r[0], (r[1] + k) % 8, (r[2] + k) % 8)

class Tables:
    """All columns + per-family per-output-bit ordered match lists."""
    def __init__(self, ex):
        self.ex = ex
        self.N = len(ex)
        self.outcol = [''.join(str(bitc(o, j)) for _, o in ex) for j in range(8)]
        self.coutcol = [comp(c) for c in self.outcol]
        self.incol = [''.join(str(bitc(i, a)) for i, _ in ex) for a in range(8)]
        self.paircol = {}          # (fam,a,b) -> column
        for fam, f in SYM.items():
            for grp in sym_pairs():
                for a, b in grp:
                    self.paircol[(fam, a, b)] = ''.join(
                        str(f(int(x), int(y))) for x, y in zip(self.incol[a], self.incol[b]))
        for grp in asym_pairs():
            for a, b in grp:
                self.paircol[('AND-NOT', a, b)] = ''.join(
                    str(int(x) & (1 - int(y))) for x, y in zip(self.incol[a], self.incol[b]))
        # OR-NOT (complement of AND-NOT with swapped args) and XOR-NOT (complement of XOR)
        for grp in asym_pairs():
            for a, b in grp:
                self.paircol[('OR-NOT', a, b)] = comp(self.paircol[('AND-NOT', b, a)])
                ab = (a, b) if ('XOR', a, b) in self.paircol else (b, a)
                self.paircol[('XOR-NOT', a, b)] = comp(self.paircol[('XOR',) + ab])
        # ordered match lists
        self.match = {fam: [[] for _ in range(8)] for fam in FAMS}
        for j in range(8):
            oc, cc = self.outcol[j], self.coutcol[j]
            for a in range(8):
                if self.incol[a] == oc: self.match['Identity'][j].append(('I', a))
                if self.incol[a] == cc: self.match['NOT'][j].append(('NOT', a))
            for c in (0, 1):
                if oc == str(c) * self.N: self.match['Constant'][j].append(('C', c))
            for fam in ('AND', 'OR', 'XOR'):
                for grp in sym_pairs():
                    for a, b in grp:
                        if self.paircol[(fam, a, b)] == oc:
                            self.match[fam][j] += [(fam, a, b), (fam, b, a)]
            for grp in asym_pairs():
                for a, b in grp:
                    if self.paircol[('AND-NOT', a, b)] == oc:
                        self.match['AND-NOT'][j].append(('AND-NOT', a, b))
                    if self.paircol[('AND-NOT', a, b)] == cc:        # cmatch -> OR-NOT ba
                        self.match['OR-NOT'][j].append(('OR-NOT', b, a))
            for grp in sym_pairs():
                for a, b in grp:
                    if self.paircol[('XOR', a, b)] == cc:
                        self.match['XOR-NOT'][j] += [('XOR-NOT', a, b), ('XOR-NOT', b, a)]

    def has(self, fam, j, r):
        return r in self.match[fam][j]

# ---------------- runs ----------------
def run_from(T, fam, anchor, j0, step):
    """maximal stride run from anchor rule at output bit j0 going by step.
    returns (rules list anchor-first, failed extension rule or None)."""
    rules = [anchor]
    j, k = j0, 0
    while len(rules) < 8:
        k += step
        j = j0 + k
        if j < 0 or j > 7: return rules, None
        nxt = shift(anchor, k)
        ok = T.has(fam, j, nxt)
        if fam in SYM or fam == 'XOR-NOT':
            ok = ok or T.has(fam, j, (fam, nxt[2], nxt[1]))
        if not ok: return rules, nxt
        rules.append(nxt)
    return rules, None

def run_line(fam, rules, fail):
    def tok(r):
        if r[0] in ('I', 'NOT'): return str(r[1])
        if r[0] == 'C': return str(r[1])
        return '%d%d' % (r[1], r[2])
    parts = [tok(r) for r in rules]
    if fail is not None and rules[0][0] != 'C':
        parts.append(tok(fail) + 'x')
    return ' '.join(parts)

def same(r, s):
    """rule equality up to symmetric-operand order."""
    if r == s: return True
    return r and s and r[0] == s[0] and r[0] in ('AND', 'OR', 'XOR', 'XOR-NOT') \
        and len(r) == 3 and (r[0], r[2], r[1]) == s

def best_run(runs, pref=None, j0=0):
    """runs = list of (rules, fail); pick longest; ties broken by agreement with
    the stride-prior rule at the anchor bit, then first."""
    if not runs: return None
    def key(rf):
        i = runs.index(rf)
        agree = 1 if (pref and same(rf[0][0], pref[j0])) else 0
        return (len(rf[0]), agree, -i)
    return max(runs, key=key)

def prefer(cands, p):
    """pick the stride-prior rule from candidates if present, else the first."""
    if p is not None:
        for c in cands:
            if same(c, p): return c
    return cands[0]

# ---------------- the renderer ----------------
def render(ex, q, pref=None):
    T = Tables(ex)
    N = T.N
    if pref is None: pref = [None] * 8
    L = []
    add = L.append
    add('We need to deduce the transformation by matching the example outputs.')
    add('')
    add('Output bit columns (with bitsum as hash)')
    for j in range(8): add('%d %s %s' % (j, T.outcol[j], hashv(T.outcol[j])))
    add('')
    add('Output bit columns complemented')
    for j in range(8): add('%d %s %s' % (j, T.coutcol[j], hashv(T.coutcol[j])))
    add('')

    def suffixes(col):
        s = ''
        m = [j for j in range(8) if col == T.outcol[j]]
        if m: s += ' match ' + ' '.join(map(str, m))
        return s, [j for j in range(8) if col == T.coutcol[j]]

    def csuffix(cm):
        return (' cmatch ' + ' '.join(map(str, cm))) if cm else ''

    def matching_block(fam):
        add('Matching output with %s' % fam)
        for j in range(8):
            es = T.match[fam][j]
            if not es:
                add('%d absent' % j)
            elif fam in ('Identity', 'NOT'):
                add('%d %s' % (j, ' '.join(str(r[1]) for r in es)))
            elif fam == 'Constant':
                add('%d %s' % (j, ' '.join('C%d' % r[1] for r in es)))
            else:
                add('%d %s' % (j, ' '.join('%d%d' % (r[1], r[2]) for r in es)))
        add('')

    best = {}                     # (fam, side) -> (rules, fail) or None
    def lr_blocks(fam):
        for side, j0, step in (('Left', 0, 1), ('Right', 7, -1)):
            add(side)
            runs = []
            for anchor in T.match[fam][j0]:
                rules, fail = run_from(T, fam, anchor, j0, step)
                runs.append((rules, fail))
                add(run_line(fam, rules, fail))
            if not runs: add('none')
            b = best_run(runs, pref, j0)
            best[(fam, side)] = b
            add('Best: %s' % ('%s: %d' % (' '.join(rname(r) for r in b[0]), len(b[0])) if b else 'none'))
            add('')

    # Identity
    add('Identity')
    for a in range(8):
        s, cm = suffixes(T.incol[a])
        add('%d %s %s%s%s' % (a, T.incol[a], hashv(T.incol[a]), s, csuffix(cm)))
    add('')
    lr_blocks('Identity')
    # NOT (derived)
    add('NOT (Identity cmatch)')
    matching_block('NOT'); lr_blocks('NOT')
    # Constant
    add('Constant')
    for c in (0, 1):
        col = str(c) * N
        s, _ = suffixes(col)
        add('%d %s a%s' % (c, col, s))
    add('')
    lr_blocks('Constant')
    # AND OR XOR
    for fam in ('AND', 'OR', 'XOR'):
        add(fam)
        for grp in sym_pairs():
            for a, b in grp:
                col = T.paircol[(fam, a, b)]
                s, cm = suffixes(col)
                cs = csuffix(cm) if fam == 'XOR' else ''
                add('%d%d %d%d %s %s%s%s' % (a, b, b, a, col, hashv(col), s, cs))
            add('')
        lr_blocks(fam)
    # AND-NOT
    add('AND-NOT')
    for grp in asym_pairs():
        for a, b in grp:
            col = T.paircol[('AND-NOT', a, b)]
            s, cm = suffixes(col)
            add('%d%d %s %s%s%s' % (a, b, col, hashv(col), s, csuffix(cm)))
        add('')
    lr_blocks('AND-NOT')
    # OR-NOT / XOR-NOT (derived)
    add('OR-NOT (AND-NOT cmatch, swapped)')
    matching_block('OR-NOT'); lr_blocks('OR-NOT')
    add('XOR-NOT (XOR cmatch)')
    matching_block('XOR-NOT'); lr_blocks('XOR-NOT')

    # ---------------- Selecting ----------------
    add('Selecting')
    add('')
    bl = best_run([b for b in (best[(f, 'Left')] for f in FAMS) if b], pref, 0)
    br = best_run([b for b in (best[(f, 'Right')] for f in FAMS) if b], pref, 7)
    ll = len(bl[0]) if bl else 0
    rl = len(br[0]) if br else 0
    add('Left longest: %d' % ll)
    add('Right longest: %d' % rl)
    add('')
    order = ('left', 'right') if ll >= rl else ('right', 'left')
    bestline = {'left': bl, 'right': br}
    for s in order:
        b = bestline[s]
        add('Best %s: %s' % (s, '%s: %d' % (' '.join(rname(r) for r in b[0]), len(b[0])) if b else 'none'))
    # truncation: weaker side cut so runs do not overlap
    lrules = list(bl[0]) if bl else []
    rrules = list(br[0]) if br else []
    if len(lrules) + len(rrules) > 8:
        if order[0] == 'left': rrules = rrules[:8 - len(lrules)]
        else: lrules = lrules[:8 - len(rrules)]
    trunc = {'left': lrules, 'right': rrules}
    for s in order:
        rs = trunc[s]
        add('Truncated %s: %s' % (s, ('%s: %d' % (' '.join(rname(r) for r in rs), len(rs))) if rs else 'none'))
    add('')
    assign = [None] * 8
    for k, r in enumerate(rrules): assign[7 - k] = r
    for k, r in enumerate(lrules): assign[k] = r
    add('Tentative')
    for j in range(8): add('%d %s' % (j, rname(assign[j]) if assign[j] else 'pending'))
    add('')
    # placeholders continue the winning run through all pending bits
    win = order[0]
    wrun = trunc[win] if trunc[win] else trunc[order[1]]
    wanchor, w0 = (wrun[0], 0) if (win == 'left' or not trunc[win]) else (wrun[0], 7)
    if not trunc[win]:
        win = order[1]; w0 = 0 if win == 'left' else 7
    ph = {}
    for j in range(8):
        if assign[j] is None: ph[j] = shift(wanchor, j - w0)
    def phtok(r):
        if r[0] in ('I', 'NOT'): return '?%d?' % r[1]
        if r[0] == 'C': return '?C%d' % r[1]
        return '?%d%d' % (r[1], r[2])
    add('Preferred from %s' % win)
    for j in range(8): add('%d %s' % (j, rname(assign[j]) if assign[j] else phtok(ph[j])))
    add('')

    # ---------------- Matching at placeholders ----------------
    def ph_check(fam, j, p):
        """hit rule for family fam at output j testing only placeholder positions."""
        if fam == 'Constant':
            for r in T.match['Constant'][j]: return r
            return None
        if p[0] == 'C': return None
        a = p[1]; b = p[2] if len(p) > 2 else None
        if fam in ('Identity', 'NOT'):
            r = ('I' if fam == 'Identity' else 'NOT', a)
            return r if T.has(fam, j, r) else None
        if b is None:                       # unary placeholder: any pair using a
            for r in T.match[fam][j]:
                if a in (r[1], r[2]): return r
            return None
        for r in ((fam, a, b), (fam, b, a)):
            if T.has(fam, j, r): return r
        return None

    pend = sorted(ph)
    hits = {}                                # j -> {fam: rule}
    add('Matching')
    for j in range(8):
        if j not in ph:
            add('%d %s' % (j, rname(assign[j])))
            continue
        hits[j] = {}
        parts = []
        for fam in FAMS:
            h = ph_check(fam, j, ph[j])
            if h is None:
                parts.append('%s absent' % fam)
            else:
                hits[j][fam] = h
                parts.append('Constant C%d' % h[1] if fam == 'Constant' else rname(h))
        if not hits[j]: parts = ['all absent']
        add('%d %s - %s' % (j, phtok(ph[j]), ', '.join(parts)))
    add('')
    add('Perfect match')
    perfect = None
    for fam in FAMS:
        ok = bool(pend) and all(fam in hits[j] for j in pend)
        if ok and perfect is None: perfect = fam
        add('%s %s' % (fam, 'yes' if ok else 'no'))
    add('')
    add('Matched')
    for j in range(8):
        if j in ph and hits[j]:
            cands = [hits[j][fam] for fam in FAMS if fam in hits[j]]
            if perfect and perfect in hits[j] and not any(same(c, pref[j]) for c in cands):
                assign[j] = hits[j][perfect]
            else:
                assign[j] = prefer(cands, pref[j])
        add('%d %s' % (j, rname(assign[j]) if assign[j] else 'none'))
    add('')

    # ---------------- Recheck (the fix: continue the search, never punt) ------
    add('Recheck pending')
    rest = [j for j in range(8) if assign[j] is None]
    if not rest: add('none')
    for j in rest:
        parts = []
        cands = []
        for fam in FAMS:
            es = T.match[fam][j]
            if not es:
                parts.append('%s absent' % fam)
                continue
            if fam in ('Identity', 'NOT'):
                parts.append('%s %s' % (fam, ' '.join(str(r[1]) for r in es)))
            elif fam == 'Constant':
                parts.append('%s %s' % (fam, ' '.join('C%d' % r[1] for r in es)))
            else:
                parts.append('%s %s' % (fam, ' '.join('%d%d' % (r[1], r[2]) for r in es)))
            cands += es
        if not cands: return None           # 3-input bit: row not renderable
        assign[j] = prefer(cands, pref[j])
        add('%d %s -> %s' % (j, ', '.join(parts), rname(assign[j])))
    add('')
    add('Selected')
    for j in range(8): add('%d %s' % (j, rname(assign[j])))
    add('')

    # ---------------- Apply ----------------
    add('Applying to %s' % format(q, '08b'))
    add('Input')
    for p in range(8): add('%d %d' % (p, bitc(q, p)))
    add('Output')
    bits = []
    for j in range(8):
        r = assign[j]
        v = reval(r, q)
        bits.append(v)
        if r[0] == 'I': add('%d %s = %d' % (j, rname(r), v))
        elif r[0] == 'NOT': add('%d %s = NOT(%d) = %d' % (j, rname(r), bitc(q, r[1]), v))
        elif r[0] == 'C': add('%d %s = %d' % (j, rname(r), v))
        elif r[0] in SYM:
            add('%d %s = %s(%d,%d) = %d' % (j, rname(r), r[0], bitc(q, r[1]), bitc(q, r[2]), v))
        else:
            base = r[0][:-4] if r[0].endswith('-NOT') else r[0]
            add('%d %s = %s(%d,NOT(%d)) = %d' % (j, rname(r), base, bitc(q, r[1]), bitc(q, r[2]), v))
    ans = ''.join(map(str, bits))
    add('')
    add('I will now return the answer in \\boxed{}')
    add('The answer in \\boxed is')
    add('\\boxed{%s}' % ans)
    return '\n'.join(L), ans, assign

# ---------------- linter: independent re-parse of the emitted text ----------------
RULE_RE = re.compile(r'^(I|NOT|C|AND|OR|XOR|AND-NOT|OR-NOT|XOR-NOT)(\d)(\d)?$')
def parse_rule(name):
    m = RULE_RE.match(name)
    assert m, name
    f, a, b = m.group(1), int(m.group(2)), m.group(3)
    if f == 'I': return ('I', a)
    if f in ('NOT', 'C'): return (f, a)
    assert b is not None, name
    return (f, a, int(b))

def lint(cot, prompt, final):
    assert cot.isascii(), 'non-ascii'
    assert 'default' not in cot, 'default line present'
    ex, q = bs.parse(prompt)
    N = len(ex)
    sel = re.search(r'\nSelected\n((?:\d .+\n){8})', cot)
    assert sel, 'no Selected block'
    rules = {}
    for line in sel.group(1).strip().split('\n'):
        j, name = line.split(' ', 1)
        assert '?' not in name and 'default' not in name, 'unresolved Selected'
        rules[int(j)] = parse_rule(name)
    assert len(rules) == 8
    # every Selected rule verified on EVERY example pair
    for j, r in rules.items():
        for i, o in ex:
            assert reval(r, i) == bitc(o, j), 'rule %s fails example' % rname(r)
    # apply block consistency
    app = re.search(r'\nOutput\n((?:\d .+\n){8})', cot)
    assert app, 'no apply block'
    bits = {}
    for line in app.group(1).strip().split('\n'):
        m = re.match(r'^(\d) (\S+) .*?= (\d)$', line) or re.match(r'^(\d) (\S+) = (\d)$', line)
        assert m, line
        j, name, v = int(m.group(1)), m.group(2), int(m.group(3))
        assert '?' not in name
        assert parse_rule(name) == rules[j], 'apply rule != Selected at bit %d' % j
        assert reval(rules[j], q) == v, 'apply arithmetic wrong at bit %d' % j
        bits[j] = v
    ans = ''.join(str(bits[j]) for j in range(8))
    boxed = re.findall(r'\\boxed\{([01]{8})\}', cot)
    assert boxed and boxed[-1] == ans == final, 'boxed/final mismatch'
    # column sanity: emitted output columns are correct transposes
    for j in range(8):
        col = ''.join(str(bitc(o, j)) for _, o in ex)
        assert re.search(r'\nOutput bit columns \(with bitsum as hash\)\n(?:.*\n){%d}%d %s ' % (j, j, col), cot), \
            'output column %d wrong' % j
    return True

# ---------------- her-trace tier classification ----------------
def classify(path):
    try:
        raw = open(path).read()
    except OSError:
        return 'alifix_other'
    m = re.search(r'\nSelected\n(.*?)\n\n', raw, re.S)
    if not m:
        return 'alifix_trunc'
    blk = m.group(1)
    punt = 'default' in blk
    extrap = '?' in blk
    if punt and extrap: return 'alifix_punt_extrap'
    if punt: return 'alifix_punt'
    if extrap: return 'alifix_extrap'
    return 'alifix_other'

# ---------------- stride prior ----------------
OP2FAM = {'AND': 'AND', 'OR': 'OR', 'XOR': 'XOR',
          'ANDN': 'AND-NOT', 'ORN': 'OR-NOT', 'XORN': 'XOR-NOT'}
def stride_pref(assign_s):
    """bit_stride per-bit descs -> per-bit rule reprs (the stride prior)."""
    pref = [None] * 8
    for j, d in enumerate(assign_s):
        if d is None: continue
        if d[0] == 'C': pref[j] = ('C', d[1])
        elif d[0] == 'U': pref[j] = ('NOT' if d[2] else 'I', (d[1] + j) % 8)
        else: pref[j] = (OP2FAM[d[3]], (d[1] + j) % 8, (d[2] + j) % 8)
    return pref

# ---------------- her Selected block (passed mode warm-start prior) ----------
def parse_her_selected(path):
    """Per-bit rules from HER trace's Selected block; punts/placeholders/
    unparseable lines -> None at that bit."""
    her = [None] * 8
    try:
        raw = open(path, errors='replace').read()
    except OSError:
        return her
    m = re.search(r'\nSelected\n(.*?)\n\n', raw, re.S)
    if not m:
        return her
    for line in m.group(1).strip().split('\n'):
        parts = line.split(' ', 1)
        if len(parts) != 2 or not parts[0].isdigit(): continue
        j = int(parts[0])
        if not 0 <= j <= 7: continue
        name = parts[1].strip()
        if name.startswith('Identity'): name = 'I' + name[len('Identity'):]
        if '?' in name or 'default' in name or not RULE_RE.match(name): continue
        her[j] = parse_rule(name)
    return her

# ---------------- main ----------------
def main():
    csv.field_size_limit(10 ** 9)
    fail_dir = os.path.join(ROOT, 'analysis/ali_validation/failed/bit_manipulation')
    failed = sorted(f[:-4] for f in os.listdir(fail_dir) if f.endswith('.txt'))
    val_ids = {json.loads(l)['id'] for l in open(os.path.join(ROOT, 'pipeline/data/val.jsonl'))}
    rows = {r['id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
            if r['category'] == 'bit_manipulation'}
    solved = {}
    cache = os.path.join(ROOT, 'pipeline/data/bitmanip_solved.jsonl')
    if os.path.exists(cache):
        for l in open(cache):
            d = json.loads(l)
            solved[d.get('id')] = d
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, 'competition_dataset/tokenizer.json'))

    out_rows = []
    drops = {'val': 0, 'cov<8': 0, 'answer!=gold': 0, 'render_fail': 0, 'tokens>4500': 0}
    toks, tiers = [], {}
    wrong_examples = []
    gold_xcheck_bad = 0
    for rid in failed:
        if rid in val_ids:
            drops['val'] += 1; continue
        r = rows[rid]
        ex, q = bs.parse(r['prompt'])
        gold = r['answer'].strip()
        if rid in solved and solved[rid].get('gold') != gold:
            gold_xcheck_bad += 1                       # global-solver cache cross-check
        assign_s, cov = bs.solve_bits(ex)
        if cov < 8:
            drops['cov<8'] += 1; continue
        res = render(ex, q, stride_pref(assign_s))
        if res is None:
            drops['render_fail'] += 1; continue
        cot, ans, assign = res
        if ans != gold:
            drops['answer!=gold'] += 1
            wrong_examples.append(rid)
            continue
        lint(cot, r['prompt'], ans)
        nt = len(tok.encode(cot).ids)
        if nt > 4500:
            drops['tokens>4500'] += 1; continue
        tier = classify(os.path.join(fail_dir, rid + '.txt'))
        tiers[tier] = tiers.get(tier, 0) + 1
        toks.append(nt)
        out_rows.append({'id': rid, 'category': 'bit_manipulation', 'prompt': r['prompt'],
                         'cot': cot, 'final': ans, 'tier': tier, 'src': 'ali_failed_bit',
                         'n_tokens': nt})
    os.makedirs(os.path.join(ROOT, 'pipeline/data/v16'), exist_ok=True)
    out_path = os.path.join(ROOT, 'pipeline/data/v16/bit_alifix.jsonl')
    with open(out_path, 'w') as f:
        for d in out_rows:
            f.write(json.dumps(d) + '\n')
    toks_s = sorted(toks)
    stats = dict(n=len(toks), median=int(statistics.median(toks_s)) if toks else 0,
                 p95=toks_s[int(0.95 * len(toks_s))] if toks else 0,
                 max=max(toks_s) if toks else 0)
    print('emitted:', len(out_rows), 'drops:', drops, 'tiers:', tiers)
    print('tokens:', stats)
    print('gold cache cross-check mismatches:', gold_xcheck_bad)
    print('gold-mismatch ids (first 10):', wrong_examples[:10])
    print('wrote', out_path)
    return out_rows, drops, tiers, stats, wrong_examples

def main_passed():
    """Re-render Ali's PASSED bit rows in the same compressed dialect.
    Prior at choice points: her verified Selected rule > stride rule."""
    csv.field_size_limit(10 ** 9)
    pass_dir = os.path.join(ROOT, 'analysis/ali_validation/passed/bit_manipulation')
    passed = sorted(f[:-4] for f in os.listdir(pass_dir) if f.endswith('.txt'))
    val_ids = {json.loads(l)['id'] for l in open(os.path.join(ROOT, 'pipeline/data/val.jsonl'))}
    rows = {r['id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'competition_dataset/train_categorized.csv')))
            if r['category'] == 'bit_manipulation'}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, 'competition_dataset/tokenizer.json'))

    out_rows = []
    drops = {'val': 0, 'cov<8': 0, 'answer!=gold': 0, 'render_fail': 0, 'tokens>4500': 0}
    toks = []
    drop_ids = {k: [] for k in drops}
    disagree_rows = 0
    disagree_bits = 0
    her_unverified_bits = 0     # her Selected rule fails an example (table-error rules)
    her_missing_bits = 0        # punt/placeholder/unparseable in her Selected
    for rid in passed:
        if rid in val_ids:
            drops['val'] += 1; drop_ids['val'].append(rid); continue
        r = rows[rid]
        ex, q = bs.parse(r['prompt'])
        gold = r['answer'].strip()
        assign_s, cov = bs.solve_bits(ex)
        if cov < 8:
            drops['cov<8'] += 1; drop_ids['cov<8'].append(rid); continue
        spref = stride_pref(assign_s)
        her = parse_her_selected(os.path.join(pass_dir, rid + '.txt'))
        pref = list(spref)
        row_disagrees = False
        for j in range(8):
            if her[j] is None:
                her_missing_bits += 1; continue
            if not all(reval(her[j], i) == bitc(o, j) for i, o in ex):
                her_unverified_bits += 1; continue       # keep stride prior
            if not same(her[j], spref[j]):
                disagree_bits += 1; row_disagrees = True
            pref[j] = her[j]                             # prefer HERS (warm start)
        if row_disagrees: disagree_rows += 1
        res = render(ex, q, pref)
        if res is None:
            drops['render_fail'] += 1; drop_ids['render_fail'].append(rid); continue
        cot, ans, assign = res
        if ans != gold:
            drops['answer!=gold'] += 1; drop_ids['answer!=gold'].append(rid); continue
        lint(cot, r['prompt'], ans)
        nt = len(tok.encode(cot).ids)
        if nt > 4500:
            drops['tokens>4500'] += 1; drop_ids['tokens>4500'].append(rid); continue
        toks.append(nt)
        out_rows.append({'id': rid, 'category': 'bit_manipulation', 'prompt': r['prompt'],
                         'cot': cot, 'final': ans, 'src': 'alipass', 'n_tokens': nt})
    os.makedirs(os.path.join(ROOT, 'pipeline/data/v16'), exist_ok=True)
    out_path = os.path.join(ROOT, 'pipeline/data/v16/bit_alipass.jsonl')
    with open(out_path, 'w') as f:
        for d in out_rows:
            f.write(json.dumps(d) + '\n')
    toks_s = sorted(toks)
    stats = dict(n=len(toks), median=int(statistics.median(toks_s)) if toks else 0,
                 p95=toks_s[int(0.95 * len(toks_s))] if toks else 0,
                 max=max(toks_s) if toks else 0)
    print('emitted:', len(out_rows), 'of', len(passed), 'drops:', drops)
    print('tokens:', stats)
    print('her-vs-stride disagreements: %d rows, %d bits (resolved: hers)' % (disagree_rows, disagree_bits))
    print('her Selected bits unverified (table errors, stride kept):', her_unverified_bits)
    print('her Selected bits missing (default/?/unparseable, stride kept):', her_missing_bits)
    for k in ('cov<8', 'answer!=gold', 'render_fail', 'tokens>4500'):
        if drop_ids[k]: print('drop %s ids (first 10): %s' % (k, drop_ids[k][:10]))
    print('wrote', out_path)
    return out_rows, drops, stats, (disagree_rows, disagree_bits)

if __name__ == '__main__':
    main_passed() if 'passed' in sys.argv[1:] else main()
