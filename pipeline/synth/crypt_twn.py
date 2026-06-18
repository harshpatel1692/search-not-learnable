"""crypt_r9 = clean, uniform trace_we_need generator (anchor + bounded try/reject).

Skeleton (SAME every row, the point):
  1. spaced restatement
  2. operator from result-length cue
  3. reading direction
  4. anchor = most-constrained equation; list its <=12 candidate digit-assignments
  5. TRY each candidate: complete the other equations by shown arithmetic; a wrong
     candidate hits a visible contradiction -> REJECT; exactly one completes -> keep
  6. map -> verify every example -> spaced encode -> box
Every digit is witnessed by a shown arithmetic line. Rows whose anchor is too big
(>12 cands), whose completion needs >2-unknown search, or whose trace exceeds the
budget are DROPPED (not teleported).

  python3 pipeline/synth/crypt_twn.py demo [n]
  python3 pipeline/synth/crypt_twn.py build [n] [seed]
  python3 pipeline/synth/crypt_twn.py coverage         # measure on real rows
"""
import os, sys, time, json, random, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
import cryptarithm2 as C2
import cryptarithm_cot as CC
from gav_render import op_expr, op_phrase, _decode

ANCHOR_CAP = 12          # max candidate list we will render
TOK_CAP = 1600
MAX_ATT = 3              # r10: hard cap on numbered attempts, then SOLVE or BAIL (anti-loop)

import zlib

# Paraphrase bank: the PROCEDURE is fixed, the WORDING varies per row so the model
# can't learn template strings — it must reproduce the meaning. (Computed numbers
# already vary naturally; here we vary the framing + connective phrasing.)
_P = {
    'open': [
        "We need to find the hidden digit for each symbol. I rewrite everything spaced so I track single glyphs.",
        "Each symbol stands for a digit. I space out every glyph so I can handle them one at a time.",
        "Let me recover the symbol-to-digit map. First I write everything with the glyphs spaced apart.",
        "The puzzle hides a digit behind each symbol; I keep the glyphs spaced while I work.",
        "Goal: decode which digit each symbol means. I separate the glyphs with spaces to read them cleanly.",
        "Every symbol is a digit in disguise. Writing it all out spaced so the glyphs don't run together.",
        "Time to crack the symbol cipher. I'll keep each glyph spaced so I never confuse two of them.",
        "I need the digit behind each glyph. Step one: rewrite the lines with the symbols spaced out.",
        "Let me figure out the digit for every symbol, working from the spaced-out equations below.",
        "These symbols encode digits. I lay everything out spaced to handle one glyph at a time.",
        "First I space the glyphs apart, then I work out which digit each symbol hides.",
        "Decoding the map symbol by symbol. Spacing the glyphs so each stays its own token.",
        "The task is to recover each symbol's digit. I rewrite the equations glyph-spaced first.",
        "I'll uncover the digit for each glyph; everything is spaced below so the symbols stay distinct.",
    ],
    'op_intro': [
        "Find each operator. First rule out concatenation (result = the operand symbols rearranged), then read the arithmetic from the result length:",
        "Work out each operator. Concatenation first: does the result just rearrange the operand symbols? Otherwise the result length gives the operation:",
        "Identify the operators. Rule out concat (that would make the result the operands rearranged), then use the result size:",
        "Pin down each operator. Check concatenation first (result = operands written together); if not, the result length tells the operation:",
        "Determine the operators. Eliminate concat first (it would copy the operand symbols), then read the arithmetic off the result length:",
        "For each operator: is it concatenation (result is the operand symbols rearranged)? If not, infer the operation from how long the result is:",
        "Establish each operator. Concat would mean the result is just the operands; rule that out, then the result length names the operation:",
        "Read each operator. First test concatenation (does the result reuse the operand symbols?), otherwise use the result size:",
        "Sort out the operators. Rule out concat (result = operands rearranged) before reading the arithmetic from the result's length:",
        "Figure each operator. Concatenation check first; failing that, the number of result symbols points to the operation:",
        "Name each operator. Begin by excluding concatenation, then deduce the arithmetic from the result length:",
        "Decide each operator. Is the result the operand symbols rearranged (concat)? If not, its length gives the operation:",
        "Get each operator. Concat is ruled out when the result isn't the operands rearranged; then length decides the arithmetic:",
        "Resolve the operators one by one: reject concatenation first, then read the operation from the result size:",
    ],
    'concat_len': [
        "not concatenation (the result {R} has {n} symbols, concat would need 4)",
        "not concat (the result {R} is {n} symbols; a join would give all 4 operand symbols)",
        "concatenation is out ({R} has {n} symbols, not the 4 a join produces)",
        "rule out concat (result {R} is only {n} symbols, concat needs 4)",
        "this isn't concat ({R} = {n} symbols; concatenation would be 4)",
        "not a concatenation ({n}-symbol result {R}, but a join is 4 symbols)",
        "concat fails here (the result {R} has {n}, not 4, symbols)",
        "no concat ({R} is {n} symbols long; a join of two 2-digit numbers is 4)",
        "not the operands joined ({R} is {n} symbols, a join would be 4)",
        "concatenation doesn't fit ({R} has {n} symbols vs the 4 of a join)",
        "reject concatenation: {R} is {n} symbols, not 4",
        "the result {R} is {n} symbols, so not concat (which needs 4)",
        "can't be concat ({R} = {n} symbols; concat would be 4)",
        "not a join ({R} has {n} symbols, a join has 4)",
    ],
    'concat_sym': [
        "not concatenation (the result symbols {rs} are not the operand symbols {os})",
        "not concat (result symbols {rs} differ from the operands {os})",
        "concatenation is out (the result uses {rs}, not the operand symbols {os})",
        "rule out concat (the result {rs} isn't a rearrangement of {os})",
        "this isn't concat (result symbols {rs} vs operand symbols {os} don't match)",
        "not the operands joined (the result {rs} isn't the operand set {os})",
        "concat fails (result {rs} are different symbols from the operands {os})",
        "no concat (the result {rs} is not {os} rearranged)",
        "not a concatenation (the result symbols {rs} aren't the operands {os})",
        "concatenation doesn't fit ({rs} are not the operand symbols {os})",
        "reject concat: the result {rs} isn't the operands {os} reordered",
        "can't be concat (result {rs} != operands {os})",
        "not a join (the result {rs} uses symbols outside the operands {os})",
        "the result {rs} isn't {os} rearranged, so not concat",
    ],
    'is_concat': [
        "the result IS the operand symbols rearranged -> concatenation",
        "the result is exactly the operands written together -> concatenation",
        "the result reuses the operand symbols -> this is concatenation",
        "the result is the operands joined -> concatenation",
        "result = operands rearranged, so concatenation",
        "the result is just the operand symbols, so concatenation",
        "this is concatenation (the result is the operands reordered)",
        "the operand symbols make up the result -> concatenation",
    ],
    'dir_pre': [
        "Reading direction: ", "Now fix the reading direction: ", "Digit order: ",
        "Which way to read the digits: ", "Reading order: ", "Endianness: ",
        "Decide how to read each number: ", "The numbers read ",
    ],
    'dir_rev': [
        "little-endian (digits reversed); standard order makes an example fail.",
        "the numbers are little-endian (read right-to-left); standard order contradicts an example.",
        "little-endian -- each number's digits are reversed; the standard reading breaks an example.",
        "reversed (little-endian); reading left-to-right fails one of the examples.",
        "right-to-left -- the digits are reversed; standard order can't satisfy an example.",
        "little-endian. The normal left-to-right reading contradicts an example, so reverse every number.",
        "digits reversed (little-endian); a standard reading makes an example impossible.",
        "little-endian, i.e. reverse each number; standard order doesn't work on an example.",
        "the numbers are written reversed; reading them little-endian is the only consistent option.",
        "little-endian -- flip each number's digits; left-to-right breaks an example.",
        "reversed order: the rightmost glyph is the tens digit; standard order fails.",
        "little-endian throughout (standard order is ruled out by an example).",
    ],
    'dir_std': [
        "standard (left glyph = tens).",
        "normal left-to-right, leftmost glyph is the tens digit.",
        "standard order (the left glyph is tens).",
        "left-to-right as usual; the first glyph is the tens digit.",
        "standard reading -- leftmost digit is tens.",
        "ordinary order, left glyph = tens, right glyph = units.",
        "big-endian / standard: the left symbol is the tens place.",
        "read normally, left to right (tens then units).",
        "standard digit order (left = tens, right = units).",
        "left-to-right; no reversal needed.",
        "the usual order -- left glyph tens, right glyph units.",
        "standard, reading each number left to right.",
    ],
    'anchor': [
        "Anchor on the most constrained equation {eq} ('{op}'). Try its digit-assignments (distinct digits, no leading zero) in turn:",
        "Start from the tightest equation {eq} ('{op}'); test each valid assignment (distinct digits, no leading zero):",
        "The most constrained equation is {eq} ('{op}'). Go through its candidate assignments (distinct, no leading zero):",
        "Use {eq} ('{op}') as the anchor -- it pins the most. Try each digit-assignment (distinct, no leading zero):",
        "Begin with the tightest line {eq} ('{op}'). Its possible assignments (distinct digits, no leading zero), one at a time:",
        "Lead with {eq} ('{op}'), the equation with the fewest options. Test each assignment (distinct, no leading zero):",
        "Take {eq} ('{op}') first -- it constrains the most. Run through its valid maps (distinct digits, no leading zero):",
        "The narrowest equation is {eq} ('{op}'). Enumerate its digit-assignments (distinct, no leading zero):",
        "Anchor: {eq} ('{op}'). I check each assignment it allows (distinct digits, none leading with 0):",
        "Pick the hardest-to-satisfy equation {eq} ('{op}') and try its candidate maps (distinct, no leading zero):",
        "Solve {eq} ('{op}') first, the most constrained one. Each candidate assignment (distinct, no leading zero):",
        "From {eq} ('{op}') -- the tightest constraint -- list and test the assignments (distinct digits, no leading zero):",
    ],
    'complete': [
        "complete the rest from the other equations:",
        "fill in the remaining symbols from the other equations:",
        "the other equations finish the map:",
        "use the other equations to pin the rest:",
        "work the remaining symbols out from the other lines:",
        "now the other equations determine what's left:",
        "finish the map using the other equations:",
        "the rest follows from the other equations:",
        "pin the remaining digits from the other equations:",
        "let the other equations resolve the rest:",
        "carry the other equations to fill in the remaining glyphs:",
        "the leftover symbols come from the other equations:",
    ],
    'keep': [
        "all equations satisfied -> keep this one.",
        "every equation checks out -> this is the map.",
        "all checks pass -> accept this assignment.",
        "everything is consistent -> this is the answer map.",
        "all equations hold -> lock this in.",
        "no contradictions -> this assignment is correct.",
        "each equation works -> keep this map.",
        "consistent everywhere -> take this one.",
        "all lines verify -> this is it.",
        "the whole system holds -> accept this map.",
        "nothing breaks -> this assignment stands.",
        "every line is satisfied -> done with the search.",
    ],
    'map_hdr': ["Map:", "Digit map:", "So the map is:", "Recovered map:",
                "The assignment:", "Symbol -> digit:", "Final map:", "That gives:"],
    'verify': ["Verify:", "Check every example:", "Confirm against all examples:",
               "Double-check each equation:", "Verify all examples:", "Re-check the examples:",
               "Confirm every line:", "Validation:", "Check it against each example:",
               "Make sure each example holds:"],
    'ok': ["ok", "ok", "checks out", "holds", "correct", "matches", "good", "fine", "yes"],
    'apply': ["Query", "Now the query", "Apply to the query", "For the query",
              "On the query", "Solve the query", "The query"],
    'forcing': ["forcing", "which gives", "so", "pinning", "giving", "which fixes",
                "hence", "therefore", "making", "this sets", "so that's"],
    'needs': ["needs", "requires", "forces", "would need", "demands", "calls for",
              "wants", "must have"],
    'encode': ["Encode", "Map the digits back", "Back to symbols", "Convert back",
               "Translate the digits to symbols", "Write the digits as symbols",
               "Re-encode as glyphs", "Turn the digits back into symbols"],
}


def _pk(rng, key):
    return rng.choice(_P[key])


def _rng_for(prompt):
    return random.Random(zlib.crc32(prompt.encode()))


def sp(s):
    return ' '.join(s)


def _operands(L, rev):
    return ((L[1], L[0]) if rev else (L[0], L[1])), ((L[4], L[3]) if rev else (L[3], L[4]))


_MULF = {'mul', 'mul_p1', 'mul_m1', 'mul_p2', 'mul_m2'}
_ADDF = {'add', 'add_p1', 'add_m1', 'add_p2', 'add_m2'}
_ABSF = {'absdiff', 'absdiff_p1', 'absdiff_m1', 'absdiff_p2', 'absdiff_m2', 'neg_absdiff'}


def _cue(rng, op, rlen):
    """honest result-length hint + the operation, paraphrased per row."""
    if op in _MULF:
        return rng.choice([
            "the result has 4 symbols and only a product of two 2-digit numbers reaches 4 digits, so a multiplication",
            "a 4-symbol result means two 2-digit numbers multiplied -- a multiplication",
            "4 result symbols can only come from multiplying, so a multiplication",
            "two 2-digit numbers reach 4 digits only by multiplying -- a multiplication",
            "the 4-symbol result forces a product -- multiplication",
            "only a product is this big (4 symbols) -- a multiplication",
            "a 4-digit result points straight to multiplication",
        ]) if rlen >= 4 else rng.choice([
            f"the {rlen}-symbol result is a small product, so a multiplication",
            f"a short {rlen}-symbol result fits a product -- a multiplication",
            f"a {rlen}-symbol result is a small multiplication",
            f"the product is small here ({rlen} symbols) -- still a multiplication",
            f"a compact {rlen}-digit result, consistent with a product -- multiplication",
        ])
    if op in _ADDF:
        return rng.choice([
            f"the {rlen}-symbol result is the right size for a sum",
            f"a {rlen}-symbol result is what a sum gives",
            f"a {rlen}-digit result is sum-sized, so an addition",
            f"the result length ({rlen}) matches a sum",
            f"adding two 2-digit numbers gives a {rlen}-symbol result like this -- a sum",
            f"a {rlen}-symbol result fits addition",
            f"sized like a sum ({rlen} symbols) -- an addition",
        ])
    if op in ('sub_signed', 'rsub_signed'):
        return rng.choice([
            "the result is short (and may carry a sign), so a subtraction",
            "a short, possibly-signed result points to a subtraction",
            "a brief result, sometimes with a sign -- a subtraction",
            "short result with a possible minus sign -- a subtraction",
            "the small (maybe signed) result fits a subtraction",
            "a difference: the result is short and can be signed",
        ])
    if op in _ABSF:
        return rng.choice([
            "the result is short, so an absolute difference",
            "a short result fits an absolute difference",
            "the small result points to an absolute difference",
            "a brief result -- an absolute difference",
            "a difference in magnitude: the result is short",
            "short result, consistent with |a - b|",
        ])
    return rng.choice([
        "a number-theoretic operation (remainder / lcm)",
        "a remainder-or-lcm style operation",
        "a modular / lcm operation",
        "a number-theory operation (mod, lcm, or gcd)",
        "a remainder or least-common-multiple operation",
    ])


def _magnitude_block(rng, op, rl):
    """Process-of-elimination by MAGNITUDE: bound the operands from the result length
    before trying digits (the r10 model wasted tries on magnitude-impossible candidates
    like 23*21=483 for a 4-digit result). All numbers shown are real and the reasoning
    is map-free (uses only result length), so it's learnable."""
    if op in _MULF:
        if rl >= 4:
            return rng.choice([
                f"Magnitude first: a {rl}-symbol result means the product is at least {10**(rl-1)}. "
                f"Two 2-digit numbers reach that only when they are not both small -- 23 * 21 = 483 is "
                f"only 3 digits (too small), while 47 * 89 = 4183 has 4. So I need operands whose product "
                f"is {rl} digits; tiny digits in both won't do.",
                f"Bound the size: for a {rl}-digit product I need a*b >= {10**(rl-1)}. 21 * 23 = 483 is too "
                f"small (3 digits); something like 68 * 74 = 5032 fits. So at least one operand is large -- "
                f"skip the all-small assignments.",
                f"Use the length: {rl} result symbols => a*b in [{10**(rl-1)}, {10**rl-1}]. Small pairs "
                f"(e.g. 19 * 21 = 399) fall short, so the operands must be bigger; I only try digit choices "
                f"whose product has {rl} digits.",
            ])
        return rng.choice([
            f"Magnitude: a {rl}-digit product (between {10**(rl-1)} and {10**rl-1}) comes from moderate "
            f"operands, e.g. 12 * 30 = 360; not from two large numbers (which overshoot).",
            f"Size check: the product is {rl} digits, so the operands stay moderate -- 14 * 26 = 364 fits, "
            f"81 * 73 = 5913 is far too big.",
        ])
    if op in _ADDF:
        if rl >= 3:
            return rng.choice([
                "Magnitude first: two 2-digit numbers sum to at most 99 + 99 = 198, so a 3-symbol result "
                "means a + b is 100-198 and starts with 1. A small pair like 10 + 20 = 30 never reaches 3 "
                "digits -- the operands must be large enough to carry past 100.",
                "Bound it: a + b here is >= 100 (3 result symbols), and <= 198, so the leading digit is 1. "
                "Pairs like 23 + 41 = 64 are too small; I need operands that add past 100.",
                "Size: a 3-digit sum needs a + b >= 100 (max 198, so it begins with 1). 10 + 10 = 20 is far "
                "too small; both operands have to be fairly large.",
            ])
        return rng.choice([
            "Magnitude: a 2-symbol sum means a + b < 100, so the operands are on the smaller side "
            "(e.g. 31 + 24 = 55); large pairs would carry to 3 digits.",
            "Size check: the sum is 2 digits (< 100), so neither operand is large -- 40 + 35 = 75 fits.",
        ])
    if op in ('sub_signed', 'rsub_signed') or op in _ABSF:
        return rng.choice([
            "Magnitude: the result is short, so the two numbers are close and their difference is small "
            "(e.g. 71 - 68 = 3); widely separated numbers would give a bigger gap.",
            "Size: a short difference means the operands are near each other -- a small result rules out "
            "pairs that are far apart.",
        ])
    return "Use the result length to bound the operand sizes before trying digit choices."


def _eq_str(L, R):
    return f"{sp(''.join(L))} = {sp(R)}"


def _eqval(nr, op, m):
    """(A,B,val,sval,rhs,ok) for a fully-known arithmetic equation under map m."""
    l0, l1, r0, r1, res, sign, rl, _e = nr
    A, B = m[l0] * 10 + m[l1], m[r0] * 10 + m[r1]
    v = C2.OPS[op](A, B)
    if v is None:
        return A, B, None, None, None, False
    sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
    rhs = 0
    for g in res:
        rhs = rhs * 10 + m[g]
    return A, B, v, sv, rhs, (sv == rhs and sv >= 0)


def _eq_solutions(nr, op, lead, cap):
    gl = []
    for g in [nr[0], nr[1], nr[2], nr[3], *nr[4]]:
        if g not in gl:
            gl.append(g)
    if len(gl) > 7:
        return None
    out = []
    for perm in itertools.permutations(range(10), len(gl)):
        m = dict(zip(gl, perm))
        if any(m[g] == 0 for g in gl if g in lead):
            continue
        _, _, _, _, _, ok = _eqval(nr, op, m)
        if ok:
            out.append(m)
        if len(out) > cap:
            return None
    return out


def _complete(nrs, ops, glyphs, lead, seed, rng):
    """extend seed to a full consistent map using <=2-unknown equation solves +
    injectivity. returns (full_map, lines) or (None, fail_line)."""
    m = dict(seed)
    used = set(m.values())
    if len(used) != len(m):
        return None, "digit reused in the candidate"
    lines = []
    progress = True
    while progress:
        progress = False
        # injectivity: last digit
        rem = [g for g in glyphs if g not in m]
        if len(used) == 9 and len(rem) == 1 and not (rem[0] in lead and (set(range(10)) - used) == {0}):
            d = (set(range(10)) - used).pop()
            m[rem[0]] = d; used.add(d)
            lines.append(rng.choice([
                f"  digit {d} is the only one left, so '{rem[0]}' = {d}",
                f"  '{rem[0]}' = {d} -- the last unused digit",
                f"  only {d} remains, so '{rem[0]}' = {d}",
                f"  every other digit is taken, so '{rem[0]}' = {d}",
                f"  by elimination '{rem[0]}' = {d} (the one digit left)",
                f"  the last free digit is {d}, so '{rem[0]}' = {d}",
                f"  {d} is unused, so '{rem[0]}' = {d}",
                f"  that leaves {d} for '{rem[0]}', so '{rem[0]}' = {d}",
                f"  '{rem[0]}' takes the remaining digit {d}",
            ]))
            progress = True
            continue
        # pick the arithmetic equation with fewest (1..2) unknowns
        best = None
        for (L, R), nr, op in zip(*nrs):
            if op in C2.CONCATS:
                continue
            eg = [nr[0], nr[1], nr[2], nr[3], *nr[4]]
            unk = [g for g in dict.fromkeys(eg) if g not in m]
            if 1 <= len(unk) <= 2 and (best is None or len(unk) < len(best[3])):
                best = ((L, R), nr, op, unk)
        if best is None:
            break
        (L, R), nr, op, unk = best
        avail = [d for d in range(10) if d not in used]
        sols = []
        for combo in itertools.permutations(avail, len(unk)):
            asg = dict(zip(unk, combo))
            if any(asg[g] == 0 for g in unk if g in lead):
                continue
            mm = dict(m); mm.update(asg)
            if _eqval(nr, op, mm)[5]:
                sols.append(asg)
        if not sols:
            l0, l1, r0, r1, res, sign, rl, _e = nr
            # 1-unknown: show the digit it is forced to and why it's rejected
            if len(unk) == 1:
                g = unk[0]
                for d in range(10):
                    mm = dict(m); mm[g] = d
                    A, B, v, sv, rhs, ok = _eqval(nr, op, mm)
                    if ok:                                # arithmetic works at g=d
                        neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
                        why = ("it would lead a number, so it cannot be 0" if (g in lead and d == 0)
                               else f"but {d} is already used")
                        return None, (f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {neg}{sv} needs "
                                      f"'{g}' = {d}, {why} -> reject")
                # no digit makes the arithmetic hold: show the closest computed value
                if all(x in m for x in (l0, l1, r0, r1)):
                    A, B = m[l0] * 10 + m[l1], m[r0] * 10 + m[r1]
                    v = C2.OPS[op](A, B)
                    sv = abs(v) if (v is not None and (op in C2.SIGNED or op in C2.NEGPRE)) else v
                    return None, (f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {sv}, which can't match "
                                  f"the result -> reject")
                return None, f"  {_eq_str(L, R)}: no digit for '{g}' makes the arithmetic hold -> reject"
            # 2-unknown with both operands known -> show the product/sum mismatch
            if all(g in m for g in (l0, l1, r0, r1)):
                A, B = m[l0] * 10 + m[l1], m[r0] * 10 + m[r1]
                v = C2.OPS[op](A, B)
                if v is not None:
                    sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
                    neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
                    if len(str(sv)) != rl:
                        return None, (f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {neg}{sv} has "
                                      f"{len(str(sv))} digits but the result has {rl} symbols -> reject")
                    return None, (f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {neg}{sv}, whose digits "
                                  f"clash with the fixed symbols -> reject")
            return None, f"  {_eq_str(L, R)}: the open symbols can't be filled consistently -> reject"
        if len(sols) == 1:
            asg = sols[0]
            for g, d in asg.items():
                m[g] = d; used.add(d)
            mm = m
            A, B, v, sv, rhs, _ = _eqval(nr, op, m)
            neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
            lines.append(f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {neg}{sv}, {_pk(rng, 'forcing')} "
                         + ", ".join(f"'{g}' = {d}" for g, d in asg.items()))
            progress = True
        # if >1 solutions, leave for a later equation to disambiguate
    if all(g in m for g in glyphs):
        return m, lines
    return None, "  could not finish without a deeper search -> drop"


def _render_concat(eqs, qL, ops, rng):
    """structural trace for a concat-query row: no digit map needed, the answer is
    the operand symbols rearranged."""
    qa, qb, qop, qc, qd = qL
    qt = ops[qop]
    wit = None
    for (L, R) in eqs:
        if L[2] == qop:
            wit = (L, R)
            break
    if wit is None:
        return None, None, None
    ans = (qa + qb + qc + qd) if qt == 'concat_fwd' else (qc + qd + qa + qb)
    L, R = wit
    out = [_pk(rng, 'open')]
    for (a, b) in eqs:
        out.append("  " + _eq_str(a, b))
    out.append(f"Query:  {sp(''.join(qL))}")
    out.append("")
    order = rng.choice(["in order", "operand 1 then operand 2", "left pair then right pair"]) \
        if qt == 'concat_fwd' else rng.choice(["second pair then first", "right pair then left pair",
                                               "operand 2 first, then operand 1"])
    out.append(rng.choice([
        f"Operator '{qop}': in {_eq_str(L, R)} the result is exactly the operand symbols written "
        f"together ({order}), with no arithmetic, so '{qop}' = concatenation.",
        f"Look at {_eq_str(L, R)}: the result just reuses the operand symbols ({order}) -- no "
        f"computation. So '{qop}' is concatenation.",
        f"In {_eq_str(L, R)} the right side is the operands written side by side ({order}); that "
        f"makes '{qop}' = concatenation, not arithmetic.",
        f"'{qop}': the example {_eq_str(L, R)} shows the result is the operands joined ({order}), "
        f"so '{qop}' = concatenation."]))
    out.append(rng.choice([
        f"{_pk(rng, 'apply')} {sp(''.join(qL))}: join the operands {order} -> {sp(ans)}.",
        f"So for {sp(''.join(qL))}, write the operands together {order}: {sp(ans)}.",
        f"Applying that to {sp(''.join(qL))} gives the operands joined {order}: {sp(ans)}."]))
    out.append(f"\\boxed{{{ans}}}")
    return "\n".join(out), ans, 'concat'


def _expand(nrs, ops, glyphs, lead, seed):
    """forward-force from seed (<=1-unknown solves + injectivity), no rendering.
    returns the partial map (may be incomplete)."""
    m = dict(seed)
    used = set(m.values())
    if len(used) != len(m):
        return m
    changed = True
    while changed:
        changed = False
        rem = [g for g in glyphs if g not in m]
        if len(used) == 9 and len(rem) == 1 and not (rem[0] in lead and (set(range(10)) - used) == {0}):
            d = (set(range(10)) - used).pop()
            m[rem[0]] = d; used.add(d); changed = True; continue
        for (L, R), nr, op in zip(*nrs):
            if op in C2.CONCATS:
                continue
            unk = [g for g in dict.fromkeys([nr[0], nr[1], nr[2], nr[3], *nr[4]]) if g not in m]
            if len(unk) == 1:
                g = unk[0]
                c = [d for d in range(10) if d not in used and not (g in lead and d == 0)
                     and _eqval(nr, op, {**m, g: d})[5]]
                if len(c) == 1:
                    m[g] = c[0]; used.add(c[0]); changed = True
    return m


def _reject_witness(eqs, norms, opseq, m, anchor_nr, rng):
    """a real-arithmetic line showing why the (forward-expanded) candidate map m is
    wrong, or None -> then the ROW IS DROPPED. We never emit a constant-string reject:
    every reject shows computed numbers that vary by candidate (no token to memorize)."""
    # (a) an equation with both operands known whose arithmetic contradicts
    for (L, R), nr, op in zip(eqs, norms, opseq):
        if nr is anchor_nr or op in C2.CONCATS:
            continue
        l0, l1, r0, r1, res, sign, rl, _e = nr
        if not all(g in m for g in (l0, l1, r0, r1)):
            continue
        A, B = m[l0] * 10 + m[l1], m[r0] * 10 + m[r1]
        v = C2.OPS[op](A, B)
        if v is None:
            return f"  {_eq_str(L, R)}: {op_expr(op, A, B)} is undefined -> reject"
        sv = abs(v) if (op in C2.SIGNED or op in C2.NEGPRE) else v
        neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
        if sv < 0:
            return f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {sv} is negative but no sign symbol -> reject"
        pre = f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {neg}{sv}"
        k = len(str(sv))
        if k != rl:
            return pre + rng.choice([
                f" has {k} digits but the result has {rl} symbols -> reject",
                f" is {k} digits, the result is {rl} symbols -> reject",
                f" -- that's {k} digits, but the result is {rl} symbols long -> reject",
                f", a {k}-digit number, can't fill a {rl}-symbol result -> reject",
                f" is {k} digits wide; the result needs {rl} -> reject",
                f" ({k} digits) doesn't match the {rl}-symbol result -> reject",
                f", but the result is {rl} symbols and this is {k} digits -> reject",
            ])
        nd = _pk(rng, 'needs')
        for g, ch in zip(res, str(sv)):
            d = int(ch)
            if g in m and m[g] != d:
                return pre + rng.choice([
                    f" {nd} '{g}' = {d}, but '{g}' = {m[g]} -> reject",
                    f", but that wants '{g}' = {d} not {m[g]} -> reject",
                    f" {nd} '{g}' = {d}, which clashes with '{g}' = {m[g]} -> reject",
                    f", so '{g}' should be {d}, but it's already {m[g]} -> reject",
                    f" {nd} '{g}' = {d}; that contradicts '{g}' = {m[g]} -> reject",
                    f", yet '{g}' is {m[g]}, not the {d} this {nd} -> reject",
                ])
            if g not in m and d in m.values():
                other = next(s for s, dd in m.items() if dd == d)
                return pre + rng.choice([
                    f" {nd} '{g}' = {d}, but {d} is already '{other}' -> reject",
                    f" {nd} '{g}' = {d}, and {d} is taken by '{other}' -> reject",
                    f" {nd} '{g}' = {d}, but {d} is used for '{other}' -> reject",
                    f", so '{g}' = {d}, clashing with '{other}' = {d} -> reject",
                    f" {nd} '{g}' = {d}; {d} already belongs to '{other}' -> reject",
                    f", meaning '{g}' = {d}, but '{other}' is {d} -> reject",
                ])
    return None


def render(prompt, ops, rev, deadline_s=10.0):
    rng = _rng_for(prompt)
    eqs, qL = C2.parse(prompt)
    if ops[qL[2]] in C2.CONCATS:                # concat query = structural, no map needed
        return _render_concat(eqs, qL, ops, rng)
    glyphs, lead = set(), set()
    norms = []
    for (L, R) in eqs:
        nr = C2.normalize(L, R, rev, L[2])
        if nr is None:
            return None, None, None
        norms.append(nr)
        glyphs |= {nr[0], nr[1], nr[2], nr[3], *nr[4]}
        lead |= {nr[0], nr[2]}
    opseq = [ops[L[2]] for L, R in eqs]
    bundle = ([e for e in eqs], norms, opseq)
    # ground-truth map (to confirm the survivor) via committed-op propagation
    import propagation as P
    opchars = {L[2] for L, R in eqs} | {qL[2]}
    try:
        eng = P.Eng(eqs, qL, {g: [ops[g]] for g in opchars},
                    opchars <= set('+-*'), time.time() + deadline_s)
        eng.propagate()
    except Exception:
        return None, None, None
    if any(len(eng.dom[s]) != 1 for s in eng.syms):
        return None, None, None
    gold_map = {s: next(iter(eng.dom[s])) for s in eng.syms}
    ans, _ = eng.answer()
    if ans is None:
        return None, None, None

    # choose anchor = arithmetic eq with fewest standalone solutions (<= cap)
    anchor = None
    for (L, R), nr, op in zip(eqs, norms, opseq):
        if op in C2.CONCATS:
            continue
        s = _eq_solutions(nr, op, lead, ANCHOR_CAP)
        if s is None:
            continue
        if anchor is None or len(s) < len(anchor[3]):
            anchor = ((L, R), nr, op, s)
    if anchor is None or len(anchor[3]) > ANCHOR_CAP:
        return None, None, None
    (aL, aR), anr, aop, acands = anchor

    out = [_pk(rng, 'open')]
    for (L, R) in eqs:
        out.append("  " + _eq_str(L, R))
    out.append(f"Query:  {sp(''.join(qL))}")
    out.append("")
    out.append(_pk(rng, 'op_intro'))
    seen = set()
    for ((L, R), op) in zip(eqs, opseq):
        if L[2] in seen:
            continue
        seen.add(L[2])
        opnd = [L[0], L[1], L[3], L[4]]
        if len(R) != 4:
            cc = _pk(rng, 'concat_len').format(R=sp(R), n=len(R))
        elif sorted(R) != sorted(opnd):
            cc = _pk(rng, 'concat_sym').format(rs=' '.join(R), os=' '.join(opnd))
        else:
            out.append(f"  '{L[2]}': {_pk(rng, 'is_concat')}.")
            continue
        out.append(f"  '{L[2]}': {cc}; {_cue(rng, op, len(R))} -> '{L[2]}' = {op_phrase(op)}.")
    out.append("")
    out.append(_pk(rng, 'dir_pre') + (_pk(rng, 'dir_rev') if rev else _pk(rng, 'dir_std')))
    out.append("")
    out.append(_magnitude_block(rng, aop, len(aR)))
    out.append(_pk(rng, 'anchor').format(eq=_eq_str(aL, aR), op=op_phrase(aop)) +
               f" (I allow at most {MAX_ATT} tries, then commit.)")
    # order candidates so the VALID-magnitude ones (incl. the survivor) come first --
    # operands sorted descending so big-operand assignments lead (matches the magnitude reasoning)
    ordered = sorted(acands, key=lambda c: -sum(c.values()))

    def _tag(c):
        return ", ".join(f"{g}={c[g]}" for g in sorted(c))

    survivor_cand = next((c for c in ordered if all(gold_map.get(g) == d for g, d in c.items())), None)
    surv_pos = ordered.index(survivor_cand) if survivor_cand is not None else 999

    if survivor_cand is not None and surv_pos < MAX_ATT:
        # ---- SOLVE: <=3 numbered attempts, last one closes ----
        att = 0
        att_lines = []
        for c in ordered[:surv_pos]:
            w = _reject_witness(eqs, norms, opseq, _expand(bundle, ops, glyphs, lead, c), anr, rng)
            if w is None:
                return None, None, None
            att += 1
            att_lines.append(f"Attempt {att} ({_tag(c)}): {w.strip()}")
        full, slines = _complete(bundle, ops, glyphs, lead, survivor_cand, rng)
        if full is None or full != gold_map:
            return None, None, None
        att += 1
        out.extend(att_lines)
        out.append(f"Attempt {att} ({_tag(survivor_cand)}): {_pk(rng, 'complete')}")
        out.extend(slines)
        out.append("  " + _pk(rng, 'keep'))
        survivor = full
        out.append("")
        out.append(_pk(rng, 'map_hdr') + "  " + ",  ".join(f"{s} = {survivor[s]}" for s in sorted(survivor)))
        out.append(_pk(rng, 'verify'))
        okw = _pk(rng, 'ok')
        for (L, R), nr, op in zip(eqs, norms, opseq):
            if op in C2.CONCATS:
                out.append(f"  {_eq_str(L, R)}: concatenation, {okw}")
                continue
            A, B, v, sv, rhs, _ = _eqval(nr, op, survivor)
            neg = '-' if (op in C2.NEGPRE or (op in C2.SIGNED and v < 0)) else ''
            out.append(f"  {_eq_str(L, R)}: {op_expr(op, A, B)} = {neg}{sv}, {okw}")
        qa, qb, qop, qc, qd = qL
        qt = ops[qop]
        (a, b) = _operands(qL, rev)
        A, B = _decode(a, survivor), _decode(b, survivor)
        v = C2.OPS[qt](A, B); sv = abs(v) if (qt in C2.SIGNED or qt in C2.NEGPRE) else v
        neg = '-' if (qt in C2.NEGPRE or (qt in C2.SIGNED and v < 0)) else ''
        inv = {d: s for s, d in survivor.items()}
        enc = ", ".join(f"{ch}->{inv.get(int(ch), '(new)')}" for ch in str(sv))
        out.append(f"{_pk(rng, 'apply')} {sp(''.join(qL))}: {op_expr(qt, A, B)} = {neg}{sv}. "
                   f"{_pk(rng, 'encode')} ({enc}){', reversed for little-endian' if rev else ''} -> {sp(ans)}")
        out.append(f"\\boxed{{{ans}}}")
        return "\n".join(out), ans, 'solve'

    # ---- BAIL: exactly MAX_ATT witnessed rejects, then STOP and commit a best guess ----
    att = 0
    for c in ordered:
        if att >= MAX_ATT:
            break
        w = _reject_witness(eqs, norms, opseq, _expand(bundle, ops, glyphs, lead, c), anr, rng)
        if w is None:
            continue
        att += 1
        out.append(f"Attempt {att} ({_tag(c)}): {w.strip()}")
    if att < MAX_ATT:
        return None, None, None
    qa, qb, qop, qc, qd = qL
    qt = ops[qop]
    guess = None
    for c in ordered:
        em = _expand(bundle, ops, glyphs, lead, c)
        if not all(g in em for g in (qa, qb, qc, qd)):
            continue
        (a, b) = _operands(qL, rev)
        A, B = _decode(a, em), _decode(b, em)
        if A is None or B is None or A < 10 or B < 10:
            continue
        v = C2.OPS[qt](A, B)
        if v is None:
            continue
        sv = abs(v) if (qt in C2.SIGNED or qt in C2.NEGPRE) else v
        inv = {d: s for s, d in em.items()}
        if all(int(ch) in inv for ch in str(sv)):
            guess = ''.join(inv[int(ch)] for ch in str(sv))
            break
    if guess is None:
        return None, None, None
    out.append(rng.choice([
        f"That's {MAX_ATT} tries and none closes every equation, and a full search won't fit the budget. "
        f"I stop and commit the closest-fitting reading.",
        f"After {MAX_ATT} attempts nothing is consistent everywhere; rather than loop I take the best fit so far.",
        f"None of the first {MAX_ATT} candidates works and I won't search further -- I commit my best guess.",
        f"{MAX_ATT} tries, no clean fit. I stop here and give the most consistent answer I have."]))
    out.append(f"\\boxed{{{guess}}}")
    return "\n".join(out), guess, 'bail'


def _gen(n, seed, emit):
    rng = random.Random(seed)
    tok = CC.tokenizer()
    got = tried = 0
    seen = set()
    while got < n and tried < n * 60:
        tried += 1
        p = CC.gen_puzzle_r5(rng)
        if p is None or p['prompt'] in seen:
            continue
        try:
            cot, ans, kind = render(p['prompt'], p['ops'], p['rev'])
        except Exception:
            continue
        # solve/concat must box the gold; bail intentionally boxes a guess (teaches STOP)
        if cot is None or (kind != 'bail' and ans != p['answer']):
            continue
        nt = len(tok.encode(cot).ids)
        if nt > TOK_CAP:
            continue
        seen.add(p['prompt'])
        got += 1
        emit(p, cot, ans, nt, tried)
    return got, tried


def demo(n=3):
    def e(p, cot, ans, nt, tried):
        print('=' * 72); print(f"gold={ans!r} ntok={nt}"); print(cot); print()
    got, tried = _gen(n, 7, e)
    print(f"[keep rate ~ {got}/{tried}]")


def build(n=3000, seed=950):
    out = os.path.join(ROOT, 'pipeline', 'data', 'crypt_r9', 'crypt_twn_synth.jsonl')
    rows = []
    def e(p, cot, ans, nt, tried):
        rows.append({'id': f'twn-{seed}-{len(rows):05d}', 'category': 'cryptarithm_deduce',
                     'prompt': p['prompt'], 'cot': cot, 'final': ans, 'ntok': nt})
    got, tried = _gen(n, seed, e)
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    import statistics
    nt = [r['ntok'] for r in rows]
    print(f"wrote {len(rows)} rows (keep {got}/{tried} = {got/tried:.2f}) -> {out}")
    print(f"ntok median {int(statistics.median(nt))} p95 {int(sorted(nt)[int(len(nt)*.95)])} max {max(nt)}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    if cmd == 'demo':
        demo(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == 'build':
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 3000,
              int(sys.argv[3]) if len(sys.argv) > 3 else 950)
