"""crypt_guess renderer driver (r4/r5) — real policy-hit rows + synthetic top-up.

r4 bet (distinct-ops structure): glyph ops are drawn WITHOUT replacement, so the
guess bet = the most common generator rule NOT already used by the example glyphs
(first op in C2.PRIORITY outside the example attributions, restricted to
RVOCAB + concat_fwd). Measured hit-rate on the 161 gold guess rows:
0.205 (old always-sub bet) -> 0.304. The bet itself lives in
render_trace_r4 (cryptarithm_cot.py, is_guess branch); this driver only feeds it.

r5 bet (two-regime glyph model): on CANONICAL guess rows (all operator glyphs
literally '+','-','*') the unseen query glyph's CHAR reveals its family — bet
the top famA[q_glyph] op not used by the examples; scrambled rows keep the
exclusion bet over the opB order. Measured qt-hit on the 162 gold guess rows:
0.333 -> 0.593 (100/162 guess rows are canonical). The bet lives in
render_trace_r5 (is_guess branch); run `python3 crypt_guess_render.py r5 [n]`.

Only policy-HIT rows are rendered (boxed == gold, enforced inside the renderer)
so the training signal is consistent. Real rows reuse the gold-conditioned meta
from pipeline/data/cryptarithm_gold_meta.jsonl (cross-checks + lint); synthetic
rows are force_guess generator draws pre-filtered to hidden-draw == policy bet.
Output: pipeline/data/crypt_r4/crypt_guess.jsonl (r4, lint_r4-clean) or
pipeline/data/crypt_r5/crypt_guess.jsonl (r5, lint_r5-clean).
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
from synth import cryptarithm_cot as CC
from solvers import cryptarithm2 as C2

BETPOOL = CC.RVOCAB + ['concat_fwd']

def exclusion_bet(used_ops):
    """most common generator rule not used by the examples (the r4 guess bet)."""
    return next((o for o in C2.PRIORITY if o not in used_ops and o in BETPOOL), None)

def family_bet(p):
    """r5 bet for a synthetic force_guess draw p (mirrors render_trace_r5)."""
    used = {op for g, op in p['ops'].items() if g != p['qglyph']}
    chars = set(p['ops'].keys())
    if all(c in CC.CANON_CHARS for c in chars):
        return next((o for o in CC.famA_order(p['qglyph']) if o not in used), None)
    return next((o for o in CC.opB_order() if o not in used and o in BETPOOL), None)

def gold_guess_meta():
    out = {}
    with open(f"{ROOT}/pipeline/data/cryptarithm_gold_meta.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d['cat'] == 'cryptarithm_guess':
                out[d['id']] = d
    return out

def main(n_synth=120, seed=21, out=None):
    tk = CC.tokenizer()
    vids = CC.val_ids()
    gm = gold_guess_meta()
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    # --- real guess rows (policy hits only; render_trace_r4 drops bet != gold) ---
    for r in CC.load_real('cryptarithm_guess'):
        if r['id'] in vids:
            continue
        d = gm.get(r['id'])
        if d is None or not d.get('ok'):
            drop['gold-unsolvable'] = drop.get('gold-unsolvable', 0) + 1
            continue
        meta = {'mapping': d['mapping'], 'rev': d['rev'],
                'ops': d['ops'], 'qop': d['qop']}
        CC._emit4(kept, drop, tl, infos, r['prompt'], r['answer'].strip(), meta,
                  tk, r['id'], 'cryptarithm_guess')
    n_real = len(kept)
    # --- synthetic top-up: force_guess; keep only generator draws that hit the
    # exclusion bet (cheap pre-filter; the render itself re-enforces boxed == gold)
    rng = random.Random(seed)
    tries = 0
    while len(kept) - n_real < n_synth and tries < n_synth * 80:
        tries += 1
        p = CC.gen_puzzle(rng, force_guess=True)
        if not p:
            continue
        used = {op for g, op in p['ops'].items() if g != p['qglyph']}
        if p['qop'] != exclusion_bet(used):
            continue
        CC._emit4(kept, drop, tl, infos, p['prompt'], p['answer'],
                  CC.synth_meta(p), tk,
                  f"synth4g-s{seed}-{len(kept) - n_real:04d}", 'cryptarithm_guess')
    if out is None:
        out = f"{ROOT}/pipeline/data/crypt_r4/crypt_guess.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    toks = sorted(x['ntok'] for x in kept)
    print(f"crypt_guess r4: real policy-hits {n_real}, synth {len(kept)-n_real},"
          f" total {len(kept)} -> {out}")
    print(f"drops: {json.dumps(drop)}")
    print(f"episodes: {json.dumps(CC.episode_stats(infos))}")
    if toks:
        print(f"tok med={toks[len(toks)//2]} p95={toks[int(len(toks)*.95)]} max={toks[-1]}")

def main5(n_synth=120, seed=22, out=None):
    """r5 flow: render_trace_r5 family-aware bet, lint_r5, two-regime synth."""
    import zlib, random as _rnd
    tk = CC.tokenizer()
    vids = CC.val_ids()
    gm = gold_guess_meta()
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    n_attempt = 0
    for r in CC.load_real('cryptarithm_guess'):
        if r['id'] in vids:
            continue
        d = gm.get(r['id'])
        if d is None or not d.get('ok'):
            drop['gold-unsolvable'] = drop.get('gold-unsolvable', 0) + 1
            continue
        n_attempt += 1
        meta = {'mapping': d['mapping'], 'rev': d['rev'],
                'ops': d['ops'], 'qop': d['qop']}
        CC._emit5(kept, drop, tl, infos, r['prompt'], r['answer'].strip(), meta,
                  tk, r['id'], 'cryptarithm_guess')
    n_real = len(kept)
    rng = random.Random(seed)
    tries = 0
    while len(kept) - n_real < n_synth and tries < n_synth * 80:
        tries += 1
        p = CC.gen_puzzle_r5(rng, force_guess=True)
        if not p:
            continue
        if p['qop'] != family_bet(p):
            continue        # cheap pre-filter; the render re-enforces boxed == gold
        CC._emit5(kept, drop, tl, infos, p['prompt'], p['answer'],
                  CC.synth_meta(p), tk,
                  f"synth5g-s{seed}-{len(kept) - n_real:04d}", 'cryptarithm_guess')
    if out is None:
        out = f"{ROOT}/pipeline/data/crypt_r5/crypt_guess.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    toks = sorted(x['ntok'] for x in kept)
    canon = sum(i.get('canonical', 0) for i in infos)
    print(f"crypt_guess r5: real policy-hits {n_real}/{n_attempt} attempted,"
          f" synth {len(kept)-n_real}, total {len(kept)} -> {out}")
    print(f"canonical traces: {canon}/{len(kept)}")
    print(f"drops: {json.dumps(drop)}")
    print(f"episodes: {json.dumps(CC.episode_stats(infos))}")
    if toks:
        print(f"tok med={toks[len(toks)//2]} p95={toks[int(len(toks)*.95)]} max={toks[-1]}")

def main6(n_synth=120, seed=23, out=None):
    """r6 flow: r5 family-aware bet + FIX A (positional table construction) and
    FIX B (monotone try-counters) via render_trace_r6/lint_r6. No bail-outs in
    guess (FIX C is a deduce-only construction)."""
    tk = CC.tokenizer()
    vids = CC.val_ids()
    gm = gold_guess_meta()
    kept, drop, infos = [], {}, []
    tl = {1: [], 2: [], 3: []}
    n_attempt = 0
    for r in CC.load_real('cryptarithm_guess'):
        if r['id'] in vids:
            continue
        d = gm.get(r['id'])
        if d is None or not d.get('ok'):
            drop['gold-unsolvable'] = drop.get('gold-unsolvable', 0) + 1
            continue
        n_attempt += 1
        meta = {'mapping': d['mapping'], 'rev': d['rev'],
                'ops': d['ops'], 'qop': d['qop']}
        CC._emit6(kept, drop, tl, infos, r['prompt'], r['answer'].strip(), meta,
                  tk, r['id'], 'cryptarithm_guess')
    n_real = len(kept)
    rng = random.Random(seed)
    tries = 0
    while len(kept) - n_real < n_synth and tries < n_synth * 80:
        tries += 1
        p = CC.gen_puzzle_r5(rng, force_guess=True)
        if not p:
            continue
        if p['qop'] != family_bet(p):
            continue        # cheap pre-filter; the render re-enforces boxed == gold
        CC._emit6(kept, drop, tl, infos, p['prompt'], p['answer'],
                  CC.synth_meta(p), tk,
                  f"synth6g-s{seed}-{len(kept) - n_real:04d}", 'cryptarithm_guess')
    if out is None:
        out = f"{ROOT}/pipeline/data/crypt_r6/crypt_guess.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        for k in kept:
            f.write(json.dumps(k) + '\n')
    toks = sorted(x['ntok'] for x in kept)
    canon = sum(i.get('canonical', 0) for i in infos)
    print(f"crypt_guess r6: real policy-hits {n_real}/{n_attempt} attempted,"
          f" synth {len(kept)-n_real}, total {len(kept)} -> {out}")
    print(f"canonical traces: {canon}/{len(kept)}")
    print(f"drops: {json.dumps(drop)}")
    print(f"episodes: {json.dumps(CC.episode_stats(infos))}")
    if toks:
        print(f"tok med={toks[len(toks)//2]} p95={toks[int(len(toks)*.95)]} max={toks[-1]}")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'r6':
        main6(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    elif len(sys.argv) > 1 and sys.argv[1] == 'r5':
        main5(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
