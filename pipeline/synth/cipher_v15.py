"""cipher v15 renderer — Ali-grammar traces with the FINAL-REDECODE fix.

The 0.86 adapter's cipher grammar (99.2%) enumerates the 77-word vocab in-trace,
aligns every example pair char-by-char, builds Mapping/Inverse/Unknown tables,
decodes the query, vocab-snaps unknown letters... but never RE-DECODES the query
words after snap resolutions land (its 12 fails are exactly cross-word
resolutions assembled from memory: '?atches' + x->w boxed as 'chases').

This renderer reproduces that grammar verbatim (incl. the proven 「」 brackets)
and appends a deterministic final re-decode block before \boxed{}.

Usage:
  python3 pipeline/synth/cipher_v15.py real    # render all real train rows -> data/v15/cipher_real.jsonl
  python3 pipeline/synth/cipher_v15.py synth N # render N synthetic rows     -> data/v15/cipher_synth.jsonl
"""
import csv, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
csv.field_size_limit(10**8)

# ---------------------------------------------------------------- vocab + rows
def load_rows():
    return [r for r in csv.DictReader(open(f"{ROOT}/competition_dataset/train_categorized.csv"))
            if r['category'] == 'cipher']

def build_vocab(rows):
    v = set()
    for r in rows:
        for line in r['prompt'].splitlines():
            if '->' in line:
                for w in line.split('->', 1)[1].split():
                    v.add(w.lower())
        for w in r['answer'].split():
            v.add(w.lower())
    return sorted(v)

def parse_prompt(prompt):
    """-> (examples [(cipher_words, plain_words)], query_words)"""
    ex = []
    for line in prompt.splitlines():
        if '->' in line:
            l, r = line.split('->', 1)
            ex.append((l.split(), r.split()))
    q = re.search(r'decrypt the following text:\s*(.+)$', prompt, re.M).group(1).strip().split()
    return ex, q

# ---------------------------------------------------------------- solver (oracle)
def solve(examples, query, vocab):
    """example-alignment + vocab constraint propagation. Returns full map for query letters."""
    m, inv = {}, {}
    for cws, pws in examples:
        for cw, pw in zip(cws, pws):
            if len(cw) == len(pw):
                for a, b in zip(cw, pw):
                    m[a] = b; inv[b] = a
    changed = True
    while changed:
        changed = False
        for cw in query:
            if all(c in m for c in cw):
                continue
            cands = candidates(cw, m, inv, vocab)
            if len(cands) == 1:
                w = cands[0]
                for i, c in enumerate(cw):
                    if c not in m:
                        m[c] = w[i]; inv[w[i]] = c; changed = True
    return m

def candidates(cw, m, inv, vocab):
    out = []
    for w in vocab:
        if len(w) != len(cw):
            continue
        ok, mm = True, {}
        for i, c in enumerate(cw):
            if c in m:
                if m[c] != w[i]: ok = False; break
            else:
                if w[i] in inv and inv[w[i]] != c: ok = False; break
                if c in mm and mm[c] != w[i]: ok = False; break
                mm[c] = w[i]
        if ok:
            out.append(w)
    return out

# ---------------------------------------------------------------- renderer
def dashes(w): return '-'.join(w)

def render(prompt, vocab):
    examples, query = parse_prompt(prompt)
    L = []
    L.append("We need to find the encryption mapping from the examples. It looks like a substitution cipher.")
    L.append("")
    L.append("Listing the input words:")
    L.append("")
    for cws, _ in examples:
        L.append(f"「{' '.join(cws)}」")
        for w in cws[:1]: L.append(w)
        for w in cws[1:]: L.append(f" {w}")
        L.append("")
    L.append(f"「 {' '.join(query)}」")
    L.append(f" {query[0]}")
    for w in query[1:]: L.append(f" {w}")
    L.append("")
    L.append("Breaking down into characters:")
    L.append("")
    for cws, _ in examples:
        L.append(f"「{' '.join(cws)}」")
        for w in cws: L.append(dashes(w))
        L.append("")
    L.append(f"「{' '.join(query)}」")
    for w in query: L.append(dashes(w))
    L.append("")
    L.append("Wonderland words:")
    L.extend(vocab)
    L.append("")
    # per-example alignment
    m, inv = {}, {}
    for cws, pws in examples:
        L.append(f"「{' '.join(cws)}」 -> 「{' '.join(pws)}」 / " + ' '.join(f"「{w}」" for w in pws) + ":")
        first = True
        for cw, pw in zip(cws, pws):
            L.append(f"「{'' if first else ' '}{cw}」->「{pw}」")
            first = False
            L.append(f"{dashes(cw)}->{dashes(pw)}")
            for a, b in zip(cw, pw):
                L.append(f"{a}->{b}")
                m[a] = b; inv[b] = a
        L.append("")
    # mapping tables
    L.append("Mapping so far")
    for c in "abcdefghijklmnopqrstuvwxyz":
        L.append(f"{c}->{m.get(c, '?')}")
    L.append("Inverse mapping")
    for c in "abcdefghijklmnopqrstuvwxyz":
        L.append(f"{c}->{inv.get(c, '?')}")
    unknown = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in m]
    L.append("Unknown characters")
    L.extend(unknown)
    L.append("Unmapped target letters")
    L.extend([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in inv])
    L.append("")
    # decode query with partials
    L.append(f"Now decrypting 「 {' '.join(query)}」:")
    partial = {}
    for qi, cw in enumerate(query):
        L.append(f"「{' ' if qi == 0 else ' '}{cw}」" if qi == 0 else f"「 {cw}」")
        L.append(dashes(cw))
        dec = []
        for c in cw:
            L.append(f"{c}->{m.get(c, '?')}")
            dec.append(m.get(c, '?'))
        if '?' in dec:
            shown = '-'.join(f"({c})" if m.get(c) is None else m[c] for c in cw)
            L.append(f"{shown}->「{''.join(dec)}」")
        else:
            L.append(f"{dashes(''.join(dec))}->「{''.join(dec)}」")
        partial[qi] = dec
    L.append("")
    L.append("Iterating over the unknown letters to see if they are in the question")
    qletters = set(c for w in query for c in w)
    need = []
    for c in unknown:
        inq = c in qletters
        L.append(f"{c} ({'yes' if inq else 'no'})")
        if inq: need.append(c)
    L.append("")
    if need:
        L.append("The unknown letters")
        L.extend(need)
        L.append("")
        L.append("Let me find the best matching wonderland words:")
        L.append("")
        # sequential vocab snap (resolutions propagate)
        for qi, cw in enumerate(query):
            if all(c in m for c in cw):
                continue
            dec = ''.join(m.get(c, '?') for c in cw)
            shown = '-'.join(f"({c})" if c not in m else m[c] for c in cw)
            cands = candidates(cw, m, inv, vocab)
            L.append(f"{shown} / 「{dec}」 could be: " + ', '.join(f"「{w}」" for w in cands) if cands
                     else f"{shown} / 「{dec}」 could be: (none)")
            rej = [w for w in cands if w != cands[0]] if len(cands) > 1 else []
            # bijection rejection reasons for all but the surviving candidate
            survivors = []
            for w in cands:
                bad = []
                for i, c in enumerate(cw):
                    if c not in m and w[i] in inv and inv[w[i]] != c:
                        bad.append(w[i])
                if bad:
                    L.append(f"Rejected: 「{w}」 ({', '.join(bad)} already mapped)")
                else:
                    survivors.append(w)
            if not rej and len(cands) <= 1:
                L.append("Rejected: None")
            pick = survivors[0] if survivors else (cands[0] if cands else None)
            if pick is None:
                continue
            L.append(f"Remaining: 「{pick}」")
            L.append(f"Best match: 「{pick}」")
            L.append("New mappings")
            for i, c in enumerate(cw):
                if c not in m:
                    L.append(f"{c}->{pick[i]}")
                    m[c] = pick[i]; inv[pick[i]] = c
            L.append("")
    # >>> THE FIX: deterministic final re-decode of every query word <<<
    L.append("Final mapping applied to the query:")
    final_words = []
    for cw in query:
        dec = ''.join(m.get(c, '?') for c in cw)
        L.append(f"「{cw}」-> {dashes(dec)} -> 「{dec}」")
        final_words.append(dec)
    ans = ' '.join(final_words)
    L.append("")
    L.append("I will now return the answer in \\boxed{}")
    L.append("The answer in \\boxed is")
    L.append(f"\\boxed{{{ans}}}")
    return '\n'.join(L), ans

# ---------------------------------------------------------------- synthetic
SENT_CACHE = None
def synth_row(vocab, rows, rng):
    """fresh substitution alphabet over real sentence distribution."""
    global SENT_CACHE
    if SENT_CACHE is None:
        SENT_CACHE = []
        for r in rows:
            ex, q = parse_prompt(r['prompt'])
            for _, pws in ex: SENT_CACHE.append(pws)
            SENT_CACHE.append(r['answer'].split())
    perm = list("abcdefghijklmnopqrstuvwxyz")
    while True:
        rng.shuffle(perm)
        enc = dict(zip("abcdefghijklmnopqrstuvwxyz", perm))
        if all(enc[c] != c for c in enc):  # derangement, like the generator
            break
    n_ex = rng.choice([3, 4, 5])
    sents = rng.sample(SENT_CACHE, n_ex + 1)
    def encw(ws): return ' '.join(''.join(enc[c] for c in w) for w in ws)
    lines = ["In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:"]
    for s in sents[:-1]:
        lines.append(f"{encw(s)} -> {' '.join(s)}")
    lines.append(f"Now, decrypt the following text: {encw(sents[-1])}")
    return '\n'.join(lines), ' '.join(sents[-1])

# ---------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'real'
    rows = load_rows()
    vocab = build_vocab(rows)
    assert len(vocab) == 77, f"vocab size {len(vocab)} != 77"
    os.makedirs(f"{ROOT}/pipeline/data/v15", exist_ok=True)
    val_ids = set()
    try:
        for line in open(f"{ROOT}/pipeline/data/val.jsonl"):
            val_ids.add(json.loads(line).get('id'))
    except FileNotFoundError:
        pass

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/competition_dataset/tokenizer.json")

    if mode == 'real':
        out, nok, toks = [], 0, []
        for r in rows:
            cot, ans = render(r['prompt'], vocab)
            good = ans == r['answer'].strip()
            nok += good
            if good and r['id'] not in val_ids:
                toks.append(len(tok.encode(cot).ids))
                out.append({"id": r['id'], "category": "cipher", "prompt": r['prompt'],
                            "cot": cot, "final": ans})
        with open(f"{ROOT}/pipeline/data/v15/cipher_real.jsonl", 'w') as f:
            for o in out: f.write(json.dumps(o) + '\n')
        toks.sort()
        print(f"real: {nok}/{len(rows)} correct ({100*nok/len(rows):.2f}%); "
              f"written {len(out)} (val excluded); tok med={toks[len(toks)//2]} "
              f"p95={toks[int(len(toks)*.95)]} max={toks[-1]}")
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
        rng = random.Random(15)
        out, toks = [], []
        for i in range(n):
            prompt, gold = synth_row(vocab, rows, rng)
            cot, ans = render(prompt, vocab)
            if ans != gold:
                continue  # only keep verified traces
            toks.append(len(tok.encode(cot).ids))
            out.append({"id": f"synthcipher{i:05d}", "category": "cipher", "prompt": prompt,
                        "cot": cot, "final": ans})
        with open(f"{ROOT}/pipeline/data/v15/cipher_synth.jsonl", 'w') as f:
            for o in out: f.write(json.dumps(o) + '\n')
        toks.sort()
        print(f"synth: {len(out)}/{n} verified+written; tok med={toks[len(toks)//2]} "
              f"p95={toks[int(len(toks)*.95)]} max={toks[-1]}")

if __name__ == '__main__':
    main()
