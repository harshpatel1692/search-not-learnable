"""Deep hypothesis test: the base burns its whole budget COMPUTING taps (decimal, obsessive re-derivation) and
never SEARCHES. So: hand it the pre-computed tap tables and ask ONLY to find which <=3 taps combine to the output.
If it converges now -> the bottleneck is tap-computation verbosity (CoT should front-load taps; SFT teaches terse
computation). If it still fails -> the search itself is beyond it. Isolates the two failure sources."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N, bit_global as G
N.set_experiment("probe_precomputed")
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))

def taps(x):
    ib=[int(c) for c in x]; out=[]
    for k in range(1,8): out.append((f"ROTL{k}", "".join(str(G.get_source_bit(ib,p,('rot',k))) for p in range(8))))
    for k in range(1,8): out.append((f"ROTR{k}", "".join(str(G.get_source_bit(ib,p,('rot',8-k))) for p in range(8))))
    for k in range(1,8): out.append((f"SHL{k}",  "".join(str(G.get_source_bit(ib,p,('shl',k))) for p in range(8))))
    for k in range(1,8): out.append((f"SHR{k}",  "".join(str(G.get_source_bit(ib,p,('shr',k))) for p in range(8))))
    return out
def table(x): return "  "+ "; ".join(f"{n}={v}" for n,v in taps(x)) + f"  (IDENTITY={x})"

def build_prompt(prompt):
    rows=re.findall(r'([01]{8})\s*->\s*([01]{8})',prompt)
    q=re.search(r'output for:\s*([01]{8})',prompt).group(1)
    s=("A hidden rule maps 8-bit input->output. The rule = a Boolean function (AND/OR/XOR/NAND/NOR/XNOR, "
       "asymmetric NOT-variants, or 3-input majority/choice/parity/nested) applied to up to 3 'taps'. A tap is the "
       "input transformed by ROTL/ROTR/SHL/SHR by some k. The SAME function+taps apply to every output bit (work "
       "on whole 8-bit values). I have ALREADY COMPUTED every tap for you below — do NOT recompute shifts, just "
       "find which 2 or 3 taps and which operation reproduce the output for ALL examples, then apply to the query.\n\n")
    for i,(xi,yi) in enumerate(rows[:4]):
        s+=f"Example {i+1}: input {xi} -> output {yi}\n  taps of input {xi}:\n{table(xi)}\n"
    s+=f"\nQuery input {q}\n  taps of query:\n{table(q)}\n\n"
    s+=("Find the operation and the 2-3 taps such that applying the operation to those taps equals the output in "
        "every example (check by reading the tap values above and combining them). Then apply the same to the "
        "query taps. Give the 8-bit answer in \\boxed{}.")
    return s
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
for sub in ['2tap-XOR-shift','3tap-XNOR-shift']:
    e=ex[sub]
    r=N.ask(build_prompt(e['prompt']), model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=7680, temperature=0.0,
            meta={"sub":sub,"rule":e['expr'],"td":e['td']})
    got=norm(r.get("answer","")); gold=e['answer']
    print(f"{sub:18s} {'CORRECT' if got==gold else 'WRONG':7s} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
