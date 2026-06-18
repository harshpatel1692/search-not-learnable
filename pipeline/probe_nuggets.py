"""More nuggets (fast capped, <1min): directed list-match, asymmetric/XNOR apply, 1-tap SHIFT solve open-ended,
all-zero structural inference, and chain tolerance. Find what else shapes the CoT."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_nuggets")
M=0xFF
def shr(v,k):return v>>k
def shl(v,k):return (v<<k)&M
def i2b(v):return format(v,'08b')
def ask(tag,prompt,gold,mt=2000,check='word'):
    r=N.ask(prompt,model="nvidia/nemotron-3-nano-30b-a3b",max_tokens=mt,temperature=0.0,meta={"tag":tag})
    a=(r.get('answer','') or '')
    if check=='word': got=a.strip().upper(); ok=gold.upper() in got
    elif check=='bits': mm=re.findall(r'[01]{8}',a); got=mm[-1] if mm else a.strip(); ok=(got==gold)
    else: got=a.strip(); ok=(gold in got)
    print(f"[{tag}] {'OK ' if ok else 'BAD'} got={got[:26]!r} gold={gold!r} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)

# A: directed list-match (pick from explicit candidates) — core of verify-scan
X=0b10101001; lst="; ".join(f"SHR{k}={i2b(shr(X,k))}" for k in range(1,8))
ask("A-list-match", f"Given these values: {lst}. Which label equals 00101010? Answer the label only.", "SHR2")

# B: asymmetric op apply (NOT A) AND B
A=0b10110010; B=0b11001100; gB=i2b((~A & M)&B)
ask("B-notA-and-B", f"Compute, on 8-bit values, (NOT A) AND B where A={i2b(A)}, B={i2b(B)}. NOT flips all 8 bits. "
    f"Give the 8-bit result in \\boxed{{}}.", gB, check='bits')

# C: XNOR apply
gC=i2b((~(A^B))&M)
ask("C-xnor", f"Compute XNOR(A,B) = NOT(A XOR B) on 8-bit values, A={i2b(A)}, B={i2b(B)}. Give 8-bit in \\boxed{{}}.", gC, check='bits')

# E: structural inference from all-zero example
ask("E-zero-struct", "A bit rule maps input->output. One example is 00000000 -> 00000000. Does this prove the rule has "
    "no constant flip (no XOR-with-a-fixed-mask term)? Answer YES or NO.", "YES")

# F: 1-tap SHIFT solve, open-ended (only ~21 candidates)
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))
e=ex.get('1tap-IDENTITY-shift') or ex['1tap-IDENTITY']
td=list(e['td'].values())[0]; tn={'shl':'SHL','shr':'SHR','rot':'ROTL'}[td[0]]+str(td[1])
ask("F-1tap-shift", "The output is a SINGLE transform of the input (one of: ROTLk, SHLk, SHRk for k=1..7; shifts fill "
    f"with 0). Find which one. Examples:\n{e['prompt']}\nAnswer just the transform name (e.g. SHL3).", tn)
