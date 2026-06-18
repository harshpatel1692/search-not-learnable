"""What ELSE does the base already know? Fast capped probes (<=2000 tok, <1min) for skills a generate-and-verify
CoT would lean on: popcount, permutation-detection, REVERSE-RECOGNITION (is V a shift of X? which?), and one
INVERTIBILITY step (OUT^tap -> recognize as a tap). These decide if the narrowed loop is built on native skills."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_native")
M=0xFF
def shl(v,k):return (v<<k)&M
def shr(v,k):return v>>k
def rotl(v,k):k&=7;return ((v<<k)|(v>>(8-k)))&M if k else v
def i2b(v):return format(v,'08b')
def ask(tag,prompt,gold,mt=1800,check='word'):
    r=N.ask(prompt, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=mt, temperature=0.0, meta={"tag":tag})
    a=(r.get('answer','') or '')
    if check=='word': got=a.strip().upper(); ok=gold.upper() in got
    elif check=='bits': mm=re.findall(r'[01]{8}',a); got=mm[-1] if mm else a.strip(); ok=(got==gold)
    else: got=a.strip(); ok=(gold in got)
    print(f"[{tag}] {'OK ' if ok else 'BAD'} got={got[:24]!r} gold={gold!r} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)

# P-popcount (sanity)
ask("popcount","Count the number of 1-bits in each: 10110110, 01000001, 11111110. Answer as three numbers separated by spaces.",
    "5 2 7", check='sub')

# P-permutation-detect: do these outputs preserve #1s vs input? (rotation example) yes
ask("perm-detect","For each pair, does the OUTPUT have the SAME number of 1-bits as the INPUT? 10101001->01010011; "
    "00110011->01100110; 11110000->11100001. Answer YES or NO (one word).","YES")

# P-reverse-recognition (KEY): is V a shift/rotation of X, and which?
X=0b10101001; V=shr(X,2)
ask("rev-recognize", f"Let X={i2b(X)}. Consider these transforms of X: SHL k=(X*2^k)mod256, SHR k=X>>k, ROTLk=rotate. "
    f"Is the value {i2b(V)} equal to one of them? If yes answer which (like SHR2). Answer the transform name only.","SHR2")

# P-invertibility step (KEY): OUT ^ SHL3(X) -> recognize. rule XOR(SHL3,SHR2) so OUT^SHL3 = SHR2(X)
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))['2tap-XOR-shift']
rows=re.findall(r'([01]{8})\s*->\s*([01]{8})',ex['prompt']); x1,y1=rows[0]; X1=int(x1,2); Y1=int(y1,2)
ask("invert-step", f"X={x1}, OUT={y1}. Compute R = OUT XOR SHL3(X), where SHL3(X)=(X*8)mod256 (8-bit). Then say whether R "
    f"equals a SHR (right shift) of X, and which one (e.g. SHR2). Give R as 8 bits and the transform name.",
    "SHR2", check='word')
