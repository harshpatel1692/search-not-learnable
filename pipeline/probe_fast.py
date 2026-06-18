"""Fast capped probes (<=2200 tokens each, ~<1min): H2/H4 apply-given-rule, H7 op-id, H16 verify.
Decouples whether the base can APPLY/VERIFY/CLASSIFY (vs the hard FIND search)."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_fast")
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))
MT=2200
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
def ask(tag,prompt,gold,check='bits'):
    r=N.ask(prompt, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=MT, temperature=0.0, meta={"tag":tag})
    a=r.get('answer','') or ''
    if check=='bits': got=norm(a); ok=(got==gold)
    else: got=a.strip().upper()[:20]; ok=gold.upper() in got
    print(f"[{tag}] {'OK ' if ok else 'BAD'} got={got!r} gold={gold!r} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)

# H2: apply a 2-tap XOR rule stated explicitly (decimal)
e=ex['2tap-XOR-shift']; q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1)
ask("H2-apply-2tapXOR",
 f"x is an 8-bit number written {q} (binary). Compute output = SHL3(x) XOR SHR2(x), where SHL3(x)=(x*8) mod 256 "
 f"and SHR2(x)=x>>2 (integer divide by 4). Give the 8-bit binary output in \\boxed{{}}.", e['answer'])

# H4: apply a 3-tap rule stated explicitly
e3=ex['3tap-XNOR-shift']; q3=re.search(r'output for:\s*([01]{8})',e3['prompt']).group(1)
ask("H4-apply-3tap",
 f"x is the 8-bit number {q3}. Compute, on whole 8-bit values: A=SHL7(x)=(x*128) mod256; B=SHR6(x)=x>>6; "
 f"C=ROTL1(x)=((x*2)|(x>>7)) mod256; then M = (NOT B) AND C (bitwise, 8-bit); then output = XNOR(A, M) "
 f"(bitwise, =NOT(A XOR M)). Give the 8-bit output in \\boxed{{}}.", e3['answer'])

# H7: identify the OP from bit-counts only
eo=ex['2tap-OR-shift']
rows=re.findall(r'([01]{8})\s*->\s*([01]{8})',eo['prompt'])
tbl="; ".join(f"{a}->{b}" for a,b in rows[:6])
ask("H7-op-id-OR",
 f"A rule combines two shifted copies of the input with ONE bitwise operation. Examples: {tbl}. Compare the "
 f"NUMBER of 1-bits in each output vs its input. If outputs usually have MORE 1s the op is OR; FEWER -> AND; "
 f"about HALF -> XOR. Which single operation is it? Answer one word: AND, OR, or XOR.", "OR", check='word')

# H16: verify a CORRECT rule, then a WRONG rule
ex2=ex['2tap-XOR-shift']; rws=re.findall(r'([01]{8})\s*->\s*([01]{8})',ex2['prompt'])[:5]
tbl2="; ".join(f"{a}->{b}" for a,b in rws)
ask("H16-verify-correct",
 f"Candidate rule: output = SHL3(x) XOR SHR2(x) (SHL3=(x*8)mod256, SHR2=x>>2). Does it reproduce ALL these "
 f"examples? {tbl2}. Answer YES or NO.", "YES", check='word')
ask("H16-verify-wrong",
 f"Candidate rule: output = SHL1(x) XOR SHR1(x) (SHL1=(x*2)mod256, SHR1=x>>1). Does it reproduce ALL these "
 f"examples? {tbl2}. Answer YES or NO.", "NO", check='word')
