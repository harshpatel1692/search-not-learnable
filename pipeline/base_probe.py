"""Probe the COMPETITION BASE model (nvidia/nemotron-3-nano-30b-a3b) on bit-op competence:
Tier1 primitives, Tier2 chained execution (drift), Tier3 full task (does it search or give up?).
Validates the curriculum hypothesis: teach only the blocks the base is weak on."""
import sys,re; sys.path.insert(0,'pipeline')
import nvidia_api as N
M=0xFF; B="nvidia/nemotron-3-nano-30b-a3b"
def rotl(x,k):k&=7;return ((x<<k)|(x>>(8-k)))&M if k else x
def b8(x):return format(x,'08b')
def norm(s):  # pull last 8-bit token from answer
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()

N.set_experiment("base_probe")
T1=[("ROTL3 01011010", "Rotate the 8-bit value 01011010 left by 3 positions (bits that fall off the left wrap to the right). Give only the 8-bit result.", b8(rotl(0b01011010,3))),
    ("XOR",           "Compute the bitwise XOR of the two 8-bit values 01011010 and 11001100. Give only the 8-bit result.", b8(0b01011010^0b11001100)),
    ("SHL1",          "Shift the 8-bit value 11010011 left by 1 (drop the overflow bit, fill with 0 on the right). Give only the 8-bit result.", b8((0b11010011<<1)&M)),
    ("NOT",           "Compute the bitwise NOT (complement each bit) of the 8-bit value 01011010. Give only the 8-bit result.", b8((~0b01011010)&M))]
# Tier2: execute a known 3-op chain, show steps (tests bit tracking across composition)
v=0b01011010
s1=(v<<1)&M; s2=s1^(s1>>6); s3=s2^M
T2=[("chain shl1;x^shr6;notall",
     "Start with the 8-bit value 01011010. Step 1: shift left by 1 (drop overflow). Step 2: XOR that result with itself-shifted-right-by-6. Step 3: XOR that with 11111111. Show each step and give the final 8-bit result.", b8(s3))]
res=[]
for nm,p,gold in T1+T2:
    r=N.ask(p, model=B, max_tokens=3000, temperature=0.0, meta={"tier":"prim","name":nm})
    got=norm(r.get("answer","")); ok=(got==gold)
    res.append((nm,ok,got,gold,len(r.get("reasoning",''))))
    print(f"[{'PASS' if ok else 'FAIL'}] {nm:24s} got={got} gold={gold} think={len(r.get('reasoning',''))}c", flush=True)
print("PRIMITIVE+CHAIN:", sum(1 for x in res if x[1]), "/", len(res))
