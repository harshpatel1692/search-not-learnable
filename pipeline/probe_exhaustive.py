"""Give the base EXHAUSTIVE context about the bit_manip generator, then a real puzzle, and let it reason FREELY
(no prescribed procedure). Goal: observe HOW it naturally attacks the problem with full knowledge -> informs CoT.
Captures the full reasoning trace (logged to nvidia_logs/probe_exhaustive.jsonl)."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_exhaustive")
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))

CONTEXT="""You are solving a "bit manipulation" puzzle. Below is COMPLETE background on exactly how these puzzles are generated, then the puzzle. Reason carefully about the most efficient strategy for YOU, explain your strategy, then carry it out.

=== HOW THE PUZZLE IS GENERATED (exact) ===
- A single secret rule maps an 8-bit input to an 8-bit output. The SAME rule produces every output bit (it is one fixed Boolean circuit applied to the whole 8-bit value).
- The rule reads up to THREE "taps". A tap is a transformed copy of the whole 8-bit input, using ONE fixed transform:
    * ROTL k / ROTR k : rotate all 8 bits left/right by k (1..7); bits wrap around the ends.
    * SHL k / SHR k   : shift left/right by k (1..7); vacated positions are filled with 0 (bits do NOT wrap, they fall off).
- The rule combines the taps with a Boolean function from this library:
    * 0 taps: constant 00000000 or 11111111.
    * 1 tap : IDENTITY (output = the tap) or NOT (bitwise complement).
    * 2 taps A,B: AND, OR, XOR, NAND, NOR, XNOR, or asymmetric ((NOT A) AND B), (A AND (NOT B)), ((NOT A) OR B), (A OR (NOT B)).
    * 3 taps A,B,C: MAJORITY(A,B,C); CHOICE (if A then B else C); PARITY (A XOR B XOR C); or nested ((A op B) op C).
- Because each tap is a rigid shift/rotation, the chosen function is applied bitwise across all 8 positions at once -> you can reason on WHOLE 8-bit values, not individual bit positions.

=== STATISTICS FROM 1602 REAL PUZZLES ===
- Taps: 56% use 2 taps, 34% use 3 taps, 9% use 1 tap, <1% constant.
- 92% involve at least one SHIFT (expect runs of 0 at one end of a tap).
- Operations: XOR ~28%, OR ~22%, XNOR ~18%, AND ~12%, IDENTITY ~9%, asymmetric NOT-variants ~11%.

=== HOW TO FIND THE RULE (logic) ===
- The correct rule must reproduce EVERY example exactly (all 8 bits, every example). That is a very strong constraint; usually exactly one rule fits.
- Bit-count signal: OR makes MORE 1s than its inputs, AND makes FEWER, XOR about half, XNOR = NOT(XOR).
- XOR is invertible: if OUT = XOR(TA,TB) then OUT XOR TA = TB. So compute OUT XOR (a candidate tap) and see if the result is itself a clean shift/rotation of the input.
- Shifts show as runs of 0s at one end; rotations preserve the number of 1s.
- You can compute transformed copies as whole 8-bit values (e.g. SHL3 of 10101001 = 01001000) reliably.

=== THE PUZZLE ===
"""
TAIL="\n\nThink step by step about the most EFFICIENT way to identify the rule with the fewest computations, state your strategy, execute it, and give the final 8-bit output in \\boxed{}."

def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
for sub in ['2tap-XOR-shift','3tap-XNOR-shift']:
    e=ex[sub]
    r=N.ask(CONTEXT+e['prompt']+TAIL, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=7680, temperature=0.0,
            meta={"sub":sub,"rule":e['expr'],"td":e['td']})
    got=norm(r.get("answer","")); gold=e['answer']
    print(f"{sub:18s} {'CORRECT' if got==gold else 'WRONG':7s} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
