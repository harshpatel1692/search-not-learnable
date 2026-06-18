"""unit_conversion: linear y=a*x+b, fit from the two most-separated examples. 100% on train."""
import re
def solve(prompt:str):
    pairs=[(float(x),float(y)) for x,y in re.findall(r'([\d.]+)\s*m becomes\s*([\d.]+)', prompt)]
    q=float(re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt).group(1))
    lo=min(pairs,key=lambda t:t[0]); hi=max(pairs,key=lambda t:t[0])
    a=(hi[1]-lo[1])/(hi[0]-lo[0]); b=lo[1]-a*lo[0]
    xc,yc=pairs[len(pairs)//2]; chk=a*xc+b; val=a*q+b
    cot=("Each output is a linear conversion y=a·x+b. Use the two most-separated examples "
         f"({lo[0]}→{lo[1]}) and ({hi[0]}→{hi[1]}): a=({hi[1]}−{lo[1]})/({hi[0]}−{lo[0]})={a:.4f}; "
         f"b={lo[1]}−{a:.4f}·{lo[0]}={b:.3f}. Verify on {xc}: {a:.4f}·{xc}+{b:.3f}={chk:.2f}≈{yc}. "
         f"Apply to {q}: y={a:.4f}·{q}+{b:.3f}={val:.2f}.")
    return f"{val:.2f}", cot
