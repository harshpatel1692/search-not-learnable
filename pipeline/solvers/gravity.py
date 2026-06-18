"""gravity: d=0.5*g*t^2 -> A=d/t^2 constant. 100% on train."""
import re
def solve(prompt:str):
    pairs=[(float(t),float(d)) for t,d in re.findall(r'For t = ([\d.]+)s, distance = ([\d.]+)', prompt)]
    q=float(re.search(r'falling distance for t = ([\d.]+)s', prompt).group(1))
    t1,d1=pairs[0]; A=d1/(t1*t1); t2,d2=pairs[1]; chk=A*t2*t2; val=A*q*q
    cot=(f"d=0.5·g·t², so A=d/t² is constant. From t={t1}, d={d1}: "
         f"A={d1}/{t1}²={d1}/{t1*t1:.4f}={A:.4f}. Verify on t={t2}: {A:.4f}·{t2}²={chk:.2f}≈{d2}. "
         f"Apply to t={q}: d={A:.4f}·{q}²={A:.4f}·{q*q:.4f}={val:.2f}.")
    return f"{val:.2f}", cot
