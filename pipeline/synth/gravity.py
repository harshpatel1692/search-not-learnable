"""Synthetic gravity rows. d=A*t^2, A in [2.45,9.79], t in [1,5]."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from solvers import gravity
H="In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:"
def gen(rng):
    A=round(rng.uniform(2.45,9.79),4); k=rng.choice([3,4,5])
    ts=[round(rng.uniform(1,5),2) for _ in range(k)]; q=round(rng.uniform(1,5),2)
    prompt=H+"\n"+"\n".join(f"For t = {t:.2f}s, distance = {A*t*t:.2f} m" for t in ts)+f"\nNow, determine the falling distance for t = {q:.2f}s given d = 0.5*g*t^2."
    ans,c=gravity.solve(prompt); return {"category":"gravity","prompt":prompt,"answer":ans,"final":ans,"cot":c}
