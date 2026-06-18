"""Synthetic numeral rows (Roman). Random N in 1-100, 3-5 examples."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from solvers import numeral
H="In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:"
def gen(rng):
    k=rng.choice([3,4,5]); ex=rng.sample(range(1,101),k); q=rng.randint(1,100)
    prompt=H+"\n"+"\n".join(f"{n} -> {numeral.to_roman(n)}" for n in ex)+f"\nNow, write the number {q} in the Wonderland numeral system."
    a,c=numeral.solve(prompt); return {"category":"numeral","prompt":prompt,"answer":a,"final":a,"cot":c}
