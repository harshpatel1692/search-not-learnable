"""Synthetic unit_conversion rows. y=a*x (b=0 per data), a in [0.5,2.0], x in [5,50]."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from solvers import unit_conversion
H="In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"
def gen(rng):
    a=round(rng.uniform(0.5,2.0),4); k=rng.choice([3,4,5])
    xs=[round(rng.uniform(5,50),2) for _ in range(k)]; q=round(rng.uniform(5,50),2)
    prompt=H+"\n"+"\n".join(f"{x:.2f} m becomes {a*x:.2f}" for x in xs)+f"\nNow, convert the following measurement: {q:.2f} m"
    ans,c=unit_conversion.solve(prompt); return {"category":"unit_conversion","prompt":prompt,"answer":ans,"final":ans,"cot":c}
