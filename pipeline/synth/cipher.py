"""Synthetic cipher rows. Random letter substitution over the fixed 77-word vocab."""
import sys, os, string; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from solvers import cipher as cipher_solver
VOCAB="above alice ancient around beyond bird book bright castle cat cave chases clever colorful creates crystal curious dark discovers door dragon draws dreams explores follows forest found garden golden hatter hidden imagines in inside island key king knight library magical map message mirror mountain mouse mysterious near ocean palace potion princess puzzle queen rabbit reads school secret sees silver story strange student studies teacher the through tower treasure turtle under valley village watches wise wizard wonderland writes".split()
H="In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:"
def gen(rng):
    letters=list(string.ascii_lowercase); perm=letters[:]; rng.shuffle(perm)
    sub=dict(zip(letters,perm))
    enc=lambda w:''.join(sub[c] for c in w)
    def sent(): return rng.sample(VOCAB, rng.choice([3,4,5]))
    k=rng.choice([3,4,5]); exs=[sent() for _ in range(k)]; q=sent()
    lines=[" ".join(enc(w) for w in s)+" -> "+" ".join(s) for s in exs]
    prompt=H+"\n"+"\n".join(lines)+"\nNow, decrypt the following text: "+" ".join(enc(w) for w in q)
    voc=set(VOCAB); ans,c=cipher_solver.solve(prompt, voc)
    return {"category":"cipher","prompt":prompt,"answer":ans,"final":ans,"cot":c}
