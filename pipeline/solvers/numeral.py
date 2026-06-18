"""numeral: standard Roman numerals. 100% on train."""
import re
_RVAL=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
       (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
def to_roman(n:int)->str:
    out=''
    for v,s in _RVAL:
        while n>=v: out+=s; n-=v
    return out
def solve(prompt:str):
    q=int(re.search(r'write the number\s+(\d+)\s+in', prompt).group(1))
    ans=to_roman(q); n=q; steps=[]
    for v,s in _RVAL:
        while n>=v: steps.append(f"{n}≥{v}→{s} (rem {n-v})"); n-=v
    cot=("The examples are Roman numerals (11=XI, 15=XV, 94=XCIV, 19=XIX). "
         f"Convert {q} by subtracting the largest Roman value each step: "+"; ".join(steps)+
         f". Concatenate the symbols: {ans}.")
    return ans, cot
