"""bit_manipulation solver via the WINNING per-bit pair+stride method (~85%).
Each output bit = const | unary(in[a], opt NOT) | binary OP(in[a],in[b]) for OP in
{AND,OR,XOR,ANDN,ORN,XORN}. As output bit index increases by 1, input positions stride +1 (mod 8)
because the transforms are rotations/shifts. Greedy stride set-cover over the 8 bits; gaps -> const.
Bits needing 3 input bits (the ~15% tail) are left uncovered -> default 1 (winner's fallback)."""
import re
def bit(x,p): return (x>>(7-p))&1            # MSB-first position p in 0..7
def mk(bits): return sum(bits[p]<<(7-p) for p in range(8))
OPS={'AND':lambda a,b:a&b,'OR':lambda a,b:a|b,'XOR':lambda a,b:a^b,
     'ANDN':lambda a,b:a&(1-b),'ORN':lambda a,b:a|(1-b),'XORN':lambda a,b:a^(1-b)}
def parse(p):
    ex=[(int(a,2),int(b,2)) for a,b in re.findall(r'([01]{8})\s*->\s*([01]{8})',p)]
    q=re.search(r'output for:\s*([01]{8})',p)
    return ex,(int(q.group(1),2) if q else None)

def _bit_consistent_unary(ex,j,a0,neg):
    a=(a0+j)%8
    return all((bit(i,a)^neg)==bit(o,j) for i,o in ex)
def _bit_consistent_binary(ex,j,a0,b0,op):
    a=(a0+j)%8; b=(b0+j)%8; f=OPS[op]
    return all(f(bit(i,a),bit(i,b))==bit(o,j) for i,o in ex)
def _bit_consistent_const(ex,j,c):
    return all(bit(o,j)==c for _,o in ex)

def _runs(bitset):
    """Maximal CONTIGUOUS cyclic runs (list of bit-lists) within bitset over positions 0..7."""
    if not bitset: return []
    if len(bitset)==8: return [list(range(8))]
    present=[j for j in range(8) if j in bitset]
    runs=[]; cur=[present[0]]
    for j in present[1:]:
        if j==cur[-1]+1: cur.append(j)
        else: runs.append(cur); cur=[j]
    runs.append(cur)
    # merge wrap-around (run ending at 7 + run starting at 0)
    if len(runs)>1 and runs[0][0]==0 and runs[-1][-1]==7:
        runs[0]=runs[-1]+runs[0]; runs.pop()
    return runs

def solve_bits(ex):
    """Return (assignment[8], covered_count). Segments = (param, contiguous run); cover by longest run first
    (the winner's 'longest stride' rule), simpler params break ties."""
    rank={'C':0,'U':1,'B':2}
    segs=[]   # (run_len, rank, param, run_bits)
    def add(desc):
        if desc[0]=='C': bs=set(j for j in range(8) if _bit_consistent_const(ex,j,desc[1]))
        elif desc[0]=='U': bs=set(j for j in range(8) if _bit_consistent_unary(ex,j,desc[1],desc[2]))
        else: bs=set(j for j in range(8) if _bit_consistent_binary(ex,j,desc[1],desc[2],desc[3]))
        for run in _runs(bs): segs.append((len(run),rank[desc[0]],desc,run))
    for c in (0,1): add(('C',c))
    for a0 in range(8):
        for neg in (0,1): add(('U',a0,neg))
    for a0 in range(8):
        for b0 in range(8):
            if a0==b0: continue
            for op in OPS: add(('B',a0,b0,op))
    segs.sort(key=lambda s:(-s[0],s[1]))      # longest run first, then simpler param
    assign=[None]*8; covered=set()
    for _,_,desc,run in segs:
        new=[j for j in run if j not in covered]
        if not new: continue
        for j in new: assign[j]=desc; covered.add(j)
        if len(covered)==8: break
    return assign, len(covered)

def apply_bit(desc,x,j):
    if desc is None: return 1                      # winner's fallback for 3-input tail
    if desc[0]=='C': return desc[1]
    if desc[0]=='U': return bit(x,(desc[1]+j)%8)^desc[2]
    _,a0,b0,op=desc; return OPS[op](bit(x,(a0+j)%8),bit(x,(b0+j)%8))

def solve(prompt):
    ex,q=parse(prompt)
    if len(ex)<2 or q is None: return None
    assign,cov=solve_bits(ex)
    bits=[apply_bit(assign[j],q,j) for j in range(8)]
    ans=format(mk(bits),'08b')
    return ans, assign, cov

if __name__=="__main__":
    import csv,sys
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
    n=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
    rows=rows[:n]; ok=wrong=0; full=0
    for r in rows:
        ans,assign,cov=solve(r['prompt'])
        if cov==8: full+=1
        if ans==r['answer'].strip(): ok+=1
        else: wrong+=1
    print(f"STRIDE solver: correct {ok} ({100*ok/len(rows):.1f}%) | wrong {wrong} | fully-covered {full} ({100*full/len(rows):.1f}%) | n={len(rows)}")
