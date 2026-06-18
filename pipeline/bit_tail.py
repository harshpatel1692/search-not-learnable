"""Attempt the bit_manip 3-input-bit TAIL: fill bits left uncovered by the binary stride solver with a
ternary stride param G3(p,q,r)=op2(op1(p,q),r) (opt NOT on p), positions striding +1. Measure recover rate
and spurious rate (does it fit examples but mispredict query?)."""
import csv,sys; sys.path.insert(0,'pipeline')
import bit_stride as S
OPS=S.OPS; bit=S.bit; mk=S.mk
def g3_forms():
    forms=[]
    for o1 in OPS:
        for o2 in OPS:
            for negp in (0,1):
                forms.append((o1,o2,negp))
    return forms
FORMS=g3_forms()
def eval_g3(form,p,q,r):
    o1,o2,negp=form
    return OPS[o2](OPS[o1](p^negp,q),r)
def fill_tail(ex,assign):
    U=[j for j in range(8) if assign[j] is None]
    if not U: return assign,[]
    # try ternary stride params consistent on ALL uncovered bits across examples
    best=None
    for a0 in range(8):
        for b0 in range(8):
            for c0 in range(8):
                if len({a0,b0,c0})<3: continue
                for form in FORMS:
                    okall=True
                    for j in U:
                        a=(a0+j)%8;b=(b0+j)%8;c=(c0+j)%8
                        if not all(eval_g3(form,bit(i,a),bit(i,b),bit(i,c))==bit(o,j) for i,o in ex):
                            okall=False;break
                    if okall:
                        best=(a0,b0,c0,form); break
                if best:break
            if best:break
        if best:break
    if not best: return assign,U
    a0,b0,c0,form=best
    for j in U: assign[j]=('T',a0,b0,c0,form)
    return assign,[]
def apply_full(assign,q):
    bits=[]
    for j in range(8):
        d=assign[j]
        if d is None: bits.append(1)
        elif d[0]=='T':
            _,a0,b0,c0,form=d
            bits.append(eval_g3(form,bit(q,(a0+j)%8),bit(q,(b0+j)%8),bit(q,(c0+j)%8)))
        else: bits.append(S.apply_bit(d,q,j))
    return format(mk(bits),'08b')
if __name__=="__main__":
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
    n=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
    rows=rows[:n]
    base_ok=tail_ok=tail_wrong=still_unc=0; recovered=0
    for idx,r in enumerate(rows):
        ex,q=S.parse(r['prompt']); assign,cov=S.solve_bits(ex)
        if cov==8: base_ok+=1; continue              # already covered by binary pass
        assign2,U=fill_tail(ex,[a for a in assign])
        ans=apply_full(assign2,q)
        if not U:   # tail filled
            if ans==r['answer'].strip(): tail_ok+=1
            else: tail_wrong+=1
        else: still_unc+=1
        if (idx+1)%200==0: print(f"  ..{idx+1}/{len(rows)} tail_ok={tail_ok} tail_wrong={tail_wrong} still_unc={still_unc}",flush=True)
    print(f"of uncovered rows: tail-filled CORRECT={tail_ok} WRONG={tail_wrong} still-uncovered={still_unc}")
