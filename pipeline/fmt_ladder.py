import json, sys
from pathlib import Path
src="/tmp/hint_ladder_full.jsonl"
out=Path("analysis/reports/hint_ladder_60ed3f31.md")
out.parent.mkdir(parents=True, exist_ok=True)
recs=[json.loads(l) for l in open(src)]
L=[]
L.append("# Hint-ladder trace — cryptarithm_deduce id=60ed3f31\n")
L.append(f"Query gold = `{recs[0]['gold']}`.  Teacher = nvidia/nemotron-3-super-120b-a12b, temp=0.6, reasoning_budget=3500, max_tokens=8000.\n")
L.append("Goal: lowest hint rung that yields a CORRECT, hint-free trace.\n")
L.append("## Verdict summary\n")
L.append("| rung | answer | gold | converged | leak | tokens |")
L.append("|---|---|---|---|---|---|")
for r in recs:
    a=str(r['answer']); a=a if len(a)<=14 else a[:14]+"…"
    L.append(f"| {r['rung']} | `{a}` | `{r['gold']}` | {'YES' if r['correct'] else 'no'} | {','.join(r['leaks']) or 'clean'} | {r['n_tokens']} |")
L.append("")
for r in recs:
    L.append("\n" + "="*100)
    L.append(f"## {r['rung']}  —  converged={r['correct']}  leak={r['leaks'] or 'clean'}  tokens={r['n_tokens']}  answer=`{r['answer'] if len(str(r['answer']))<40 else str(r['answer'])[:40]+'…'}`")
    L.append("\n### SYSTEM MESSAGE")
    L.append("```\n"+ (r.get('system') or '(logged in nvidia_logs)') +"\n```")
    L.append("### REASONING (CoT)")
    L.append("```\n"+ (r.get('reasoning') or '(none)') +"\n```")
    L.append("### FINAL CONTENT (answer span)")
    L.append("```\n"+ (r.get('content') or '(none)')[:1500] +"\n```")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes, {len(recs)} rungs)")
