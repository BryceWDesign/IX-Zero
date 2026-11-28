"""Read an externally supplied ARC-AGI JSON directory. A bounded transform
synthesizer sees only train pairs and test inputs. Exact-match scorer runs
outside the solver and counts abstentions. Not an official competition score.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .exploration import grid, freeze, execute, candidates

ATOMS=('id','hflip','vflip','transpose','rotate','crop','tileh','tilev')+tuple(f'color:{a}:{b}' for a in range(10) for b in range(10) if a!=b)


def solve(train,inputs,budget=1000,depth=2):
    if type(budget) is not int or not 1<=budget<=20000 or type(depth) is not int or not 1<=depth<=4 or not 1<=len(train)<=100 or not 1<=len(inputs)<=100:raise ValueError('ARC input or search budget')
    train=[(grid(freeze(a)),grid(freeze(b))) for a,b in train];inputs=[grid(freeze(x)) for x in inputs]
    valid=[];attempted=0
    for code in candidates(ATOMS,(),depth,budget):
        attempted+=1
        if all(execute(code,a,{})==b for a,b in train):valid.append(code)
    output=[]
    for x in inputs:
        values={execute(code,x,{}) for code in valid}
        output.append(next(iter(values)) if len(values)==1 else None)
    return {'predictions':output,'candidates':len(valid),'attempted':attempted}


def evaluate(folder,budget=1000,depth=2):
    files=sorted(Path(folder).glob('*.json'))
    if not files:raise ValueError('no external ARC JSON tasks found')
    rows=[]
    for file in files:
        blob=file.read_bytes();task=json.loads(blob)
        train=[(r['input'],r['output']) for r in task['train']];inputs=[r['input'] for r in task['test']]
        if not train or not inputs:raise ValueError('empty ARC partition')
        r=solve(train,inputs,budget,depth)
        expected=[grid(freeze(v['output'])) for v in task['test']]
        correct=sum(a==b for a,b in zip(r['predictions'],expected));abstain=sum(a is None for a in r['predictions'])
        rows.append({'id':file.stem,'sha256':hashlib.sha256(blob).hexdigest(),'exact_task':correct==len(expected),'correct':correct,'wrong':len(expected)-correct-abstain,'abstentions':abstain,**r})
    return {'tasks':len(rows),'exact_tasks':sum(r['exact_task'] for r in rows),'correct_pairs':sum(r['correct'] for r in rows),'wrong_pairs':sum(r['wrong'] for r in rows),'abstentions':sum(r['abstentions'] for r in rows),'candidate_budget':budget,'depth':depth,'rows':rows,'claim':'external-directory evaluation; not a private or official leaderboard score'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');p.add_argument('--budget',type=int,default=1000);p.add_argument('--depth',type=int,default=2);p.add_argument('--output');a=p.parse_args()
    if not 1<=a.budget<=20000 or not 1<=a.depth<=4:p.error('invalid budget or depth')
    report=evaluate(a.folder,a.budget,a.depth);text=json.dumps(report,indent=2)
    if a.output:Path(a.output).write_text(text+'\n',encoding='utf-8')
    else:print(text)

if __name__=='__main__':main()
