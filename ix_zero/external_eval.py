"""External ARC-format evaluation. The candidate is executed in a distinct
process with only demonstrations and test inputs in its request. This is a
logical evaluation separation, not a hostile-code security boundary. No task
IDs, answer files, or test outputs are passed to that process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from .exploration import grid,freeze


def worker():
    raw=sys.stdin.read(4_000_001)
    if len(raw)>4_000_000:raise ValueError('worker request size')
    p=json.loads(raw)
    if set(p)!={'mode','train','inputs','budget'}:raise ValueError('worker request contract')
    if p['mode']=='original':
        from .arc_eval import solve
        result=solve(p['train'],p['inputs'],budget=p['budget'],depth=2)
    elif p['mode'] in ('objects','pixels'):
        from .object_induction import solve
        result=solve(p['train'],p['inputs'],budget=p['budget'],representation='all' if p['mode']=='objects' else 'pixels')
    elif p['mode']=='relational':
        from .grid_relational import infer
        result=infer(p['train'],p['inputs'],budget=p['budget'])
    else:raise ValueError('solver mode')
    print(json.dumps(result,allow_nan=False))


def run_worker(mode,train,inputs,budget,timeout=30):
    p={'mode':mode,'train':train,'inputs':inputs,'budget':budget}
    proc=subprocess.run([sys.executable,'-m','ix_zero.external_eval','--worker'],input=json.dumps(p),text=True,capture_output=True,timeout=timeout,check=True)
    return json.loads(proc.stdout)


def evaluate(folder,budget=5000,limit=None,seed=0,modes=('original','objects','pixels')):
    if type(budget) is not int or not 1<=budget<=20000 or limit is not None and (type(limit) is not int or limit<1) or not modes or not set(modes)<=set(('original','objects','pixels','relational')):raise ValueError('evaluation options')
    files=sorted(Path(folder).glob('*.json'))
    if not files:raise ValueError('no external ARC tasks')
    if limit is not None and limit<len(files):files=sorted(random.Random(seed).sample(files,limit))
    rows=[]
    for path in files:
        blob=path.read_bytes();task=json.loads(blob);train=[(r['input'],r['output']) for r in task['train']];inputs=[r['input'] for r in task['test']];expected=[grid(freeze(r['output'])) for r in task['test']]
        for mode in modes:
            try:
                r=run_worker(mode,train,inputs,budget);pred=[None if x is None else grid(freeze(x)) for x in r['predictions']]
                if len(pred)!=len(expected):raise ValueError('wrong number of predictions')
                error=None
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError,KeyError) as e:
                r={};pred=[None]*len(expected);error=type(e).__name__+': '+str(e)[:400]
            hits=sum(x==y for x,y in zip(pred,expected));abstain=sum(x is None for x in pred)
            rows.append({'task':path.stem,'file_sha256':hashlib.sha256(blob).hexdigest(),'mode':mode,'exact_task':hits==len(expected),'correct':hits,'wrong':len(expected)-hits-abstain,'abstentions':abstain,'error':error,'expected':expected,'predictions':pred,'candidate_report':r})
    summary={m:{'tasks':len(files),'exact_tasks':sum(r['exact_task'] for r in rows if r['mode']==m),**{k:sum(r[k] for r in rows if r['mode']==m) for k in ('correct','wrong','abstentions')},'execution_errors':sum(r['error'] is not None for r in rows if r['mode']==m)} for m in modes}
    return {'candidate_cap':budget,'selection':{'available':len(list(Path(folder).glob('*.json'))),'limit':limit,'seed':seed,'task_ids':[p.stem for p in files]},'summary':summary,'rows':rows,'claim':'externally supplied ARC files, not an official score; candidate caps are equal but actual grammar sizes and evaluation cost differ'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',nargs='?');p.add_argument('--worker',action='store_true');p.add_argument('--budget',type=int,default=5000);p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=0);p.add_argument('--output');p.add_argument('--modes',nargs='+',choices=('original','objects','pixels','relational'),default=['original','objects','pixels']);a=p.parse_args()
    if a.worker:worker();return
    if not a.folder:p.error('folder required')
    r=evaluate(a.folder,a.budget,a.limit,a.seed,tuple(a.modes));s=json.dumps(r,indent=2,allow_nan=False)+'\n'
    if a.output:Path(a.output).write_text(s,encoding='utf-8')
    print(json.dumps(r['summary'],indent=2))

if __name__=='__main__':main()
