"""Paired synthetic experiment; teacher does not call learner executor."""
import argparse
import json
import random
import time
from pathlib import Path
from .core import Mind,Space,Task,PRIMITIVES

def teacher(program, x, m):
    v=list(x)
    for op in program:
        if op=='rot': v.append(v.pop(0)); continue
        if op=='rev': v.reverse(); continue
        a,b=v[:2]
        v[0]={'inc':a+1,'dec':a-1,'mix':a+b,'unmix':a-b,'mul':a*b,'zero':0,'flip':-1-a,'gate':a+int(b==0)}[op]%m
    return tuple(v)

def train(t,oracle,pool,budget,active,rng):
    pool=list(pool); rng.shuffle(pool)
    for _ in range(budget):
        if t.blocked: break
        if active:
            x=t.query(pool)
            if x is None: break
        else:
            x=next((v for v in pool if v not in t.observations),None)
            if x is None: break
        t.observe(x,oracle(x))
    return len(t.observations)

def score(t,oracle,holdout):
    predictions=[(x,t.predict(x)) for x in holdout]
    actual=[(x,p,oracle(x)) for x,p in predictions]
    return {'total':len(actual),'correct':sum(p==y for x,p,y in actual),'wrong':sum(p is not None and p!=y for x,p,y in actual),'abstained':sum(p is None for x,p,y in actual),'observations':len(t.observations),'candidates':len(t.candidates),'blocked':t.blocked,'records':actual}

def run_seed(seed):
    rng=random.Random(seed); m=Mind(); source=Space(2,5); target=Space(3,5) if seed%2 else Space(2,7)
    originals=[tuple(rng.choice(PRIMITIVES) for _ in range(3)) for _ in range(3)]
    donors=[]
    for i,p in enumerate(originals):
        t=m.new(f'source{i}',source,reuse=False); oracle=lambda x,p=p:teacher(p,x,source.modulus)
        pool=list(source.states); rng.shuffle(pool); audit,pool=pool[:3],pool[3:]
        train(t,oracle,pool,8,True,rng)
        if t.resolved():
            for x in audit:
                if not t.audit(x,oracle(x)): break
            if t.concept() is not None and len(t.concept())>=2: m.promote(f'source{i}')
        donors.append({'teacher_program':p,'summary':t.summary()})
    target_program=originals[0]+originals[1]
    oracle=lambda x:teacher(target_program,x,target.modulus)
    inputs=list(target.states); rng.shuffle(inputs); holdout,pool=inputs[:12],inputs[12:]
    rows={}
    for label,lib,active in [('reuse_active',m.library(),True),('cold_active',(),True),('reuse_passive',m.library(),False)]:
        t=Task(target,lib); train(t,oracle,pool,8,active,random.Random(seed+1000))
        rows[label]=score(t,oracle,holdout)
    retention=sum(t.predict(x)==y for t in m.tasks.values() for x,y in t.observations.items())
    return {'seed':seed,'source_space':[source.width,source.modulus],'target_space':[target.width,target.modulus],'donors':donors,'library':m.library(),'target_program':target_program,'holdout':holdout,'arms':rows,'source_observed_retained':retention,'source_observed_total':sum(len(t.observations) for t in m.tasks.values())}

def campaign(seeds):
    start=time.perf_counter(); rows=[run_seed(s) for s in seeds]; arms={}
    for arm in ('reuse_active','cold_active','reuse_passive'):
        arms[arm]={k:sum(r['arms'][arm][k] for r in rows) for k in ('total','correct','wrong','abstained','observations')}
        arms[arm]['perfect_tasks']=sum(r['arms'][arm]['correct']==r['arms'][arm]['total'] for r in rows)
    return {'protocol':'synthetic-composition-transfer-v1','seeds':list(seeds),'arms':arms,'rows':rows,'seconds':time.perf_counter()-start}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,default=20261124); p.add_argument('--trials',type=int,default=8); p.add_argument('--out',default='results/campaign.json'); a=p.parse_args()
    if not 1<=a.trials<=30: p.error('trials must be 1..30')
    r=campaign(range(a.seed,a.seed+a.trials)); Path(a.out).write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
