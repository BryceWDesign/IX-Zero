"""Fresh-evidence program revision diagnostic. The evaluator supplies targeted
signed/zero observations after a positive-domain failure; the learner selects
an executable wrapper. Targeted repair is not general self-modification.
"""
import argparse
import json
import math
from pathlib import Path
import random
from .program_induction import ProgramMemory,revise
from .exploration import Measurements
from .completion_experiment import packet,score,aggregate


def run(trials,seed):
    rows=[]
    for s in range(seed,seed+trials):
        m=ProgramMemory()
        for i,(name,f) in enumerate((('gcd',math.gcd),('lcm',math.lcm))):
            t,p,a,h,shift=packet(s,f,i*180);r=m.learn(name,t,Measurements(p),Measurements(a),budget=12000)
            if not r['promoted']:raise RuntimeError('base acquisition failed: '+name)
        for name,f in (('gcd',math.gcd),('lcm',math.lcm)):
            rng=random.Random(s+43);x=[(a,b) for a in range(-20,21) for b in range(-20,21) if a<=0 or b<=0];rng.shuffle(x);x.remove((0,0));x=[(0,0)]+x
            train=x[:18]+[(60,42),(27,51)];pool=x[18:58];audit=x[58:88]+[(63,77),(48,49)];hold=x[88:188]+[(104,143),(-70,-70),(72,72),(0,75),(-75,0)]
            # Explicit boundary coverage is an experimenter intervention, not
            # independently invented diagnostic input generation by Zero.
            extra=[(-30,-30),(70,70),(0,35),(-35,0),(35,0),(0,-35),(35,-35),(-35,35)]
            for mode in ('random_pool','stratified_pool'):
                key=name+'-'+mode;inputs=pool+(extra if mode=='stratified_pool' else []);p=Measurements([(v,f(*v)) for v in inputs]);a=Measurements([(v,f(*v)) for v in audit]);r=revise(m,name,key,[(v,f(*v)) for v in train],p,a)
                rows.append({'seed':s,'target':name,'mode':mode,'report':r,'train':[(v,f(*v)) for v in train],'pool_inputs':p.inputs,'audit_inputs':a.inputs,'unchanged':score([(v,f(*v)) for v in hold],lambda v:m.predict(name,v)),'revised':score([(v,f(*v)) for v in hold],lambda v:m.predict(key,v))})
    summary={}
    for name in ('gcd','lcm'):
        summary[name]={}
        for mode in ('random_pool','stratified_pool'):
            rs=[r for r in rows if r['target']==name and r['mode']==mode];summary[name][mode]={'promotions':sum(r['report']['promoted'] for r in rs),'unchanged':aggregate([r['unchanged'] for r in rs]),'revised':aggregate([r['revised'] for r in rs])}
    return {'seeds':list(range(seed,seed+trials)),'summary':summary,'rows':rows,'claim':'targeted fresh-evidence revision; signed/zero input strata supplied by experimenter, no host source modification'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=291100);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=50:p.error('trials 1..50')
    r=run(a.trials,a.seed);print(json.dumps(r['summary'],indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if a.check:
        for target in r['summary'].values():
            for v in target.values():
                assert all(sum(z.values())==105*a.trials for z in (v['unchanged'],v['revised']))

if __name__=='__main__':main()
