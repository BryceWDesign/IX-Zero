"""Second, disjoint-seed campaign with calibration withheld from acquisition.
Compare raw and audit-gated outputs on identical learned models. This is not
an externally blinded evaluation. Every failure is retained.
"""
import argparse
import json
import random
import time
from pathlib import Path
from .core import Mind, Space, Task, PRIMITIVES
from .experiment import teacher, train, score

def run(seed):
    rng=random.Random(seed); m=Mind(); source=Space(2,5); target=Space(3,5) if seed%2 else Space(2,7)
    programs=[tuple(rng.choice(PRIMITIVES) for _ in range(3)) for _ in range(3)]
    sources=[]
    for i,p in enumerate(programs):
        t=m.new(f'source{i}',source,reuse=False); oracle=lambda x,p=p:teacher(p,x,source.modulus)
        pool=list(source.states); rng.shuffle(pool); audit,pool=pool[:3],pool[3:]
        train(t,oracle,pool,8,True,rng)
        if t.resolved():
            for x in audit:
                if not t.audit(x,oracle(x)): break
            if t.concept() is not None and len(t.concept())>=2: m.promote(f'source{i}')
        sources.append({'program':p,'summary':t.summary(),'training':list(t.observations.items()),'audit':list(t.audits.items())})
    target_program=programs[0]+programs[1]; oracle=lambda x:teacher(target_program,x,target.modulus)
    inputs=list(target.states); rng.shuffle(inputs); holdout,cal,pool=inputs[:12],inputs[12:15],inputs[15:]
    rows={}
    for name,lib,active in [('reuse_active',m.library(),True),('cold_active',(),True),('reuse_passive',m.library(),False)]:
        t=Task(target,lib); initial=len(t.candidates); train(t,oracle,pool,5,active,random.Random(seed+1000))
        raw_predictions=[(x,t.predict(x)) for x in holdout]
        # Calibration is evaluated first. Holdout labels remain unused.
        for x in cal:
            if t.blocked: break
            t.audit(x,oracle(x))
        guarded_predictions=[(x,t.validated_predict(x)) for x in holdout]
        truth={x:oracle(x) for x in holdout}
        def measure(preds):
            return {'total':len(preds),'correct':sum(p==truth[x] for x,p in preds),'wrong':sum(p is not None and p!=truth[x] for x,p in preds),'abstained':sum(p is None for x,p in preds),'records':[(x,p,truth[x]) for x,p in preds]}
        rows[name]={'raw':measure(raw_predictions),'guarded':measure(guarded_predictions),'initial_candidates':initial,'training':list(t.observations.items()),'calibration':list(t.audits.items()),'blocked':t.blocked}
    return {'seed':seed,'source_space':[source.width,source.modulus],'target_space':[target.width,target.modulus],'sources':sources,'library':m.library(),'target_program':target_program,'holdout':holdout,'calibration_pool':cal,'arms':rows}

def campaign(seeds):
    start=time.perf_counter(); seeds=list(seeds); rows=[run(s) for s in seeds]; arms={}
    for a in ('reuse_active','cold_active','reuse_passive'):
        arms[a]={mode:{k:sum(r['arms'][a][mode][k] for r in rows) for k in ('total','correct','wrong','abstained')} for mode in ('raw','guarded')}
        arms[a]['training_labels']=sum(len(r['arms'][a]['training']) for r in rows)
        arms[a]['calibration_labels']=sum(len(r['arms'][a]['calibration']) for r in rows)
        arms[a]['source_labels_available']=sum(sum(len(s['training'])+len(s['audit']) for s in r['sources']) for r in rows) if a.startswith('reuse') else 0
        arms[a]['initial_candidates_total']=sum(r['arms'][a]['initial_candidates'] for r in rows)
    return {'protocol':'paired-calibration-v1','seeds':seeds,'arms':arms,'rows':rows,'seconds':time.perf_counter()-start}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,default=20261201); p.add_argument('--trials',type=int,default=16); p.add_argument('--out',default='results/validation.json'); a=p.parse_args()
    if not 1<=a.trials<=30: p.error('trials must be 1..30')
    r=campaign(range(a.seed,a.seed+a.trials)); Path(a.out).write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
