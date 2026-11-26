"""Reproducible evaluator for the integrated neural-symbolic runtime.

All sensor images and task dynamics are synthetic. Target programs remain in
this evaluator; only calibration labels, raster observations, and independent
measurements cross the learner interface. Holdouts never enter training.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from .checkpoint import canonical
from .hybrid import Hybrid, bits, DIRECTIONS
from .operators import Domain


def digest(x):return hashlib.sha256(canonical(x)).hexdigest()


class Sensor:
    """Three-channel raster; per-trial polarities, noise and exposure vary."""
    def __init__(self,seed):
        rng=Random(seed);self.polarity=rng.choice(((-1,-1,1),(-1,1,-1),(1,-1,-1),(1,1,1)))
    def pixel(self,label,rng):
        return [round(max(0,min(1,.5+sign*(2*label-1)*rng.uniform(.19,.40)+rng.uniform(-.10,.10))),6) for sign in self.polarity]
    def calibration(self,seed):
        rng=Random(seed);rows=[(self.pixel(y,rng),y) for y in (0,1) for _ in range(48)];rng.shuffle(rows);return rows
    def raster(self,family,w,y,rng):
        values=bits(y,w) if family=='word' else [int((i%w,i//w)==tuple(y)) for i in range(w*w)]
        return [self.pixel(v,rng) for v in values]


def trial(seed,family,mode,negative=None,*,prior=None,return_mind=False):
    rng=Random(seed); sensor=Sensor(seed+901);scope=[2,3,4,5] if family=='word' else [3,4,5]
    # Fixed authored families; disjoint seeds vary task rules, sensor and partitions.
    law=rng.choice(('shift2','right2','nand','x')) if family=='word' else (rng.randrange(4),rng.choice((-1,1)),rng.choice((1,2)),rng.choice((False,True)))
    def truth(x):
        w,a,b=x[:3]
        if negative=='outside_family':
            return (a+b)% (1<<w) if family=='word' else ((a+1)%w,(b+1)%w)
        if family=='word':return {'shift2':(a<<2)&((1<<w)-1),'right2':a>>2,'nand':((1<<w)-1)^(a&b),'x':a}[law]
        # Separate evaluator implementation, not a call to the candidate executor.
        rot,reflect,stride,wrap=law;dx,dy=DIRECTIONS[(rot+reflect*x[3])%4]
        vals=(a+dx*stride,b+dy*stride)
        return tuple(v%w if wrap else max(0,min(w-1,v)) for v in vals)
    partitions={}
    for w in scope:
        xs=list(Domain(w).inputs) if family=='word' else list(product(range(w),range(w),range(4)))
        Random(seed+w*71).shuffle(xs);n=min(96,len(xs)-7) if family=='word' else len(xs)//2
        partitions[w]=dict(train=xs[:n],audit=xs[n:n+4],hold=xs[n+4:])
    def pool(w):return {k:partitions[w][k] for k in ('train','audit')}
    m=Hybrid(family,scope,scope[0],**pool(scope[0]),seed=seed,mode=mode,budget=2) if prior is None else prior.transfer(family,scope,scope[0],**pool(scope[0]),seed=seed,budget=2)
    initial=m.weights();cal=sensor.calibration(seed+17);test=sensor.calibration(seed+19+(100 if family=='grid' else 0))
    recognition_before=sum(m.read(x)==y for x,y in test)
    if prior is None:m.calibrate(cal)
    recognition_after=sum(m.read(x)==y for x,y in test)
    hold=[(w,*x) for w in scope for x in partitions[w]['hold']]
    initial_predictions=[m.neural_prediction(x) for x in hold];requests=[];replay_mid=False
    for i in range(100):
        r=m.next();action=r['action']
        if action in ('complete','failed'):break
        if action=='environment':
            requests.append(r)
            if negative=='deny':m.deny();continue
            m.answer(pool(r['input']))
        else:
            x=r['input'];y=truth(x)
            if negative=='audit' and action=='audit':y=(y^1) if family=='word' else ((y[0]+1)%x[0],y[1])
            # Independent per-input rendering makes arms comparable regardless of request order.
            raster=sensor.raster(family,x[0],y,Random(seed+int(digest(x)[:12],16)))
            value=y if action=='audit' or mode=='oracle' else raster
            requests.append(dict(r,measurement=y,raster=raster if action=='train' and mode!='oracle' else None))
            m.answer(value)
        if i==1:
            with TemporaryDirectory() as d:
                p=Path(d)/'state.json';m.save(p);m=Hybrid.load(p);replay_mid=True
    else:raise RuntimeError('bounded query limit')
    rows=[dict(input=x,expected=truth(x),predicted=m.predict(x),neural_before=b,neural_after=m.neural_prediction(x)) for x,b in zip(hold,initial_predictions)]
    plans=[]
    if family=='grid' and m.inquiry.validated() and m.inquiry.finished:
        w=scope[-1];prng=Random(seed+222)
        for _ in range(12):
            start=tuple(prng.randrange(w) for _ in range(2));goal=tuple(prng.randrange(w) for _ in range(2));path=m.plan(start,goal,w);actual=start
            if path is not None:
                for a in path:actual=truth((w,*actual,a))
            # Independent breadth-first reachability and shortest distance.
            frontier={start};seen={start};distance=0
            while goal not in seen and frontier:
                frontier={truth((w,*p,a)) for p in frontier for a in range(4)}-seen;seen|=frontier;distance+=1
            expected_distance=distance if goal in seen else None
            plans.append(dict(start=start,goal=goal,actions=path,actual=actual,expected_distance=expected_distance,valid=(path is None and expected_distance is None) or (path is not None and actual==goal and len(path)==expected_distance)))
    invalidated=None
    if negative=='later' and m.inquiry.finished and m.inquiry.validated():
        x=hold[0];y=truth(x);wrong=y^1 if family=='word' else ((y[0]+1)%x[0],y[1]);invalidated=not m.challenge(x,wrong) and m.predict(x) is None
    with TemporaryDirectory() as d:
        p=Path(d)/'state.json';m.save(p);restored=Hybrid.load(p);replay=canonical(m.payload())==canonical(restored.payload())
    final=m.weights()
    result=dict(seed=seed,family=family,mode=mode,negative=negative,transferred=prior is not None,hidden_law=law,sensor_polarity=sensor.polarity,
                calibration_holdout=len(test),recognition_before=recognition_before,recognition_after=recognition_after,
                initial_weights={k:digest(v) for k,v in initial.items()},final_weights={k:digest(v) for k,v in final.items()},updates={k:v['updates'] for k,v in final.items()},
                final_network_state=final,calibration=cal if prior is None else [],calibration_test=test,partitions=partitions,requests=requests,rows=rows,plans=plans,replay=replay,replay_mid=replay_mid,
                promoted=m.inquiry.finished and m.inquiry.validated(),invalidated=invalidated,correct=sum(r['predicted']==r['expected'] for r in rows),wrong=sum(r['predicted'] is not None and r['predicted']!=r['expected'] for r in rows),abstain=sum(r['predicted'] is None for r in rows),neural_before=sum(r['neural_before']==r['expected'] for r in rows),neural_after=sum(r['neural_after']==r['expected'] for r in rows))

    return (result,m) if return_mind else result


def campaign(trials=4,seed=410000):
    if not 1<=trials<=12:raise ValueError('trials must be 1..12')
    rows=[trial(seed+i,f,m) for i in range(trials) for f in ('word','grid') for m in ('hybrid','unguided','oracle','centroid','symbolic','frozen')]
    transfers=[]
    for i in range(trials):
        for source,target in (('word','grid'),('grid','word')):
            source_result,prior=trial(seed+i,source,'hybrid',return_mind=True)
            target_result=trial(seed+i,target,'hybrid',prior=prior)
            transfers.append(dict(source=source,target=target,source_final=source_result['final_weights'],target_result=target_result))
    negatives=[trial(seed+1000,f,'hybrid',n) for f in ('word','grid') for n in ('audit','deny','later','outside_family')]
    summary={}
    for f in ('word','grid'):
        for m in ('hybrid','unguided','oracle','centroid','symbolic','frozen'):
            rs=[r for r in rows if r['family']==f and r['mode']==m]
            summary[f+'/'+m]={k:sum(r[k] for r in rs) for k in ('promoted','correct','wrong','abstain','recognition_before','recognition_after','calibration_holdout','neural_before','neural_after')}
            summary[f+'/'+m].update(queries=sum(len(r['requests']) for r in rs),plans=sum(len(r['plans']) for r in rs),valid_plans=sum(p['valid'] for r in rs for p in r['plans']))
    return dict(protocol='authored finite word and spatial task families; disjoint held-out inputs; matched partitions; not blinded or an AGI benchmark',seed=seed,trials=trials,summary=summary,rows=rows,negative_controls=negatives,transfers=transfers)


def check(result):
    rows=result['rows'];h=[r for r in rows if r['mode'] in ('hybrid','unguided')];oracle=[r for r in rows if r['mode'] in ('oracle','centroid')]
    return (all(r['wrong']==0 for r in rows) and all(r['promoted'] and r['wrong']==0 and r['abstain']==0 and r['replay'] and r['replay_mid'] for r in h+oracle)
            and all(all(r['updates'][k]>0 and r['initial_weights'][k]!=r['final_weights'][k] for k in r['updates']) and r['recognition_after']==r['calibration_holdout'] for r in h)
            and all(p['valid'] for r in h+oracle for p in r['plans'])
            and all(t['target_result']['promoted'] and t['target_result']['wrong']==0 and t['target_result']['abstain']==0 and t['target_result']['replay'] and t['target_result']['replay_mid'] and not t['target_result']['calibration'] and t['target_result']['recognition_before']==t['target_result']['calibration_holdout'] and t['target_result']['initial_weights']['recognition']==t['source_final']['recognition'] and t['target_result']['initial_weights']['strategy']==t['source_final']['strategy'] for t in result['transfers'])
            and all(r['replay'] and not r['promoted'] and (r['invalidated'] if r['negative']=='later' else r['abstain']==len(r['rows'])) for r in result['negative_controls']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=4);p.add_argument('--seed',type=int,default=410000);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    r=campaign(a.trials,a.seed);print(json.dumps(r['summary'],indent=2));ok=check(r);print('hybrid evaluation:', 'PASS' if ok else 'FAIL')
    if a.output:a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return 1 if a.check and not ok else 0


if __name__=='__main__':raise SystemExit(main())
