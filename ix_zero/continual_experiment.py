"""External synthetic evaluator. Truth functions are never passed to the learner."""
import argparse
from itertools import product
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from .continual import Continual, digest
from .hybrid_experiment import Sensor
from .operators import Domain

class Host:
    def __init__(self, family, widths, seed, law, polarity=0, exception=None):
        self.family,self.widths,self.seed,self.law=family,widths,seed,law
        self.sensor=Sensor(seed+901); self.sensor.polarity=(1,-1,1) if polarity==0 else (-1,1,-1)
        self.exception=exception; self.partitions={}
    def inputs(self,w):
        return list(Domain(w).inputs) if self.family=='word' else list(product(range(w),range(w),range(4)))
    def pool(self,w,attempt):
        key=(w,attempt)
        if key not in self.partitions:
            xs=[x for x in self.inputs(w) if (w,*x)!=self.exception]
            Random(self.seed+w*71+attempt*7919).shuffle(xs)
            n=min(96,len(xs)-8) if self.family=='word' else min(96,len(xs)//2)
            self.partitions[key]=dict(train=xs[:n],audit=xs[n:n+4])
        return self.partitions[key]
    def truth(self,x):
        w,a,b=x[:3]
        if self.family=='word':
            if self.law=='add': y=(a+b)%(1<<w)
            elif self.law=='identity': y=a
            elif self.law=='nand':y=((1<<w)-1)^(a&b)
            elif self.law=='left_right':y=((a<<2)&((1<<w)-1))>>2
            elif self.law.startswith('left'):y=(a<<int(self.law[4:]))&((1<<w)-1)
            elif self.law.startswith('right'):y=a>>int(self.law[5:])
            else:raise ValueError('unknown host word law')
            return y^int(tuple(x)==self.exception)
        rotation,reflection,stride,wrap=self.law
        d=(rotation+reflection*x[3])%4
        da=(int(d==0)-int(d==2))*stride;db=(int(d==1)-int(d==3))*stride
        return tuple(v%w if wrap else max(0,min(w-1,v)) for v in (a+da,b+db))
    def answer(self,q,attempt):
        a=q['action']
        if a in ('sensor_check','calibrate','verify_sensor'):
            offset={'sensor_check':17,'calibrate':19,'verify_sensor':23}[a]
            return self.sensor.calibration(self.seed+offset+attempt*1000)
        if a=='recover':return self.pool(q['initial'],attempt+1)
        x=q['input']
        if a=='environment':return self.pool(x,attempt)
        y=self.truth(x)
        if a in ('audit','probe'):return y
        return self.sensor.raster(self.family,x[0],y,Random(self.seed+int(digest(x)[:10],16)+attempt*1000))
    def held(self,trace):
        excluded={(w,*x) for (w,_),p in self.partitions.items() for k in ('train','audit') for x in p[k]}
        excluded.update(tuple(e['request']['input']) for e in trace if e['request']['action']=='probe')
        return [(w,*x) for w in self.widths for x in self.inputs(w) if (w,*x) not in excluded]

def drive(m,h,*,start=True,budget=1):
    if start:m.begin(h.family,h.widths,h.widths[0],**h.pool(h.widths[0],0),seed=h.seed,budget=budget)
    trace=[]
    for _ in range(400):
        q=m.next()
        if q['action'] in ('complete','failed'):break
        v=h.answer(q,m.retry);trace.append(dict(request=q,answer=v));m.answer(v)
    else:raise RuntimeError('continual request budget exceeded')
    k=m.active;r=m.status()['records'][k];rows=[]
    for x in h.held(trace):
        expected=h.truth(x);predicted=m.predict(k,x)
        rows.append(dict(input=x,expected=expected,predicted=predicted))
    return dict(episode=k,law=h.law,family=h.family,widths=h.widths,seed=h.seed,polarity=h.sensor.polarity,
        state=r['state'],budget=r['budget'],dependencies=r['dependencies'],expression=r['expr'],sensor_source=r['sensor_source'],strategy_source=r['strategy_source'],representation_sources=r['representation_sources'],recognition_depth=r.get('recognition_depth'),recognition_updates=r.get('recognition_updates'),
        requests=len(trace),calibrations=sum(e['request']['action']=='calibrate' for e in trace),recoveries=sum(e['request']['action']=='recover' for e in trace),probes=sum(e['request']['action']=='probe' for e in trace),
        correct=sum(z['predicted']==z['expected'] for z in rows),wrong=sum(z['predicted'] is not None and z['predicted']!=z['expected'] for z in rows),abstained=sum(z['predicted'] is None for z in rows),trace=trace,rows=rows)

def replay(m):
    with TemporaryDirectory() as d:
        p=Path(d)/'state.json';m.save(p);r=Continual.load(p)
        return r,m.fingerprint()==r.fingerprint()

def paired(seed,reuse,reuse_skills=True,progressive=True):
    m=Continual(reuse=reuse,max_budget=2,reuse_skills=reuse_skills,progressive=progressive);results=[]
    for i,law in enumerate(('left2','left4','right2','right4','left_right')):
        results.append(drive(m,Host('word',[3,5],seed+i,law,i%2),budget=2))
    m,ok=replay(m)
    return dict(seed=seed,reuse=reuse,reuse_skills=reuse_skills,progressive=progressive,replayed=ok,episodes=results,bank=m.status()['sensor_bank'])

def recovery(seed):
    m=Continual(max_budget=2);old=Host('grid',[4,6],seed,(1,-1,2,True));first=drive(m,old,budget=2)
    changed=Host('grid',[4,6],seed,(0,1,1,False),1)
    x=next((w,*x) for w in changed.widths for x in changed.inputs(w) if old.truth((w,*x))!=changed.truth((w,*x)))
    old_id=m.active;accepted=m.challenge(old_id,x,changed.truth(x));after=m.predict(old_id,x)
    second=drive(m,changed,start=False);m,ok=replay(m)
    return dict(seed=seed,original=first,challenge=dict(input=x,old=old.truth(x),new=changed.truth(x),accepted=accepted,after=after),recovered=second,replayed=ok,old_still_invalid=not m.live(old_id))

def boundary(seed):
    m=Continual(max_budget=1);h=Host('word',[3,5],seed,'identity',exception=(3,7,7));r=drive(m,h)
    found=any(z['request']['action']=='probe' and tuple(z['request']['input'])==h.exception for z in r['trace'])
    return dict(seed=seed,exception=h.exception,detected=found,result=r)

def lifetime(seed,episodes=100):
    m=Continual(max_budget=2);records=[];retained=[]
    for i in range(episodes):
        if i%2==0:h=Host('word',[3,5],seed+i,('left2','right2','nand','identity')[i//2%4],(i//4)%2)
        else:h=Host('grid',[4,6],seed+i,(i%4,1 if i%3 else -1,1+i%2,bool(i%3)),(i//4)%2)
        r=drive(m,h,budget=2);records.append(r)
        if r['state']=='validated':retained.append((r['episode'],h))
    m,ok=replay(m);checks=[]
    for k,h in retained:
        xs=[(w,*x) for w in h.widths for x in h.inputs(w)]
        checks.append(dict(episode=k,total=len(xs),correct=sum(m.predict(k,x)==h.truth(x) for x in xs)))
    return dict(seed=seed,episodes=records,replayed=ok,retention=checks,bank=m.status()['sensor_bank'])

def campaign(seed=510000,trials=3,episodes=100):
    if not 1<=trials<=6 or not 1<=episodes<=100:raise ValueError('experiment bound')
    return dict(protocol='authored finite synthetic experiments; not independently blinded',seed=seed,
        paired=[paired(seed+i*100,reuse,skills,progressive) for i in range(trials) for reuse,skills,progressive in ((True,True,True),(False,False,False),(True,False,True),(True,True,False))],
        recovery=[recovery(seed+1000+i) for i in range(trials)],boundary=[boundary(seed+2000+i) for i in range(trials)],lifetime=lifetime(seed+3000,episodes))

def common_scores(c):
    # Match the evaluation inputs even when failed arms request more pools.
    scores=[]
    for seed in sorted({r['seed'] for r in c['paired']}):
        arms=[r for r in c['paired'] if r['seed']==seed]
        for i in range(len(arms[0]['episodes'])):
            maps=[{tuple(z['input']):z for z in r['episodes'][i]['rows']} for r in arms]
            common=set.intersection(*(set(m) for m in maps))
            scores.append(dict(seed=seed,law=arms[0]['episodes'][i]['law'],total=len(common),arms=[dict(reuse=r['reuse'],reuse_skills=r['reuse_skills'],progressive=r['progressive'],correct=sum(m[x]['predicted']==m[x]['expected'] for x in common),wrong=sum(m[x]['predicted'] is not None and m[x]['predicted']!=m[x]['expected'] for x in common),abstained=sum(m[x]['predicted'] is None for x in common)) for r,m in zip(arms,maps)]))
    return scores


def depth_control(seed):
    rows=[]
    for limit in (1,2):
        m=Continual(max_budget=limit,reuse=False)
        rows.append(drive(m,Host('word',[3,5],seed,'left2'),budget=1))
    m=Continual(max_budget=3)
    outside=drive(m,Host('word',[3,5],seed+1,'add'),budget=1)
    return dict(seed=seed,fixed=rows[0],adaptive=rows[1],outside=outside)


def summary(c):
    def count(rows):return dict(episodes=len(rows),validated=sum(r['state']=='validated' for r in rows),correct=sum(r['correct'] for r in rows),wrong=sum(r['wrong'] for r in rows),abstained=sum(r['abstained'] for r in rows),calibrations=sum(r['calibrations'] for r in rows),requests=sum(r['requests'] for r in rows))
    life=c['lifetime'];return dict(reuse=count([e for r in c['paired'] if r['reuse'] and r['reuse_skills'] and r['progressive'] for e in r['episodes']]),no_reuse=count([e for r in c['paired'] if not r['reuse'] for e in r['episodes']]),skill_disabled=count([e for r in c['paired'] if r['reuse'] and not r['reuse_skills'] for e in r['episodes']]),
        lateral_disabled=count([e for r in c['paired'] if r['reuse'] and r['reuse_skills'] and not r['progressive'] for e in r['episodes']]),
        recovery=count([r['recovered'] for r in c['recovery']]),boundary_detected=sum(r['detected'] for r in c['boundary']),boundary_trials=len(c['boundary']),boundary_promoted=sum(r['result']['state']=='validated' for r in c['boundary']),lifetime=count(life['episodes']),retention_correct=sum(r['correct'] for r in life['retention']),retention_total=sum(r['total'] for r in life['retention']),replayed=sum(r['replayed'] for r in c['paired']+c['recovery']+[life]),replay_total=len(c['paired'])+len(c['recovery'])+1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=510000);p.add_argument('--trials',type=int,default=3);p.add_argument('--episodes',type=int,default=100);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    c=campaign(a.seed,a.trials,a.episodes);c['common_holdouts']=common_scores(c);c['depth_controls']=[depth_control(a.seed+4000+i) for i in range(a.trials)];s=summary(c);c['summary']=s
    if a.output:a.output.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(s,indent=2))
    if a.check:
        assert s['reuse']['validated']>s['no_reuse']['validated'] and s['reuse']['validated']>s['skill_disabled']['validated'],s
        assert all(r['fixed']['state']!='validated' and r['adaptive']['state']=='validated' and r['outside']['state']!='validated' for r in c['depth_controls'])
        assert s['recovery']['validated']==a.trials and all(r['old_still_invalid'] and not r['challenge']['accepted'] and r['challenge']['after'] is None for r in c['recovery']),s
        assert s['boundary_detected']==a.trials and s['boundary_promoted']==0,s
        assert s['lifetime']['validated']==a.episodes and s['lifetime']['wrong']==0 and s['retention_correct']==s['retention_total'],s
        assert s['replayed']==s['replay_total'],s

if __name__=='__main__':main()
