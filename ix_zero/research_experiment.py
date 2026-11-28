"""Reproducible authored experiments. Final labels never enter the learner.
A conventional enumerator and random query control are run at explicit budgets.
Failures remain reportable; passing --check means protocol checks, not AGI.
"""
import argparse
import json
import random
from pathlib import Path
from .exploration import Learner, Measurements, execute, parse_grammar, route
from .audit import AuditConsensus
from .pair_inquiry import inquire as pair_inquire, predict as pair_predict
from itertools import product
from math import gcd


def episode(learner,train,pool,audit,hold,**kwargs):
    m=Measurements(pool);n=AuditConsensus(audit) if isinstance(audit,dict) else Measurements(audit)
    r=learner.fit(train,m,n,**kwargs)
    # Final scorer independent of candidate selection and promotion.
    predictions=[learner.predict(r['skill'],x,kwargs.get('grammar')) if r['promoted'] else None for x,y in hold]
    return {'fit':r,'query_count':len(m.asked),'audit_count':len(n.asked),'raw_audit_observations':sum(len(v.asked) for v in n.sources.values()) if isinstance(n,AuditConsensus) else len(n.asked),'audit_source_votes':n.log if isinstance(n,AuditConsensus) else [],'held_out':[{'input':x,'expected':y,'predicted':p} for (x,y),p in zip(hold,predictions)],'correct':sum(p==y for p,(_,y) in zip(predictions,hold)),'wrong':sum(p is not None and p!=y for p,(_,y) in zip(predictions,hold)),'abstentions':sum(p is None for p in predictions)}


def active_campaign(seeds):
    out=[]
    for seed in seeds:
        rng=random.Random(seed)
        # Ambiguity across three independently defined target functions.
        for name,f in [('absolute',abs),('square',lambda x:x*x),('identity',lambda x:x)]:
            points=[-1,2,3,4,5,-2,-3,6];rng.shuffle(points)
            train=[(x,f(x)) for x in [0,1]];pool=[(x,f(x)) for x in points];audit=[(x,f(x)) for x in [-4,7,8]];hold=[(x,f(x)) for x in [-9,-8,-7,9,10,11]]
            for mode in ('active','random','enumerative'):
                out.append({'seed':seed,'family':name,'mode':mode,**episode(Learner(),train,pool,audit,hold,depth=1,budget=8,query_budget=1,mode=mode,seed=seed)})
    return out


def macro_campaign(seeds):
    out=[]
    for seed in seeds:
        rng=random.Random(seed);values=list(range(2,52));rng.shuffle(values)
        train,pool,audit,hold=values[:4],values[4:10],values[10:16],values[16:]
        for reuse in (True,False):
            m=Learner();rows=[]
            for f in (lambda x:2*x+1,lambda x:(2*x+1)**2):
                rows.append(episode(m,[(x,f(x)) for x in train],[(x,f(x)) for x in pool],[(x,f(x)) for x in audit],[(x,f(x)) for x in hold],depth=2,budget=72,query_budget=1,reuse=reuse,seed=seed))
            out.append({'seed':seed,'reuse':reuse,'episodes':rows})
    return out


def noise_campaign(seeds):
    rows=[]
    for seed in seeds:
        for percent in (0,1,5,10,20):
            for hidden in (False,True):
                for adversarial in (False,True):
                    rng=random.Random(seed);f=lambda x:x+1+(7 if hidden and x<0 else 0)
                    train=[(x,f(x)) for x in (1,2,3)];pool=[(x,f(x)) for x in range(4,10)]
                    audit_x=[-1]+list(range(10,109));selected=set(rng.sample(audit_x,percent))
                    if adversarial and percent and -1 not in selected:selected.remove(min(selected));selected.add(-1)
                    audit=[(x,(x+1 if hidden and x==-1 else f(x)+7) if x in selected else f(x)) for x in audit_x]
                    held=[(x,f(x)) for x in range(-101,-1)];r=episode(Learner(),train,pool,audit,held,depth=1,budget=8,query_budget=1,seed=seed)
                    rows.append({'seed':seed,'corruption_percent':percent,'hidden_regime':hidden,'targeted_corruption':adversarial,'false_promotion':r['fit']['promoted'] and r['wrong']>0,'clean_target_withheld':not hidden and not r['fit']['promoted'],**r})
    return rows


def consensus_campaign(seeds):
    rows=[]
    for seed in seeds:
        for hidden in (False,True):
            for corrupted_sources in (1,2,3):
                f=lambda x:x+1+(7 if hidden and x<0 else 0)
                xs=[-1]+list(range(10,109));clean=[(x,f(x)) for x in xs]
                altered=[(x,(x+1 if hidden else f(x)+7) if x==-1 else f(x)) for x in xs]
                sources={f'observer{i}':altered if i<corrupted_sources else clean for i in range(3)}
                r=episode(Learner(),[(x,f(x)) for x in (1,2,3)],[(x,f(x)) for x in range(4,10)],sources,[(x,f(x)) for x in range(-101,-1)],depth=1,budget=8,query_budget=1,seed=seed)
                rows.append({'seed':seed,'hidden_regime':hidden,'corrupted_sources':corrupted_sources,'false_promotion':r['fit']['promoted'] and r['wrong']>0,**r})
    return rows


def external_grammar_campaign(seeds):
    # Authored simulation of independently supplied grammar API, NOT external authorship.
    rows=[]
    for seed in seeds:
        rng=random.Random(seed);symbols=[f'z{n}' for n in range(32)];a=symbols[:];b=symbols[:];rng.shuffle(a);rng.shuffle(b)
        spec={'external:alpha':list(zip(symbols,a)),'external:beta':list(zip(symbols,b))};g=parse_grammar(spec)
        order=symbols[:];rng.shuffle(order);f=lambda x:execute(('external:alpha','external:beta'),x,g)
        rows.append({'seed':seed,**episode(Learner(),[(x,f(x)) for x in order[:4]],[(x,f(x)) for x in order[4:10]],[(x,f(x)) for x in order[10:16]],[(x,f(x)) for x in order[16:]],grammar=spec,depth=2,budget=6,query_budget=1,seed=seed)})
    return rows


def pair_campaign(seeds):
    rows=[]
    for seed in seeds:
        xs=list(product(range(8),repeat=2));random.Random(seed).shuffle(xs)
        for family,f in [('product_plus_two',lambda x:x[0]*x[1]+2),('nand',lambda x:7^(x[0]&x[1])),('less_equal',lambda x:int(x[0]<=x[1])),('gcd',lambda x:gcd(*x))]:
            for mechanism in ('combined','scalar_only'):
                t=Measurements([(x,f(x)) for x in xs[6:30]]);a=Measurements([(x,f(x)) for x in xs[30:36]])
                r,h=pair_inquire([(x,f(x)) for x in xs[:6]],t,a,mechanisms=('scalar','word','predicate') if mechanism=='combined' else ('scalar',),query_budget=4)
                pred=[(x,f(x),pair_predict(h,x)) for x in xs[36:]]
                rows.append({'seed':seed,'family':family,'mode':mechanism,'report':r,'query_count':len(t.asked),'audit_count':len(a.asked),'held_out':pred,'correct':sum(y==z for _,y,z in pred),'wrong':sum(z is not None and y!=z for _,y,z in pred),'abstentions':sum(z is None for _,_,z in pred)})
    return rows


def run(trials,seed):
    seeds=list(range(seed,seed+trials));active=active_campaign(seeds);macros=macro_campaign(seeds);noise=noise_campaign(seeds);grammar=external_grammar_campaign(seeds);consensus=consensus_campaign(seeds);pairs=pair_campaign(seeds)
    def aggregate(rs):return {k:sum(r[k] for r in rs) for k in ('correct','wrong','abstentions','query_count','audit_count')}
    summary={'active':{m:aggregate([r for r in active if r['mode']==m]) for m in ('active','random','enumerative')},'macro_second':{str(b):aggregate([r['episodes'][1] for r in macros if r['reuse']==b]) for b in (True,False)},'external_grammar':aggregate(grammar),'noise':[],'consensus':[],'cross_mechanism':{m:{f:aggregate([r for r in pairs if r['mode']==m and r['family']==f]) for f in ('product_plus_two','nand','less_equal','gcd')} for m in ('combined','scalar_only')}}
    for p in (0,1,5,10,20):
        for targeted in (False,True):
            rs=[r for r in noise if r['corruption_percent']==p and r['targeted_corruption']==targeted];summary['noise'].append({'percent':p,'targeted':targeted,'trials':len(rs),'promotions':sum(r['fit']['promoted'] for r in rs),'false_promotions':sum(r['false_promotion'] for r in rs),'clean_target_withheld':sum(r['clean_target_withheld'] for r in rs),**aggregate(rs)})
    for count in (1,2,3):
        rs=[r for r in consensus if r['corrupted_sources']==count];summary['consensus'].append({'corrupted_sources':count,'trials':len(rs),'false_promotions':sum(r['false_promotion'] for r in rs),'promotions':sum(r['fit']['promoted'] for r in rs),'raw_audit_observations':sum(r['raw_audit_observations'] for r in rs),**aggregate(rs)})
    routes=[route('start','goal',[('start','walk','side'),('side','walk','goal')],['walk']),route('start','goal',[('start','walk','side')],['walk'])]
    return {'protocol':'authored finite campaigns; not independent benchmark','seeds':seeds,'summary':summary,'active':active,'macros':macros,'noise':noise,'external_grammar':grammar,'consensus':consensus,'cross_mechanism':pairs,'routes':routes}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=10);p.add_argument('--seed',type=int,default=950000);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=100:p.error('trials 1..100')
    r=run(a.trials,a.seed);s=r['summary'];print(json.dumps(s,indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    if a.check:
        assert all(v['correct']+v['wrong']+v['abstentions']==18*a.trials for v in s['active'].values())
        assert r['routes'][0]['reframed'] and r['routes'][1]['status']=='unreachable'
        assert all(e['fit']['attempted']<=72 for rr in r['macros'] for e in rr['episodes'])
        assert all(x['false_promotions']<=x['promotions'] for x in s['noise'])

if __name__=='__main__':main()
