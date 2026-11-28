"""Continuation research: actual iterative induction, learned calls, independent
standard-library target functions, object representations and evidence controls.
The evaluator owns the target functions and final labels. Authored partitions
are not an externally defined benchmark suite. No results are hardcoded.
"""
import argparse
import json
import math
import operator
from pathlib import Path
import random
from .exploration import Learner, Measurements
from .audit import AuditConsensus
from .program_induction import ProgramMemory
from .tool_induction import ToolMemory
from .object_induction import solve as grid_solve
from .exploration import freeze


def score(rows,predict):
    answers=[{'input':x,'expected':y,'predicted':predict(x)} for x,y in rows]
    return {'rows':answers,'correct':sum(r['predicted']==r['expected'] for r in answers),'wrong':sum(r['predicted'] is not None and r['predicted']!=r['expected'] for r in answers),'abstentions':sum(r['predicted'] is None for r in answers)}


def aggregate(rows):return {k:sum(r[k] for r in rows) for k in ('correct','wrong','abstentions')}


def packet(seed,f,offset):
    pts=[(a,b) for a in range(1,49) for b in range(1,49)];rng=random.Random(seed);rng.shuffle(pts);pts=pts[offset:offset+180]
    rows=[(x,f(*x)) for x in pts];shift=random.Random(seed+offset+997);large=shift.sample([(a,b) for a in range(100,257) for b in range(100,257)],40)
    return rows[:14],rows[14:44],rows[44:64],rows[64:],[(x,f(*x)) for x in large]


def program_campaign(seeds):
    rows=[];searches=[];boundaries=((0,0),(0,3),(3,0),(-5,10),(10,-5),(-4,-6))
    for seed in seeds:
        for mode,loops,reuse in [('native',True,True),('no_reuse',True,False),('no_loop',False,False)]:
            m=ProgramMemory();episode=[]
            for i,(name,f) in enumerate((('gcd',math.gcd),('lcm',math.lcm))):
                t,p,a,h,shift=packet(seed,f,i*180);pool=Measurements(p);audit=Measurements(a)
                report=m.learn(name,t,pool,audit,reuse=reuse,allow_loops=loops,budget=12000,seed=seed)
                result=score(h,lambda x:m.predict(name,x));shift_result=score(shift,lambda x:m.predict(name,x));boundary=score([(x,f(*x)) for x in boundaries],lambda x:m.predict(name,x))
                record={'seed':seed,'arm':mode,'task':name,'train':t,'pool_inputs':pool.inputs,'audit_inputs':audit.inputs,'queries':len(pool.asked),'audit_count':len(audit.asked),'report':report,**result,'shifted':shift_result,'boundary':boundary};rows.append(record);episode.append(record)
            if mode=='native' and all(r['report']['promoted'] for r in episode):
                # A counterfactual deliberately false challenge tests dependency
                # withdrawal; it is not an observation about the math function.
                rev=m.challenge('gcd',(12,8),999);episode[-1]['deliberate_contradiction_control']={'report':rev,'dependent_prediction_after':m.predict('lcm',(12,8))}
        t,p,a,h,_=packet(seed,math.gcd,0)
        for mode in ('enumerative','random','evolutionary'):
            m=ProgramMemory();pool=Measurements(p);audit=Measurements(a);report=m.learn('q',t,pool,audit,budget=6000,search=mode,seed=seed)
            searches.append({'seed':seed,'mode':mode,'report':report,'queries':len(pool.asked),'audit_count':len(audit.asked),**score(h,lambda x:m.predict('q',x))})
    return rows,searches


def tool_campaign(seeds):
    rows=[]
    for seed in seeds:
        for name,f,points in [('comb',math.comb,[(a,b) for a in range(1,17) for b in range(1,17)]),('isqrt',math.isqrt,[(i,) for i in range(1,129)]),('xor',operator.xor,[(a,b) for a in range(1,17) for b in range(1,17)]),('gcd',math.gcd,[(a,b) for a in range(1,30) for b in range(1,30)])]:
            random.Random(seed).shuffle(points);examples=[(x,f(*x)) for x in points[:104]];p=Measurements(examples[14:44]);a=Measurements(examples[44:64]);m=ToolMemory();r=m.learn('q',examples[:14],p,a,modules=['math','operator'])
            rows.append({'seed':seed,'target':name,'report':r,'queries':len(p.asked),'audit_count':len(a.asked),**score(examples[64:],lambda x:m.predict('q',x))})
    return rows


def grid_example(rng,kind):
    h,w=7,9;color,distractor=rng.sample(range(1,10),2);g=[[0]*w for _ in range(h)];rh,rw=rng.randint(2,3),rng.randint(2,3);side=rng.randrange(2)
    top,left=(1,1) if side==0 else (h-rh-1,w-rw-1);dy,dx=(h-1,w-1) if side==0 else (0,0)
    for i in range(top,top+rh):
        for j in range(left,left+rw):g[i][j]=color
    g[dy][dx]=distractor
    if kind=='largest_crop':y=[[color]*rw for _ in range(rh)]
    elif kind=='smallest_mask':y=[[distractor if (i,j)==(dy,dx) else 0 for j in range(w)] for i in range(h)]
    elif kind=='scale':y=[[v for v in row for _ in range(2)] for row in g for _ in range(2)]
    elif kind=='translate_wrap':y=[[g[(i-1)%h][(j-1)%w] for j in range(w)] for i in range(h)]
    else:raise ValueError(kind)
    return g,y


def grid_campaign(seeds):
    rows=[]
    for seed in seeds:
        for kind in ('largest_crop','smallest_mask','scale','translate_wrap'):
            rng=random.Random(seed);pairs=[];seen=set()
            while len(pairs)<7:
                x,y=grid_example(rng,kind);key=freeze(x)
                if key not in seen:seen.add(key);pairs.append((x,y))
            for mode in ('all','pixels'):
                r=grid_solve(pairs[:3],[x for x,y in pairs[3:]],representation=mode,budget=5000)
                predictions=iter(r['predictions']);s=score([(freeze(x),freeze(y)) for x,y in pairs[3:]],lambda _:next(predictions));rows.append({'seed':seed,'family':kind,'mode':mode,'train':pairs[:3],'candidate_report':r,**s})
    return rows


def reliability_campaign(seeds):
    rows=[]
    for seed in seeds:
        critical=-random.Random(seed).randint(1,50);audit_x=[critical]+list(range(10,109));test=[x for x in range(-101,0) if x!=critical]
        for hidden in (False,True):
            def f(x):return x+1+(7 if hidden and x<0 else 0)
            clean=[(x,f(x)) for x in audit_x];altered=[(x,(x+1 if hidden else f(x)+7) if x==critical else f(x)) for x in audit_x]
            for corrupt in range(4):
                sources={f'observer{i}':altered if i<corrupt else clean for i in range(3)}
                for policy in ('majority','unanimous'):
                    m=Learner();audit=AuditConsensus(sources,policy=policy);pool=Measurements([(x,f(x)) for x in range(4,10)]);r=m.fit([(x,f(x)) for x in (1,2,3)],pool,audit,depth=1,budget=8,query_budget=1)
                    s=score([(x,f(x)) for x in test],lambda x:m.predict(r['skill'],x) if r['promoted'] else None)
                    rows.append({'seed':seed,'hidden':hidden,'corrupted_sources':corrupt,'policy':policy,'critical_input':critical,'promoted':r['promoted'],'false_promotion':r['promoted'] and s['wrong']>0,'clean_withheld':not hidden and not r['promoted'],'raw_audit_observations':sum(len(o.asked) for o in audit.sources.values()),'audit_conflicts':r['audit_conflicts'],'audit_source_votes':audit.log,**s})
    return rows


def run(trials,seed):
    seeds=list(range(seed,seed+trials));program,search=program_campaign(seeds);tools=tool_campaign(seeds);grids=grid_campaign(seeds);reliability=reliability_campaign(seeds)
    summary={'program':{},'search':{},'tool_assisted':{},'grid':{},'reliability':[]}
    for arm in ('native','no_reuse','no_loop'):
        summary['program'][arm]={}
        for task in ('gcd','lcm'):
            rs=[r for r in program if r['arm']==arm and r['task']==task];summary['program'][arm][task]={**aggregate(rs),'promotions':sum(r['report']['promoted'] for r in rs),'attempts':sum(r['report']['attempted'] for r in rs),'shifted':aggregate([r['shifted'] for r in rs]),'boundary':aggregate([r['boundary'] for r in rs])}
    for mode in ('enumerative','random','evolutionary'):
        rs=[r for r in search if r['mode']==mode];summary['search'][mode]={**aggregate(rs),'promotions':sum(r['report']['promoted'] for r in rs),'attempts':sum(r['report']['attempted'] for r in rs),'queries':sum(r['queries'] for r in rs),'audit_count':sum(r['audit_count'] for r in rs)}
    for task in ('comb','isqrt','xor','gcd'):
        rs=[r for r in tools if r['target']==task];summary['tool_assisted'][task]={**aggregate(rs),'promotions':sum(r['report']['promoted'] for r in rs),'selected':sorted({r['report']['selected'] for r in rs if r['report']['selected']})}
    for mode in ('all','pixels'):
        summary['grid'][mode]={kind:{**aggregate([r for r in grids if r['mode']==mode and r['family']==kind]),'attempts':sum(r['candidate_report']['attempted'] for r in grids if r['mode']==mode and r['family']==kind)} for kind in ('largest_crop','smallest_mask','scale','translate_wrap')}
    for policy in ('majority','unanimous'):
        for corrupt in range(4):
            rs=[r for r in reliability if r['policy']==policy and r['corrupted_sources']==corrupt]
            summary['reliability'].append({'policy':policy,'corrupted_sources':corrupt,'false_promotions':sum(r['false_promotion'] for r in rs),'clean_withheld':sum(r['clean_withheld'] for r in rs),'promotions':sum(r['promoted'] for r in rs),'raw_audit_observations':sum(r['raw_audit_observations'] for r in rs),**aggregate(rs)})
    return {'seeds':seeds,'protocol':'independently authored Python standard-library target functions, authored partitions and grid/reliability diagnostics; not an independent benchmark suite or general intelligence measurement','summary':summary,'program':program,'search':search,'tools':tools,'grids':grids,'reliability':reliability}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=271100);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=50:p.error('trials 1..50')
    r=run(a.trials,a.seed);print(json.dumps(r['summary'],indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if a.check:
        for arm in r['summary']['program'].values():
            for v in arm.values():assert v['correct']+v['wrong']+v['abstentions']==116*a.trials and v['attempts']==12000*a.trials
        for v in r['summary']['search'].values():assert v['attempts']==6000*a.trials and v['correct']+v['wrong']+v['abstentions']==116*a.trials
        for row in r['reliability']:assert row['raw_audit_observations']==300
        # Negative outcomes are evidence, not failed engineering CI.
        assert any(x['false_promotion'] for x in r['reliability'])

if __name__=='__main__':main()
