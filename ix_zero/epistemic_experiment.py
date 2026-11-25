"""External synthetic epistemic challenge; learner never receives truth expressions.

The evaluator grants only unlabeled environment pools and answers requests.
Each run keeps its untouched holdout partition separate from all supplied pools.
"""
import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from .developmental import Development
from .operators import Domain


def tally(m,name,rows):
    predictions=[dict(input=x,expected=y,predicted=m.predict(name,x)) for x,y in rows]
    return dict(correct=sum(r['predicted']==r['expected'] for r in predictions),
                incorrect=sum(r['predicted'] is not None and r['predicted']!=r['expected'] for r in predictions),
                abstained=sum(r['predicted'] is None for r in predictions),rows=predictions)


def late_audit(seed,mode):
    # Same partition and hidden law as the original external adversarial probe.
    rng=random.Random(seed);special=(15,15);xs=[x for x in Domain(4).inputs if x!=special];rng.shuffle(xs)
    train=xs[:100];audit=xs[100:103];held=xs[103:]
    if mode=='late_exception':audit+=[special]
    elif mode=='early_exception':audit=[special]+audit
    m=Development();m.add('probe',dict(kind='operator',deps=[],width=4,budget=4,train=train,audit=audit))
    def truth(x):return 1 if mode!='control' and tuple(x)==special else x[0]&(~x[1]&15)
    requests=[]
    for _ in range(110):
        r=m.next()
        if r['action']!='request':break
        y=truth(r['input']);requests.append(dict(phase=r['phase'],input=r['input'],label=y));m.answer(y)
    else:raise RuntimeError('step limit')
    result=tally(m,'probe',[(x,truth(x)) for x in held+([special] if mode=='control' else [])])
    return dict(seed=seed,mode=mode,status=m.status()['probe'],requests=requests,
                exception_audited=any(r['phase']=='audit' and tuple(r['input'])==special for r in requests),
                exception_prediction=m.predict('probe',special),exception_expected=truth(special),**result)


def environment_trial(seed,mode,*,return_mind=False):
    rng=random.Random(seed);scope=[2,3,4,5];m=Development();partitions={};requests=[]
    # Shift-two and shift-four are distinct hidden laws; zero is an adversarial
    # control. All agree on the entire initial 2-bit environment.
    shift={'shift2':2,'shift4':4,'deny':2,'zero':0,'later_contradiction':2,'audit_contradiction':2}[mode]
    def truth(x):
        w,a,b=x
        return 0 if mode=='zero' else (a<<shift)&((1<<w)-1)
    def pools(w):
        xs=list(Domain(w).inputs);rng.shuffle(xs)
        n=min(128,len(xs)-7);audit=xs[n:n+4];train=xs[:n];hold=xs[n+4:]
        partitions[w]=(train,audit,hold)
        return dict(train=train,audit=audit)
    m.add('law',dict(kind='epistemic',deps=[],width=2,widths=scope,budget=4 if mode=='shift4' else 2,**pools(2)))
    for _ in range(100):
        r=m.next()
        if r['action']!='request':break
        if r['phase']=='environment':
            requests.append(r)
            if mode=='deny':m.abandon();continue
            m.answer(pools(r['input']))
        else:
            x=r['input'];y=truth(x)
            if mode=='audit_contradiction' and r['phase']=='audit' and x[0]!=2:y^=1
            requests.append(dict(**r,label=y));m.answer(y)
    else:raise RuntimeError('step limit')
    # Scope boundary check: no predictions in an environment not actually audited.
    ungranted=[dict(width=w,predicted=m.predict('law',[w,1,0])) for w in scope if w not in partitions]
    hold=[((w,*x),truth((w,*x))) for w,(_,_,h) in partitions.items() for x in h]
    scored=tally(m,'law',hold);before=m.status()['law'];later=None
    with TemporaryDirectory() as t:
        path=Path(t)/'journal';m.save(path);restored=Development.load(path)
        replay=restored.events==m.events and restored.status()==m.status() and tally(restored,'law',hold)==scored
        if mode=='later_contradiction' and m.live('law'):
            x,y=hold[0];later=dict(input=x,accepted=m.challenge('law',x,y^1),prediction_after=m.predict('law',x))
            m.save(path);restored=Development.load(path);replay=replay and restored.events==m.events and restored.status()==m.status()
    result=dict(seed=seed,mode=mode,requests=requests,status_before=before,status_after=m.status()['law'],
                replay=replay,ungranted=ungranted,later=later,**scored)
    return (m,result) if return_mind else result


def transfer_trial(seed,reuse):
    """Paired learner with identical evidence partitions and two-gate budget."""
    m=Development(reuse=reuse);scope=[2,3,4,5];partitions={}
    def pools(name,w):
        key=(name,w)
        if key not in partitions:
            # Per-width deterministic partition independent of request ordering.
            xs=list(Domain(w).inputs);random.Random(seed+w*100+1000*(name=='target')).shuffle(xs)
            n=min(128,len(xs)-7);partitions[key]=(xs[:n],xs[n:n+4],xs[n+4:])
        t,a,h=partitions[key];return dict(train=t,audit=a)
    def run(name,shift,deps):
        m.add(name,dict(kind='epistemic',deps=deps,width=2,widths=scope,budget=2,**pools(name,2)))
        for _ in range(100):
            r=m.next()
            if r['action']!='request':break
            if r['phase']=='environment':m.answer(pools(name,r['input']))
            else:
                w,a,b=r['input'];m.answer((a<<shift)&((1<<w)-1))
        else:raise RuntimeError('transfer step limit')
        # Fixed held-out partitions across both arms, even for an abstaining learner.
        for w in scope:pools(name,w)
        hold=[((w,*x),(x[0]<<shift)&((1<<w)-1)) for w in scope for x in partitions[name,w][2]]
        return dict(status=m.status()[name],**tally(m,name,hold))
    source=run('source',2,[]);target=run('target',4,['source'])
    e=m.jobs['target'].episode;deps=list(e.models[0].dependencies) if len(e.models)==1 else []
    return dict(seed=seed,reuse=reuse,source=source,target=target,learned_dependencies=deps)


def campaign(trials=6,seed=310000):
    if type(trials) is not int or not 1<=trials<=32:raise ValueError('trials must be 1..32')
    audit=[late_audit(seed+i,mode) for i in range(trials) for mode in ('control','late_exception','early_exception')]
    env=[environment_trial(seed+1000+i,mode) for i in range(trials) for mode in ('shift2','shift4','zero','deny','audit_contradiction','later_contradiction')]
    def summarize(rows,field):
        return {mode:dict(trials=len(rs),live=sum(r[field]['live'] for r in rs),correct=sum(r['correct'] for r in rs),incorrect=sum(r['incorrect'] for r in rs),abstained=sum(r['abstained'] for r in rs)) for mode in sorted({r['mode'] for r in rows}) if (rs:=[r for r in rows if r['mode']==mode])}
    a=summarize(audit,'status');e=summarize(env,'status_before')
    transfer=[transfer_trial(seed+2000+i,reuse) for i in range(trials) for reuse in (True,False)]
    transfer_summary={str(reuse):dict(trials=trials,live=sum(r['target']['status']['live'] for r in transfer if r['reuse']==reuse),**{k:sum(r['target'][k] for r in transfer if r['reuse']==reuse) for k in ('correct','incorrect','abstained')}) for reuse in (True,False)}
    ok=all(r['status']['live']==(r['mode']=='control') and r['incorrect']==0 and (r['mode']=='control' or r['exception_audited']) for r in audit)
    ok=ok and all(r['replay'] and all(v['predicted'] is None for v in r['ungranted']) and r['incorrect']==0 and r['status_before']['live']==(r['mode'] not in ('deny','audit_contradiction')) and any(q['phase']=='environment' for q in r['requests']) and (r['mode']!='shift4' or 5 in r['status_before']['epistemic']['granted']) and (r['mode']!='later_contradiction' or (r['later'] is not None and not r['later']['accepted'] and not r['status_after']['live'])) for r in env)
    # A counterexample unavailable in every supplied pool is deliberately not
    # discoverable: report the inevitable wrong prediction, not a fake pass.
    negative=[]
    for r in env:
        if r['mode']=='shift2' and r['rows']:
            p=next(x for x in r['rows'] if x['input'][0]==5)
            negative.append(dict(seed=r['seed'],input=p['input'],predicted=p['predicted'],alternative_truth=p['expected']^1,wrong=p['predicted']!=(p['expected']^1)))
    ok=ok and all(r['source']['status']['live'] for r in transfer) and all(r['target']['status']['live'] and 'source' in r['learned_dependencies'] and r['target']['incorrect']==0 for r in transfer if r['reuse'])
    return dict(python=platform.python_version(),seed=seed,trials=trials,passed=ok,audit_summary=a,environment_summary=e,transfer_summary=transfer_summary,unobservable_exception_counterfactual=negative,audit_records=audit,environment_records=env,transfer_records=transfer)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=6);p.add_argument('--seed',type=int,default=310000);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    r=campaign(a.trials,a.seed)
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if not k.endswith('_records')},indent=2))
    if a.check and not r['passed']:raise SystemExit(1)

if __name__=='__main__':main()
