"""Reproducible synthetic evaluation; target functions live only in this harness."""
import argparse
import json
import random
from .operators import Domain, OperatorMind, OperatorEpisode, execute


def partitions(width, seed):
    inputs=list(Domain(width).inputs); random.Random(seed).shuffle(inputs)
    return inputs[:14],inputs[14:20],inputs[20:]


def run_episode(t, fn, seed, observe=None, audit=None):
    train,checks,hold=partitions(t.domain.width,seed)
    observe=observe or t.observe; audit=audit or t.audit
    for x in train: observe(x,fn(*x))
    for x in checks:
        if not t.resolved() or not audit(x,fn(*x)): break
    predictions=[(x,fn(*x),t.predict(x)) for x in hold]
    return dict(correct=sum(y==p for _,y,p in predictions),incorrect=sum(p is not None and p!=y for _,y,p in predictions),
                abstained=sum(p is None for _,y,p in predictions),status=t.status(),predictions=predictions,
                training=[[x,y] for x,y in t.training.items()],audits=[[x,y,p] for x,(y,p) in t.audits.items()])


def campaign(trials=12,seed=8400):
    records=[]; active=[]
    for i in range(trials):
        s=seed+i
        for width in (3,4):
            mask=Domain(width).mask
            rng=random.Random(s+100000*width)
            random_mapping={x:rng.randrange(mask+1) for x in Domain(width).inputs}
            targets={'and':lambda a,b:a&b,'or':lambda a,b:a|b,'xor':lambda a,b:a^b,
                     'xnor':lambda a,b:mask^(a^b),'nor':lambda a,b:mask^(a|b),'implication':lambda a,b:(mask^a)|b,
                     'shifted_intersection':lambda a,b:(a>>1)&b,
                     'addition':lambda a,b:(a+b)&mask,'minimum':min,'random_mapping':lambda a,b:random_mapping[a,b]}
            for name,fn in targets.items():
                t=OperatorEpisode(width,6)
                records.append(dict(group='discovery',family=name,width=width,seed=s,**run_episode(t,fn,s)))
        m=OperatorMind();m.apply('new','source',3,5,False);t=m.episodes['source']
        source=run_episode(t,lambda a,b:a^b,s,lambda x,y:m.apply('observe','source',x,y),lambda x,y:m.apply('audit','source',x,y))
        decision=m.advance('source')
        records.append(dict(group='source',family='xor',width=3,seed=s,policy_action=decision['action'],**source))
        for width in (3,4):
            mask=Domain(width).mask
            for family,fn in {'shifted_xor':lambda a,b:a^(b>>1),'nested_xor':lambda a,b:(a^b)^((a<<1)&mask)}.items():
                for reuse in (False,True):
                    t=OperatorEpisode(width,3,tuple(m.promoted.items()) if reuse else ())
                    records.append(dict(group='reuse' if reuse else 'cold',family=family,width=width,seed=s,**run_episode(t,fn,s+5000)))
        # Policy controls the actual inquiry and promotion, with labels restricted to
        # an evaluator-defined training partition. Audits and holdouts stay separate.
        m=OperatorMind();m.apply('new','active',3,5,False);t=m.episodes['active'];train,checks,hold=partitions(3,s+10000)
        trace=[];j=0
        for _ in range(25):
            d=m.advance('active',train);trace.append(d)
            if d['action']=='inquire':
                x=d['result'];m.apply('observe','active',x,x[0]^x[1])
            elif d['action']=='audit':
                x=checks[j];j+=1;m.apply('audit','active',x,x[0]^x[1])
            else:break
        predictions=[(x,x[0]^x[1],t.predict(x)) for x in hold]
        active.append(dict(seed=s,status=t.status(),promoted='active' in m.promoted,trace=trace,predictions=predictions,
                           training=[[x,y] for x,y in t.training.items()],audits=[[x,y,p] for x,(y,p) in t.audits.items()]))
    totals={}
    for r in records:
        k=f"{r['group']}/{r['family']}/{r['width']}";a=totals.setdefault(k,dict(correct=0,incorrect=0,abstained=0,training=0,audits=0,episodes=0,truncated=0))
        for f in ('correct','incorrect','abstained'):a[f]+=r[f]
        a['training']+=r['status']['training'];a['audits']+=r['status']['audits'];a['episodes']+=1;a['truncated']+=r['status']['truncated']
    a=dict(episodes=len(active),promoted=sum(r['promoted'] for r in active),correct=0,incorrect=0,abstained=0,training=sum(r['status']['training'] for r in active),audits=sum(r['status']['audits'] for r in active))
    for r in active:
        for _,y,p in r['predictions']:
            a['correct' if p==y else 'abstained' if p is None else 'incorrect']+=1
    return dict(trials=trials,seed=seed,totals=totals,active_totals=a,records=records,active=active)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=8400);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=100:p.error('trials must be 1..100')
    r=campaign(a.trials,a.seed)
    if a.output:
        with open(a.output,'w',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ('trials','seed','totals','active_totals')},indent=2))
    if a.check:
        for family in ('shifted_xor','nested_xor'):
            for w in (3,4):
                reuse=r['totals'][f'reuse/{family}/{w}'];cold=r['totals'][f'cold/{family}/{w}']
                if not reuse['correct']>cold['correct']:raise SystemExit('reuse comparison failed')
                if reuse['incorrect']:raise SystemExit('reuse produced errors')

if __name__=='__main__':main()
