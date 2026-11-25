"""Seeded, held-out representation selection and abstraction experiments."""
import argparse,json,random
from itertools import product
from .operators import OperatorMind
from .representation import Episode, Abstraction, fit_with_abstraction
from .scalar import ScalarEpisode


def source(seed):
    m=OperatorMind()
    for name,fn in [('word_a',lambda a,b:a^b),('word_b',lambda a,b:a&b),('word_c',lambda a,b:a|b)]:
        m.apply('new',name,3,5,False); ep=m.episodes[name]
        xs=list(ep.domain.inputs);random.Random(seed).shuffle(xs)
        for a,b in xs[:24]:m.apply('observe',name,(a,b),fn(a,b))
        for a,b in xs[24:27]:
            if not m.apply('audit',name,(a,b),fn(a,b)):raise RuntimeError('source audit')
        m.apply('promote',name)
    return m


def teacher(labels, family, multiplier, shift, fn):
    # Evaluator implementation is separate from learner Syntax.mapping.
    labels=tuple(sorted(labels)); n=len(labels)
    def code(i):
        j=(multiplier*i+shift)%n
        return j^(j//2) if family=='gray' else j
    enc={v:code(i) for i,v in enumerate(labels)}; dec={v:k for k,v in enc.items()}
    return lambda x:dec[fn(enc[x[0]],enc[x[1]])]


def split(labels,seed):
    xs=list(product(labels,repeat=2));random.Random(seed).shuffle(xs)
    return xs[:20],xs[20:23],xs[23:]


def train(e,fn,pool,audits, budget=12):
    chosen=[]
    for _ in range(budget):
        action=e.advance(pool)
        if action['action']!='inquire':break
        x=action['request'];e.observe(x,fn(x));chosen.append(x)
    if e.live() and len(e.training)>=6:
        for x in audits:
            if not e.audit(x,fn(x)):break
    return chosen


def fixed(e,fn,train_x,audits):
    for x in train_x:
        if not e.live():break
        e.observe(x,fn(x))
    if e.live() and len(e.training)>=6:
        for x in audits:
            if not e.audit(x,fn(x)):break


def score(e,fn,held):
    rows=[]
    for x in held:
        y=e.predict(x);truth=fn(x); rows.append(dict(x=x,prediction=y,expected=truth))
    return dict(correct=sum(r['prediction']==r['expected'] for r in rows),incorrect=sum(r['prediction'] is not None and r['prediction']!=r['expected'] for r in rows),abstained=sum(r['prediction'] is None for r in rows),rows=rows,status=e.status())


def trial(seed,novel=False):
    rng=random.Random(seed); m=source(seed)
    def labels():return tuple(sorted(rng.sample(range(-32,0),8)))
    xor=lambda a,b:a^b
    parents=[]
    for shift in (0,1):
        lab=labels(); fn=teacher(lab,'ordinal',1,shift,xor); p,a,h=split(lab,rng.randrange(10**8));e=Episode(m,lab)
        train(e,fn,p,a); parents.append(e)
    abstract=Abstraction.induce(*parents)
    results=[]
    for fam,mult,shift,op in [('ordinal',1,2 if novel else 0,xor),('ordinal',1,7 if novel else 1,xor),('ordinal',3,2,xor),('gray',3,5,lambda a,b:a&b),('ordinal',5,3,lambda a,b:a|b)]:
        lab=labels(); f=teacher(lab,fam,mult,shift,op);p,a,h=split(lab,rng.randrange(10**8));e=Episode(m,lab)
        tr=train(e,f,p,a); t=Episode(m,lab,abstract);fixed(t,f,tr,a);raw=Episode(m,lab,raw=True)
        adaptive,cost=fit_with_abstraction(m,lab,abstract,[(x,f(x)) for x in tr])
        if adaptive.live() and len(adaptive.training)>=6:
            for x in a:
                if not adaptive.audit(x,f(x)):break
        scalar=ScalarEpisode()
        for x in tr:
            if scalar.blocked:break
            scalar.observe(x,f(x))
        if not scalar.blocked and len(scalar.training)>=2:
            for x in a:
                if not scalar.audit(x,f(x)):break
        scalar_rows=[dict(x=x,prediction=scalar.predict(x),expected=f(x)) for x in h]
        scalar_score=dict(correct=sum(r['prediction']==r['expected'] for r in scalar_rows),incorrect=sum(r['prediction'] is not None and r['prediction']!=r['expected'] for r in scalar_rows),abstained=sum(r['prediction'] is None for r in scalar_rows),rows=scalar_rows,status=scalar.status())
        ad=score(adaptive,f,h);ad['status']['attempts']=cost['candidate_programs'];ad['expanded']=cost['expanded']
        results.append(dict(family=fam,mult=mult,shift=shift,training=tr,audits=a,full=score(e,f,h),abstract=score(t,f,h),raw=score(raw,f,h),adaptive=ad,scalar=scalar_score))
    # Independent evaluator's random mapping is not expressible by the family.
    lab=labels();p,a,h=split(lab,rng.randrange(10**8)); table={x:rng.choice(lab) for x in product(lab,repeat=2)}
    e=Episode(m,lab);train(e,table.__getitem__,p,a);negative=score(e,table.__getitem__,h)
    # Hidden exception intentionally outside training and audit.
    lab=labels();f=teacher(lab,'ordinal',1,0,xor);p,a,h=split(lab,rng.randrange(10**8));e=Episode(m,lab);train(e,f,p,a)
    x=h[0]; before=e.predict(x); wrong=next(v for v in lab if v!=f(x)); blocked=False
    if before is not None:e.audit(x,wrong);blocked=e.predict(h[1]) is None
    return dict(seed=seed,template=abstract.status(),parents=[e.status() for e in parents],results=results,negative=negative,shift=dict(prediction=before,expected=wrong,blocked=blocked))


def campaign(seed,trials,novel=False):
    rows=[trial(seed+i,novel) for i in range(trials)]; out={}
    for arm in ('full','abstract','raw','adaptive','scalar'):
        cases=[r[arm] for t in rows for r in t['results']];out[arm]={k:sum(r[k] for r in cases) for k in ('correct','incorrect','abstained')};out[arm]['candidate_programs']=sum(r['status']['attempts'] for r in cases) if arm!='scalar' else None
    out['negative']={k:sum(t['negative'][k] for t in rows) for k in ('correct','incorrect','abstained')}
    out['hidden_exception']=dict(errors=sum(t['shift']['prediction'] is not None and t['shift']['prediction']!=t['shift']['expected'] for t in rows),quarantined=sum(t['shift']['blocked'] for t in rows))
    # Separate in-family transfer and out-of-family negative transfer.
    for name,subset in [('matched',slice(0,2)),('other',slice(2,5))]:
        out[name]={}
        for arm in ('full','abstract','raw','adaptive','scalar'):
            cases=[r[arm] for t in rows for r in t['results'][subset]];out[name][arm]={k:sum(r[k] for r in cases) for k in ('correct','incorrect','abstained')};out[name][arm]['candidate_programs']=sum(r['status']['attempts'] for r in cases) if arm!='scalar' else None
    return dict(seed=seed,trials=trials,summary=out,episodes=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=184000);p.add_argument('--trials',type=int,default=24);p.add_argument('--output');p.add_argument('--check',action='store_true');p.add_argument('--novel',action='store_true');a=p.parse_args();r=campaign(a.seed,a.trials,a.novel);print(json.dumps(r['summary'],indent=2))
    if a.output:
        with open(a.output,'w',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    if a.check:
        assert r['summary']['full']['correct']>r['summary']['raw']['correct'] and r['summary']['matched']['abstract']['correct']>0
        assert r['summary']['matched']['abstract']['candidate_programs']<r['summary']['matched']['full']['candidate_programs']
        assert r['summary']['hidden_exception']['errors']==r['summary']['hidden_exception']['quarantined']

if __name__=='__main__':main()
