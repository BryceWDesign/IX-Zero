"""Developer-authored, budget-capped component comparisons, not an AGI ranking.
Original Sally source is optional; no stand-in result is reported when unavailable.
"""
import argparse
import json
import random
from itertools import product
from .unified import UnifiedMind
from .unified_experiment import build_joint
from .scalar import ScalarEpisode


def coupled_truth(s,v,a,modulus,kind):
    """Evaluator only: numeric and symbolic channels affect each other."""
    s=set(s);name,x=a;u,w=v
    if kind=='gate':
        if ('raw',x) in s and u==w:
            s.remove(('raw',x));s.add(('ready',x));u+=2
        elif ('raw',x) in s:u+=1
    elif kind=='threshold':
        if ('raw',x) in s and w>=modulus//2:
            s.remove(('raw',x));s.add(('ready',x));u+=2
        elif ('raw',x) in s:u+=1
    elif kind=='reset':
        if ('raw',x) in s:s.remove(('raw',x));s.add(('ready',x));u,w=0,0
        else:u,w=1,1
    else:raise ValueError('unknown simulator')
    return frozenset(s),(u%modulus,w%modulus)


def row(seed,kind,modulus,phase,i,tag,v):
    x=f'{phase}{seed}_{i}';s=frozenset({(tag,x)});a=('prep',x)
    t,y=coupled_truth(s,v,a,modulus,kind);return s,v,a,t,y


def learn_coupled(seed,kind,modulus):
    m=UnifiedMind();m.apply('coupled_start',('prep','example'),2,modulus)
    # Same fixed full crossing of contexts and coordinates, with different names
    # for fitting, auditing and final testing. Held-out identities, not numeric ranges.
    combos=list(product(('raw','ready'),product(range(modulus),repeat=2)))
    rng=random.Random(seed);rng.shuffle(combos)
    for i,(tag,v) in enumerate(combos):m.apply('coupled_observe',*row(seed,kind,modulus,'train',i,tag,v))
    for i,(tag,v) in enumerate(combos[:6]):
        if m.coupled[('prep',1)].blocked:break
        m.apply('coupled_audit',*row(seed,kind,modulus,'audit',i,tag,v))
    return m,combos


def coupled_campaign(seeds):
    results=[]
    for seed in seeds:
        for kind in ('gate','threshold','reset'):
            modulus=3+2*(seed%2);m,combos=learn_coupled(seed,kind,modulus)
            counts=dict(correct=0,incorrect=0,abstained=0,plans=0,plan_success=0,plan_wrong=0,plan_abstained=0)
            for i,(tag,v) in enumerate(combos):
                s,v,a,t,y=row(seed,kind,modulus,'test',i,tag,v);p=m.predict((s,v),a)
                counts['abstained' if p is None else 'correct' if p==(t,y) else 'incorrect']+=1
                if tag!='raw':continue
                # Replay the evaluator-known state after one action. A plan may be empty if already satisfied.
                counts['plans']+=1
                q=m.plan((s,v),[a],present=t,absent=s-t,vector=y)['path']
                if q is None:counts['plan_abstained']+=1;continue
                f,z=s,v
                for ac in q:f,z=coupled_truth(f,z,ac,modulus,kind)
                counts['plan_success' if (f,z)==(t,y) else 'plan_wrong']+=1
            results.append(dict(seed=seed,kind=kind,modulus=modulus,**counts,model=m.coupled[('prep',1)].status()))
    return results


def sparse_campaign(seeds):
    results=[]
    for seed in seeds:
        for kind in ('gate','threshold','reset'):
            m=UnifiedMind();mod=5;m.apply('coupled_start',('prep','x'),2,mod)
            combos=list(product(('raw','ready'),product(range(mod),repeat=2)));random.Random(seed).shuffle(combos)
            for i,(tag,v) in enumerate(combos[:32]):m.apply('coupled_observe',*row(seed,kind,mod,'sparsefit',i,tag,v))
            for i,(tag,v) in enumerate(combos[32:38]):
                if m.coupled[('prep',1)].blocked:break
                m.apply('coupled_audit',*row(seed,kind,mod,'sparseaudit',i,tag,v))
            counts=dict(correct=0,incorrect=0,abstained=0)
            for i,(tag,v) in enumerate(combos[38:]):
                s,v,a,t,y=row(seed,kind,mod,'sparsetest',i,tag,v);p=m.predict((s,v),a)
                counts['abstained' if p is None else 'correct' if p==(t,y) else 'incorrect']+=1
            results.append(dict(seed=seed,kind=kind,fit_combinations=32,audit_combinations=6,test_combinations=12,**counts,model=m.coupled[('prep',1)].status()))
    return results


def recovery_campaign(seeds):
    results=[]
    for seed in seeds:
        old=build_joint(seed);a=('prep',f'shift{seed}');s=frozenset({('raw',a[1])});v=(0,0)
        t,y=coupled_truth(s,v,a,5,'gate');before=old.predict((s,v),a);old.apply('joint_audit',s,v,a,t,y)
        blocked=old.predict((s,v),a) is None
        # Recover from a known blocked separable model without reusing old audits.
        old.apply('coupled_start',a,2,5)
        combos=list(product(('raw','ready'),product(range(5),repeat=2)))
        for i,(tag,vec) in enumerate(combos):old.apply('coupled_observe',*row(seed,'gate',5,'relearn',i,tag,vec))
        for i,(tag,vec) in enumerate(combos[:6]):
            if old.coupled[('prep',1)].blocked:break
            old.apply('coupled_audit',*row(seed,'gate',5,'revalidate',i,tag,vec))
        outcomes=[0,0,0]
        for i,(tag,vec) in enumerate(combos):
            f,z,b,tf,tv=row(seed,'gate',5,'retest',i,tag,vec);p=old.predict((f,z),b)
            outcomes[2 if p is None else 0 if p==(tf,tv) else 1]+=1
        results.append(dict(seed=seed,wrong_before=before is not None and before!=(t,y),blocked_before_recovery=blocked,correct=outcomes[0],incorrect=outcomes[1],abstained=outcomes[2]))
    return results


SCALAR_WORLDS=(('product',lambda x,y:x*y),('difference',lambda x,y:x-y),('composition',lambda x,y:x*y+x),('absolute',lambda x,y:abs(x-y)),('offset',lambda x,y:x*y+2),('scaled',lambda x,y:3*(x-y)),('remainder',lambda x,y:x%y if y else 0),('division',lambda x,y:x//y if y else 0),('unstructured',None))


def scalar_campaign(seeds,with_sally=False):
    if with_sally:
        from ix_sally.cognition.developmental_learning import InductiveLearner
        from ix_sally.foundation import FoundationError
        sally=InductiveLearner()
    rows=[]
    for seed in seeds:
        for name,fn in SCALAR_WORLDS:
            rng=random.Random(f'{seed}-{name}');pool=list(product(range(-3,4),repeat=2));held=list(product((-8,-7,-6,-5,5,6,7,8),repeat=2));rng.shuffle(pool);rng.shuffle(held);pool=tuple(pool[:32]);held=tuple(held[:32]);mapping={x:rng.randrange(-50,51) for x in pool+held}
            oracle=(lambda p:mapping[p]) if fn is None else (lambda p:fn(*p))
            zero=ScalarEpisode();used=[]
            for _ in range(7):
                q=zero.query(pool)
                if q is None:break
                zero.observe(q,oracle(q));used.append(q)
            for _ in range(3):
                q=next((p for p in pool if p not in used),None)
                if q is None or zero.blocked:break
                ok=zero.audit(q,oracle(q));used.append(q)
                if not ok:break
            arms=[('IX-Zero',zero.predict,len(used))]
            if with_sally:
                sq=[]
                def observed(p):sq.append(p);return oracle(p)
                try:
                    model,*_=sally.infer(pool=pool,observe=observed,budget=10,active=True)
                    pred=model.evaluate
                except FoundationError:pred=lambda p:None
                arms.append(('IX-Sally original InductiveLearner',pred,len(sq)))
            for arm,predict,probes in arms:
                counts=dict(correct=0,incorrect=0,abstained=0)
                for p in held:
                    y=predict(p);counts['abstained' if y is None else 'correct' if y==oracle(p) else 'incorrect']+=1
                rows.append(dict(seed=seed,world=name,arm=arm,probes=probes,**counts))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=900);p.add_argument('--with-sally',action='store_true');p.add_argument('--check',action='store_true',help='fail on known IX-Zero fixture regression');a=p.parse_args()
    if not 1<=a.trials<=32:p.error('trials must be 1..32')
    seeds=range(a.seed,a.seed+a.trials)
    result=dict(seeds=list(seeds),coupled=coupled_campaign(seeds),sparse=sparse_campaign(seeds),recovery=recovery_campaign(seeds),scalar=scalar_campaign(seeds,a.with_sally),native_sally=a.with_sally,native_julia='not executed by this harness')
    print(json.dumps(result,indent=2))
    if a.check:
        for k in ('coupled','sparse','recovery'):
            if any(r['incorrect'] or r['abstained'] or not r['correct'] for r in result[k]):raise SystemExit('joint fixture regression: '+k)
        if any(r['plan_wrong'] or r['plan_abstained'] or r['plans']!=r['plan_success'] for r in result['coupled']):raise SystemExit('planning regression')
        if not all(r['wrong_before'] and r['blocked_before_recovery'] for r in result['recovery']):raise SystemExit('recovery control regression')
        for r in result['scalar']:
            if r['arm']!='IX-Zero':continue
            if r['world']=='unstructured':
                if r['abstained']!=32:raise SystemExit('unstructured control regression')
            elif r['correct']!=32:raise SystemExit('scalar fixture regression')

if __name__=='__main__':main()
