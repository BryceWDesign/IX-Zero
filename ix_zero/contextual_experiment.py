"""Reproduce contextual recovery with held-out evidence and negative controls."""
import argparse,json,random
from itertools import product
from .acce import Workspace
from .contextual import Domain, Episode
from .representation_experiment import source


def sources(seed):
    mind=source(seed); ws=Workspace().operators(mind)
    return mind, ws._caps['op_word_a'], (ws._caps['op_word_b'],ws._caps['op_word_c'])


def evaluate(ep, teacher, xs):
    rows=[dict(input=x, prediction=ep.predict(x), expected=teacher(x)) for x in xs]
    return dict(correct=sum(r['prediction']==r['expected'] for r in rows),
                incorrect=sum(r['prediction'] is not None and r['prediction']!=r['expected'] for r in rows),
                abstained=sum(r['prediction'] is None for r in rows),rows=rows)


def trial(seed):
    rng=random.Random(seed); mind,base,alternatives=sources(seed)
    labels=tuple(range(8)); contexts=('ordinary','changed','other'); d=Domain(labels,contexts)
    normals=[x for x in d.inputs if x[0]!='changed'];rng.shuffle(normals)
    rules=[('operation',lambda x: x[0]=='changed',lambda a,b:a&b),
           ('ordering',lambda x:x[0]=='changed' and x[1]>x[2],lambda a,b:a|b),
           ('coordinate',lambda x:x[0]=='changed' and x[1]==seed%8,lambda a,b:a&b),
           ('isolated',lambda x:x==('changed',seed%8,(seed+3)%8),lambda a,b:7)]
    result=[]
    for name,guard,alternate in rules:
        def truth(x):return alternate(*x[1:]) if guard(x) else x[1]^x[2]
        possible=[x for x in d.inputs if truth(x)!=x[1]^x[2]]
        if not possible: # isolated constant can equal baseline; use a different explicit teacher constant
            special=('changed',seed%8,(seed+3)%8)
            def truth(x):return (special[1]^special[2])^1 if x==special else x[1]^x[2]
            possible=[special]
        rng.shuffle(possible); counterexample=possible[0]
        old=Episode(d,base,alternatives)
        for x in normals[:12]:old.observe(x,truth(x))
        for x in normals[12:15]:assert old.audit(x,truth(x))
        before=old.predict(counterexample);old.contradict(counterexample,truth(counterexample));blocked=old.predict(normals[15]) is None
        # Partition by input, not prediction. Reserve three changed and three ordinary audits.
        available=[x for x in d.inputs if x not in normals[:15] and x!=counterexample];rng.shuffle(available)
        changed=[x for x in available if x[0]=='changed']; stable=[x for x in available if x[0]!='changed']
        audit=changed[:4]+stable[:4]; pool=changed[4:36]+stable[4:20]; held=[x for x in available if x not in audit and x not in pool]
        def learn(active, budget=None):
            ep=old.recover(); questions=[]
            for _ in range(24 if budget is None else budget):
                if not ep.live():break
                if active:
                    a=ep.advance(pool)
                    if a['action']!='inquire':break
                    q=a['inquiry'];x=q['input'];questions.append(q)
                else:
                    x=next((x for x in pool if x not in ep.training),None)
                    if x is None:break
                ep.observe(x,truth(x))
            if ep.live():
                for x in audit:
                    if not ep.audit(x,truth(x)):break
            return ep,questions
        active,qs=learn(True); passive,_=learn(False,len(qs));r=dict(family=name,counterexample=dict(input=counterexample,prediction=before,expected=truth(counterexample)),old_blocked=blocked,questions=qs,audit_inputs=audit,active=evaluate(active,truth,held),passive=evaluate(passive,truth,held),active_status=active.status(),passive_status=passive.status())
        result.append(r)
    # Unsupported two-context exception (single-context guard cannot represent it).
    old=Episode(d,base,alternatives)
    def other_truth(x):return x[1]&x[2] if x[0]!='ordinary' else x[1]^x[2]
    ordinary=[x for x in d.inputs if x[0]=='ordinary'];
    for x in ordinary[:12]:old.observe(x,other_truth(x))
    for x in ordinary[12:15]:old.audit(x,other_truth(x))
    wrong=[x for x in d.inputs if other_truth(x)!=x[1]^x[2]]
    old.contradict(wrong[0],other_truth(wrong[0])); rec=old.recover()
    for x in wrong[1:]:
        if not rec.live():break
        rec.observe(x,other_truth(x))
    negative=evaluate(rec,other_truth,[x for x in d.inputs if x not in rec.training]);negative['status']=rec.status()
    return dict(seed=seed,episodes=result,negative=negative)


def campaign(seed,trials):
    rows=[trial(seed+i) for i in range(trials)]; episodes=[e for r in rows for e in r['episodes']]
    summary={arm:{k:sum(e[arm][k] for e in episodes) for k in ('correct','incorrect','abstained')} for arm in ('active','passive')}
    summary['families']={name:{arm:{k:sum(e[arm][k] for e in episodes if e['family']==name) for k in ('correct','incorrect','abstained')} for arm in ('active','passive')} for name in ('operation','ordering','coordinate','isolated')}
    summary['initial_errors']=sum(e['counterexample']['prediction']!=e['counterexample']['expected'] for e in episodes)
    summary['quarantined']=sum(e['old_blocked'] for e in episodes)
    summary['questions']=sum(len(e['questions']) for e in episodes)
    summary['informative_questions']=sum(q['disagreement']>0 for e in episodes for q in e['questions'])
    summary['negative']={k:sum(r['negative'][k] for r in rows) for k in ('correct','incorrect','abstained')}
    return dict(seed=seed,trials=trials,summary=summary,episodes=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=196000);p.add_argument('--trials',type=int,default=24);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args();r=campaign(a.seed,a.trials);print(json.dumps(r['summary'],indent=2))
    if a.output:
        with open(a.output,'w',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    if a.check:
        s=r['summary']; assert s['initial_errors']==s['quarantined']==a.trials*4
        assert s['active']['correct']>0 and s['negative']['correct']==s['negative']['incorrect']==0

if __name__=='__main__':main()
