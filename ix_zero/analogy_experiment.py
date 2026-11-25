"""Seeded finite relational transfer experiment; target labels held by evaluator."""
import argparse
from itertools import permutations,product
import json
import random
from pathlib import Path
from .analogical import AnalogicalMind


def composition(a,b): return tuple(a[i] for i in b)


def fixtures(family,seed):
    rng=random.Random(seed)
    if family in ('rotation_phase','wrong_operation','wrong_graph','random_target','hidden_exception'):
        n=7; op=lambda a,b:(a+b)%n
        rows=[tuple((a+1)%n for a in range(n)),tuple((a+2)%n for a in range(n))]
        source_names=['north','nne','ene','east','ese','sse','south']
        target_names=['phaseA','phaseB','phaseC','phaseD','phaseE','phaseF','phaseG']
        target_op=(lambda a,b:(a-b)%n) if family=='wrong_operation' else op
        if family=='random_target':
            random_table={(a,b):rng.randrange(n) for a,b in product(range(n),repeat=2)}
            target_op=lambda a,b:random_table[a,b]
        if family=='hidden_exception':
            hidden=(rng.randrange(n),rng.randrange(n))
            target_op=lambda a,b:(op(a,b)+1)%n if (a,b)==hidden else op(a,b)
        target_rows=rows if family!='wrong_graph' else [rows[0],tuple(range(n))]
    elif family=='resource_signal':
        n=6;op=min;target_op=op
        rows=[tuple(min(a+1,n-1) for a in range(n)),tuple(max(a-1,0) for a in range(n))];target_rows=rows
        source_names=['empty','trace','low','medium','high','full'];target_names=['mute','faint','quiet','clear','loud','peak']
    elif family=='switch_membership':
        n=4;op=lambda a,b:a^b;target_op=op
        rows=[tuple(a^1 for a in range(n)),tuple(a^2 for a in range(n))];target_rows=rows
        source_names=['offoff','onoff','offon','onon'];target_names=['emptyset','alpha','beta','both']
    elif family=='permutation_tiles':
        ps=tuple(permutations(range(3)));n=len(ps)
        op=lambda a,b:ps.index(composition(ps[a],ps[b]));target_op=op
        rows=[tuple(op(a,ps.index((1,0,2))) for a in range(n)),tuple(op(a,ps.index((1,2,0))) for a in range(n))];target_rows=rows
        source_names=['identity','swapBC','swapAB','rotateABC','rotateACB','swapAC'];target_names=['layoutA','layoutB','layoutC','layoutD','layoutE','layoutF']
    else:raise ValueError('unknown family')
    # Randomize the names assigned to latent states and order of observations.
    rng.shuffle(source_names);rng.shuffle(target_names)
    def encode(names,rs,prefix):
        order=list(range(n));rng.shuffle(order)
        relation_order=list(range(len(rs)));rng.shuffle(relation_order)
        return [names[i] for i in order],{f'{prefix}{k}':[names[rs[j][i]] for i in order] for k,j in enumerate(relation_order)}
    s,rs=encode(source_names,rows,'sourceRel');t,rt=encode(target_names,target_rows,'targetRel')
    measured=[[[source_names[a],source_names[b]],source_names[op(a,b)]] for a,b in product(range(n),repeat=2)]
    rng.shuffle(measured)
    all_target=[[[target_names[a],target_names[b]],target_names[target_op(a,b)]] for a,b in product(range(n),repeat=2)]
    rng.shuffle(all_target)
    return s,rs,measured,t,rt,all_target


def run_case(family,seed,mode):
    s,rs,labels,t,rt,targets=fixtures(family,seed)
    m=AnalogicalMind();m.apply('source','source',s,rs,labels);m.apply('new','target',t,rt,mode)
    # Fixed label partition, identical across all arms. Audit labels never filter models.
    train,audit,test=targets[:2],targets[2:5],targets[5:]
    e=m.episodes['target']
    for x,y in train:m.apply('observe','target',x,y)
    aud=[]
    for x,y in audit:
        if e.blocked or e._prediction(e.graph.pair(x)) is None:break
        passed=m.apply('audit','target',x,y);aud.append([x,y,passed])
    preds=[[x,y,e.predict(x)] for x,y in test]
    c=sum(p==y for _,y,p in preds);w=sum(p is not None and p!=y for _,y,p in preds)
    before=e.status(); feedback=None
    if w:
        x,y,p=next(row for row in preds if row[2] is not None and row[2]!=row[1])
        passed=m.apply('audit','target',x,y)
        feedback=dict(input=x,observed=y,prior_prediction=p,audit_passed=passed,
                      quarantined=e.blocked,remaining_predictions=sum(e.predict(xx) is not None for xx,_,_ in preds))
    return dict(family=family,seed=seed,mode=mode,correct=c,incorrect=w,abstained=len(test)-c-w,
                source=[s,rs,labels],target_graph=[t,rt],training=train,audits=aud,audit_partition=audit,
                predictions=preds,status=before,feedback=feedback)


def run_autonomous(family,seed):
    s,rs,labels,t,rt,targets=fixtures(family,seed)
    m=AnalogicalMind();m.apply('source','source',s,rs,labels);m.apply('new','target',t,rt,'structural')
    lookup={tuple(x):y for x,y in targets};e=m.episodes['target'];steps=[]
    # The evaluator answers policy inquiries and chooses fresh audit inputs.
    # Remaining unqueried inputs are the test set; this is an integration check,
    # not a fixed-partition comparison.
    for _ in range(12):
        r=m.advance('target');a=r['action'];steps.append(r)
        if a=='inquire':x=r['result'];m.apply('observe','target',x,lookup[tuple(x)])
        elif a=='audit':
            x=next((x for x,y in targets if e.graph.pair(x) not in e.training and e.graph.pair(x) not in e.audits and e._prediction(e.graph.pair(x)) is not None),None)
            if x is None:break
            m.apply('audit','target',x,lookup[tuple(x)])
        else:break
    predictions=[[x,y,e.predict(x)] for x,y in targets if e.graph.pair(x) not in e.training and e.graph.pair(x) not in e.audits]
    return dict(family=family,seed=seed,promoted='target' in m.sources,steps=steps,status=e.status(),predictions=predictions,
                events=m.events)


def run_operator_chain(seed):
    from .operators import OperatorMind
    om=OperatorMind();om.apply('new','rule',2,5,False);t=om.episodes['rule']
    xs=list(t.domain.inputs);random.Random(seed).shuffle(xs)
    for x in xs[:10]:om.apply('observe','rule',x,x[0]^x[1])
    for x in xs[10:13]:
        if not t.resolved() or not om.apply('audit','rule',x,x[0]^x[1]):break
    if not t.resolved() or len(t.audits)<3:
        return dict(seed=seed,promoted=False,operator_events=om.events,first=[],second=[],quarantined=False)
    om.apply('promote','rule')
    m=AnalogicalMind();s=['offoff','onoff','offon','onon'];r={'flipA':[s[i^1] for i in range(4)],'flipB':[s[i^2] for i in range(4)]}
    m.apply('operator_source','bit_source',s,r,om.events,'rule')
    _,_,_,target,rel,labels=fixtures('switch_membership',seed)
    m.apply('new','first',target,rel,'structural')
    def fit(name,labels):
        e=m.episodes[name]
        for x,y in labels[:2]:m.apply('observe',name,x,y)
        for x,y in labels[2:5]:
            if e._prediction(e.graph.pair(x)) is None or e.blocked:break
            if not m.apply('audit',name,x,y):break
        preds=[[x,y,e.predict(x)] for x,y in labels[5:]]
        if not e.blocked and len(e.audits)>=3 and len(e.models)==1:m.apply('promote',name)
        return preds
    first=fit('first',labels)
    # Second transfer selects the newly acquired target alone; the original
    # bit-vector model is not among its candidate sources.
    if 'first' not in m.sources:return dict(seed=seed,promoted=False,operator_events=om.events,first=first,second=[],quarantined=False,events=m.events)
    _,_,_,target,rel,labels=fixtures('switch_membership',seed+20000)
    rename={x:'layout_'+x for x in target};target=[rename[x] for x in target]
    rel={k:[rename[x] for x in row] for k,row in rel.items()}
    labels=[[[rename[x] for x in pair],rename[y]] for pair,y in labels]
    m.apply('new','second',target,rel,'structural',['first']);second=fit('second',labels)
    om.apply('retract','rule');m.reconcile_operators(om)
    return dict(seed=seed,promoted='second' in m.sources,operator_events=om.events,first=first,second=second,
                quarantined=m.episodes['first'].blocked and m.episodes['second'].blocked,events=m.events)


def campaign(trials,seed):
    positive=('rotation_phase','resource_signal','switch_membership','permutation_tiles')
    families=positive+('wrong_operation','wrong_graph','random_target','hidden_exception')
    rows=[run_case(f,seed+i,mode) for f in families for i in range(trials) for mode in ('structural','unstructured','local')]
    auto=[run_autonomous(f,seed+10000+i) for f in positive for i in range(trials)]
    chain=[run_operator_chain(seed+30000+i) for i in range(trials)]
    totals={}
    for r in rows:
        k=r['family']+'/'+r['mode'];v=totals.setdefault(k,dict(correct=0,incorrect=0,abstained=0,initial_models=0,mappings=0))
        for field in ('correct','incorrect','abstained'):v[field]+=r[field]
        for field in ('initial_models','mappings'):v[field]+=r['status'][field]
    return dict(seed=seed,trials=trials,totals=totals,cases=rows,autonomous=auto,
                autonomous_promotions=sum(r['promoted'] for r in auto),autonomous_trials=len(auto),
                operator_chain=chain,chain_promotions=sum(r['promoted'] for r in chain),
                chain_quarantined=sum(r['quarantined'] for r in chain),
                chain_correct=sum(y==p for r in chain for _,y,p in r['first']+r['second']),
                chain_predictions=sum(len(r['first'])+len(r['second']) for r in chain))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=88000);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=100:p.error('trials must be 1..100')
    r=campaign(a.trials,a.seed)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ('cases','autonomous','operator_chain')},indent=2))
    if a.check:
        for f in ('rotation_phase','resource_signal','switch_membership','permutation_tiles'):
            v=r['totals'][f+'/structural'];assert v['incorrect']==0 and v['correct']>0,(f,v)
        assert r['totals']['wrong_graph/structural']['correct']==0
        assert r['autonomous_promotions']>0
        assert r['chain_promotions']>0 and r['chain_quarantined']==r['chain_promotions']

if __name__=='__main__':main()
