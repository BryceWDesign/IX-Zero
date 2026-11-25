"""Held-out dispatch evaluation with randomized grounded vocabularies and aliases.
Source operators and scalar/relational models are fixed across seeds. Dispatch
bindings and language episodes are fitted independently on each seed.
"""
import argparse, json, random
from itertools import product
from pathlib import Path
from .acce import Workspace, CognitiveDispatch
from .acce_demo import build, objects, teach
from .language import LanguageMind
from .language_experiment import fixtures

ORDER=lambda a,b:(a>b)-(a<b)
PAIRS=list(product(range(4),repeat=2))


def run(seed,om,um,am):
    rng=random.Random(seed); f=fixtures(seed); lm=LanguageMind(); lm.apply('new','test')
    # Teacher names are arbitrary and distinct from the capability catalog.
    alias={k:f'goal_{seed}_{i}' for i,k in enumerate(('inspect','compare','rule'))}
    for t,s,y in f['train']:lm.apply('observe','test',t,s,[alias[y[0]],*y[1:]])
    for t,s,y in f['audit']:
        if not lm.apply('audit','test',t,s,[alias[y[0]],*y[1:]]):raise RuntimeError('linguistic audit failed')
    ws=Workspace().primitives().operators(om).scalars(um).analogies(am,om)
    d=CognitiveDispatch(ws); no=CognitiveDispatch(ws)
    # Distinct numerical pairs in training, audit and test. All labels come from
    # the evaluator. The composition-free arm cannot express object projections.
    tasks={'inspect':([(0,),(1,)],[(2,),(3,),(4,)],lambda a:a),
           'compare':([(0,3),(3,0)],[(1,1),(1,3),(2,0)],ORDER),
           'rule':([(1,1),(1,2)],[(0,2),(2,3),(3,3)],lambda a,b:a^b)}
    evidence={}; partitions={}; traces={}
    for verb,(tr,au,truth) in tasks.items():
        ins=('object',)*len(tr[0]); b=d.new(alias[verb],ins,'integer'); baseline=no.new(alias[verb],ins,'integer',False)
        traces[verb]=[b.advance([objects(*x) for x in tr])]
        for x in tr:b.observe(objects(*x),truth(*x))
        traces[verb].append(b.advance([objects(*x) for x in au]))
        for x in au:
            if not b.audit(objects(*x),truth(*x)):raise RuntimeError('dispatch audit failed')
        traces[verb].append(b.advance())
        # No compatible program is available to the ablation for binary tasks.
        # Unary direct value is already compatible and receives the same labels.
        if baseline.models:teach(baseline,[(objects(*x),truth(*x)) for x in tr],[(objects(*x),truth(*x)) for x in au])
        evidence[verb]=b.status(); partitions[verb]=dict(train=tr,audit=au)
    records=[]; unsupported=[]
    for text,rows,y in f['test']:
        verb,*refs=y; tr,au,truth=tasks[verb]; arity=len(tr[0])
        if len(refs)!=arity:
            r=d.interpret(lm.episodes['test'],text,rows,{})
            unsupported.append([text,y,r]);continue
        pool=[x for x in ([(n,) for n in range(5,33)] if arity==1 else PAIRS) if x not in tr+au]
        x=rng.choice(pool); vals={i:rng.randrange(4) for i,_,_ in rows}; vals.update(zip(refs,x))
        p=d.interpret(lm.episodes['test'],text,rows,vals);q=no.interpret(lm.episodes['test'],text,rows,vals)
        legacy=lm.compute('test',text,rows,vals,om)
        records.append(dict(text=text,scene=rows,intent=y,values=vals,expected=truth(*x),prediction=p.get('value'),ablation=q.get('value'),legacy=legacy,programs=p.get('evidence',{}).get('programs',[]),inputs=x))
    # Structured typed tasks also route to an existing signed scalar learner and
    # an existing relational learner. These are not language-comprehension tests.
    numeric=d.new('numeric_alias',('object','object'),'integer')
    teach(numeric,[(objects(a,b),a*b) for a,b in [(2,3),(3,3)]],[(objects(a,b),a*b) for a,b in [(4,-2),(5,2),(2,2)]])
    scalar=[]
    for x in rng.sample([x for x in product(range(-8,9),repeat=2) if x not in [(2,3),(3,3),(4,-2),(5,2),(2,2)]],20):
        scalar.append([x,x[0]*x[1],d.execute('numeric_alias',objects(*x)).get('value')])
    relation=d.new('relation_alias',('object','object'),'relation')
    train=[(1,1),(1,2)];audit=[(0,2),(2,3),(3,3)]
    teach(relation,[(objects(*x,True),'s'+str(x[0]^x[1])) for x in train],[(objects(*x,True),'s'+str(x[0]^x[1])) for x in audit])
    rel=[[x,'s'+str(x[0]^x[1]),d.execute('relation_alias',objects(*x,True)).get('value')] for x in PAIRS if x not in train+audit]
    # Hidden exception is deliberately introduced after successful validation.
    b=d.bindings[(alias['compare'],('object','object'))];args=objects(1,2);before=b.predict(args);corrective=0
    passed=b.audit(args,corrective);after=b.predict(objects(2,3))
    return dict(seed=seed,alias=alias,evidence=evidence,partitions=partitions,traces=traces,language_train=f['train'],language_audit=f['audit'],records=records,unsupported=unsupported,scalar=scalar,relational=rel,hidden_exception=dict(before=before,expected=corrective,audit_passed=passed,after=after))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=140000);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=50:p.error('trials must be 1..50')
    _,om,um,am,_,_,_=build();cases=[run(a.seed+i,om,um,am) for i in range(a.trials)]
    rows=[r for c in cases for r in c['records']];unsupported=[r for c in cases for r in c['unsupported']]
    def score(key):return dict(total=len(rows),correct=sum(r[key]==r['expected'] for r in rows),incorrect=sum(r[key] is not None and r[key]!=r['expected'] for r in rows),abstained=sum(r[key] is None for r in rows))
    def structured(key):
        r=[r for c in cases for r in c[key]];return dict(total=len(r),correct=sum(y==p for _,y,p in r),incorrect=sum(p is not None and y!=p for _,y,p in r),abstained=sum(p is None for _,_,p in r))
    summary=dict(composed=score('prediction'),no_composition=score('ablation'),exact_name_baseline=score('legacy'),unsupported_total=len(unsupported),unsupported_abstained=sum(r[2].get('abstained',False) for r in unsupported),scalar=structured('scalar'),relational=structured('relational'),hidden_errors=sum(c['hidden_exception']['before']!=c['hidden_exception']['expected'] for c in cases),quarantined=sum(c['hidden_exception']['after'] is None and not c['hidden_exception']['audit_passed'] for c in cases))
    print(json.dumps(summary,indent=2))
    if a.output:a.output.write_text(json.dumps(dict(seed=a.seed,trials=a.trials,summary=summary,cases=cases,operator_events=om.events,scalar_events=um.events,analogy_events=am.events),default=lambda v:v.__dict__,separators=(',',':'))+'\n',encoding='utf-8')
    if a.check:
        assert summary['composed']['correct']==len(rows) and summary['composed']['incorrect']==0
        assert summary['no_composition']['correct']<len(rows) and summary['exact_name_baseline']['correct']==0
        assert summary['unsupported_total']==summary['unsupported_abstained']
        assert summary['scalar']['correct']==summary['scalar']['total'] and summary['relational']['correct']==summary['relational']['total']
        assert summary['hidden_errors']==summary['quarantined']==a.trials
if __name__=='__main__':main()
