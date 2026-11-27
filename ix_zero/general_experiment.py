"""Authored native reasoning evaluation with hidden functions and matched controls."""
import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from .general import General, canonical
from .continual_experiment import Host


def call(m,command,**kwargs):return m.execute(dict(command=command,**kwargs))


def scope_for(seed):
    r=Random(seed)
    xs=[(),*((i,) for i in range(-8,16))]
    while len(xs)<96:
        x=tuple(r.randint(-9,9) for _ in range(r.randint(2,7)))
        if x not in xs:xs.append(x)
    r.shuffle(xs)
    return xs,xs[:48],xs[48:60],xs[60:]


def learn(m,key,seed,oracle,*,corrupt=False):
    xs,train,audit,held=scope_for(seed)
    start=call(m,'begin',key=key,scope=xs,train=train,audit=audit,budget=2)
    trace=[]
    for _ in range(100):
        q=call(m,'next')
        if q['action'] in ('validated','failed'):break
        x=q['input']; y=tuple(oracle(x))
        if corrupt and q['action']=='audit':y=(999999,)
        trace.append({'request':q,'measurement':y});call(m,'answer',y=y)
    else:raise RuntimeError('request limit')
    rows=[]
    for x in held:
        # The evaluator does not call the primitive interpreter for truth.
        expected=tuple(oracle(x));predicted=m.programs.predict(key,x)
        rows.append(dict(input=x,expected=expected,predicted=predicted))
    return dict(skill=key,terminal=q,initial_candidates=start['candidates'],trace=trace,rows=rows,
                correct=sum(z['predicted']==z['expected'] for z in rows),wrong=sum(z['predicted'] is not None and z['predicted']!=z['expected'] for z in rows),abstained=sum(z['predicted'] is None for z in rows))


def program_chain(seed,reuse):
    m=General(reuse)
    # Compose two different independently acquired two-step transformations.
    def first(x):return tuple(v+1 for v in reversed(x))
    def second(x):return tuple(sum(x[:i+1]) for i in range(len(x)) if sum(x[:i+1])%2==0)
    def combined(x):return second(first(x))
    rows=[learn(m,'first',seed,first),learn(m,'second',seed+1,second),learn(m,'combined',seed+2,combined)]
    with TemporaryDirectory() as d:
        p=Path(d)/'workspace.json';m.save(p);restored=General.load(p);same=m.fingerprint()==restored.fingerprint()
    invalidation=None
    if 'combined' in m.programs.skills:
        x=m.programs.skills['first']['scope'][0];p=m.programs.predict('first',x)
        call(m,'challenge',key='first',x=x,y=[999999] if p!=(999999,) else [])
        invalidation=m.programs.skills['combined']['state']
    return dict(reuse=reuse,rows=rows,replay=same,invalidation=invalidation)


def language_math(seed):
    m=General();r=Random(seed);rows=[]
    # Use fresh identifiers to avoid memorized facts, and exact rational answers.
    names=['n'+str(v) for v in r.sample(range(10000),4)]
    a,b,c,d=names
    call(m,'ingest',source='source_a',text=f'{a} is inside {b}.\n{b} is inside {c}.')
    call(m,'ingest',source='source_b',text=f'{c} is inside {d}.')
    for text,expected in [(f'Is {a} inside {d}?',True),(f'Is {d} inside {a}?',None)]:
        result=call(m,'ask',text=text);rows.append(dict(query=text,expected=expected,result=result,correct=result.get('value')==expected))
    call(m,'ingest',source='conflict',text=f'{b} is not inside {c}.')
    blocked=call(m,'ask',text=f'Is {a} inside {d}?')
    conflict=call(m,'ask',text=f'Is {b} inside {c}?')
    call(m,'retract',source='conflict')
    restored=call(m,'ask',text=f'Is {a} inside {d}?')
    for _ in range(24):
        a0,b0,c0,d0=[r.randint(-20,20) for _ in range(4)]
        if a0==c0:c0+=1
        expected=str(Fraction(d0-b0,a0-c0));text=f'Solve {a0}*x + ({b0}) = {c0}*x + ({d0})'
        result=call(m,'ask',text=text);rows.append(dict(query=text,expected=expected,result=result,correct=result.get('solution')==expected))
    listing=call(m,'ask',text=f'Which items are inside {d}?')
    listing_ok={e['entity'] for e in listing['matches']}=={a,b,c}
    conditional=call(m,'ask',text=f'If {a} is inside {d} then calculate 4*(3+2)')
    unsupported=call(m,'ask',text='Describe the history of astronomy')
    return dict(rows=rows,blocked=blocked,conflict=conflict,restored=restored,conditional=conditional,unsupported=unsupported,listing=listing,listing_ok=listing_ok)


def integrated(seed):
    m=General();h=Host('word',[2,3],seed,'nand');trace=[]
    def core(command,**a):return call(m,'core',request=dict(command=command,**a))
    core('begin',family=h.family,widths=h.widths,initial=h.widths[0],**h.pool(h.widths[0],0),seed=seed,budget=1)
    for _ in range(400):
        q=core('next')
        if q['action'] in ('complete','failed'):break
        v=h.answer(q,m.core.retry);trace.append(dict(request=q,answer=v));core('answer',value=v)
    else:raise RuntimeError('core request limit')
    episode=m.core.active
    skill=learn(m,'transform',seed+9,lambda x:tuple(v+1 for v in x))
    rows=[]
    for a,b in product(range(8),repeat=2):
        text=f'Apply transform to core {episode} at width 3 with {a} and {b}'
        result=call(m,'ask',text=text);expected=(8-(a&b),)
        rows.append(dict(query=text,result=result,expected=expected,correct=result.get('value')==expected))
    # Plan a new goal, using the independently induced increment rule, not oracle transitions.
    plans=[]
    for start,goal in [(0,4),(-2,3),(7,7),(3,-1),(14,16)]:
        result=call(m,'ask',text=f'Plan from [{start}] to [{goal}]')
        expected=goal-start if goal>=start else None
        steps=result.get('steps');ok=(steps is None and expected is None) or (steps is not None and len(steps)==expected and all(z['after']==(z['before'][0]+1,) for z in steps) and (not steps or steps[0]['before']==(start,) and steps[-1]['after']==(goal,)))
        plans.append(dict(start=start,goal=goal,expected_steps=expected,result=result,correct=ok))
    math_bridge=call(m,'ask',text='Apply transform to result of calculate 2*3')
    with TemporaryDirectory() as d:
        p=Path(d)/'workspace.json';m.save(p);restored=General.load(p);same=m.fingerprint()==restored.fingerprint()
    # A changed observation must block the composed answer even when the new skill is valid.
    x=(3,0,0);core('challenge',episode=episode,input=x,value=0)
    after=call(m,'ask',text=f'Apply transform to core {episode} at width 3 with 0 and 0')
    return dict(core_terminal=q,core_trace=trace,skill=skill,rows=rows,replay=same,after=after,plans=plans,math_bridge=math_bridge)


def constraint_tasks(seed):
    rng=Random(seed);m=General();rows=[]
    # Exhaustive reference enumeration is independent of the search implementation.
    for _ in range(18):
        domains={k:sorted(rng.sample(range(-3,6),4)) for k in 'abc'}
        total=rng.randint(-5,10);diff=rng.randint(-4,4)
        rules=[{'op':'distinct','vars':list('abc')},{'op':'sum_eq','vars':list('abc'),'value':total},{'op':'diff_eq','vars':['a','b'],'value':diff}]
        truth=[dict(a=a,b=b,c=c) for a,b,c in product(*(domains[k] for k in 'abc')) if len({a,b,c})==3 and a+b+c==total and a-b==diff]
        expected='unsatisfiable' if not truth else 'unique' if len(truth)==1 else 'ambiguous'
        result=call(m,'solve_constraints',domains=domains,constraints=rules)
        correct=result['status']==expected and len(result['solutions'])==min(2,len(truth)) and all(x in truth for x in result['solutions'])
        rows.append(dict(domains=domains,constraints=rules,expected=expected,oracle_solutions=truth,result=result,correct=correct))
    return rows

def campaign(seed,trials):
    cases=[]
    for i in range(trials):
        s=seed+100*i
        cases.append(dict(seed=s,full=program_chain(s,True),control=program_chain(s,False),language_math=language_math(s),constraints=constraint_tasks(s),integrated=integrated(s),negative=learn(General(),'corrupt',s,lambda x:x[::-1],corrupt=True),unsupported_program=learn(General(),'unsupported',s,lambda x:tuple(v**3 for v in x))))
    return dict(seed=seed,trials=trials,cases=cases)


def summary(c):
    cases=c['cases'];out={}
    for arm in ('full','control'):
        rows=[r for v in cases for r in v[arm]['rows']];out[arm]=dict(acquisitions=sum(r['terminal']['action']=='validated' for r in rows),correct=sum(r['correct'] for r in rows),wrong=sum(r['wrong'] for r in rows),abstained=sum(r['abstained'] for r in rows),final=sum(v[arm]['rows'][-1]['terminal']['action']=='validated' for v in cases),replays=sum(v[arm]['replay'] for v in cases))
    out.update(language_math_correct=sum(r['correct'] for v in cases for r in v['language_math']['rows']),language_math_total=sum(len(v['language_math']['rows']) for v in cases),integrated_correct=sum(r['correct'] for v in cases for r in v['integrated']['rows']),integrated_total=sum(len(v['integrated']['rows']) for v in cases),integration_replays=sum(v['integrated']['replay'] for v in cases),conflicts_blocked=sum(v['language_math']['blocked']['status']=='unknown' and v['language_math']['conflict']['status']=='conflict' and v['language_math']['restored'].get('value') is True for v in cases),conditional_correct=sum(v['language_math']['conditional'].get('result',{}).get('value')=='20' for v in cases),unsupported_language=sum(v['language_math']['unsupported']['status']=='unsupported' for v in cases),contradiction_blocked=sum(v['integrated']['after']['status']=='unknown' for v in cases),negative_blocked=sum(v['negative']['terminal']['action']=='failed' for v in cases),unsupported_program_blocked=sum(v['unsupported_program']['terminal']['action']=='failed' for v in cases),dependent_invalidated=sum(v['full']['invalidation']=='invalid' for v in cases))
    out.update(constraint_correct=sum(r['correct'] for v in cases for r in v['constraints']),constraint_total=sum(len(v['constraints']) for v in cases),plans_correct=sum(p['correct'] for v in cases for p in v['integrated']['plans']),plans_total=sum(len(v['integrated']['plans']) for v in cases),math_bridge_correct=sum(v['integrated']['math_bridge'].get('value')==(7,) for v in cases),document_list_correct=sum(v['language_math']['listing_ok'] for v in cases))
    return out


def check(c):
    s=summary(c);n=c['trials']
    return (s['full']['acquisitions']==3*n and s['full']['wrong']==0 and s['full']['correct']==108*n and s['full']['replays']==n and s['control']['final']==0 and s['control']['wrong']==0 and s['control']['acquisitions']==2*n and s['language_math_correct']==s['language_math_total']==26*n and s['integrated_correct']==s['integrated_total']==64*n and s['plans_correct']==s['plans_total']==5*n and s['constraint_correct']==s['constraint_total']==18*n and all(s[k]==n for k in ('integration_replays','conflicts_blocked','conditional_correct','unsupported_language','contradiction_blocked','negative_blocked','unsupported_program_blocked','dependent_invalidated','math_bridge_correct','document_list_correct')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=810000);p.add_argument('--trials',type=int,default=2);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=12:p.error('trials must be in 1..12')
    c=campaign(a.seed,a.trials);c['summary']=summary(c)
    if a.output:Path(a.output).write_bytes(canonical(c)+b'\n')
    print(json.dumps(c['summary'],indent=2));return int(a.check and not check(c))


if __name__=='__main__':raise SystemExit(main())
