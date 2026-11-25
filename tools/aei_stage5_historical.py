"""Run from extracted IX-Zero root: PYTHONPATH=. python /path/to/aei_stage5_historical.py.
External synthetic evaluator; no modifications to the learner. Python 3.11+.
"""
import json,random,itertools,platform,sys
from ix_zero.developmental import Development
from ix_zero.operators import OperatorEpisode,execute,Domain

def truth(pair,exception=False):
    a,b=pair
    return 1 if exception and (a,b)==(15,15) else a & (~b & 15)

def run(seed,mode):
    domain=list(itertools.product(range(16),repeat=2)); special=(15,15)
    normal=[x for x in domain if x!=special]; random.Random(seed).shuffle(normal)
    train=normal[:100]; audits=normal[100:103]; held=normal[103:]
    if mode=='late_exception': audits=audits+[special]
    elif mode=='early_exception': audits=[special]+audits
    else: held=held+[special]
    m=Development();m.add('probe',dict(kind='operator',deps=[],width=4,budget=4,train=train,audit=audits))
    requested=[]
    for _ in range(110):
        r=m.next()
        if r['action']!='request':break
        x=tuple(r['input']);y=truth(x,mode!='control');requested.append(dict(phase=r['phase'],input=x,label=y));m.answer(y)
    else: raise RuntimeError('step limit')
    # Same fixed independent held-out normal set in all arms. Probe the exception separately.
    hold=[x for x in held if x!=special]
    predictions=[dict(input=x,predicted=m.predict('probe',x),expected=truth(x,mode!='control')) for x in hold]
    n=dict(correct=sum(p['predicted']==p['expected'] for p in predictions),incorrect=sum(p['predicted'] is not None and p['predicted']!=p['expected'] for p in predictions),abstained=sum(p['predicted'] is None for p in predictions))
    e=m.jobs['probe'].episode
    return dict(seed=seed,mode=mode,status=m.status(),expression=e.status()['expression'],requested=requested,normal=n,predictions=predictions,special=dict(predicted=m.predict('probe',special),expected=truth(special,mode!='control')),special_audited=any(tuple(p['input'])==special and p['phase']=='audit' for p in requested))

def collision():
    # Two different ground truths are identical on the complete 2-bit domain.
    a=('zero',); b=('shl',('shl',('x',)))
    e=OperatorEpisode(2,3)
    xs=Domain(2).inputs
    for x in xs[:6]:e.observe(x,0)
    for x in xs[6:9]:e.audit(x,0)
    learned=e.simplest()[0].expr
    rows=[]
    for width in (2,3,4):
        inputs=Domain(width).inputs
        rows.append(dict(width=width,total=len(inputs),equivalent=sum(execute(a,x,width)==execute(b,x,width) for x in inputs),learned_vs_shift=sum(execute(learned,x,width)==execute(b,x,width) for x in inputs)))
    return dict(learned=learned,alternate=b,comparisons=rows)

def boundaries():
    out={}
    for label,fn in [('width_5',lambda:Domain(5)),('empty_environment',lambda:Development().next()['action']),('unknown_task_schema',lambda:Development().add('grid',dict(kind='grid',deps=[],train=[[1]],audit=[[2],[3],[4]])))]:
        try:out[label]={'returned':fn()}
        except ValueError as e:out[label]={'rejected':str(e)}
    return out

if __name__=='__main__':
    rows=[run(seed,mode) for seed in range(310000,310006) for mode in ('control','late_exception','early_exception')]
    summary={mode:dict(trials=len(rs),live=sum(r['status']['probe']['live'] for r in rs),normal={k:sum(r['normal'][k] for r in rs) for k in ('correct','incorrect','abstained')},special_audited=sum(r['special_audited'] for r in rs),special_wrong=sum(r['special']['predicted'] is not None and r['special']['predicted']!=r['special']['expected'] for r in rs),special_correct=sum(r['special']['predicted']==r['special']['expected'] for r in rs),special_abstained=sum(r['special']['predicted'] is None for r in rs),training=[r['status']['probe']['training'] for r in rs],audits=[r['status']['probe']['audits'] for r in rs]) for mode in ('control','late_exception','early_exception') if (rs:=[r for r in rows if r['mode']==mode])}
    result=dict(python=platform.python_version(),summary=summary,collision=collision(),boundaries=boundaries(),records=rows)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if len(sys.argv)>1:
        with open(sys.argv[1],'w') as f:json.dump(result,f,indent=2)
