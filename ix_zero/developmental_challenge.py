"""Negative controls for the integrated developmental curriculum."""
import argparse,json,random
from itertools import product
from pathlib import Path
from .developmental import key
from .developmental_experiment import trial, score
from .representation_experiment import teacher


def challenge(seed):
    m,scene,values,relations=trial(seed,True,demo=True)
    rng=random.Random(seed+800000)
    def shuffled(x):x=list(x);rng.shuffle(x);return x
    def drive(name,rows):
        lookup={key(x):y for x,y in rows}
        for _ in range(300):
            r=m.next()
            if r['action']!='request':return
            if r['job']!=name:raise RuntimeError('unexpected curriculum request')
            m.answer(lookup[key(r['input'])])
        raise RuntimeError('negative control step budget')
    # An unstructured table on a new domain; none of its labels are supplied in the objective.
    labels=tuple(sorted(rng.sample(range(-500,-200),8)));xs=shuffled(product(labels,repeat=2));rows=[(x,rng.choice(labels)) for x in xs]
    m.offer('random-control',labels,xs[:24],xs[24:27]);drive('random-control',rows[:27]);random_result=score(m,'random-control',rows[27:])
    # Two contextual changes cannot be represented by a single context-conditioned substitution.
    lab=m.jobs['target-base'].episode.labels
    base=teacher(lab,'ordinal',1,3,lambda a,b:a^b);alt=teacher(lab,'gray',3,5,lambda a,b:a&b)
    truth=lambda x:(base if x[0]=='ordinary' else alt)(x[1:])
    ordinary=shuffled(('ordinary',a,b) for a,b in product(lab,repeat=2));changed=shuffled(('changed',a,b) for a,b in product(lab,repeat=2));other=shuffled(('other',a,b) for a,b in product(lab,repeat=2))
    m.add('two-context',dict(kind='context',deps=['target-base','target-alt'],labels=lab,contexts=['ordinary','changed','other'],train=ordinary[:12],audit=ordinary[12:15]));drive('two-context',[(x,truth(x)) for x in ordinary[:15]])
    counter=next(x for x in changed if m.predict('two-context',x)!=truth(x));initial=m.predict('two-context',counter)
    # Allocate incompatible contexts in both inquiry and audit partitions.
    changed.remove(counter);train=changed[:25]+other[:25];audit=changed[25:29]+other[25:29]+ordinary[15:19];held=changed[29:]+other[29:]+ordinary[19:]
    m.recover('two-context','two-context-repair',counter,truth(counter),train,audit);drive('two-context-repair',[(x,truth(x)) for x in train+audit]);context_result=score(m,'two-context-repair',[(x,truth(x)) for x in held])
    # Earlier free-form questions remain unsupported, rather than answered with invented text.
    prompts=['what would you ask me about the world','if you could invent a new kind of intelligence what would it do','inspect orange sphere']
    unsupported=[{'text':p,'response':m.execute('grammar',p,scene,values,relations)} for p in prompts]
    return dict(seed=seed,random=random_result,two_context=context_result,initial=dict(input=counter,prediction=initial,expected=truth(counter)),quarantined=m.predict('two-context',counter) is None,unsupported=unsupported,status=m.status(),events=m.events)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=210000);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=24:p.error('invalid trials')
    rows=[challenge(a.seed+i) for i in range(a.trials)];summary={n:{k:sum(r[n][k] for r in rows) for k in ('correct','incorrect','abstained')} for n in ('random','two_context')};summary['quarantined']=sum(r['quarantined'] for r in rows);summary['unsupported']=sum(p['response'].get('abstained',False) for r in rows for p in r['unsupported'])
    out=dict(seed=a.seed,trials=a.trials,summary=summary,records=rows);print(json.dumps(summary,indent=2))
    if a.output:Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if a.check:
        assert summary['random']['correct']==summary['random']['incorrect']==0
        assert summary['two_context']['correct']==summary['two_context']['incorrect']==0
        assert summary['quarantined']==a.trials and summary['unsupported']==3*a.trials

if __name__=='__main__':main()
