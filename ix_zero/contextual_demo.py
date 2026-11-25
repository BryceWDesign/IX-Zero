"""Grounded recursive reasoning with recovered contextual knowledge.

The synthetic teacher is external to the learner. The demo exposes its inquiry
trace and runs the recovered capability through the existing ACCE grammar.
"""
import argparse,json,random
from .contextual import Domain,Episode
from .representation_demo import build as representation_build
from .representation import Episode as Representation
from .representation_experiment import teacher,split,train
from .acce import Workspace,CognitiveDispatch
from .acce_demo import objects,teach


def build():
    grammar,_,rows,values,base_ep,base_fn=representation_build()
    labels=base_ep.labels; mind=base_ep.mind
    alt_fn=teacher(labels,'ordinal',3,2,lambda a,b:a&b)
    alt=Representation(mind,labels);p,a,h=split(labels,196102);train(alt,alt_fn,p,a)
    if not alt.resolved() or len(alt.audits)<3:raise RuntimeError('alternative representation not validated')
    d=Domain(labels,('ordinary','changed'))
    old=Episode(d,base_ep.capability('source1'),(alt.capability('source2'),))
    def truth(x):return (alt_fn if x[0]=='changed' else base_fn)(x[1:])
    normal=[x for x in d.inputs if x[0]=='ordinary'];changed=[x for x in d.inputs if x[0]=='changed'];random.Random(196103).shuffle(normal);random.Random(196104).shuffle(changed)
    for x in normal[:12]:old.observe(x,truth(x))
    for x in normal[12:15]:assert old.audit(x,truth(x))
    counter=next(x for x in changed if old.predict(x)!=truth(x)); before=old.predict(counter);old.contradict(counter,truth(counter))
    rec=old.recover();changed.remove(counter)
    pool=changed[:30]+normal[15:30];audits=changed[30:36]+normal[30:34];held=changed[36:]
    questions=[]
    for _ in range(30):
        action=rec.advance(pool)
        if action['action']!='inquire':break
        q=action['inquiry'];questions.append(q);x=q['input'];rec.observe(x,truth(x))
    for x in audits:
        if not rec.audit(x,truth(x)):raise RuntimeError('recovery audit failed')
    ws=Workspace().primitives();ws.register(rec.capability('context7','changed'));dispatch=CognitiveDispatch(ws)
    for sig in (('object','object'),('integer','integer'),('object','integer'),('integer','object')):
        def args(x):return tuple(objects(v)[0] if t=='object' else v for t,v in zip(sig,x))
        teach(dispatch.new('rule',sig,'integer'),[(args(x[1:]),truth(x)) for x in held[:8]],[(args(x[1:]),truth(x)) for x in held[8:12]])
        compare=lambda x:(x[0]>x[1])-(x[0]<x[1])
        train_pairs=[(-24,-3),(-3,-24),(-9,-9)];audit_pairs=[(-21,-18),(-15,-24),(-6,-6)]
        teach(dispatch.new('compare',sig,'integer'),[(args(x),compare(x)) for x in train_pairs],[(args(x),compare(x)) for x in audit_pairs])
    teach(dispatch.new('inspect',('object',),'integer'),[(objects(v),v) for v in labels[:2]],[(objects(v),v) for v in labels[2:5]])
    return grammar,dispatch,rows,values,old,rec,questions,dict(input=counter,original=before,observed=truth(counter)),truth


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    g,d,rows,v,old,rec,qs,feedback,truth=build()
    print('IX-Zero | contextual recovery and recursive cognition');print('Scene:',json.dumps(rows));print('Values:',json.dumps(v));print('Contradiction:',json.dumps(feedback));print('Diagnosis:',old.diagnose());print('Generated questions:',json.dumps(qs));print('Recovered:',json.dumps(rec.status()))
    if a.check:
        y=truth(('changed',v['B'],v['D']));z=(v['C']>y)-(v['C']<y)
        for text,expected in [('rule green cube blue cone',y),('compare inspect red sphere rule green cube blue cone',z),('inspect orange sphere',None)]:
            r=g.execute(text,rows,v,d);print(text,json.dumps(r));assert r.get('value')==expected and (expected is not None or r.get('abstained'))
        assert old.predict(feedback['input']) is None and rec.live();return
    print('Commands: questions / hypotheses / diagnosis / or a supported grounded sentence.')
    while True:
        try:
            text=input('You > ').strip()
            if text.lower() in ('quit','exit'):break
            if text=='questions':r=qs
            elif text=='hypotheses':r=rec.hypotheses()
            elif text=='diagnosis':r={'old':old.diagnose(),'current':rec.diagnose()}
            else:r=g.execute(text,rows,v,d)
            print('IX-Zero > '+json.dumps(r))
        except (EOFError,KeyboardInterrupt):break

if __name__=='__main__':main()
