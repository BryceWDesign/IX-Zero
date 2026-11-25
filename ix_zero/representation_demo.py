"""Execute a learned alternative representation through recursive ACCE language."""
import argparse,json
from .representation import Episode
from .representation_experiment import source,teacher,split,train
from .compositional_demo import build as grammar_build
from .acce import Workspace,CognitiveDispatch
from .acce_demo import objects,teach


def build():
    grammar,_,rows,_=grammar_build()
    m=source(184000); labels=(-24,-21,-18,-15,-12,-9,-6,-3)
    fn=teacher(labels,'ordinal',3,2,lambda a,b:a^b)
    p,a,h=split(labels,184001); e=Episode(m,labels);train(e,fn,p,a)
    if not e.resolved() or len(e.audits)<3:raise RuntimeError('representation not validated')
    ws=Workspace().primitives();ws.register(e.capability('representation7'));d=CognitiveDispatch(ws)
    # Separate evidence for dispatch, drawn from inputs reserved by the representation learner.
    for sig in (('object','object'),('integer','integer'),('object','integer'),('integer','object')):
        def args(x):return tuple(objects(v)[0] if t=='object' else v for t,v in zip(sig,x))
        teach(d.new('rule',sig,'integer'),[(args(x),fn(x)) for x in h[:6]],[(args(x),fn(x)) for x in h[6:9]])
        compare=lambda x:(x[0]>x[1])-(x[0]<x[1])
        train_pairs=[(-24,-3),(-3,-24),(-9,-9)]
        audit_pairs=[(-21,-18),(-15,-24),(-6,-6)]
        teach(d.new('compare',sig,'integer'),[(args(x),compare(x)) for x in train_pairs],[(args(x),compare(x)) for x in audit_pairs])
    teach(d.new('inspect',('object',),'integer'),[(objects(v),v) for v in labels[:2]],[(objects(v),v) for v in labels[2:5]])
    values=dict(zip(('A','B','C','D'),labels[3:7]))
    return grammar,d,rows,values,e,fn


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    g,d,rows,v,e,fn=build()
    print('IX-Zero | learned representation + recursive dispatch');print('Scene:',json.dumps(rows));print('Values:',json.dumps(v));print('Representation:',json.dumps(e.status()))
    if a.check:
        val=fn((v['B'],v['D'])); comp=(v['C']>val)-(v['C']<val)
        for text,expected in [('rule green cube blue cone',val),('compare inspect red sphere rule green cube blue cone',comp),('inspect orange sphere',None)]:
            r=g.execute(text,rows,v,d);print(text,json.dumps(r));assert r.get('value')==expected and (expected is not None or r.get('abstained'))
        e.mind.apply('retract','word_a');r=g.execute('rule green cube blue cone',rows,v,d);assert r.get('abstained');print('source retracted:',json.dumps(r));return
    while True:
        try:
            text=input('You > ').strip()
            if text.lower() in ('quit','exit'):break
            print('IX-Zero > '+json.dumps(g.execute(text,rows,v,d)))
        except (EOFError,KeyboardInterrupt):break

if __name__=='__main__':main()
