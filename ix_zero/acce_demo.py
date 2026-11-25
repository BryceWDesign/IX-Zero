"""Interactive typed cognitive dispatch over a synthetic grounded scene."""
import argparse, json
from .acce import Workspace, CognitiveDispatch, Object
from .language_demo import build as language_build
from .operators import OperatorMind
from .unified import UnifiedMind
from .analogical import AnalogicalMind


def objects(a,b=None,relation=False):
    def obj(v):
        return Object('item'+str(v),'sample','neutral',v if not relation else None,'s'+str(v) if relation else None)
    return (obj(a),) if b is None else (obj(a),obj(b))


def teach(binding,train,audit):
    for x,y in train:binding.observe(x,y)
    for x,y in audit:
        if not binding.audit(x,y):raise RuntimeError('dispatch validation failed')
    return binding


def build():
    lm,original,rows,values=language_build()
    # Rename the learned operator by replaying its original measured acquisition.
    # No matching language verb is registered in the capability catalog.
    om=OperatorMind()
    for action,name,*args in original.events:om.apply(action,'word7',*args)
    um=UnifiedMind();um.apply('scalar_start','numeric4')
    for a,b in [(2,3),(-2,3),(3,4),(0,5),(4,-2),(1,2)]:um.apply('scalar_observe','numeric4',(a,b),a*b)
    for a,b in [(5,2),(-3,-2),(3,3)]:
        if not um.apply('scalar_audit','numeric4',(a,b),a*b):raise RuntimeError('scalar validation failed')
    am=AnalogicalMind();states=['s0','s1','s2','s3'];source=['b0','b1','b2','b3']
    am.apply('operator_source','bits',source,{'r':[source[i^1] for i in range(4)],'s':[source[i^2] for i in range(4)]},om.events,'word7')
    am.apply('new','structure9',states,{'p':[states[i^1] for i in range(4)],'q':[states[i^2] for i in range(4)]},'structural')
    for a,b in [(0,1),(1,2)]:am.apply('observe','structure9',[states[a],states[b]],states[a^b])
    for a,b in [(0,2),(1,3),(3,0)]:
        if not am.apply('audit','structure9',[states[a],states[b]],states[a^b]):raise RuntimeError('relational validation failed')
    ws=Workspace().primitives().operators(om).scalars(um).analogies(am,om)
    d=CognitiveDispatch(ws)
    binary=('object','object')
    # Teacher supplies observed outcomes, not capability identifiers or programs.
    teach(d.new('compare',binary,'integer'),[(objects(a,b), (a>b)-(a<b)) for a,b in [(0,3),(3,0)]],[(objects(a,b),(a>b)-(a<b)) for a,b in [(1,1),(1,3),(2,0)]])
    teach(d.new('rule',binary,'integer'),[(objects(a,b),a^b) for a,b in [(1,1),(1,2)]],[(objects(a,b),a^b) for a,b in [(0,2),(2,3),(3,3)]])
    teach(d.new('inspect',('object',),'integer'),[(objects(a),a) for a in (0,1)],[(objects(a),a) for a in (2,3,4)])
    teach(d.new('measure',binary,'integer'),[(objects(a,b),a*b) for a,b in [(2,3),(3,3)]],[(objects(a,b),a*b) for a,b in [(4,-2),(5,2),(2,2)]])
    teach(d.new('combine',binary,'relation'),[(objects(a,b,True),'s'+str(a^b)) for a,b in [(1,1),(1,2)]],[(objects(a,b,True),'s'+str(a^b)) for a,b in [(0,2),(2,3),(3,3)]])
    return lm,om,um,am,d,rows,values


def reply(text,lm,dispatch,rows,values):
    try:return dispatch.interpret(lm.episodes['demo'],text,rows,values)
    except ValueError as e:return dict(abstained=True,reason=str(e))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    lm,om,um,am,d,rows,values=build()
    print('IX-Zero | grounded capability dispatch | synthetic scene')
    print('Scene:',json.dumps(rows));print('Values:',json.dumps(values));print('Comparison result: -1 first smaller, 0 equal, 1 first larger.')
    if a.check:
        cases=[('compare red sphere green cube',1),('rule blue cone green cube',2),('inspect red cube',0),('inspect orange sphere',None)]
        for text,y in cases:
            r=reply(text,lm,d,rows,values);print(text,json.dumps(r));assert r.get('value')==y and (y is not None or r.get('abstained'))
        return
    print('Examples: compare red sphere green cube / rule blue cone green cube / inspect red cube')
    while True:
        try:
            text=input('You > ').strip()
            if text.lower() in ('exit','quit'):break
            print('IX-Zero > '+json.dumps(reply(text,lm,d,rows,values)))
        except (EOFError,KeyboardInterrupt):break

if __name__=='__main__':main()
