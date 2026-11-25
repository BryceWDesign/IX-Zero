"""Interactive recursive objectives over the grounded ACCE demonstration."""
import argparse,json
from .acce_demo import build as base_build, teach, objects
from .compositional import Grammar


def build():
    lm,om,um,am,d,rows,values=base_build()
    # Learn typed dispatch for intermediate integer results as well as objects.
    for name,fn,train,audit in (
        ('compare',lambda a,b:(a>b)-(a<b),[(0,3),(3,0)],[(1,1),(1,3),(2,0)]),
        ('rule',lambda a,b:a^b,[(1,1),(1,2)],[(0,2),(2,3),(3,3)]),
        ('measure',lambda a,b:a*b,[(2,3),(3,3)],[(4,-2),(5,2),(2,2)])):
        for sig in (('integer','integer'),('object','integer'),('integer','object')):
            def args(a,b):return tuple(objects(v)[0] if t=='object' else v for t,v in zip(sig,(a,b)))
            teach(d.new(name,sig,'integer'),[(args(a,b),fn(a,b)) for a,b in train],[(args(a,b),fn(a,b)) for a,b in audit])
    # More than four referents are needed for disjoint unary training and audits.
    training_rows=[['a','cube','red'],['b','cube','green'],['c','sphere','red'],['d','cone','blue'],['e','sphere','blue'],['f','cone','green']]
    refs={i:a+' '+k for i,k,a in training_rows}
    g=Grammar(lm.episodes['demo'])
    forms=[('inspect',('inspect',0)),('compare',('compare',0,1)),('rule',('rule',0,1)),('compare',('please','compare','the',0,'with','the',1)),('compare',(0,'versus',1)),('rule',('combine',0,'with',1)),('measure',('multiply',0,'by',1))]
    audit_examples=[]
    for intent,pattern in forms:
        arity=sum(isinstance(x,int) for x in pattern)
        examples=[('a',),('b',),('c',),('d',),('e',)] if arity==1 else [('a','d'),('d','a'),('b','d'),('c','a'),('e','f')]
        for n,args in enumerate(examples):
            text=' '.join(refs[args[x]] if isinstance(x,int) else x for x in pattern); y=(intent,*args)
            if n<2:g.observe(text,training_rows,y)
            else:audit_examples.append((text,training_rows,y))
    for text,s,y in audit_examples:
        if not g.audit(text,s,y):raise RuntimeError('grammar audit failed')
    return g,d,rows,values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    g,d,rows,values=build()
    print('IX-Zero | recursive grounded objectives | synthetic scene');print('Scene:',json.dumps(rows));print('Values:',json.dumps(values))
    cases=[('please compare the red sphere with the green cube',1),('red sphere versus green cube',1),('compare inspect red sphere rule green cube blue cone',0),('combine inspect red sphere with rule green cube blue cone',0),('inspect orange sphere',None),('what would you ask me about the world',None)]
    if a.check:
        for text,y in cases:
            r=g.execute(text,rows,values,d);print(text,json.dumps(r));assert r.get('value')==y and (y is not None or r.get('abstained'))
        return
    print('Examples:', ' / '.join(t for t,_ in cases[:4]))
    while True:
        try:
            t=input('You > ').strip()
            if t.lower() in ('exit','quit'):break
            print('IX-Zero > '+json.dumps(g.execute(t,rows,values,d)))
        except (EOFError,KeyboardInterrupt):break

if __name__=='__main__':main()
