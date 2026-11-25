"""Fresh shallow-to-deep generalization probes for learned surface rules."""
import argparse,json,random
from collections import Counter
from pathlib import Path
from .compositional import Grammar,depth
from .compositional_demo import build


def run(trials=24,seed=180000):
    base,d,rows,values=build(); source=base.source
    train_rows=[['a','cube','red'],['b','cube','green'],['c','sphere','red'],['d','cone','blue'],['e','sphere','blue'],['f','cone','green']]
    refs={i:a+' '+k for i,k,a in train_rows}; refs.update({i:a+' '+k for i,k,a in rows})
    record=[]; counts={}; original=Counter(); flat=Counter(); neg=Counter(); depths=Counter()
    def tally(kind,result):counts.setdefault(kind,Counter())[result]+=1
    for k in range(trials):
        rng=random.Random(seed+k); g=Grammar(source)
        # Distinct arbitrary words and independently shuffled argument order.
        aliases=rng.sample(range(100000,999999),6); w=['v'+str(a) for a in aliases]
        slots=[0,1];rng.shuffle(slots)
        forms={'inspect':(w[0],0),'rule':(w[1],*slots),'compare':(w[2],*slots),'polite':(w[3],slots[0],w[4],slots[1]),'postfix':(*slots,w[5])}
        meanings={'inspect':'inspect','rule':'rule','compare':'compare','polite':'compare','postfix':'rule'}
        def render(form,args):return ' '.join(args[x] if isinstance(x,int) else x for x in forms[form])
        audits=[]
        for form in forms:
            pairs=[('a',),('b',),('c',),('d',),('e',)] if form=='inspect' else [('a','d'),('d','a'),('b','d'),('c','a'),('e','f')]
            for n,ids in enumerate(pairs):
                t=render(form,[refs[i] for i in ids]); y=(meanings[form],*ids)
                if n<2:g.observe(t,train_rows,y)
                else:audits.append((t,y))
        for t,y in audits:assert g.audit(t,train_rows,y)
        def make(x):
            if isinstance(x,str):return refs[x],values[x]
            form=x[0]; child=[make(c) for c in x[1:]]; v=[c[1] for c in child]
            out=v[0] if form=='inspect' else v[0]^v[1] if form in ('rule','postfix') else (v[0]>v[1])-(v[0]<v[1])
            return render(form,[c[0] for c in child]),out
        def sem(x):return x if isinstance(x,str) else (meanings[x[0]],*(sem(c) for c in x[1:]))
        def evaluate(kind,x):
            t,y=make(x); r=g.execute(t,rows,values,d); p=r.get('value'); m=sem(x)
            status='abstained' if r.get('abstained') else 'correct' if p==y and r.get('meaning')==m else 'incorrect'
            tally(kind,status);depths[(depth(m),status)]+=1
            flat[status if depth(m)==1 else 'abstained']+=1
            try:old= d.interpret(source,t,rows,values)
            except ValueError:old={'abstained':True}
            ov=old.get('value');original['abstained' if ov is None else 'correct' if ov==y else 'incorrect']+=1
            record.append(dict(seed=seed+k,kind=kind,text=t,expected=y,meaning=m,prediction=p,predicted_meaning=r.get('meaning'),status=status))
        # All shallow pairs with training/audit semantic pairs excluded, including relabeled scenes.
        excluded={('A','D'),('D','A'),('B','D'),('C','A')}
        for form in ('compare','rule','polite','postfix'):
            for a in values:
                for b in values:
                    if (a,b) not in excluded:evaluate('shallow', (form,a,b))
        # Entire nesting depths absent from training; retain distinct trees per depth.
        for dep in (2,3,4,5):
            trees=set()
            while len(trees)<24:
                x=rng.choice(tuple(values))
                for _ in range(dep-1):x=('rule',x,rng.choice(tuple(values))) if rng.randrange(2) else ('rule',rng.choice(tuple(values)),x)
                x=('compare',x,('inspect',rng.choice(tuple(values))))
                trees.add(x)
            for x in sorted(trees,key=repr):evaluate('nested',x)
        # Explicit unknown lexical token and ambiguous reference, plus no binding for inspect(integer).
        cases=[(render('inspect',['unknown cube']),rows), (render('inspect',['red cube']),rows+[['E','cube','red']]), (render('inspect',[render('inspect',['red cube'])]),rows)]
        for t,s in cases:
            r=g.execute(t,s,values,d);neg['abstained' if r.get('abstained') else 'unexpected']+=1
        # Changed meaning after audits: wrong old parse, then contradiction blocks grammar.
        t=render('compare',['red cube','red sphere']); before=g.predict(t,rows)
        assert before==('compare','A','C'); wrong=('compare','C','A')
        passed=g.audit(t,rows,wrong); assert not passed
        neg['shift_wrong_before']+=1;neg['quarantined' if g.predict(t,rows) is None else 'unexpected']+=1
    return dict(trials=trials,seed=seed,counts=counts,original_fixed_vocabulary=original,flat_grammar_ablation=flat,negative=neg,depths={f'{a}:{b}':v for (a,b),v in sorted(depths.items())},records=record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=24);p.add_argument('--seed',type=int,default=180000);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=100:raise ValueError('trial budget')
    r=run(a.trials,a.seed); summary={k:v for k,v in r.items() if k!='records'};print(json.dumps(summary,sort_keys=True,indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    if a.check:
        assert all(v.get('incorrect',0)==0 and v.get('abstained',0)==0 for v in r['counts'].values()),summary
        assert r['negative'].get('unexpected',0)==0 and r['negative']['quarantined']==a.trials

if __name__=='__main__':main()
