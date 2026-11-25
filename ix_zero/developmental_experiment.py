"""External synthetic curriculum evaluator. Held-out labels never enter Development."""
import argparse
from itertools import product
import json
from pathlib import Path
import random
import tempfile
from .developmental import Development, key, normal
from .representation_experiment import teacher
from .language_experiment import fixtures


def score(m,name,rows):
    predictions=[dict(input=x,expected=y,prediction=m.predict(name,x)) for x,y in rows]
    return dict(correct=sum(normal(r['prediction'])==normal(r['expected']) for r in predictions),
                incorrect=sum(r['prediction'] is not None and normal(r['prediction'])!=normal(r['expected']) for r in predictions),
                abstained=sum(r['prediction'] is None for r in predictions),rows=predictions)


def trial(seed,reuse=True,*,demo=False):
    rng=random.Random(seed);m=Development(reuse=reuse);labels={};hold={};specs={}
    def shuffle(xs):
        xs=list(xs);rng.shuffle(xs);return xs
    def add(n,k,deps,train,audit,held,**fields):
        specs[n]=dict(kind=k,deps=deps,train=[x for x,y in train],audit=[x for x,y in audit],**fields)
        labels[n]={key(x):y for x,y in train+audit};hold[n]=held
    def table(xs,f):return [(x,f(x)) for x in xs]
    def run():
        for _ in range(2000):
            r=m.next()
            if r['action']!='request':return r['action']
            m.answer(labels[r['job']][key(r['input'])])
        raise RuntimeError('evaluation step limit')
    def word(n,f,deps,budget):
        xs=shuffle(product(range(8),repeat=2)); rows=table(xs,lambda x:f(*x))
        add(n,'operator',deps,rows[:24],rows[24:27],rows[27:],width=3,budget=budget)
    word('word-a',lambda a,b:a^b,[],5);word('word-b',lambda a,b:a&b,[],5)
    word('word-c',lambda a,b:a^(b>>1),['word-a'],3)
    xs=shuffle(product(range(4),repeat=2));rows=table(xs,lambda x:x[0]|x[1]);add('word-d','operator',[],rows[:10],rows[10:13],rows[13:],width=2,budget=5)
    source_states=['s'+str(i) for i in range(4)]; target_states=['r'+str(i) for i in range(4)];rng.shuffle(target_states)
    ordered=shuffle(target_states); idx={v:i for i,v in enumerate(target_states)}
    sr={'edge':[source_states[min(i+1,3)] for i in range(4)]};tr={'link':[target_states[min(idx[v]+1,3)] for v in ordered]}
    graph_rows=table(shuffle(product(target_states,repeat=2)),lambda x:target_states[idx[x[0]]|idx[x[1]]])
    add('graph','analogy',['word-d'],graph_rows[:5],graph_rows[5:8],graph_rows[8:],states=ordered,relations=tr,source_states=source_states,source_relations=sr)
    def lab():return tuple(sorted(rng.sample(range(-120,-1),8)))
    target=lab();fbase=teacher(target,'ordinal',1,3,lambda a,b:a^b);falt=teacher(target,'gray',3,5,lambda a,b:a&b)
    def rep(n,vs,f,deps):
        xs=shuffle(product(vs,repeat=2));rows=table(xs,f)
        add(n,'representation',deps,rows[:24],rows[24:27],rows[27:],labels=vs)
    for n,c in [('parent-a',0),('parent-b',1)]:
        vs=lab();rep(n,vs,teacher(vs,'ordinal',1,c,lambda a,b:a^b),['word-a','word-b'])
    rep('target-base',target,fbase,['word-a','word-b','parent-a','parent-b'])
    rep('target-alt',target,falt,['word-a','word-b','parent-a','parent-b'])
    derived_labels=lab();rep('derived-domain',derived_labels,teacher(derived_labels,'gray',5,2,lambda a,b:a^(b>>1)),['word-c'])
    def truth(x):return (falt if x[0]=='changed' else fbase)(x[1:])
    ordinary=shuffle(('ordinary',a,b) for a,b in product(target,repeat=2));changed=shuffle(('changed',a,b) for a,b in product(target,repeat=2))
    rows=table(ordinary,truth);add('world','context',['target-base','target-alt'],rows[:12],rows[12:15],rows[15:],labels=target,contexts=['ordinary','changed'])
    # Native grounded lexicon and compositional grammar are learned as curriculum jobs.
    f=fixtures(seed,english=True); add('lex','lexicon',[],[(r[:2],r[2]) for r in f['train']],[(r[:2],r[2]) for r in f['audit']],[(r[:2],r[2]) for r in f['test']])
    scene=[['a','cube','red'],['b','cube','green'],['c','sphere','red'],['d','cone','blue'],['e','sphere','blue'],['f','cone','green']]
    refs={i:a+' '+k for i,k,a in scene};train=[];audit=[]
    for intent,pat in [('inspect',('inspect',0)),('rule',('combine',0,'with',1)),('compare',('compare',0,1)),('join',('join',0,'with',1))]:
        arity=sum(isinstance(x,int) for x in pat);examples=[('a',),('b',),('c',),('d',),('e',)] if arity==1 else [('a','d'),('d','a'),('b','d'),('c','a'),('e','f')]
        for i,parts in enumerate(examples):
            text=' '.join(refs[parts[v]] if isinstance(v,int) else v for v in pat);row=([text,scene],[intent,*parts]);(train if i<2 else audit).append(row)
    add('grammar','grammar',['lex'],train,audit,[])
    # Deliberately register dependencies out of order.
    for n in reversed(list(specs)):m.add(n,specs[n])
    first=run()
    new_labels=lab();new_fn=teacher(new_labels,'gray',7,4,lambda a,b:a^b);new_rows=table(shuffle(product(new_labels,repeat=2)),new_fn)
    offered=m.offer('offered-domain',new_labels,[x for x,y in new_rows[:24]],[x for x,y in new_rows[24:27]])
    labels['offered-domain']={key(x):y for x,y in new_rows[:27]};hold['offered-domain']=new_rows[27:]
    first=run();before={n:score(m,n,rows) for n,rows in hold.items()};status_before=m.status()
    # External counterexample; the coordinator never receives the truth function.
    counter=next(x for x in changed if fbase(x[1:])!=truth(x));prior=m.predict('world',counter)
    if prior is None or prior==truth(counter):
        return dict(seed=seed,reuse=reuse,first=first,before=before,status=status_before,failed='missing validated incumbent')
    available=[x for x in changed if x!=counter]
    # Keep separate partitions for recovery, dispatch, and downstream evaluation.
    repair_train=available[:30]+ordinary[15:27];repair_audit=available[30:34]+ordinary[27:31]
    bind_train=available[34:43];bind_audit=available[43:47];test=available[47:]+ordinary[31:]
    m.recover('world','recovered',counter,truth(counter),repair_train,repair_audit)
    labels['recovered']={key(x):truth(x) for x in repair_train+repair_audit}
    hold['recovered']=table(test,truth)
    def obj(v):return dict(identity='item'+str(v),kind='sample',attribute='neutral',value=v)
    # Learned root name is deliberately different from linguistic intent.
    for i,sig in enumerate((['object','object'],['integer','integer'],['integer','object'],['object','integer'])):
        def args(x):return [obj(v) if t=='object' else v for t,v in zip(sig,x)]
        n='bind-rule-'+str(i);add(n,'binding',['recovered'],[(args(x[1:]),truth(x)) for x in bind_train],[(args(x[1:]),truth(x)) for x in bind_audit],[],intent='rule',inputs=sig,output='integer',contexts={'recovered':'changed'});m.add(n,specs[n])
        n='bind-compare-'+str(i); xs=shuffle(product(target,repeat=2));cmp=lambda x:(x[0]>x[1])-(x[0]<x[1]);add(n,'binding',[],[(args(x),cmp(x)) for x in xs[:10]],[(args(x),cmp(x)) for x in xs[10:14]],[],intent='compare',inputs=sig,output='integer',contexts={});m.add(n,specs[n])
    n='bind-inspect';add(n,'binding',[],[([obj(v)],v) for v in target[:3]],[([obj(v)],v) for v in target[3:6]],[],intent='inspect',inputs=['object'],output='integer',contexts={});m.add(n,specs[n])
    def relargs(x):return [dict(identity='item'+v,kind='sample',attribute='neutral',relation=v) for v in x]
    n='bind-join';add(n,'binding',['graph'],[(relargs(x),y) for x,y in graph_rows[8:10]],[(relargs(x),y) for x,y in graph_rows[10:13]],[],intent='join',inputs=['object','object'],output='relation',contexts={});m.add(n,specs[n])
    second=run();rec=score(m,'recovered',hold['recovered']); lang=[]
    for i,(a,b,c) in enumerate(shuffle(product(target,repeat=3))[:32]):
        values={'a':a,'b':b,'c':c};text='compare inspect red sphere combine red cube with green cube'
        r=m.execute('grammar',text,scene,values);y=(c>falt((a,b)))-(c<falt((a,b)))
        lang.append(dict(text=text,values=values,expected=y,prediction=r.get('value')))
    graph_lang=[]
    for x,y in graph_rows[13:]:
        rels={'a':x[0],'b':x[1]};r=m.execute('grammar','join red cube with green cube',scene,{},rels)
        graph_lang.append(dict(relations=rels,expected=y,prediction=r.get('value')))
    status_pre_retract=m.status()
    replay=False
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'state.json';m.save(p);clone=Development.load(p)
        replay=(clone.status()==m.status() and all(clone.predict('recovered',x)==m.predict('recovered',x) for x,y in hold['recovered']) and all(clone.execute('grammar',r['text'],scene,r['values']).get('value')==r['prediction'] for r in lang) and all(clone.execute('grammar','join red cube with green cube',scene,{},r['relations']).get('value')==r['prediction'] for r in graph_lang))
    if demo:
        if not replay:raise RuntimeError("developmental demo replay failed")
        return clone,scene,lang[0]["values"],graph_lang[0]["relations"]
    # Source invalidation must propagate through learned dispatch after restart.
    before_retract=m.execute('grammar',lang[0]['text'],scene,lang[0]['values']).get('value');m.retract('word-a')
    blocked=m.execute('grammar',lang[0]['text'],scene,lang[0]['values']).get('value') is None and m.predict('recovered',test[0]) is None
    return dict(seed=seed,reuse=reuse,offered=offered,first=first,second=second,before=before,status_before=status_before,status_pre_retract=status_pre_retract,status=m.status(),recovery=rec,language=lang,replay=replay,quarantined=m.predict('world',counter) is None,initial=dict(input=counter,expected=truth(counter),prediction=prior),dependency_blocked=blocked,graph_language=graph_lang,previous_language=before_retract,events=m.events)


def campaign(seed,trials):
    rows=[trial(seed+i,reuse) for i in range(trials) for reuse in (True,False)]
    summary={}
    for reuse in (True,False):
        rr=[r for r in rows if r['reuse']==reuse]; ss={}
        for n in ('word-a','word-b','word-c','word-d','graph','parent-a','parent-b','target-base','target-alt','derived-domain','offered-domain','world','lex'):
            ss[n]={k:sum(r['before'][n][k] for r in rr) for k in ('correct','incorrect','abstained')}
        ss['recovery']={k:sum(r.get('recovery',{}).get(k,0) for r in rr) for k in ('correct','incorrect','abstained')}
        ls=[l for r in rr for l in r.get('language',[])];ss['language']={'correct':sum(l['prediction']==l['expected'] for l in ls),'incorrect':sum(l['prediction'] is not None and l['prediction']!=l['expected'] for l in ls),'abstained':sum(l['prediction'] is None for l in ls)}
        gs=[l for r in rr for l in r.get('graph_language',[])];ss['graph_language']={'correct':sum(l['prediction']==l['expected'] for l in gs),'incorrect':sum(l['prediction'] is not None and l['prediction']!=l['expected'] for l in gs),'abstained':sum(l['prediction'] is None for l in gs)}
        ss['replays']=sum(r.get('replay',False) for r in rr);ss['dependency_blocks']=sum(r.get('dependency_blocked',False) for r in rr);ss['terminal_runs']=sum(r.get('second') in ('complete','wait') and not r.get('failed') for r in rr)
        ss['all_objectives_live_except_contradicted_world']=sum(bool(r.get('status_pre_retract')) and all(v['live'] for n,v in r['status_pre_retract'].items() if n!='world') for r in rr)
        ss['representation_candidates']=sum(r['status_before'][n]['candidate_programs'] for r in rr for n in ('target-base','target-alt'))
        summary['reuse' if reuse else 'no_reuse']=ss
    return dict(seed=seed,trials=trials,summary=summary,records=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=208000);p.add_argument('--trials',type=int,default=12);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=32:p.error('trials must be 1..32')
    r=campaign(a.seed,a.trials);print(json.dumps(r['summary'],indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if a.check:
        s=r['summary']['reuse'];assert s['replays']==s['dependency_blocks']==a.trials
        assert s['word-c']['correct']>r['summary']['no_reuse']['word-c']['correct'] and s['recovery']['correct']>0 and s['language']['correct']>0
        assert s['all_objectives_live_except_contradicted_world']==a.trials and s['derived-domain']['correct']==s['offered-domain']['correct']==37*a.trials

if __name__=='__main__':main()
