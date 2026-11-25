"""Synthetic comparisons, not an independent or blinded AGI benchmark."""
import argparse,json,random
from pathlib import Path
from collections import Counter
from .core import Space,Task
from .discovery import Episode,DiscoveryMind

def score(predict,inputs,f):
    c=Counter(correct=0,incorrect=0,abstained=0); cases=[]
    for x in inputs:
        y=predict(x); truth=f(x); status='abstained' if y is None else 'correct' if y==truth else 'incorrect';c[status]+=1;cases.append(dict(input=x,prediction=y,truth=truth,status=status))
    return dict(c,cases=cases)

def oracle(family,s,rng):
    m=s.modulus; perms=[rng.sample(range(m),m) for _ in range(s.width)]; table={x:tuple(rng.randrange(m) for _ in x) for x in s.states}
    return {'cross':lambda x:((x[0]+x[1])*x[0]%m,(x[0]*x[1]+x[1])%m), 'conditional':lambda x:(int(x[0]<x[1])*x[1]%m,(x[0]+int(x[0]==x[1]))%m), 'projection':lambda x:tuple(perms[j][x[(j+1)%s.width]] for j in range(s.width)), 'random':table.__getitem__, 'three':lambda x:((x[0]+x[2])*x[1]%m,(x[2]+x[1])%m,int(x[0]<x[2]))}[family]

def campaign(seeds,arms=('legacy','symbolic','projection','adaptive'),force_expand=False):
    rows=[]
    for seed in seeds:
        for family in ('cross','conditional','projection','random','three'):
            for regime in ('random','balanced') if family!='three' else ('random',):
                rng=random.Random(seed); s=Space(3 if family=='three' else 2,5); f=oracle(family,s,rng); pool=list(s.states); rng.shuffle(pool)
                if regime=='balanced': train=[(x,(x+k)%5) for k in rng.sample(range(5),2) for x in range(5)]
                else: train=pool[:24 if family=='three' else 10]
                rest=[x for x in pool if x not in set(train)]; n=6 if family=='three' else 4; audit,test=rest[:n],rest[n:]
                for arm in arms:
                    t=Task(s) if arm=='legacy' else Episode(s,mode=arm)
                    for x in train:t.observe(x,f(x))
                    if arm!='legacy' and t.stage==3 and (force_expand or any(not h for h in t.symbolic)):t.expand()
                    for x in audit:
                        if not t.blocked:t.audit(x,f(x))
                    p=t.validated_predict if arm=='legacy' else lambda x:t.predict(x,True)
                    rows.append(dict(seed=seed,family=family,regime=regime,arm=arm,train=len(train),audit_requested=len(audit),audit_consumed=len(t.audits),status=t.summary() if arm=='legacy' else t.status(),training=list(t.observations.items()),audits=[(x,p,y) for x,(p,y) in t.audits.items()],**score(p,test,f)))
    return rows

def transfer(seeds):
    rows=[]
    for seed in seeds:
        rng=random.Random(seed); mind=DiscoveryMind(); mind.apply('new','source',2,5,False,'adaptive'); t=mind.episodes['source']; f=lambda x:((x[0]+x[1])*x[0]%5,x[1]); train=[(a,(a+k)%5) for k in rng.sample(range(5),2) for a in range(5)]; rest=[x for x in t.space.states if x not in train]; rng.shuffle(rest)
        for x in train:mind.apply('observe','source',x,f(x))
        if any(not h for h in t.symbolic):mind.apply('expand','source')
        for x in rest[:4]:
            if not t.blocked:mind.apply('audit','source',x,f(x))
        acquired=bool(t.learned())
        if acquired:mind.apply('promote','source')
        for reuse in (False,True):
            name='target'+str(reuse);mind.apply('new',name,2,7,reuse,'adaptive');t=mind.episodes[name];f=lambda x:(((x[0]+x[1])*x[0]+1)%7,x[1]);train=[(a,(a+k)%7) for k in random.Random(seed+10000).sample(range(7),2) for a in range(7)];rest=[x for x in t.space.states if x not in train];random.Random(seed+20000).shuffle(rest)
            for x in train:mind.apply('observe',name,x,f(x))
            if any(not h for h in t.symbolic):mind.apply('expand',name)
            for x in rest[:4]:
                if not t.blocked:mind.apply('audit',name,x,f(x))
            rows.append(dict(seed=seed,reuse=reuse,source_acquired=acquired,source_examples=10,source_audits=4,target_examples=14,target_audits=4,status=t.status(),training=list(t.observations.items()),audits=[(x,p,y) for x,(p,y) in t.audits.items()],**score(lambda x:t.predict(x,True),rest[4:],f)))
    return rows

def active(seeds):
    rows=[]
    for seed in seeds:
        rng=random.Random(seed);s=Space(2,5);f=oracle('conditional',s,rng);pool=list(s.states);rng.shuffle(pool); train,audit,test=pool[:15],pool[15:19],pool[19:]
        for adaptive in (False,True):
            t=Episode(s);t.expand()
            for i in range(10):
                x=t.query(train) if adaptive else train[i];t.observe(x,f(x))
            for x in audit:
                if not t.blocked:t.audit(x,f(x))
            rows.append(dict(seed=seed,active=adaptive,train=10,status=t.status(),training=list(t.observations.items()),audits=[(x,p,y) for x,(p,y) in t.audits.items()],**score(lambda x:t.predict(x,True),test,f)))
    return rows

def aggregate(rows,keys):
    result={}
    for r in rows:
        k='/'.join(str(r[x]) for x in keys);v=result.setdefault(k,dict(correct=0,incorrect=0,abstained=0,episodes=0))
        for n in ('correct','incorrect','abstained'):v[n]+=r[n]
        v['episodes']+=1
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=100);p.add_argument('--out',type=Path);p.add_argument('--strict',action='store_true');args=p.parse_args()
    if not 1<=args.trials<=100:p.error('trials must be 1..100')
    seeds=list(range(args.seed,args.seed+args.trials))
    if args.strict:
        r=campaign(seeds,('adaptive','strict'),True);data=dict(seeds=seeds,notes='Follow-up designed after development campaign. Both arms use stage 5 on all tasks. Not independent/blinded.',summary=aggregate(r,('arm',)),by_family=aggregate(r,('arm','family','regime')),campaign=r)
        if args.out:args.out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(data['summary'],indent=2));return
    c=campaign(seeds);t=transfer(seeds);a=active(seeds);data=dict(seeds=seeds,campaign=c,transfer=t,active=a,summary=dict(campaign=aggregate(c,('arm',)),by_family=aggregate(c,('arm','family','regime')),transfer=aggregate(t,('reuse',)),active=aggregate(a,('active',))))
    if args.out:args.out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data['summary'],indent=2))
if __name__=='__main__':main()
