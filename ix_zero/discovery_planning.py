"""Test plan proposals against a separate deterministic simulated transition oracle."""
import argparse,json,random
from pathlib import Path
from .core import Space
from .discovery import DiscoveryMind

def run(seeds):
    rows=[]
    for seed in seeds:
        r=random.Random(seed);m=DiscoveryMind();s=Space(2,5)
        oracle={'nonlinear':lambda x:((x[0]+x[1])*x[0]%5,x[1]),'increment':lambda x:(x[0],(x[1]+1)%5)}
        for name,f in oracle.items():
            m.apply('new',name,2,5,False,'strict'); t=m.episodes[name];train=[(a,(a+k)%5) for k in (0,1,2) for a in range(5)];rest=[x for x in s.states if x not in train];r.shuffle(rest)
            for x in train:m.apply('observe',name,x,f(x))
            for x in rest[:4]:
                if not t.blocked:m.apply('audit',name,x,f(x))
        eligible=[n for n,t in m.episodes.items() if t.passes>=3 and not t.blocked]
        for _ in range(12):
            x=r.choice(s.states);y=x;ground=[]
            for _ in range(r.randint(2,7)):
                name=r.choice(list(oracle));y=oracle[name](y);ground.append(name)
            p=m.plan(s,x,y,eligible);actual=x;matches=True
            if p is not None:
                for name,predicted in p:
                    actual=oracle[name](actual);matches=matches and actual==predicted
            rows.append(dict(seed=seed,start=x,goal=y,oracle_path=ground,proposal=p,observed_final=actual,
                             result='abstained' if p is None else 'correct' if matches and actual==y else 'incorrect'))
    return {'seeds':list(seeds),'summary':{k:sum(r['result']==k for r in rows) for k in ('correct','incorrect','abstained')},'cases':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=12);p.add_argument('--out',type=Path);a=p.parse_args()
    if not 1<=a.trials<=100:p.error('trials must be 1..100')
    r=run(range(300,300+a.trials));print(json.dumps(r['summary']))
    if a.out:a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
