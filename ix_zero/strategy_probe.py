"""Post-adoption probes on different task families; no feedback into selection."""
import argparse
import json
import random
from .operators import Domain
from .strategy import Strategy,BASELINE,episode,compare


def probe(selected,seed=57000,trials=4):
    cases=[]
    for i in range(trials):
        for width in (3,4):
            d=Domain(width);m=d.mask;rng=random.Random(seed+100*i+width)
            mapping={x:rng.randrange(m+1) for x in d.inputs}
            targets={'reverse_implication':lambda a,b:a|(m^b),'shift_nor':lambda a,b:m^((a>>1)|b),
                     'shifted_xor':lambda a,b:a^(b>>1),'double_shift':lambda a,b:(a<<2)&m,
                     'minimum':min,'random_mapping':lambda a,b:mapping[a,b],
                     'hidden_switch':lambda a,b:a^b if a<m else (a+b)&m}
            for j,(name,fn) in enumerate(targets.items()):
                xs=list(d.inputs);random.Random(seed+1000*i+10*width+j).shuffle(xs)
                rows=lambda seq:[[list(x),fn(*x)] for x in seq]
                cases.append(dict(id=f'{name}-{width}-{i}',family=name,width=width,budget=5,
                                  train=rows(xs[:12]),audit=rows(xs[12:15]),hold=rows(xs[15:])))
    baseline=[episode(t,BASELINE,True) for t in cases];candidate=[episode(t,selected,True) for t in cases]
    return dict(seed=seed,trials=trials,verdict=compare(candidate,baseline),baseline=baseline,candidate=candidate)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report');p.add_argument('--seed',type=int,default=57000);p.add_argument('--trials',type=int,default=4);p.add_argument('--output');a=p.parse_args()
    if not 1<=a.trials<=12:p.error('trials must be 1..12')
    with open(a.report,encoding='utf-8') as f:selected=Strategy(**json.load(f)['selected'])
    r=probe(selected,a.seed,a.trials)
    if a.output:
        with open(a.output,'w',encoding='utf-8') as f:json.dump(r,f,separators=(',',':'));f.write('\n')
    print(json.dumps(r['verdict'],indent=2))

if __name__=='__main__':main()
