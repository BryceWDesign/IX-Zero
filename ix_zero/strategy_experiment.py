"""Synthetic evaluator. Target functions are not part of the learner."""
import argparse
import json
import random
from .operators import Domain
from .strategy import choose


def corpus(seed=17000,trials=5,mode="exact"):
    result={'selection':[],'confirmation':[],'challenge':[],'mode':mode}
    for i in range(trials):
        for phase in ('selection','confirmation','challenge'):
            for width in (3,4):
                d=Domain(width);m=d.mask
                if phase=='selection':
                    targets={'and':lambda a,b:a&b,'or':lambda a,b:a|b,'xor':lambda a,b:a^b,
                             'shift_and':lambda a,b:(a>>1)&b,'nor':lambda a,b:m^(a|b)}
                elif phase=='confirmation':
                    targets={'xnor':lambda a,b:m^(a^b),'implication':lambda a,b:(m^a)|b,
                             'shift_or':lambda a,b:(a<<1|b)&m,'shift_nand':lambda a,b:m^((a>>1)&b),
                             'addition':lambda a,b:(a+b)&m}
                else:
                    rng=random.Random(seed+i*1000+width);mapping={x:rng.randrange(m+1) for x in d.inputs}
                    targets={'reverse_implication':lambda a,b:a|(m^b),'shift_nor':lambda a,b:m^((a>>1)|b),
                             'minimum':min,'random_mapping':lambda a,b:mapping[a,b],
                             'hidden_switch':lambda a,b:a^b if a<m else (a+b)&m}
                for j,(family,fn) in enumerate(targets.items()):
                    s=seed+100*i+10*width+j+({'selection':0,'confirmation':100000,'challenge':200000}[phase])
                    xs=list(d.inputs);random.Random(s).shuffle(xs)
                    rows=lambda seq:[[list(x),fn(*x)] for x in seq]
                    result[phase].append(dict(id=f'{phase}-{family}-{width}-{i}',family=family,width=width,budget=5,
                                              train=rows(xs[:12]),audit=rows(xs[12:15]),hold=rows(xs[15:])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=17000);p.add_argument('--trials',type=int,default=5);p.add_argument('--mode',choices=('exact','heuristic'),default='exact');p.add_argument('--check',action='store_true');p.add_argument('--output');p.add_argument('--corpus');a=p.parse_args()
    if not 1<=a.trials<=9:p.error('trials must be 1..9')
    data=corpus(a.seed,a.trials,a.mode);r=choose(data,details=True)
    if a.output:
        with open(a.output,'w',encoding='utf-8') as f:json.dump(r,f,separators=(',',':'));f.write('\n')
    if a.corpus:
        with open(a.corpus,'w',encoding='utf-8') as f:json.dump(data,f,separators=(',',':'));f.write('\n')
    print(json.dumps({k:r[k] for k in ('incumbent','selected','challenger','decision','confirmation_verdict','challenge_verdict')},indent=2))
    print(json.dumps([dict(strategy=c['strategy'],verdict=c['verdict']) for c in r['selection_candidates']],indent=2))
    if a.check:
        if a.mode!='exact':p.error('--check requires exact mode')
        if r['decision']!='adopt' or r['selected']['kernel']!='packed':raise SystemExit('exact improvement not adopted')
        for base,candidate in ((r['selection_baseline'],r['selection_candidates'][0]['rows']),(r['confirmation_baseline'],r['confirmation_candidate']),(r['challenge_baseline'],r['challenge_candidate'])):
            if len(base)!=len(candidate):raise SystemExit('incomplete comparison')
            for b,c in zip(base,candidate):
                if any(b[k]!=c[k] for k in ('id','predictions','training_rows','audit_rows','promoted','query_work')):raise SystemExit('exact equivalence failed')

if __name__=='__main__':main()
