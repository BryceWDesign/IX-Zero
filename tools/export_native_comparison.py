"""Export the same grounded symbolic fixtures for original Julia donor components.
No Julia output is fabricated. Execute native_relational.jl separately.
"""
import argparse
import csv
import json
from pathlib import Path
from ix_zero.unified_experiment import symbolic_fixture
from ix_zero.relational import WorldMemory


def encode(fs):return ';'.join('/'.join(f) for f in sorted(fs))


def generate(seeds,path):
    results=[]
    with open(path,'w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(('case','phase','before','action','after'))
        for seed in seeds:
            for kind in ('local','nonlocal'):
                training,audits,held=symbolic_fixture(seed,kind);m=WorldMemory();counts=dict(correct=0,incorrect=0,abstained=0)
                for phase,rows in [('train',training),('audit',audits),('test',held)]:
                    for s,a,t in rows:
                        writer.writerow((f'{seed}-{kind}',phase,encode(s),'/'.join(a),encode(t)))
                        if phase=='train':m.observe(s,a,t)
                        elif phase=='audit':m.audit(s,a,t)
                        else:
                            p=m.predict(s,a);counts['abstained' if p is None else 'correct' if p==t else 'incorrect']+=1
                results.append(dict(case=f'{seed}-{kind}',**counts))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=12);p.add_argument('--out',type=Path,default=Path('native_fixture.tsv'));a=p.parse_args()
    if not 1<=a.trials<=32:p.error('trials must be 1..32')
    print(json.dumps(generate(range(900,900+a.trials),a.out),indent=2))
