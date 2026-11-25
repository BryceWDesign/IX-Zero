"""Paired process-CPU timing of checked vs packed inquiry scoring only."""
import argparse
import json
import platform
import random
import statistics
import time
from .operators import OperatorEpisode
from .strategy import BASELINE,Strategy,query


def benchmark(seed=77000,repeats=7):
    rng=random.Random(seed);fixtures=[]
    for width in (2,3,4):
        for budget in (3,5):
            t=OperatorEpisode(width,budget);pool=list(t.domain.inputs);rng.shuffle(pool);pool=pool[:12]
            for step in range(4):
                fixtures.append((t,pool.copy()));x=query(t,pool,BASELINE)
                # Clone state by replay; fixture must remain immutable during timing.
                nt=OperatorEpisode(width,budget)
                for inp,out in t.training.items():nt.observe(inp,out)
                nt.observe(x,x[0]|x[1]);t=nt
                if not t.models:break
    candidate=Strategy(kernel='packed');rows=[]
    for i in range(repeats):
        times={};answers={}
        for name,policy in ([('checked',BASELINE),('packed',candidate)] if i%2==0 else [('packed',candidate),('checked',BASELINE)]):
            start=time.process_time_ns();answers[name]=[query(t,pool,policy) for t,pool in fixtures];times[name]=time.process_time_ns()-start
        if answers['checked']!=answers['packed']:raise RuntimeError('timed inquiry mismatch')
        rows.append(times)
    med={k:statistics.median(r[k] for r in rows) for k in ('checked','packed')}
    return dict(seed=seed,repeats=repeats,fixtures=len(fixtures),python=platform.python_version(),platform=platform.platform(),process_cpu_ns=rows,median_ns=med,speedup=med['checked']/med['packed'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');a=p.parse_args();r=benchmark()
    if a.output:
        with open(a.output,'w',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
