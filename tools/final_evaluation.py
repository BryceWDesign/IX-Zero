"""Frozen-engine evaluation. No learner tuning, external task solving, or test-label fitting.
Public examples are diagnostic subsets, not official benchmark scores.
Run from the repository root: python -m tools.final_evaluation
"""
import json
import math
import random
from itertools import product
from pathlib import Path
from collections import Counter
from ix_zero.scalar import ScalarEpisode
from ix_zero.operators import OperatorEpisode
from ix_zero.core import Space, Task
from ix_zero.language_demo import build, reply

OUT = Path('results/milestone5/frozen_evaluation.json')
SEEDS = tuple(range(202611240, 202611252))

def row(x, y, p, status=None):
    return dict(input=x, expected=y, prediction=p, result=status or ('abstained' if p is None else 'correct' if p==y else 'incorrect'))

def counts(rows):
    return dict(sorted(Counter(r['result'] for r in rows).items()))

def scalar_fit(train, audits):
    ep=ScalarEpisode()
    for x,y in train:
        if ep.blocked:break
        ep.observe(x,y)
    for x,y in audits:
        if ep.blocked:break
        ep.audit(x,y)
    return ep

def external():
    # All numeric pairs from the public simple_arithmetic_json example task.
    # Transcribed from its public source; no input text parsing by the learner.
    vals=[(1,4,5),(4,9,13),(5,0,5),(6,4,10),(8,3,11),(5,6,11),(5,2,7),(8,1,9),(0,4,4),(3,3,6),
          (47,8,55),(30,89,119),(30,58,88),(31,46,77),(16,72,88),(15,63,78),(34,31,65),(34,17,51),(9,74,83),(86,33,119),
          (486,141,627),(76,812,888),(13,476,489),(94,129,223),(635,613,1248),(516,962,1478),(749,750,1499),(389,442,831),(773,546,1319),(348,227,575)]
    train=[((a,b),a+b) for a,b in [(-3,2),(2,-1),(-4,-3),(0,0),(2,5),(-2,0)]]
    audit=[((a,b),a+b) for a,b in [(-1,3),(6,-2),(7,7)]]
    ep=scalar_fit(train,audit); rows=[]
    for a,b,y in vals:
        try: rows.append(row((a,b),y,ep.predict((a,b))))
        except ValueError as e: rows.append(dict(row((a,b),y,None,'unsupported'),reason=str(e)))
    # ARC public evaluation task, actual 2x2 demonstration and required 6x6 output.
    # Do not flatten or construct a task-specific solver. Try the actual native API.
    arc_input=[[8,6],[6,4]]
    arc_output=[[8,6,8,6,8,6],[6,4,6,4,6,4],[6,8,6,8,6,8],[4,6,4,6,4,6],[8,6,8,6,8,6],[6,4,6,4,6,4]]
    arc=dict(source='https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/00576224.json',task='00576224',input_shape=[2,2],output_shape=[6,6])
    try: Task(Space(2,7)).observe(arc_input,arc_output);arc['result']='accepted_demonstration_not_scored'
    except ValueError as e:arc.update(result='unsupported',reason=str(e))
    m,om,scene,values=build(); scan=[]
    # Four published README example commands; only capability probes, not a SCAN split.
    for text,n in [('jump',1),('jump left',2),('jump around right',8),('turn left twice',2)]:
        try: p=reply(text,m,om,scene,values);scan.append(dict(command=text,expected_action_count=n,output=p,result='unsupported_output_schema' if not p.get('abstained') else 'abstained'))
        except ValueError as e:scan.append(dict(command=text,expected_action_count=n,result='unsupported',reason=str(e)))
    return dict(arithmetic=dict(source='https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/simple_arithmetic_json/task.json',protocol='adapted supervised numeric diagnostic; not official zero-shot text protocol',training=train,audit=audit,status=ep.status(),predictions=rows,counts=counts(rows)),arc=arc,scan=dict(source='https://github.com/brendenlake/SCAN',protocol='public example input/output compatibility only; no SCAN training or official split',rows=scan))

SCALAR={
 'sum':lambda a,b:a+b, 'product':lambda a,b:a*b,
 'sum_squares':lambda a,b:a*a+b*b,'cube':lambda a,b:a*a*a,
 'min':min,'gcd':math.gcd,'xor':lambda a,b:a^b,
 'offset_7':lambda a,b:a+b+7,'remainder':lambda a,b:a%b if b else 0,
 'hidden_threshold':lambda a,b:a+b+(11 if a>=10 else 0),
 'equality':lambda a,b:int(a==b),'unstructured':None}

def scalar_campaign():
    episodes=[]
    for seed in SEEDS:
        for name,fn in SCALAR.items():
            rng=random.Random(f'{seed}:{name}'); p=list(product(range(-4,5),repeat=2));rng.shuffle(p)
            out=list(product(tuple(range(-20,-11))+tuple(range(12,21)),repeat=2));rng.shuffle(out)
            fit=p[:12];audit=p[12:15];inside=p[15:39];outside=out[:24]
            labels={x:rng.randint(-100,100) if fn is None else fn(*x) for x in fit+audit+inside+outside}
            train=[(x,labels[x]) for x in fit];aud=[(x,labels[x]) for x in audit]; ep=scalar_fit(train,aud)
            tests={k:[row(x,labels[x],ep.predict(x)) for x in xs] for k,xs in [('inside',inside),('outside',outside)]}
            # First wrong fresh prediction becomes corrective feedback; do not use it for initial scoring.
            wrong=next((r for rows in tests.values() for r in rows if r['result']=='incorrect'),None); feedback=None
            if wrong is not None:
                ep.audit(wrong['input'],wrong['expected']); feedback=dict(blocked=ep.blocked,after=ep.predict(wrong['input']))
            episodes.append(dict(seed=seed,family=name,training=train,audit=aud,actual_training=len(ep.training),predictions=tests,feedback=feedback))
    return episodes

BIT={ 'and':lambda a,b:a&b,'or':lambda a,b:a|b,'xor':lambda a,b:a^b,'nor':lambda a,b:7^(a|b),
      'addition':lambda a,b:(a+b)&7,'minimum':min,'left_rotate':lambda a,b:((a<<1)|(a>>2))&7,
      'hidden_exception':lambda a,b:(a^b)^1 if (a,b)==(7,7) else a^b }

def bit_campaign():
    episodes=[]
    for seed in SEEDS:
        for name,fn in BIT.items():
            rng=random.Random(f'{seed}:{name}');pool=list(product(range(8),repeat=2));rng.shuffle(pool)
            # Exception intentionally unseen during training/audit in this one negative control.
            if name=='hidden_exception':pool.remove((7,7));pool.append((7,7))
            fit=pool[:12];audit=pool[12:15];test=pool[15:];ep=OperatorEpisode(3,5)
            for x in fit:ep.observe(x,fn(*x))
            for x in audit:
                if ep.blocked or not ep.resolved():break
                ep.audit(x,fn(*x))
            preds=[row(x,fn(*x),ep.predict(x)) for x in test];status=ep.status()
            wrong=next((r for r in preds if r['result']=='incorrect'),None); feedback=None
            if wrong:
                ep.audit(wrong['input'],wrong['expected']);feedback=dict(blocked=ep.blocked,after=ep.predict(wrong['input']))
            episodes.append(dict(seed=seed,family=name,training=[(x,fn(*x)) for x in fit],audit=[(x,fn(*x)) for x in audit],status=status,predictions=preds,feedback=feedback))
    return episodes

def language_probes():
    m,om,scene,values=build(); rows=[]
    cases=[('inspect red cube',scene,('inspect','A')),('inspect green cube',scene,('inspect','B')),('inspect red sphere',scene,('inspect','C')),
           ('inspect blue cone',scene,('inspect','D')),('inspect red cube',list(reversed(scene)),('inspect','A')),
           ('inspect red cube',[scene[0]],('inspect','A')),('inspect red cube',scene+[['X','cube','red']],None),
           ('inspect orange cube',scene,None),('please inspect red cube',scene,('inspect','A')),
           ('inspect the red cube',scene,('inspect','A')),('red cube inspect',scene,('inspect','A')),
           ('inspect cube red',scene,('inspect','A')),('do not inspect red cube',scene,('negated_inspect','A')),
           ('inspect red cube then inspect red sphere',scene,('sequence','A','C'))]
    for text,s,y in cases:
        try:
            p=m.episodes['demo'].predict(text,s)
            r=row(text,y,p,'appropriate_abstention' if p is None and y is None else None)
        except ValueError as e:r=dict(row(text,y,None,'unsupported'),reason=str(e))
        r['scene']=s;rows.append(r)
    return rows

def main():
    result=dict(protocol='frozen milestone 4; fixed evaluation families, deterministic seeds; authored probes not blinded',seeds=SEEDS,external=external(),scalar=scalar_campaign(),operators=bit_campaign(),language=language_probes())
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('External arithmetic',result['external']['arithmetic']['counts']);print('ARC',result['external']['arc']['result']);print('SCAN',counts(result['external']['scan']['rows']))
    for family in SCALAR:
        print('scalar',family,{k:counts([r for ep in result['scalar'] if ep['family']==family for r in ep['predictions'][k]]) for k in ('inside','outside')})
    for family in BIT:print('operator',family,counts([r for ep in result['operators'] if ep['family']==family for r in ep['predictions']]))
    print('language',counts(result['language']))
if __name__=='__main__':main()
