"""Grounding experiment: independently measured transitions, opaque lexemes,
self-generated inquiry inputs and transferred plans. Authored benchmark.
The learner never receives the environment's effect descriptions or final labels.
"""
import argparse
import json
import random
from pathlib import Path
from .grounded_semantics import SemanticMemory, generate_probes, input_case, input_key, state_key


class Environment:
    def __init__(self,seed):
        self.rng=random.Random(seed)
        self.fields=[f'f{self.rng.randrange(10**8):08d}' for _ in range(2)]
        self.verbs=[f'v{self.rng.randrange(10**8):08d}' for _ in range(5)]
        self.slots=[self.rng.randrange(3) for _ in self.verbs]
        self.serial=0
    def state(self,size=3,spread=8,unequal=False):
        self.serial+=1
        s={f'e{self.serial:05d}_{i}':{f:self.rng.randint(-spread,spread) for f in self.fields} for i in range(size)}
        if unequal:
            a,b=list(s)[:2];f,g=self.fields
            if s[a][g]==s[b][g]:s[b][g]+=1
        return s
    def words(self,kind,a,b):
        w=[a,b];w.insert(self.slots[kind],self.verbs[kind]);return w
    def observe(self,raw):
        s,w=raw;out={n:dict(fs) for n,fs in s.items()}
        # The environment owns the semantics, not the model. The following
        # independent reference assignments are never passed as learner code.
        verb=next(t for t in w if t in self.verbs);kind=self.verbs.index(verb)
        a,b=(t for t in w if t!=verb);f,g=self.fields
        if kind==0:out[b][f]=s[a][f]
        elif kind==1:out[a][g],out[b][g]=s[b][g],s[a][g]
        elif kind==2:
            if s[a][g]==s[b][g]:out[b][f]=s[a][f]
        elif kind==3:out[a][f]=s[a][f]+s[b][f];out[b][f]=0
        elif kind==4:out[b][f]=s[a][f]*s[b][f]
        else:raise ValueError('unknown reference action')
        return out
    def case(self,kind,size=3,spread=8,unequal=False,force_equal=False):
        s=self.state(size,spread,unequal=unequal);a,b=list(s)[:2]
        if force_equal:s[b][self.fields[1]]=s[a][self.fields[1]]
        return [s,self.words(kind,a,b)]


class LiveMeasurements:
    """A measurement-only facade. Labels are computed at request time."""
    def __init__(self,env,inputs):
        self.inputs=inputs;self._env=env;self._allowed={input_key(input_case(x)) for x in inputs};self.log=[]
    def observe(self,x):
        k=input_key(input_case(x))
        if k not in self._allowed:raise ValueError('measurement outside designated partition')
        y=self._env.observe(x);self.log.append([x,y]);return y


def packet(env,kind):
    # Gated family has no equality in initial data; only the learner-generated
    # probe set can expose the omitted case. Audits/holdouts use fresh entity IDs.
    train_inputs=[env.case(kind,unequal=kind==2) for _ in range(6)]
    train=[[x,env.observe(x)] for x in train_inputs]
    pool=generate_probes(train,limit=144)
    audit=[env.case(kind,force_equal=(kind==2 and i%2==0)) for i in range(12)]
    holdout=[env.case(kind,size=5,spread=40,force_equal=(kind==2 and i%2==0)) for i in range(32)]
    return train,pool,audit,holdout


def score(m,env,inputs):
    rows=[]
    for x in inputs:
        y=m.predict(*x);truth=env.observe(x)
        rows.append({'input':x,'prediction':y,'expected':truth})
    return {'correct':sum(r['prediction']==r['expected'] for r in rows),'wrong':sum(r['prediction'] is not None and r['prediction']!=r['expected'] for r in rows),'abstain':sum(r['prediction'] is None for r in rows),'rows':rows}


def plan_task(env,m):
    env.serial=9000;s=env.state(4,10);a,b,c,d=list(s);f,g=env.fields
    # Set distinct values to make goals reproducibly meaningful. Reference
    # creates a goal via two effects; the candidate gets only partial outcome.
    s[a][f],s[b][f],s[c][f],s[d][f]=11,17,23,31
    s[a][g],s[b][g],s[c][g],s[d][g]=41,43,47,53
    target={c:{f:11},d:{g:43}}
    r=m.plan(s,target,max_depth=2,state_budget=20000)
    actual=s
    for action in r['actions'] or []:actual=env.observe([actual,action])
    success=all(actual[n][k]==v for n,fs in target.items() for k,v in fs.items())
    return {'start':s,'goal':target,'plan':r,'actual':actual,'success':success}


def trial(seed):
    env=Environment(seed);packets=[packet(env,k) for k in range(5)]
    arms={}
    for label,options in [('active',{'selection':'active'}),('random',{'selection':'random'}),('no_inquiry',{'query_budget':0}),('no_conditional',{'conditional':False})]:
        m=SemanticMemory();reports=[]
        for train,pool,audit,test in packets:
            p=LiveMeasurements(env,pool);a=LiveMeasurements(env,audit)
            r=m.learn(train,p,a,seed=seed,**options)
            reports.append({'training':train,'proposed_probe_inputs':pool,'audit_inputs':audit,'learning':r,'probe_measurements':p.log,'audit_measurements':a.log,'holdout':score(m,env,test)})
        # Plan task deterministic across arms using fresh same-seed env state.
        plan_env=Environment(seed);plan=plan_task(plan_env,m)
        arms[label]={'reports':reports,'planning':plan}
    # A no-memory planner is a genuine counterfactual for acquired rules.
    arms['no_memory_planning']=plan_task(Environment(seed),SemanticMemory())
    return {'seed':seed,'arms':arms}


def campaign(seed,trials):
    runs=[trial(seed+i) for i in range(trials)];summary={}
    for arm in ('active','random','no_inquiry','no_conditional'):
        rr=[r['arms'][arm] for r in runs]
        summary[arm]={'families':[{'promoted':sum(x['reports'][i]['learning']['promoted'] for x in rr),**{k:sum(x['reports'][i]['holdout'][k] for x in rr) for k in ('correct','wrong','abstain')},'queries':sum(len(x['reports'][i]['learning']['queries']) for x in rr),'audit_labels':sum(len(x['reports'][i]['learning']['audit']) for x in rr),'candidate_attempts':sum(x['reports'][i]['learning']['candidate_attempts'] for x in rr)} for i in range(5)],'plans_achieved':sum(x['planning']['success'] for x in rr)}
    summary['no_memory_plans_achieved']=sum(r['arms']['no_memory_planning']['success'] for r in runs)
    return {'seed':seed,'trials':trials,'families':['copy','swap','equality-gated copy','add and clear','multiplication (unsupported control)'],'summary':summary,'runs':runs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=351100);p.add_argument('--trials',type=int,default=2);p.add_argument('--output');p.add_argument('--check',action='store_true');args=p.parse_args()
    if not 1<=args.trials<=50:p.error('trials must be 1–50')
    r=campaign(args.seed,args.trials);print(json.dumps(r['summary'],indent=2))
    if args.output:Path(args.output).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if args.check:
        a=r['summary']['active'];n=args.trials
        # Engineering regression gates, not a general-intelligence pass criterion.
        return int(not (all(f['correct']==32*n and f['wrong']==0 for f in a['families'][:4]) and a['families'][4]['abstain']==32*n and a['plans_achieved']==n and r['summary']['no_memory_plans_achieved']==0))
    return 0

if __name__=='__main__':raise SystemExit(main())
