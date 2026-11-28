"""Cross-lexicon relational transfer and variable-arity action acquisition.
All reference environments are authored diagnostics; action rules are separate
from the learner. No external AI model or tool-produced target is used.
"""
import argparse
import json
from pathlib import Path
from .grounded_semantics import SemanticMemory, generate_probes
from .semantic_experiment import Environment, LiveMeasurements, packet, score


class ReversedEnvironment(Environment):
    def words(self,kind,a,b):return super().words(kind,b,a)
    def observe(self,raw):
        s,w=raw;verb=next(t for t in w if t in self.verbs);i=w.index(verb)
        entities=[t for t in w if t!=verb][::-1];entities.insert(i,verb)
        return super().observe([s,entities])


class ArityEnvironment(Environment):
    def case(self,kind,size=3,spread=8):
        s=self.state(size,spread);entities=list(s)[:1 if kind==0 else 3];entities.insert(self.slots[kind]%(len(entities)+1),self.verbs[kind]);return [s,entities]
    def observe(self,raw):
        s,w=raw;out={n:dict(fs) for n,fs in s.items()};verb=next(t for t in w if t in self.verbs);roles=[t for t in w if t!=verb];f,g=self.fields
        if verb==self.verbs[0]:out[roles[0]][f]=s[roles[0]][f]+1
        elif verb==self.verbs[1]:out[roles[2]][f]=s[roles[0]][f]+s[roles[1]][g]
        else:raise ValueError('unknown arity reference action')
        return out


def measured(env,train,pool,audit,test,m,method,**kw):
    p=LiveMeasurements(env,pool);a=LiveMeasurements(env,audit)
    r=method(train,p,a,**kw)
    return {'training':train,'pool_inputs':pool,'audit_inputs':audit,'learning':r,'pool_measurements':p.log,'audit_measurements':a.log,'holdout':score(m,env,test)}


def trial(seed):
    source=Environment(seed);base=SemanticMemory();source_reports=[]
    for k in range(4):
        tr,p,a,t=packet(source,k);source_reports.append(measured(source,tr,p,a,t,base,base.learn,seed=seed))
    target=ReversedEnvironment(seed+10000000);scratch=SemanticMemory();empty=SemanticMemory();rows=[]
    for k in range(4):
        inputs=[target.case(k,unequal=k==2) for _ in range(2)]
        tr=[[x,target.observe(x)] for x in inputs];p=generate_probes(tr,limit=144)
        a=[target.case(k,force_equal=k==2 and i%2==0) for i in range(12)]
        t=[target.case(k,size=6,spread=60,force_equal=k==2 and i%2==0) for i in range(32)]
        rows.append({'transfer':measured(target,tr,p,a,t,base,base.transfer,budget=256,query_budget=4),'from_scratch':measured(target,tr,p,a,t,scratch,scratch.learn,seed=seed,query_budget=4),'empty_transfer':measured(target,tr,p,a,t,empty,empty.transfer,budget=256,query_budget=4)})
    # A deliberate contradictory measurement, labelled as a falsification
    # control rather than a real environment change, must revoke dependents.
    c=source.case(0);bad=source.observe(c);name=next(iter(bad));f=source.fields[0];bad[name][f]+=1000
    challenge=base.challenge(*c,bad)
    arity=ArityEnvironment(seed+20000000);m=SemanticMemory();arity_rows=[]
    for k in range(2):
        inputs=[arity.case(k) for _ in range(8)];tr=[[x,arity.observe(x)] for x in inputs];p=generate_probes(tr,limit=192);a=[arity.case(k) for _ in range(12)];t=[arity.case(k,size=6,spread=40) for _ in range(32)]
        arity_rows.append(measured(arity,tr,p,a,t,m,m.learn,seed=seed,query_budget=24))
    return {'seed':seed,'source':source_reports,'target':rows,'arity':arity_rows,'deliberate_contradiction':challenge,'dependencies':base.dependencies}


def campaign(seed,trials):
    runs=[trial(seed+i) for i in range(trials)];summary={}
    def aggregate(reports):
        return {'promoted':sum(r['learning']['promoted'] for r in reports),**{k:sum(r['holdout'][k] for r in reports) for k in ('correct','wrong','abstain')},'queries':sum(len(r['learning']['queries']) for r in reports),'audit_labels':sum(len(r['learning']['audit']) for r in reports),'candidate_attempts':sum(r['learning']['candidate_attempts'] for r in reports)}
    summary['source']=[aggregate([r['source'][i] for r in runs]) for i in range(4)]
    for arm in ('transfer','from_scratch','empty_transfer'):summary[arm]=[aggregate([r['target'][i][arm] for r in runs]) for i in range(4)]
    summary['arity']=[aggregate([r['arity'][i] for r in runs]) for i in range(2)]
    summary['withdrawals']=[r['deliberate_contradiction']['withdrawn'] for r in runs]
    return {'seed':seed,'trials':trials,'summary':summary,'runs':runs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=361100);p.add_argument('--trials',type=int,default=2);p.add_argument('--output');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=30:p.error('trials must be 1–30')
    r=campaign(a.seed,a.trials);print(json.dumps(r['summary'],indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if a.check:
        s=r['summary'];n=a.trials
        return int(not (all(x['correct']==32*n and x['wrong']==0 for x in s['transfer']) and all(x['abstain']==32*n for x in s['empty_transfer']) and all(x['correct']==32*n for x in s['arity'])))
    return 0

if __name__=='__main__':raise SystemExit(main())
