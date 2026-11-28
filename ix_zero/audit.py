"""Multi-observer evidence protocol. Majority is an assumption, not a truth
oracle. The host must establish actual source independence. Each observation
is retained, including disagreements. No candidate-selected ground truth.
"""
from collections import Counter
from .exploration import freeze, Measurements


class AuditConsensus:
    def __init__(self,sources,*,minimum=3,policy="majority"):
        if not isinstance(sources,dict) or not 3<=len(sources)<=7 or type(minimum) is not int or not 3<=minimum<=len(sources):raise ValueError('three or more named sources required')
        if any(not isinstance(k,str) or not k.strip() or len(k)>80 for k in sources):raise ValueError('invalid source identity')
        if policy not in ("majority","unanimous"):raise ValueError("unknown audit policy")
        self.policy=policy
        self.sources={k:Measurements(v) for k,v in sorted(sources.items())}
        sets=[set(v.inputs) for v in self.sources.values()]
        if any(s!=sets[0] for s in sets):raise ValueError('source input scopes differ')
        self.inputs=next(iter(self.sources.values())).inputs;self.minimum=minimum;self.log=[];self.asked=[]

    def observe(self,x):
        votes={k:v.observe(x) for k,v in self.sources.items()};self.asked.append(x);counts=Counter(votes.values());y,n=counts.most_common(1)[0]
        ok=(n==len(votes) if self.policy=="unanimous" else n>len(votes)/2) and len(votes)>=self.minimum
        self.log.append({'input':x,'votes':votes,'agreement':n,'accepted':ok})
        if not ok:raise AuditConflict('no majority evidence')
        return y


class AuditConflict(ValueError):pass
