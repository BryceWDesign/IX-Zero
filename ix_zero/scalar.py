"""Bounded signed integer induction extending the original Sally grammar.
Pure expressions only. Active queries and independent audit labels are supplied
by a caller; held-out targets are not used in fitting.
"""
from collections import Counter
from functools import lru_cache
from ._sally_vendor.cognition.developmental_learning import InductiveLearner

LIMIT=32


def checked(x):
    if not isinstance(x,(tuple,list)) or len(x)!=2 or any(type(i) is not int or not -LIMIT<=i<=LIMIT for i in x):raise ValueError('invalid signed pair')
    return tuple(x)


def truth_value(y):
    if type(y) is not int or abs(y)>1_000_000:raise ValueError('invalid signed outcome')
    return y


@lru_cache(maxsize=1)
def vocabulary():
    # Inherit Sally's working expression search, rather than invent a weaker replacement.
    out=[]
    for p in InductiveLearner()._hypotheses:
        cost=p.complexity;name=p.expression()
        out.append((cost,name,p.evaluate))
        # General translation and scaling by small constants. No world-specific rules.
        for c in (-3,-2,-1,1,2,3):
            out.append((cost+2,f'({name}+{c})',lambda x,p=p,c=c:p.evaluate(x)+c))
        for c in (-3,-2,2,3):
            out.append((cost+2,f'({name}*{c})',lambda x,p=p,c=c:p.evaluate(x)*c))
    for i in range(2):
        for j in range(2):
            if i==j:continue
            out.extend([(3,f'x{i}//x{j} (zero=0)',lambda x,i=i,j=j:x[i]//x[j] if x[j] else 0),
                        (3,f'x{i}%x{j} (zero=0)',lambda x,i=i,j=j:x[i]%x[j] if x[j] else 0)])
    for c in range(-3,4):out.append((1,str(c),lambda x,c=c:c))
    return tuple(sorted(out,key=lambda r:(r[0],r[1])))


class ScalarEpisode:
    def __init__(self):
        self.candidates=vocabulary();self.training={};self.audits={};self.blocked=False
    def observe(self,x,y):
        x,y=checked(x),truth_value(y)
        if self.audits or self.blocked:raise ValueError('training frozen or blocked')
        if x in self.training:
            if self.training[x]!=y:self.blocked=True
            return False
        if len(self.training)>=128:raise ValueError('training budget')
        self.training[x]=y;self.candidates=tuple(r for r in self.candidates if r[2](x)==y)
        if not self.candidates:self.blocked=True
        return True
    def simplest(self):
        if not self.candidates:return ()
        c=self.candidates[0][0];return tuple(r for r in self.candidates if r[0]==c)
    def predict(self,x,validated=True):
        x=checked(x)
        if self.blocked or len(self.training)<2 or (validated and sum(self.audits.values())<3):return None
        vals={r[2](x) for r in self.simplest()}
        return next(iter(vals)) if len(vals)==1 else None
    def query(self,pool):
        if self.blocked:return None
        pool=tuple(dict.fromkeys(checked(x) for x in pool))
        if len(pool)>256:raise ValueError('query budget')
        remain=[x for x in pool if x not in self.training and x not in self.audits]
        if not remain:return None
        models=self.simplest() if self.audits else self.candidates
        # Request high-disagreement probes even after freeze; evaluator, not the
        # learner, chooses the actual audit inputs and supplies outcomes.
        def score(x):
            counts=Counter(r[2](x) for r in models)
            return (len(counts),-max(counts.values(),default=0),-abs(x[0])-abs(x[1]),x)
        return max(remain,key=score)
    def audit(self,x,y):
        x,y=checked(x),truth_value(y)
        if self.blocked or x in self.training or x in self.audits or len(self.audits)>=128:raise ValueError('audit blocked, repeated or exceeds budget')
        ok=self.predict(x,False)==y;self.audits[x]=ok
        if not ok:self.blocked=True
        return ok
    def status(self):return {'candidates':len(self.candidates),'training':len(self.training),'audits':len(self.audits),'passes':sum(self.audits.values()),'blocked':self.blocked,'simplest':[r[1] for r in self.simplest()][:16]}
