"""Bounded cross-environment inquiry over executable hypotheses.

Equivalence is joint across declared widths, never inferred from one width.
Only the evaluator can grant a new environment and supply measured outcomes.
No evaluator functions, arbitrary code, or external authority enter the learner.
"""
from collections import Counter
from functools import lru_cache
from .operators import (Domain, Candidate, Search, execute, nodes, substitute,
                        MAX_ATTEMPTS, MAX_MODELS, MAX_NODES, MAX_LIBRARY,
                        MIN_AUDITS, MIN_TRAINING)

WIDTHS = (2, 3, 4, 5)


def widths_of(widths):
    if not isinstance(widths, (tuple, list)) or not 1 <= len(widths) <= len(WIDTHS):
        raise ValueError('invalid epistemic scope')
    if any(type(w) is not int or w not in WIDTHS for w in widths) or len(set(widths)) != len(widths):
        raise ValueError('invalid epistemic widths')
    return tuple(sorted(widths))


@lru_cache(maxsize=8)
def joint_search(widths, budget, library=()):
    """Enumerate bounded gate-cost semantics over the whole declared scope.

    Semantic signatures are tuples of complete packed tables, one per width.
    Congruent subexpressions can be merged safely across all these domains.
    A truncated layer is discarded; incomplete searches cannot be promoted.
    """
    if widths_of(widths) != widths or type(budget) is not int or not 1 <= budget <= 5 or len(library) > MAX_LIBRARY:
        raise ValueError('invalid joint search budget or scope')
    ds = [Domain(w) for w in widths]
    full = tuple((1 << (len(d.inputs)*d.width))-1 for d in ds)
    low = tuple(sum(1 << (i*d.width) for i in range(len(d.inputs))) for d in ds)
    high = tuple(l << (d.width-1) for l,d in zip(low,ds))
    def packed(e,a,b):
        op=e[0]
        if op=='x':return a
        if op=='y':return b
        if op=='zero':return (0,)*len(ds)
        if op=='one':return low
        if op=='nand':return tuple(f^(x&y) for f,x,y in zip(full,packed(e[1],a,b),packed(e[2],a,b)))
        if op=='shl':return tuple((x<<1)&(f^l) for x,f,l in zip(packed(e[1],a,b),full,low))
        if op=='shr':return tuple((x>>1)&(f^h) for x,f,h in zip(packed(e[1],a,b),full,high))
        raise ValueError('invalid joint syntax')
    levels=[[] for _ in range(budget+1)]; seen=set();attempts=0
    for e in [('x',),('y',),('zero',),('one',)]:
        tables=tuple(sum(execute(e,x,d.width)<<(i*d.width) for i,x in enumerate(d.inputs)) for d in ds)
        levels[0].append(Candidate(e,tables,0));seen.add(tables)
    class Limit(Exception):pass
    for cost in range(1,budget+1):
        local=set()
        def add(expr,tables,deps):
            nonlocal attempts
            attempts+=1
            if attempts>MAX_ATTEMPTS or len(seen)+len(local)>=MAX_MODELS:raise Limit
            if tables in seen or tables in local or nodes(expr)>MAX_NODES:return
            local.add(tables);levels[cost].append(Candidate(expr,tables,cost,tuple(sorted(set(deps)))))
        try:
            for a in levels[cost-1]:
                for op in ('shl','shr'):
                    add((op,a.expr),packed((op,('x',)),a.table,()),a.dependencies)
            for i in range(cost):
                for a in levels[i]:
                    for b in levels[cost-1-i]:
                        deps=a.dependencies+b.dependencies
                        add(('nand',a.expr,b.expr),tuple(f^(x&y) for f,x,y in zip(full,a.table,b.table)),deps)
                        for name,expr in library:
                            add(substitute(expr,a.expr,b.expr),packed(expr,a.table,b.table),deps+(name,))
        except Limit:
            return Search(tuple(c for layer in levels[:cost] for c in layer),attempts,cost-1,True)
        seen.update(local)
    return Search(tuple(c for layer in levels for c in layer),attempts,budget,False)


class Inquiry:
    def __init__(self,widths,budget,initial_width,train,audit,library=()):
        self.widths=widths_of(widths);self.budget=budget
        if type(initial_width) is not int or initial_width not in self.widths:raise ValueError('initial width outside scope')
        self.search=joint_search(self.widths,budget,tuple(library));self.models=self.search.models
        self.domains={w:Domain(w) for w in self.widths};self.indices={w:i for i,w in enumerate(self.widths)}
        self.pools={};self.training={};self.audits={};self.challenges={};self.blocked=False;self.finished=False;self.reason='';self.expansions=[]
        self.grant(initial_width,train,audit)

    @staticmethod
    def validate_pool(width,train,audit):
        d=Domain(width)
        if not isinstance(train,(list,tuple)) or not isinstance(audit,(list,tuple)) or not 1<=len(train)<=256 or not MIN_AUDITS<=len(audit)<=64:
            raise ValueError('invalid epistemic evidence pool size')
        t=tuple(d.pair(x) for x in train);a=tuple(d.pair(x) for x in audit)
        if len(set(t+a))!=len(t+a):raise ValueError('overlapping epistemic pools')
        return t,a

    def grant(self,width,train,audit):
        if type(width) is not int or width not in self.widths or width in self.pools or self.blocked or self.finished or self.audits:
            raise ValueError('invalid environment grant')
        self.pools[width]=self.validate_pool(width,train,audit)
        self.expansions.append(width)

    def input(self,x):
        if not isinstance(x,(list,tuple)) or len(x)!=3 or type(x[0]) is not int or x[0] not in self.domains:raise ValueError('expected [width,a,b] in declared scope')
        return (x[0],*self.domains[x[0]].pair(x[1:]))

    def value(self,c,x):
        w,a,b=x;return self.domains[w].value(c.table[self.indices[w]],(a,b))

    def partition(self,x):return Counter(self.value(c,x) for c in self.models)

    def rank(self,x):
        p=self.partition(x);return (len(p),-max(p.values()),-x[0],x[1],x[2])

    def distinct_at(self,w):return len({c.table[self.indices[w]] for c in self.models})

    def choose(self):
        """Generate the next knowledge objective from unresolved ambiguity."""
        if self.blocked or self.finished:return {'action':'failed' if self.blocked else 'complete','reason':self.reason}
        if not self.models or self.search.truncated:
            return {'action':'failed','reason':'hypothesis family exhausted or search incomplete'}
        if not self.audits:
            available=[(w,*x) for w,(t,_) in self.pools.items() for x in t if (w,*x) not in self.training]
            if len(self.models)>1:
                if available:
                    best=max(available,key=self.rank)
                    if len(self.partition(best))>1:return {'action':'train','input':best,'reason':'discriminate surviving hypotheses'}
                # New environment objectives are derived from semantics, not a fixed escalation schedule.
                candidates=[w for w in self.widths if w not in self.pools and self.distinct_at(w)>1]
                if candidates:
                    w=max(candidates,key=lambda v:(self.distinct_at(v),-v))
                    return {'action':'environment','input':w,'reason':'available evidence cannot discriminate; request another environment',
                            'local_distinctions':{str(v):self.distinct_at(v) for v in self.pools},'requested_distinctions':self.distinct_at(w)}
                return {'action':'failed','reason':'unresolved ambiguity with no informative available environment'}
            if len(self.training)<MIN_TRAINING:
                if available:return {'action':'train','input':min(available),'reason':'minimum training evidence'}
                return {'action':'failed','reason':'insufficient independent training evidence'}
            # A singleton within a supplied grammar is not evidence that the
            # grammar contains reality. Independently audit every declared scope.
            remaining=[w for w in self.widths if w not in self.pools]
            if remaining:
                w=max(remaining)
                return {'action':'environment','input':w,'reason':'unique bounded hypothesis; independent scope verification required',
                        'local_distinctions':{str(v):self.distinct_at(v) for v in self.pools},'requested_distinctions':self.distinct_at(w)}
        if len(self.models)!=1:return {'action':'failed','reason':'ambiguous hypothesis cannot be audited'}
        fresh=[(w,*x) for w,(_,a) in self.pools.items() for x in a if (w,*x) not in self.audits]
        if fresh:return {'action':'audit','input':min(fresh),'reason':'exhaust reserved validation evidence'}
        if self.validated():return {'action':'complete','reason':'unique joint semantics; all granted environments audited'}
        return {'action':'failed','reason':'insufficient validation'}

    def observe(self,x,y):
        x=self.input(x);w=x[0];y=self.domains[w].word(y)
        if self.blocked or self.finished or self.audits or w not in self.pools or x[1:] not in self.pools[w][0] or x in self.training:raise ValueError('invalid epistemic training evidence')
        self.training[x]=y;self.models=tuple(c for c in self.models if self.value(c,x)==y)

    def audit(self,x,y):
        x=self.input(x);w=x[0];y=self.domains[w].word(y)
        if self.blocked or self.finished or len(self.models)!=1 or len(self.training)<MIN_TRAINING or w not in self.pools or x[1:] not in self.pools[w][1] or x in self.audits:raise ValueError('invalid epistemic audit evidence')
        ok=self.value(self.models[0],x)==y;self.audits[x]=(y,ok)
        if not ok:self.blocked=True;self.reason='fresh audit contradicted joint hypothesis'
        return ok

    def validated(self):
        return not self.blocked and not self.search.truncated and len(self.models)==1 and len(self.training)>=MIN_TRAINING and set(self.pools)==set(self.widths) and all((w,*x) in self.audits and self.audits[(w,*x)][1] for w,(_,a) in self.pools.items() for x in a)

    def predict(self,x):
        x=self.input(x)
        if not self.finished or not self.validated() or x[0] not in self.pools:return None
        return execute(self.models[0].expr,x[1:],x[0])

    def challenge(self,x,y):
        """New external evidence may invalidate a previously audited law."""
        x=self.input(x);w=x[0];y=self.domains[w].word(y)
        if w not in self.pools or any(x[1:] in pool for pool in self.pools[w]) or x in self.challenges or self.predict(x) is None:raise ValueError('challenge requires a fresh input in an audited environment')
        ok=self.predict(x)==y;self.challenges[x]=(y,ok)
        if not ok:self.blocked=True;self.reason='later observation contradicted joint hypothesis'
        return ok

    def status(self):
        return dict(scope=self.widths,granted=self.expansions,remaining=len(self.models),search_models=len(self.search.models),
                    search_complete_cost=self.search.complete_cost,truncated=self.search.truncated,
                    local_distinctions={str(w):self.distinct_at(w) for w in self.widths},
                    expression=self.models[0].expr if len(self.models)==1 else None,
                    training=len(self.training),audits=len(self.audits),challenges=len(self.challenges),validated=self.finished and self.validated())
