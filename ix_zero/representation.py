"""Evidence-gated finite representation synthesis and program abstraction.

Candidate programs conjugate an acquired word operator by a reversible ordinal
encoding. Search uses unlabeled finite domains; outcomes are supplied separately.
The source operator remains live; no target-specific operation is installed.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import gcd
from importlib.resources import files
from .operators import execute
from .acce import Capability
from ._ix_vendor import parse_ix, run_ix

MIN_TRAINING = 6
MIN_AUDITS = 3


def domain_labels(labels):
    labels = tuple(labels)
    if len(labels) not in (4, 8, 16) or any(type(x) is not int or abs(x)>1_000_000 for x in labels) or len(set(labels)) != len(labels):
        raise ValueError('expected 4, 8 or 16 distinct bounded integer labels')
    return tuple(sorted(labels))


@dataclass(frozen=True, order=True)
class Syntax:
    source: str
    family: str
    multiplier: int
    shift: int

    def mapping(self, n):
        if self.family not in ('ordinal','gray') or not 0 < self.multiplier < n or gcd(self.multiplier,n)!=1 or not 0<=self.shift<n:
            raise ValueError('invalid reversible encoding')
        v=tuple((self.multiplier*i+self.shift)%n for i in range(n))
        return tuple(i^(i>>1) for i in v) if self.family=='gray' else v


@dataclass(frozen=True)
class Model:
    syntax: Syntax
    table: tuple


class Episode:
    def __init__(self, mind, labels, template=None, raw=False):
        self.mind=mind; self.labels=domain_labels(labels); self.n=len(self.labels)
        self.index={v:i for i,v in enumerate(self.labels)}
        self.sources={name:(ep,mind.promoted[name]) for name,ep in mind.episodes.items()
                      if name in mind.promoted and ep.domain.width==self.n.bit_length()-1 and self.source_live(name,ep,mind.promoted[name])}
        self.template=template; self.training={}; self.audits={}; self.blocked=False
        self.attempts=0; self.models=()
        if template is not None and (not template.live() or any(p.mind is not mind for p in template.parents)):
            self.blocked=True;return
        # Compute candidate behavior without target labels. Keep equivalent
        # syntax for abstraction; inquiry and consensus use distinct tables.
        # No target oracle is available here.
        models=[]
        for name,(ep,expr) in sorted(self.sources.items()):
            for fam,m,c in product(('ordinal','gray'),range(1,self.n,2),range(self.n)):
                s=Syntax(name,fam,m,c)
                if raw and (self.labels!=tuple(range(self.n)) or (fam,m,c)!=('ordinal',1,0)):continue
                if template is not None and not template.matches(s):continue
                self.attempts+=1
                enc=s.mapping(self.n); inv={v:i for i,v in enumerate(enc)}
                table=tuple(self.labels[inv[execute(expr,(a,b),ep.domain.width)]] for a,b in product(enc,repeat=2))
                models.append(Model(s,table))
        self.models=tuple(models)
        if not models:self.blocked=True

    def source_live(self,name,ep,expr):
        return (self.mind.episodes.get(name) is ep and self.mind.promoted.get(name)==expr and ep.resolved()
                and len(ep.audits)>=MIN_AUDITS and all(ok for _,ok in ep.audits.values()))

    def live(self):
        return (not self.blocked and bool(self.models) and (self.template is None or self.template.live())
                and all(self.source_live(name,*self.sources[name]) for name in {m.syntax.source for m in self.models}))

    def pair(self,x):
        if not isinstance(x,(tuple,list)) or len(x)!=2 or any(type(v) is not int or v not in self.index for v in x):raise ValueError('input outside representation domain')
        return tuple(x)

    def outcome(self,y):
        if type(y) is not int or y not in self.index:raise ValueError('output outside representation domain')
        return y

    def pos(self,x):return self.index[x[0]]*self.n+self.index[x[1]]

    def observe(self,x,y):
        x,y=self.pair(x),self.outcome(y)
        if not self.live() or self.audits or x in self.training:raise ValueError('training blocked, frozen or repeated')
        self.training[x]=y; p=self.pos(x)
        self.models=tuple(m for m in self.models if m.table[p]==y)
        if not self.models:self.blocked=True

    def predict(self,x,validated=True):
        p=self.pos(self.pair(x))
        if not self.live() or len(self.training)<MIN_TRAINING or (validated and len(self.audits)<MIN_AUDITS):return None
        vals={m.table[p] for m in self.models}
        return next(iter(vals)) if len(vals)==1 else None

    def query(self,pool):
        if not self.live():return None
        pool=tuple(dict.fromkeys(self.pair(x) for x in pool))
        # Weight distinct behaviors equally, not syntactic duplicates.
        tables=set(m.table for m in self.models)
        best=None
        for x in pool:
            if x in self.training or x in self.audits:continue
            c=Counter(t[self.pos(x)] for t in tables)
            rank=(len(c),-max(c.values()),x)
            if best is None or rank>best[0]:best=(rank,x)
        return best[1] if best else None

    def audit(self,x,y):
        x,y=self.pair(x),self.outcome(y)
        if not self.live() or len(self.training)<MIN_TRAINING or x in self.training or x in self.audits:raise ValueError('audit blocked, insufficient training or repeated')
        ok=self.predict(x,False)==y; self.audits[x]=ok
        if not ok:self.blocked=True
        return ok

    def resolved(self):return self.live() and len({m.table for m in self.models})==1

    def advance(self,pool=()):
        x=self.query(pool)
        facts=dict(blocked=not self.live(),validated=len(self.audits)>=MIN_AUDITS,has_query=x is not None,
                   frozen=bool(self.audits),enough=len(self.training)>=MIN_TRAINING,resolved=self.resolved())
        r=run_ix(parse_ix(files('ix_zero').joinpath('policies','acce.ix').read_text(encoding='utf-8')),agent='Composition',event='advance',inputs=facts)
        if r.approvals_required or len(r.replies)!=1:raise ValueError('invalid IX action')
        a=r.replies[0]
        allowed=('revise',) if facts['blocked'] else ('complete',) if facts['validated'] else ('inquire','exhausted') if not facts['enough'] else ('inquire','audit','exhausted')
        if a not in allowed or a=='inquire' and (x is None or facts['frozen']):raise ValueError('IX action rejected')
        return dict(action=a,request=x if a=='inquire' else None,trace=r.trace_as_dicts())

    def status(self):
        return dict(attempts=self.attempts,syntax_count=len(self.models),behaviors=len({m.table for m in self.models}),training=len(self.training),audits=len(self.audits),blocked=not self.live(),resolved=self.resolved(),syntax=[m.syntax.__dict__ for m in self.models[:12]])

    def capability(self,name):
        return Capability(name,('integer','integer'),'integer',lambda a,b:self.predict((a,b)),
                          lambda:self.live() and len(self.audits)>=MIN_AUDITS,'representation:'+name)


@dataclass(frozen=True)
class Abstraction:
    """Anti-unify canonical validated programs over distinct source domains.

    Slots are typed by the syntax field; None is a hole. A source operator must
    be common to both programs. Multiple equivalent syntax are conservatively
    reduced to one canonical syntax only after full-domain behavioral consensus.
    """
    source: str
    family: str | None
    multiplier: int | None
    shift: int | None
    parents: tuple

    @classmethod
    def induce(cls,a,b):
        if a is b or a.mind is not b.mind or a.labels==b.labels or a.n!=b.n or not all(e.resolved() and len(e.audits)>=MIN_AUDITS for e in (a,b)):
            raise ValueError('need distinct, resolved, validated source episodes')
        x,y=(min(m.syntax for m in e.models) for e in (a,b))
        if x.source!=y.source:raise ValueError('no common operator')
        def same(k):return getattr(x,k) if getattr(x,k)==getattr(y,k) else None
        return cls(x.source,same('family'),same('multiplier'),same('shift'),(a,b))

    def matches(self,s):
        return s.source==self.source and all(getattr(self,k) is None or getattr(self,k)==getattr(s,k) for k in ('family','multiplier','shift'))

    def live(self):return all(p.live() and p.resolved() and len(p.audits)>=MIN_AUDITS for p in self.parents)

    def status(self):return dict(source=self.source,family=self.family,multiplier=self.multiplier,shift=self.shift,live=self.live())


def fit_with_abstraction(mind, labels, template, observations):
    """Try an induced program family; expand search using training only.

    The returned episode is unaudited. A later audit failure must quarantine it;
    audit outcomes are never used to choose or refit the representation here.
    """
    labels=domain_labels(labels)
    rows=tuple(observations)
    if len(rows)>256:raise ValueError('training budget exceeded')
    seen=set()
    for x,y in rows:
        if not isinstance(x,(tuple,list)) or len(x)!=2 or any(type(v) is not int or v not in labels for v in x) or type(y) is not int or y not in labels or tuple(x) in seen:
            raise ValueError('invalid or repeated training observation')
        seen.add(tuple(x))
    def replay(e):
        for x,y in rows:
            if not e.live():break
            e.observe(x,y)
        return e
    e=replay(Episode(mind,labels,template))
    cost=e.attempts
    if e.resolved() and len(e.training)>=MIN_TRAINING:
        return e,dict(expanded=False,candidate_programs=cost)
    e=replay(Episode(mind,labels))
    return e,dict(expanded=True,candidate_programs=cost+e.attempts)
