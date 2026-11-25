"""Bounded representation discovery, competing model families and evidence replay.

Search programs are pure scalar expression trees, not executable host code. A
factorized table learner is a separate inductive bias, not independent evidence.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import re
from .core import Space
from .checkpoint import canonical
from ._ix_vendor import parse_ix, run_ix, IXError
from importlib.resources import files

OPS = ('add', 'sub', 'mul', 'eq', 'lt')
CAP = 6000
MAX_COST = 5
MAX_MACROS = 8
MAX_EPISODES = 24
MIN_AUDITS = 3


def size(e):
    return 1 if e[0] in ('var', 'const') else 1+size(e[1])+size(e[2])


def evaluate(e, x, m):
    op = e[0]
    if op == 'var': return x[e[1]]
    if op == 'const': return e[1] % m
    a,b = evaluate(e[1],x,m),evaluate(e[2],x,m)
    if op == 'add': return (a+b)%m
    if op == 'sub': return (a-b)%m
    if op == 'mul': return (a*b)%m
    if op == 'eq': return int(a==b)
    if op == 'lt': return int(a<b)
    raise ValueError('unsupported expression')


def combine(op,a,b,m):
    if op=='add': return bytes((u+v)%m for u,v in zip(a,b))
    if op=='sub': return bytes((u-v)%m for u,v in zip(a,b))
    if op=='mul': return bytes((u*v)%m for u,v in zip(a,b))
    if op=='eq': return bytes(u==v for u,v in zip(a,b))
    if op=='lt': return bytes(u<v for u,v in zip(a,b))
    raise ValueError('unsupported expression')


@dataclass(frozen=True)
class Candidate:
    expr: tuple
    table: bytes
    cost: int


@lru_cache(maxsize=16)
def enumerate_models(space, macros=(), max_cost=MAX_COST):
    """Deduplicate on ALL finite inputs, not only observed inputs.

    Cost counts a previously acquired macro as one token. The cap can truncate
    search; no completeness claim outside the actual enumerated model set.
    """
    if max_cost not in (3,5,7) or len(macros)>MAX_MACROS:
        raise ValueError('search budget')
    states=space.states; m=space.modulus
    levels={1:[]}; seen={}; attempts=0
    def add(e,table,cost):
        if table in seen: return
        c=Candidate(e,table,cost); seen[table]=c; levels[cost].append(c)
    for e in [('var',i) for i in range(space.width)]+[('const',i) for i in range(m)]+list(macros):
        add(e,bytes(evaluate(e,x,m) for x in states),1)
    truncated=False
    for cost in range(3,max_cost+1,2):
        levels[cost]=[]
        for left in range(1,cost-1,2):
            for a in levels[left]:
                for b in levels[cost-left-1]:
                    for op in OPS:
                        attempts+=1
                        if attempts>120_000 or len(seen)>=CAP:
                            truncated=True; break
                        if size(a.expr)+size(b.expr)+1>31: continue
                        add((op,a.expr,b.expr),combine(op,a.table,b.table,m),cost)
                    if truncated: break
                if truncated: break
            if truncated: break
        if truncated: break
    return tuple(seen.values()),truncated,attempts


@dataclass
class Episode:
    space: Space
    macros: tuple=()
    sources: tuple=()
    mode: str='adaptive'
    observations: dict=field(default_factory=dict,init=False)
    audits: dict=field(default_factory=dict,init=False)
    retracted: bool=False
    stage: int=3

    def __post_init__(self):
        if self.mode not in ('adaptive','symbolic','projection','strict'): raise ValueError('invalid mode')
        if self.mode=='strict': self.stage=5
        self.index={x:i for i,x in enumerate(self.space.states)}
        self._refresh()

    @property
    def blocked(self):
        return self.retracted or any(p is not None and p!=y for p,y in self.audits.values())

    @property
    def passes(self):
        return sum(p==y for p,y in self.audits.values())

    def _refresh(self):
        self.pool,self.truncated,self.attempts=enumerate_models(self.space,self.macros,self.stage)
        self.symbolic=[]; self.projections=[]
        for j in range(self.space.width):
            hits=[]; best=None
            for c in self.pool:
                if self.mode!='strict' and best is not None and c.cost>best: break
                if all(c.table[self.index[x]]==y[j] for x,y in self.observations.items()):
                    hits.append(c); best=c.cost
            self.symbolic.append(hits)
            # Learn minimal sufficient observed coordinate subsets. Do not invent
            # values for unseen keys. Require each selected key to have support.
            projections=[]
            for n in range(self.space.width+1):
                for cols in combinations(range(self.space.width),n):
                    table={}; counts=Counter(); valid=True
                    for x,y in self.observations.items():
                        key=tuple(x[i] for i in cols)
                        if key in table and table[key]!=y[j]: valid=False; break
                        table[key]=y[j]; counts[key]+=1
                    if valid and table and all(v>=2 for v in counts.values()):
                        projections.append((cols,table))
                if projections: break
            self.projections.append(projections)

    def observe(self,x,y):
        x,y=self.space.check(x),self.space.check(y)
        if self.audits or self.blocked: raise ValueError('frozen or blocked episode')
        if x in self.observations:
            if self.observations[x]!=y: self.retracted=True
            else: raise ValueError('duplicate observation')
            return
        self.observations[x]=y; self._refresh()

    def expand(self):
        if self.audits or self.blocked or self.stage!=3: raise ValueError('expansion unavailable')
        self.stage=5; self._refresh()

    @staticmethod
    def agreed(values):
        return next(iter(values)) if len(values)==1 else None

    def lanes(self,x):
        x=self.space.check(x); k=self.index[x]
        s=[]; p=[]
        for j in range(self.space.width):
            s.append(self.agreed({c.table[k] for c in self.symbolic[j]}))
            rows=self.projections[j]
            vals=[t.get(tuple(x[i] for i in cols)) for cols,t in rows]
            p.append(self.agreed(set(vals)) if vals and None not in vals else None)
        return tuple(s),tuple(p)

    def predict(self,x,validated=False):
        x=self.space.check(x)
        if self.blocked or not self.observations or (validated and self.passes<MIN_AUDITS): return None
        s,p=self.lanes(x); out=[]
        for a,b in zip(s,p):
            if self.mode=='symbolic': v=a
            elif self.mode=='projection': v=b
            elif self.mode=='strict' and self.symbolic[len(out)] and a is None: v=None
            elif a is not None and b is not None and a!=b: v=None
            else: v=a if a is not None else b
            if v is None: return None
            out.append(v)
        return tuple(out)

    def query(self,allowed=None):
        if self.blocked or self.audits: return None
        pool=set(self.space.states if allowed is None else (self.space.check(x) for x in allowed))-self.observations.keys()
        if not pool: return None
        def score(x):
            k=self.index[x]; total=0.0
            for hits in self.symbolic:
                if hits:
                    c=Counter(h.table[k] for h in hits); total+=1-sum((v/len(hits))**2 for v in c.values())
            s,p=self.lanes(x)
            return total+sum(a!=b for a,b in zip(s,p) if a is not None and b is not None)+0.1*sum(a is None for a in p)
        return max(sorted(pool),key=score)

    def audit(self,x,y):
        x,y=self.space.check(x),self.space.check(y)
        if self.blocked or not self.observations or x in self.observations or x in self.audits: raise ValueError('audit must be fresh and episode active')
        p=self.predict(x); self.audits[x]=(p,y)
        return p==y

    def learned(self):
        if self.blocked or self.passes<MIN_AUDITS or self.mode=='projection': return ()
        # Require symbolic agreement on every recorded audit, even if the
        # projection lane supplied a combined prediction.
        result=[]
        for j,hits in enumerate(self.symbolic):
            if len(hits)!=1: continue
            e=hits[0].expr
            if size(e)<3 or size(e)>15: continue
            if all(hits[0].table[self.index[x]]==y[j] for x,(_,y) in self.audits.items()): result.append(e)
        return tuple(dict.fromkeys(result))

    def status(self):
        return dict(mode=self.mode, sources=self.sources, stage=self.stage, candidates=[len(v) for v in self.symbolic],
                    projection_keys=[[cols for cols,_ in v] for v in self.projections],
                    observations=len(self.observations),audits=len(self.audits),passes=self.passes,
                    blocked=self.blocked,library_size=len(self.macros),truncated=self.truncated,
                    candidate_pool=len(self.pool),enumeration_attempts=self.attempts)


class DiscoveryMind:
    """Append-only replay journal; explicit source dependencies and rollback.

    Digest provides corruption detection, not authentication against local forgery.
    """
    def __init__(self):
        self.episodes={}; self.promoted={}; self.events=[]

    def _library(self,space):
        pairs=[]
        for n,expressions in self.promoted.items():
            if self.episodes[n].blocked or self.episodes[n].space.width!=space.width: continue
            for e in expressions:
                if e not in [p[1] for p in pairs]: pairs.append((n,e))
        return pairs[:MAX_MACROS]

    def _cascade(self):
        for t in self.episodes.values():
            if any(self.episodes[n].blocked for n in t.sources): t.retracted=True

    def apply(self,action,name,*args):
        if len(self.events)>=4096: raise ValueError('journal budget')
        if not isinstance(name,str): raise ValueError('invalid episode name')
        if action=='new':
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',name) or name in self.episodes or len(self.episodes)>=MAX_EPISODES: raise ValueError('invalid episode name')
            width,modulus,reuse,mode=args; space=Space(width,modulus)
            if type(reuse) is not bool: raise ValueError('reuse must be boolean')
            pairs=self._library(space) if reuse else []
            self.episodes[name]=Episode(space,tuple(e for _,e in pairs),tuple(dict.fromkeys(n for n,_ in pairs)),mode)
            result=None
        else:
            t=self.episodes[name]
            if action=='observe': result=t.observe(*args)
            elif action=='expand' and not args: result=t.expand()
            elif action=='audit': result=t.audit(*args)
            elif action=='promote' and not args:
                result=t.learned()
                if not result or name in self.promoted: raise ValueError('no new verified representation')
                self.promoted[name]=result
            elif action=='retract' and not args: t.retracted=True; result=None
            else: raise ValueError('unsupported action')
        self.events.append(json.loads(canonical([action,name,*args]))); self._cascade(); return result

    def advance(self,name,source=None):
        t=self.episodes[name]; unused=len(t.space.states)-len(t.observations)-len(t.audits)
        available=sum(t.predict(x) is not None for x in t.space.states if x not in t.observations and x not in t.audits)
        facts=dict(blocked=t.blocked,expandable=t.stage==3 and not t.audits,
                   missing=any(not h for h in t.symbolic),observations=len(t.observations),
                   audits=len(t.audits),passes=t.passes,unused=unused,available=available,
                   promotable=bool(t.learned()) and name not in self.promoted)
        src=files('ix_zero').joinpath('policies','discovery.ix').read_text(encoding='utf-8') if source is None else source
        if not isinstance(src,str) or len(src)>8192: raise ValueError('invalid IX policy')
        try: r=run_ix(parse_ix(src),agent='Discovery',event='advance',inputs=facts)
        except (IXError,ArithmeticError,TypeError,RecursionError) as e: raise ValueError(f'IX policy refused: {e}') from e
        if r.approvals_required or len(r.replies)!=1: raise ValueError('invalid policy reply')
        action=r.replies[0]
        if action=='discover' and not t.blocked and facts['expandable'] and t.observations: self.apply('expand',name); result='Expanded expression search.'
        elif action=='inquire' and not t.blocked and not t.audits and unused: result=f'Request observation {t.query()}'
        elif action=='audit' and not t.blocked and t.observations and unused and t.passes<MIN_AUDITS: result='Request fresh evaluator-selected audit input.'
        elif action=='abstract' and facts['promotable'] and not t.blocked: result=str(self.apply('promote',name))
        elif action=='complete' and not t.blocked and t.passes>=MIN_AUDITS: result='Bounded audit gate passed.'
        elif action=='revise' and t.blocked: result='Quarantined; create new episode.'
        elif action=='exhausted' and not t.blocked and (not unused or t.audits): result='Abstain; insufficient independent evidence.'
        else: raise ValueError('IX action denied by host')
        return dict(action=action,result=result,facts=facts,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())

    def plan(self,space,start,goal,actions):
        """Search only audit-gated edges. A proposal, never an external action."""
        start,goal=space.check(start),space.check(goal); names=tuple(dict.fromkeys(actions))
        if any(self.episodes[n].space!=space or self.episodes[n].blocked or self.episodes[n].passes<MIN_AUDITS for n in names):
            raise ValueError('action model not audit-gated in this space')
        queue=deque([(start,())]); seen={start}
        while queue:
            x,path=queue.popleft()
            if x==goal:return path
            for n in names:
                y=self.episodes[n].predict(x,True)
                if y is not None and y not in seen:
                    seen.add(y);queue.append((y,path+((n,y),)))
        return None

    def save(self,path):
        data={'schema':1,'events':self.events}; blob=canonical({'data':data,'sha256':hashlib.sha256(canonical(data)).hexdigest()})
        if len(blob)>2_000_000: raise ValueError('checkpoint too large')
        path=Path(path); fd,tmp=tempfile.mkstemp(prefix='.ix-discovery-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f: blob=f.read(2_000_001)
        if len(blob)>2_000_000: raise ValueError('checkpoint too large')
        obj=json.loads(blob); data=obj['data']
        if data['schema']!=1 or hashlib.sha256(canonical(data)).hexdigest()!=obj['sha256']: raise ValueError('checkpoint invalid')
        mind=cls()
        for action,name,*args in data['events']: mind.apply(action,name,*args)
        if canonical(mind.events)!=canonical(data['events']): raise ValueError('replay mismatch')
        return mind
