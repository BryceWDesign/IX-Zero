"""Evidence-gated synthesis of reusable finite-word operators from NAND and shifts.

No target operators are installed in the learner. Search is bounded, pure and
finite. The evaluator supplies labels; the library stores executable syntax.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from importlib.resources import files
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .checkpoint import canonical
from ._ix_vendor import parse_ix, run_ix, IXError

MAX_ATTEMPTS = 250_000
MAX_MODELS = 35_000
MAX_EPISODES = 32
MAX_LIBRARY = 6
MAX_NODES = 127
MIN_AUDITS = 3
MIN_TRAINING = 6


def nodes(expr):
    return 1 + sum(nodes(e) for e in expr[1:])


def substitute(expr, a, b):
    if expr[0] == 'x': return a
    if expr[0] == 'y': return b
    return (expr[0], *(substitute(e, a, b) for e in expr[1:]))


@dataclass(frozen=True)
class Domain:
    width: int

    def __post_init__(self):
        if type(self.width) is not int or not 2 <= self.width <= 5:
            raise ValueError('word width must be 2 through 5')

    @property
    def mask(self): return (1 << self.width) - 1

    def word(self, v):
        if type(v) is not int or not 0 <= v <= self.mask:
            raise ValueError('word outside domain')
        return v

    def pair(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError('expected two words')
        return tuple(self.word(v) for v in x)

    @property
    def inputs(self): return tuple(product(range(self.mask + 1), repeat=2))

    def value(self, table, x):
        a, b = self.pair(x)
        return (table >> ((a * (self.mask + 1) + b) * self.width)) & self.mask


def execute(expr, x, width):
    """Scalar interpreter for learned syntax; no eval, imports or host calls."""
    d = Domain(width); a, b = d.pair(x); m = d.mask
    if nodes(expr) > MAX_NODES: raise ValueError('expression budget')
    def go(e):
        op = e[0]
        if op == 'x': return a
        if op == 'y': return b
        if op == 'zero': return 0
        if op == 'one': return 1
        if op == 'nand': return m ^ (go(e[1]) & go(e[2]))
        if op == 'shl': return (go(e[1]) << 1) & m
        if op == 'shr': return go(e[1]) >> 1
        raise ValueError('unknown instruction')
    return go(expr)


@dataclass(frozen=True)
class Candidate:
    expr: tuple
    table: int
    cost: int
    dependencies: tuple = ()


@dataclass(frozen=True)
class Search:
    models: tuple
    attempts: int
    complete_cost: int
    truncated: bool


@lru_cache(maxsize=12)
def synthesize(width, budget, library=()):
    """Enumerate expression semantics in increasing gate cost.

    Equivalence pruning uses the complete finite input domain, with no labels.
    Retained macros are callable binary operators, not fixed-input templates.
    A truncated cost layer is discarded, never presented as fully searched.
    """
    d = Domain(width)
    if type(budget) is not int or not 0 <= budget <= 6 or len(library) > MAX_LIBRARY:
        raise ValueError('invalid search budget')
    inputs = d.inputs
    full = (1 << (len(inputs) * width)) - 1
    low = sum(1 << (i * width) for i in range(len(inputs)))
    high = low << (width - 1)
    def packed(e, a, b):
        op = e[0]
        if op == 'x': return a
        if op == 'y': return b
        if op == 'zero': return 0
        if op == 'one': return low
        if op == 'nand': return full ^ (packed(e[1], a, b) & packed(e[2], a, b))
        if op == 'shl': return (packed(e[1], a, b) << 1) & (full ^ low)
        if op == 'shr': return (packed(e[1], a, b) >> 1) & (full ^ high)
        raise ValueError('invalid operator')
    levels = [[] for _ in range(budget + 1)]; seen = set(); attempts = 0
    for e, values in [(('x',), (a for a,b in inputs)), (('y',), (b for a,b in inputs)),
                      (('zero',), (0 for _ in inputs)), (('one',), (1 for _ in inputs))]:
        table = sum(v << (i * width) for i,v in enumerate(values))
        if table not in seen:
            levels[0].append(Candidate(e, table, 0)); seen.add(table)
    class Limit(Exception): pass
    for cost in range(1, budget + 1):
        local = set()
        def add(expr, table, deps):
            nonlocal attempts
            attempts += 1
            if attempts > MAX_ATTEMPTS or len(seen) + len(local) >= MAX_MODELS: raise Limit
            if table in seen or table in local or nodes(expr) > MAX_NODES: return
            local.add(table); levels[cost].append(Candidate(expr, table, cost, tuple(sorted(set(deps)))))
        try:
            for a in levels[cost-1]:
                for op in ('shl','shr'):
                    add((op,a.expr),packed((op,('x',)),a.table,0),a.dependencies)
            for i in range(cost):
                j = cost-1-i
                for a in levels[i]:
                    for b in levels[j]:
                        deps = a.dependencies+b.dependencies
                        add(('nand',a.expr,b.expr),full^(a.table&b.table),deps)
                        for name, expr in library:
                            add(substitute(expr,a.expr,b.expr),packed(expr,a.table,b.table),deps+(name,))
        except Limit:
            return Search(tuple(c for layer in levels[:cost] for c in layer),attempts,cost-1,True)
        seen.update(local)
    return Search(tuple(c for layer in levels for c in layer),attempts,budget,False)


class OperatorEpisode:
    def __init__(self, width, budget, library=()):
        self.domain = Domain(width)
        self.budget = budget
        self.library = tuple(library)
        self.search = synthesize(width,budget,self.library)
        self.training = {}; self.audits = {}; self.blocked = False
        self.models = self.search.models

    def observe(self, x, y):
        x,y = self.domain.pair(x), self.domain.word(y)
        if self.blocked or self.audits or x in self.training: raise ValueError('training blocked, frozen or repeated')
        self.training[x] = y
        self.models = tuple(c for c in self.models if self.domain.value(c.table,x)==y)

    def simplest(self):
        if not self.models: return ()
        cost = self.models[0].cost
        return tuple(c for c in self.models if c.cost == cost)

    def resolved(self): return len(self.training)>=MIN_TRAINING and len(self.simplest())==1 and not self.blocked

    def predict(self,x,validated=True):
        x = self.domain.pair(x)
        if not self.resolved() or (validated and len(self.audits)<MIN_AUDITS): return None
        return execute(self.simplest()[0].expr,x,self.domain.width)

    def audit(self,x,y):
        x,y = self.domain.pair(x),self.domain.word(y)
        if self.blocked or not self.resolved() or x in self.training or x in self.audits:
            raise ValueError('audit requires a resolved frozen hypothesis and fresh input')
        passed = self.predict(x,False)==y
        self.audits[x]=(y,passed)
        if not passed: self.blocked=True
        return passed

    def query(self, pool=None):
        if self.blocked or self.audits or not self.models: return None
        pool = self.domain.inputs if pool is None else tuple(dict.fromkeys(self.domain.pair(x) for x in pool))
        choices = [x for x in pool if x not in self.training and x not in self.audits]
        if not choices: return None
        models = self.models
        def score(x):
            counts = Counter(self.domain.value(c.table,x) for c in models)
            return len(counts),-max(counts.values()),x
        return max(choices,key=score)

    def status(self):
        best=self.simplest(); c=best[0] if len(best)==1 else None
        return dict(width=self.domain.width,budget=self.budget,complete_cost=self.search.complete_cost,
                    truncated=self.search.truncated,attempts=self.search.attempts,search_models=len(self.search.models),
                    surviving=len(self.models),minimum_cost_models=len(best),training=len(self.training),
                    audits=len(self.audits),blocked=self.blocked,resolved=self.resolved(),
                    expression=c.expr if c else None,cost=c.cost if c else None,
                    dependencies=c.dependencies if c else (), expanded_nodes=nodes(c.expr) if c else None)


class OperatorMind:
    def __init__(self):
        self.episodes={}; self.promoted={}; self.events=[]

    def apply(self, action, name, *args):
        if action=='new':
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',name) or name in self.episodes or len(self.episodes)>=MAX_EPISODES:
                raise ValueError('invalid or duplicate episode name')
            width,budget,reuse=args
            if type(reuse) is not bool: raise ValueError('reuse must be boolean')
            library=tuple((n,e) for n,e in self.promoted.items() if not self.episodes[n].blocked) if reuse else ()
            self.episodes[name]=OperatorEpisode(width,budget,library); result=None
        else:
            t=self.episodes[name]
            if action=='observe': result=t.observe(*args)
            elif action=='audit': result=t.audit(*args)
            elif action=='retract' and not args: t.blocked=True; result=None
            elif action=='promote' and not args:
                if name in self.promoted or len(self.promoted)>=MAX_LIBRARY or not t.resolved() or len(t.audits)<MIN_AUDITS:
                    raise ValueError('operator promotion not supported')
                c=t.simplest()[0]
                if c.cost==0 or c.expr[0] in ('nand','shl','shr') and nodes(c.expr)<=3:
                    raise ValueError('primitive or trivial expression is not a new operator')
                if c.expr in self.promoted.values(): raise ValueError('duplicate operator syntax')
                self.promoted[name]=c.expr; result=c.expr
            else: raise ValueError('unsupported operator action')
        self.events.append(json.loads(canonical([action,name,*args])))
        # Conservative: invalidate episodes whose search used a contradicted library,
        # even if their final candidate did not depend on that macro.
        changed=True
        while changed:
            changed=False
            for t in self.episodes.values():
                if not t.blocked and any(self.episodes[n].blocked for n,_ in t.library):
                    t.blocked=True; changed=True
        return result

    def advance(self,name,pool=None,query_selector=None,*,audit_target=MIN_AUDITS):
        if type(audit_target) is not int or not MIN_AUDITS<=audit_target<=64:raise ValueError('invalid audit target')
        t=self.episodes[name]
        c=t.simplest(); unused=len(t.domain.inputs)-len(t.training)-len(t.audits)
        allowed=t.domain.inputs if pool is None else tuple(dict.fromkeys(t.domain.pair(x) for x in pool))
        train_unused=sum(x not in t.training and x not in t.audits for x in allowed)
        # Do not freeze solely because the cheapest hypothesis is unique. A
        # longer explanation can still disagree on an available training input.
        # Only the evaluator supplies its actual label.
        disagreement=(not t.audits and t.resolved() and any(
            any(t.domain.value(c.table,x)!=t.domain.value(t.models[0].table,x) for c in t.models[1:])
            for x in allowed if x not in t.training))
        promotable=(t.resolved() and len(t.audits)>=audit_target and name not in self.promoted
                    and len(self.promoted)<MAX_LIBRARY and c[0].cost>0 and nodes(c[0].expr)>3
                    and c[0].expr not in self.promoted.values())
        facts=dict(blocked=t.blocked,resolved=t.resolved(),audits=len(t.audits),audit_target=audit_target,unused=unused,
                   promotable=promotable,has_models=bool(t.models),train_unused=train_unused,disagreement=disagreement)
        src=files('ix_zero').joinpath('policies','operators.ix').read_text(encoding='utf-8')
        try: r=run_ix(parse_ix(src),agent='Operators',event='advance',inputs=facts)
        except (IXError,ArithmeticError,TypeError,RecursionError) as e: raise ValueError('operator policy refused') from e
        if r.approvals_required or len(r.replies)!=1: raise ValueError('invalid operator policy result')
        action=r.replies[0]; result=None
        if action=='revise' and t.blocked: result='Contradicted operator; begin a new episode.'
        elif action=='inquire' and not t.blocked and not t.audits and t.models and (not t.resolved() or disagreement) and train_unused>0:
            result=t.query(allowed) if query_selector is None else t.domain.pair(query_selector(t,allowed))
            if result not in allowed or result in t.training or result in t.audits:raise ValueError('ineligible inquiry')
        elif action=='audit' and t.resolved() and not disagreement and len(t.audits)<audit_target and unused>0: result='Supply independently measured fresh audit label.'
        elif action=='abstract' and promotable: result=self.apply('promote',name)
        elif action=='complete' and t.resolved() and len(t.audits)>=audit_target: result='Validated bounded operator.'
        elif action=='exhausted' and not t.blocked and (not t.models or unused==0 or (not t.resolved() and train_unused==0)): result='Abstain: no supported acquisition.'
        else: raise ValueError('operator policy action rejected')
        return dict(action=action,result=result,facts=facts,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())

    def save(self,path):
        data={'schema':1,'events':self.events}; blob=canonical({'data':data,'sha256':hashlib.sha256(canonical(data)).hexdigest()})
        if len(blob)>1_000_000: raise ValueError('operator checkpoint too large')
        path=Path(path); fd,tmp=tempfile.mkstemp(prefix='.ix-operators-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f: blob=f.read(1_000_001)
        if len(blob)>1_000_000: raise ValueError('operator checkpoint too large')
        obj=json.loads(blob); data=obj['data']
        if data['schema']!=1 or hashlib.sha256(canonical(data)).hexdigest()!=obj['sha256']: raise ValueError('invalid operator checkpoint')
        mind=cls()
        for action,name,*args in data['events']: mind.apply(action,name,*args)
        if canonical(mind.events)!=canonical(data['events']): raise ValueError('operator replay mismatch')
        return mind
