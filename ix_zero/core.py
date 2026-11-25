"""Finite-state experimental cognition: executable hypotheses, active tests and reuse.

No generated Python, network, shell, or physical actions. All operations are total
functions over small integer vectors. Search is complete only within its grammar.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
import re

State = tuple[int, ...]
Program = tuple[str, ...]
PRIMITIVES = ('inc','dec','rot','rev','mix','unmix','mul','zero','flip','gate')
MAX_LIBRARY = 8
MAX_DEPTH = 3
MAX_EXPANDED = 12
MAX_TASKS = 32

@dataclass(frozen=True)
class Space:
    width: int = 2
    modulus: int = 7
    def __post_init__(self):
        if type(self.width) is not int or type(self.modulus) is not int or not (2 <= self.width <= 3 and 2 <= self.modulus <= 7):
            raise ValueError('width must be 2 or 3; modulus must be 2 through 7')
    def check(self, x):
        if not isinstance(x, (tuple,list)) or len(x) != self.width or any(type(v) is not int or not 0 <= v < self.modulus for v in x):
            raise ValueError('invalid state for this space')
        return tuple(x)
    @property
    def states(self):
        return states(self.width,self.modulus)

@lru_cache(maxsize=12)
def states(width, modulus):
    return tuple(product(range(modulus), repeat=width))

def execute(p: Program, x: State, s: Space) -> State:
    x = s.check(x)
    if len(p) > MAX_EXPANDED:
        raise ValueError('program exceeds execution budget')
    m = s.modulus
    for op in p:
        a,b,*rest = x
        if op == 'inc': x = ((a+1)%m,b,*rest)
        elif op == 'dec': x = ((a-1)%m,b,*rest)
        elif op == 'rot': x = x[1:]+x[:1]
        elif op == 'rev': x = x[::-1]
        elif op == 'mix': x = ((a+b)%m,b,*rest)
        elif op == 'unmix': x = ((a-b)%m,b,*rest)
        elif op == 'mul': x = ((a*b)%m,b,*rest)
        elif op == 'zero': x = (0,b,*rest)
        elif op == 'flip': x = (m-1-a,b,*rest)
        elif op == 'gate': x = ((a+1)%m if b == 0 else a,b,*rest)
        else: raise ValueError('unknown primitive')
    return x

@lru_cache(maxsize=24)
def grammar(library: tuple[Program,...], depth: int) -> tuple[Program,...]:
    if type(depth) is not int or not 1 <= depth <= MAX_DEPTH or len(library) > MAX_LIBRARY:
        raise ValueError('grammar budget exceeded')
    if any(not 2 <= len(p) <= MAX_EXPANDED or any(op not in PRIMITIVES for op in p) for p in library):
        raise ValueError('invalid library')
    tokens = tuple((op,) for op in PRIMITIVES)+library
    # A program may have multiple tokenizations. Retain first (minimum token cost).
    found = {(): None}
    for n in range(1,depth+1):
        for parts in product(tokens,repeat=n):
            p = sum(parts,())
            if len(p) <= MAX_EXPANDED: found.setdefault(p,None)
    return tuple(found)

@dataclass
class Task:
    space: Space
    library: tuple[Program,...] = ()
    depth: int = MAX_DEPTH
    observations: dict[State,State] = field(default_factory=dict,init=False)
    audits: dict[State,tuple[State|None,State]] = field(default_factory=dict,init=False)
    contradictions: list[tuple[State,State,State]] = field(default_factory=list,init=False)
    candidates: list[Program] = field(init=False,repr=False)
    def __post_init__(self):
        self.candidates = list(grammar(self.library,self.depth))
    @property
    def blocked(self):
        return bool(self.contradictions or any(p is None or p != y for p,y in self.audits.values()) or not self.candidates)
    def observe(self, x, y):
        x,y = self.space.check(x),self.space.check(y)
        if x in self.audits: raise ValueError('audit inputs cannot become training inputs')
        if self.contradictions and x not in self.observations: raise ValueError('contradiction: start a new task')
        if x in self.observations:
            if self.observations[x] != y:
                if len(self.contradictions)>=self.space.modulus**self.space.width: raise ValueError('contradiction budget exhausted')
                self.contradictions.append((x,self.observations[x],y))
            return
        if self.audits: raise ValueError('audit started: freeze training; use a new task to revise')
        self.observations[x] = y
        self.candidates = [p for p in self.candidates if execute(p,x,self.space)==y]
    def predict(self,x):
        x = self.space.check(x)
        if self.blocked or not self.observations: return None
        it=iter(self.candidates)
        first=execute(next(it),x,self.space)
        return first if all(execute(p,x,self.space)==first for p in it) else None
    def validated_predict(self,x):
        x=self.space.check(x)
        return self.predict(x) if len(self.audits)>=3 and not self.blocked else None
    def query(self, allowed=None):
        """Minimize expected surviving fraction under uniform program prior.

        Uses no teacher outcomes. Syntactic multiplicity biases this heuristic;
        it is not a calibrated probability estimate.
        """
        if self.blocked or self.audits: return None
        pool=self.space.states if allowed is None else tuple(self.space.check(x) for x in allowed)
        best=None; score=-1
        for x in sorted(set(pool)-self.observations.keys()):
            counts=Counter(execute(p,x,self.space) for p in self.candidates)
            # constant N^2 - sum bin sizes squared
            value=len(self.candidates)**2-sum(n*n for n in counts.values())
            if value>score: best,score=x,value
        return best if score>0 or (best is not None and not self.observations) else None
    def audit(self,x,y):
        x,y=self.space.check(x),self.space.check(y)
        if self.blocked: raise ValueError('blocked task: start a new task')
        if x in self.observations or x in self.audits: raise ValueError('audit requires unused input')
        # Freeze the hypothesis set. Outcomes never select or update programs.
        prediction=self.predict(x)
        self.audits[x]=(prediction,y)
        return prediction == y
    def resolved(self):
        if self.blocked or not self.observations: return False
        return all(self.predict(x) is not None for x in self.space.states)
    def concept(self):
        if len(self.audits)<3 or not self.resolved(): return None
        # Choose short representative of equivalence class within source space.
        return min(self.candidates,key=lambda p:(len(p),p))
    def summary(self):
        return {'width':self.space.width,'modulus':self.space.modulus,'observations':len(self.observations),'audits':len(self.audits),'candidates':len(self.candidates),'blocked':self.blocked,'concept':self.concept()}

@dataclass
class Mind:
    tasks: dict[str,Task] = field(default_factory=dict,init=False)
    # provenance is task name -> learned program; invalidated sources are ignored.
    concepts: dict[str,Program] = field(default_factory=dict,init=False)
    def library(self):
        return tuple(dict.fromkeys(p for name,p in self.concepts.items() if not self.tasks[name].blocked))[:MAX_LIBRARY]
    def new(self,name,space,*,reuse=True,depth=MAX_DEPTH):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',name) or name in self.tasks or len(self.tasks)>=MAX_TASKS:
            raise ValueError('invalid, duplicate, or excessive task name')
        t=Task(space,self.library() if reuse else (),depth)
        self.tasks[name]=t
        return t
    def promote(self,name):
        t=self.tasks[name]; p=t.concept()
        if p is None or len(p)<2: raise ValueError('requires resolved multi-step concept and three passing unseen audits')
        if name not in self.concepts and len(self.concepts)>=MAX_LIBRARY: raise ValueError('library full')
        self.concepts[name]=p
        return p
    def plan(self,space,start,goal,actions):
        """Shortest consensus-supported sequence; proposal only, never dispatch."""
        start,goal=space.check(start),space.check(goal)
        actions=tuple(dict.fromkeys(actions))
        if any(self.tasks[n].space!=space or self.tasks[n].concept() is None for n in actions):
            raise ValueError('all action models must have resolved, audited target-space evidence')
        queue=deque([(start,())]); seen={start}
        while queue:
            x,path=queue.popleft()
            if x==goal: return path
            for n in actions:
                y=self.tasks[n].predict(x)
                if y is not None and y not in seen:
                    seen.add(y); queue.append((y,path+((n,y),)))
        return None
