"""Finite relational transfer by enumerated, unlabelled graph correspondences.

A source is a complete observed binary table, or a validated transferred table.
Target graph observations are separate from target operation labels. No latent
state IDs or relation correspondences are supplied to the search.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
from .checkpoint import canonical
from ._ix_vendor import parse_ix, run_ix, IXError

MIN_AUDITS = 3
MAX_EPISODES = 32
MAX_SOURCES = 8
MAX_BYTES = 1_000_000


def name_ok(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}', name):
        raise ValueError('invalid name')


@dataclass(frozen=True)
class Graph:
    states: tuple
    relations: tuple

    @classmethod
    def read(cls, states, relations):
        if not isinstance(states, (list, tuple)) or not 3 <= len(states) <= 7:
            raise ValueError('expected 3 to 7 states')
        if any(not isinstance(x, str) or not 1 <= len(x) <= 80 for x in states) or len(set(states)) != len(states):
            raise ValueError('invalid or duplicate state')
        if not isinstance(relations, dict) or not 1 <= len(relations) <= 4:
            raise ValueError('expected 1 to 4 total unary relations')
        states = tuple(states)
        rows = []
        for k, row in sorted(relations.items()):
            name_ok(k)
            if not isinstance(row, (list, tuple)) or len(row) != len(states) or any(x not in states for x in row):
                raise ValueError('invalid relation observations')
            rows.append(tuple(states.index(x) for x in row))
        return cls(states, tuple(sorted(rows)))

    def label(self, x):
        if not isinstance(x, str) or x not in self.states:
            raise ValueError('unknown state')
        return self.states.index(x)

    def pair(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError('expected two states')
        return tuple(self.label(v) for v in x)

    @property
    def n(self): return len(self.states)

    @property
    def inputs(self): return tuple(product(range(self.n), repeat=2))


@dataclass(frozen=True)
class Source:
    graph: Graph
    table: tuple
    dependencies: tuple = ()


def correspondences(source, target, structural=True):
    """Enumerate all source->target bijections, ignoring names of unary relations."""
    if source.n != target.n or (structural and len(source.relations) != len(target.relations)):
        return
    n = source.n
    for p in permutations(range(n)):
        inv = tuple(p.index(i) for i in range(n))
        if structural:
            transported = tuple(sorted(tuple(p[r[inv[i]]] for i in range(n)) for r in source.relations))
            if transported != target.relations: continue
        yield p, inv


def transport(table, p, inv):
    n = len(p)
    return tuple(p[table[inv[a]*n+inv[b]]] for a,b in product(range(n), repeat=2))


class Episode:
    def __init__(self, graph, sources, mode):
        if mode not in ('structural', 'unstructured', 'local'):
            raise ValueError('unknown transfer mode')
        self.graph, self.mode = graph, mode
        self.training, self.audits = {}, {}
        self.blocked = False
        self.models = {}
        self.mappings = 0
        n = graph.n
        if mode == 'local':
            # Target-only alternatives: projections, constants, and each observed
            # unary relation applied to either input. No source data is used.
            for side in (0,1):
                for r in (tuple(range(n)), *graph.relations):
                    self.models[tuple(r[x[side]] for x in graph.inputs)] = set()
            for k in range(n): self.models[(k,)*(n*n)] = set()
        else:
            for name, s in sources.items():
                for p, inv in correspondences(s.graph, graph, mode == 'structural'):
                    self.mappings += 1
                    t = transport(s.table,p,inv)
                    self.models.setdefault(t,set()).add(name)
        self.dependencies = tuple(sorted({name for src in self.models.values() for name in src}))
        self.initial_models = len(self.models)

    def _idx(self,x): return x[0]*self.graph.n+x[1]
    def _prediction(self,x):
        if self.blocked: return None
        values = {t[self._idx(x)] for t in self.models}
        return next(iter(values)) if len(values)==1 else None

    def observe(self,x,y):
        x,y=self.graph.pair(x),self.graph.label(y)
        if self.blocked: raise ValueError('episode quarantined')
        if x in self.training:
            if self.training[x]!=y: self.blocked=True
            return not self.blocked
        if x in self.audits:
            if self.audits[x][0]!=y: self.blocked=True
            return not self.blocked
        if self.audits: raise ValueError('training frozen after first audit')
        self.training[x]=y
        self.models={t:s for t,s in self.models.items() if t[self._idx(x)]==y}
        return bool(self.models)

    def audit(self,x,y):
        x,y=self.graph.pair(x),self.graph.label(y)
        if self.blocked or x in self.training or x in self.audits or not self.training:
            raise ValueError('audit requires fresh input and training')
        pred=self._prediction(x)
        if pred is None: raise ValueError('no agreed prediction to audit')
        passed=pred==y
        self.audits[x]=(y,passed)
        if not passed: self.blocked=True
        return passed

    def predict(self,x):
        x=self.graph.pair(x)
        if self.blocked or len(self.audits)<MIN_AUDITS: return None
        y=self._prediction(x)
        return self.graph.states[y] if y is not None else None

    def query(self,pool=None):
        if self.blocked or self.audits or not self.models: return None
        allowed=self.graph.inputs if pool is None else tuple(dict.fromkeys(self.graph.pair(x) for x in pool))
        choices=[]
        for x in allowed:
            if x in self.training: continue
            c=Counter(t[self._idx(x)] for t in self.models)
            choices.append((len(c),-max(c.values()),x))
        if not choices: return None
        x=max(choices)[2]
        return tuple(self.graph.states[i] for i in x)

    def status(self):
        return dict(mode=self.mode, mappings=self.mappings, initial_models=self.initial_models,
                    models=len(self.models), training=len(self.training), audits=len(self.audits),
                    blocked=self.blocked, sources=sorted({n for ss in self.models.values() for n in ss}))


class AnalogicalMind:
    def __init__(self):
        self.sources={}; self.episodes={}; self.retracted=set(); self.events=[]; self.operator_links={}

    def apply(self,action,name,*args):
        name_ok(name)
        if action=='operator_source':
            # Replay the exact acquisition history instead of trusting supplied syntax.
            from .operators import OperatorMind
            if name in self.sources or name in self.episodes or len(self.sources)>=MAX_SOURCES:
                raise ValueError('duplicate name or source budget')
            states,relations,events,episode=args
            g=Graph.read(states,relations)
            if g.n!=4 or not isinstance(events,list) or len(canonical(events))>MAX_BYTES//2:
                raise ValueError('operator source requires four ordered word labels and bounded evidence')
            om=OperatorMind()
            for a,n,*rest in events:om.apply(a,n,*rest)
            t=om.episodes[episode]
            if t.domain.width!=2 or t.blocked or episode not in om.promoted:
                raise ValueError('operator source must be a live promoted two-bit operator')
            table=tuple(t.predict(x) for x in g.inputs)
            if any(y is None for y in table):raise ValueError('operator source is not audit-gated')
            self.sources[name]=Source(g,table)
            self.operator_links[name]=(episode,json.loads(canonical(events)))
            result=None
        elif action in ('source','new'):
            if name in self.sources or name in self.episodes: raise ValueError('duplicate name')
            if action=='new' and len(args)==4:
                states,relations,extra,selected=args
                if not isinstance(selected,(list,tuple)) or len(set(selected))!=len(selected) or any(k not in self.sources or k in self.retracted for k in selected):
                    raise ValueError('invalid source selection')
            else:
                states,relations,extra=args;selected=None
            g=Graph.read(states,relations)
            if action=='source':
                if len(self.sources)>=MAX_SOURCES or not isinstance(extra,(list,tuple)) or len(extra)!=g.n*g.n:
                    raise ValueError('complete source table required within source budget')
                observations={}
                for x,y in extra:
                    p=g.pair(x)
                    if p in observations: raise ValueError('duplicate source observation')
                    observations[p]=g.label(y)
                self.sources[name]=Source(g,tuple(observations[x] for x in g.inputs))
            else:
                if len(self.episodes)>=MAX_EPISODES: raise ValueError('episode budget')
                live={k:v for k,v in self.sources.items() if k not in self.retracted and (selected is None or k in selected)}
                self.episodes[name]=Episode(g,live,extra)
            result=None
        elif action=='retract' and not args:
            if name not in self.sources and name not in self.episodes: raise ValueError('unknown source or episode')
            self.retracted.add(name)
            if name in self.episodes: self.episodes[name].blocked=True
            result=None
        elif action=='promote' and not args:
            t=self.episodes[name]
            if name in self.sources or len(self.sources)>=MAX_SOURCES or t.blocked or len(t.audits)<MIN_AUDITS or len(t.models)!=1 or t.mode=='local':
                raise ValueError('promotion requires validated unique transferred table')
            self.sources[name]=Source(t.graph,next(iter(t.models)),t.dependencies)
            result=True
        elif action in ('observe','audit'):
            t=self.episodes[name]; result=getattr(t,action)(*args)
            if t.blocked: self.retracted.add(name)
        else: raise ValueError('unsupported analogy action')
        self.events.append(json.loads(canonical([action,name,*args])))
        # Conservative dependency invalidation, including already promoted targets.
        while True:
            affected={k for k,t in self.episodes.items() if set(t.dependencies)&self.retracted}
            affected|={k for k,s in self.sources.items() if set(s.dependencies)&self.retracted}
            if affected<=self.retracted: break
            self.retracted|=affected
        for k in self.retracted & self.episodes.keys(): self.episodes[k].blocked=True
        return result

    def reconcile_operators(self,operator_mind):
        """Invalidate imported priors if their originating journal is absent or contradicted."""
        for name,(episode,events) in self.operator_links.items():
            if name in self.retracted:continue
            current=operator_mind.episodes.get(episode)
            if (current is None or current.blocked or episode not in operator_mind.promoted or
                    canonical(operator_mind.events[:len(events)])!=canonical(events)):
                self.apply('retract',name)

    def advance(self,name):
        t=self.episodes[name]
        unused=[x for x in t.graph.inputs if x not in t.training and x not in t.audits]
        agreed=sum(t._prediction(x) is not None for x in unused)
        f=dict(blocked=t.blocked, has_models=bool(t.models), unique=len(t.models)==1,
               training=len(t.training), audits=len(t.audits), available=len(unused), agreed=agreed,
               promotable=name not in self.sources and len(self.sources)<MAX_SOURCES and t.mode!='local')
        src=files('ix_zero').joinpath('policies','analogy.ix').read_text(encoding='utf-8')
        try:r=run_ix(parse_ix(src),agent='Analogy',event='advance',inputs=f)
        except (IXError,ArithmeticError,TypeError,RecursionError) as e:raise ValueError('analogy policy refused') from e
        if r.approvals_required or len(r.replies)!=1:raise ValueError('invalid analogy policy result')
        a=r.replies[0]
        if a=='revise' and t.blocked: result='Quarantined; new evidence epoch required.'
        elif a=='inquire' and not t.blocked and not t.audits and t.models and unused: result=t.query()
        elif a=='audit' and not t.blocked and t.training and agreed and len(t.audits)<MIN_AUDITS: result='Provide fresh measured target outcome.'
        elif a=='abstract' and not t.blocked and f['unique'] and f['promotable'] and len(t.audits)>=MIN_AUDITS: result=self.apply('promote',name)
        elif a=='complete' and not t.blocked and len(t.audits)>=MIN_AUDITS:result='Validated bounded transfer.'
        elif a=='exhausted' and not t.blocked: result='Abstain: insufficient structural or validation evidence.'
        else: raise ValueError('analogy policy action rejected')
        return dict(action=a,result=result,facts=f,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())

    def save(self,path):
        data={'schema':1,'events':self.events}; blob=canonical({'data':data,'sha256':hashlib.sha256(canonical(data)).hexdigest()})
        if len(blob)>MAX_BYTES:raise ValueError('analogy checkpoint too large')
        path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.ix-analogy-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES:raise ValueError('analogy checkpoint too large')
        obj=json.loads(blob);data=obj['data']
        if data['schema']!=1 or hashlib.sha256(canonical(data)).hexdigest()!=obj['sha256']:raise ValueError('invalid analogy checkpoint')
        m=cls()
        for a,n,*args in data['events']:m.apply(a,n,*args)
        if canonical(m.events)!=canonical(data['events']):raise ValueError('analogy replay mismatch')
        return m
