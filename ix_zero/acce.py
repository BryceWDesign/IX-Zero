"""Typed, bounded capability composition and outcome-supervised dispatch.

The host registers pure capabilities and projections; the learner never executes
arbitrary code. Meaning-to-program bindings are inferred from demonstrations,
not from matching capability names. Audit outcomes come from the caller.
"""
from dataclasses import dataclass
from itertools import product
from importlib.resources import files
from collections import Counter
import hashlib, re
from ._ix_vendor import parse_ix, run_ix
from .language import scene, symbol

MIN_AUDITS = 3
MAX_PROGRAMS = 256
MAX_EVENTS = 256

@dataclass(frozen=True)
class Object:
    identity: str
    kind: str
    attribute: str
    value: int | None = None
    relation: str | None = None


def valid(kind, v):
    if kind == 'object':
        return (isinstance(v, Object) and all(isinstance(x,str) and 0<len(x)<=40 for x in (v.identity,v.kind,v.attribute))
                and (v.value is None or valid('integer',v.value)) and (v.relation is None or valid('relation',v.relation)))
    if kind == 'integer': return type(v) is int and abs(v) <= 2**31-1
    if kind == 'relation': return isinstance(v, str) and 0 < len(v) <= 40
    return False

@dataclass(frozen=True)
class Capability:
    name: str
    inputs: tuple
    output: str
    evaluate: object
    available: object
    origin: str

    def call(self, args):
        if not self.available() or len(args) != len(self.inputs) or not all(valid(t,x) for t,x in zip(self.inputs,args)):
            return None
        try: y = self.evaluate(*args)
        except (ValueError, KeyError, ArithmeticError): return None
        return y if valid(self.output,y) else None

@dataclass(frozen=True)
class Program:
    root: Capability
    projections: tuple
    @property
    def name(self):
        return self.root.name + '(' + ','.join(p.name if p else 'identity' for p in self.projections) + ')'
    @property
    def origins(self):
        return (self.root.origin, *(p.origin for p in self.projections if p))
    def available(self):
        return self.root.available() and all(p is None or p.available() for p in self.projections)
    def call(self, args):
        if not self.available() or len(args)!=len(self.projections): return None
        a = tuple(p.call((v,)) if p else v for p,v in zip(self.projections,args))
        if any(v is None for v in a): return None
        return self.root.call(a)

class Workspace:
    """A bounded typed capability catalog; adapters retain live source identity."""
    def __init__(self): self._caps = {}; self._frozen = False
    def register(self, cap):
        if self._frozen or cap.name in self._caps or len(self._caps)>=32: raise ValueError('catalog frozen, duplicate or full')
        if not isinstance(cap.name,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",cap.name):raise ValueError("invalid capability name")
        if not 1<=len(cap.inputs)<=2 or any(t not in ('object','integer','relation') for t in (*cap.inputs,cap.output)):
            raise ValueError('unsupported signature')
        self._caps[cap.name]=cap
    def programs(self, inputs, output, compose=True):
        if not 1<=len(inputs)<=2 or any(t not in ("object","integer","relation") for t in (*inputs,output)):raise ValueError("unsupported objective signature")
        self._frozen=True; candidates=[]
        for c in self._caps.values():
            if c.output!=output or len(c.inputs)!=len(inputs): continue
            choices=[]
            for src,dst in zip(inputs,c.inputs):
                options=[None] if src==dst else []
                if compose: options += [p for p in self._caps.values() if p.inputs==(src,) and p.output==dst and src!=dst]
                choices.append(options)
            for ps in product(*choices):
                pr=Program(c,ps)
                if pr.available(): candidates.append(pr)
                if len(candidates)>MAX_PROGRAMS: raise ValueError('composition budget exceeded')
        return tuple(sorted(candidates,key=lambda p:p.name))
    def catalog(self):
        return [dict(name=c.name,inputs=c.inputs,output=c.output,origin=c.origin,available=c.available()) for c in self._caps.values()]
    def primitives(self):
        yes=lambda:True
        for name,ins,out,fn in (
            ('value',('object',),'integer',lambda o:o.value),
            ('state',('object',),'relation',lambda o:o.relation),
            ('ordering',('integer','integer'),'integer',lambda a,b:(a>b)-(a<b)),
            ('difference',('integer','integer'),'integer',lambda a,b:a-b),
            ('sum',('integer','integer'),'integer',lambda a,b:a+b),
            ('minimum',('integer','integer'),'integer',min),
            ('maximum',('integer','integer'),'integer',max),
        ): self.register(Capability(name,ins,out,fn,yes,'host primitive:'+name))
        return self
    def operators(self,mind):
        for name,expr in mind.promoted.items():
            ep=mind.episodes[name]
            def live(name=name,ep=ep,expr=expr):
                return mind.episodes.get(name) is ep and mind.promoted.get(name)==expr and ep.resolved() and len(ep.audits)>=3 and all(ok for _,ok in ep.audits.values())
            self.register(Capability('op_'+name,('integer','integer'),'integer',lambda a,b,ep=ep:ep.predict((a,b)),live,'operator:'+name))
        return self
    def scalars(self,mind):
        for name,ep in mind.scalar.items():
            def live(name=name,ep=ep): return mind.scalar.get(name) is ep and not ep.blocked and sum(ep.audits.values())>=3
            self.register(Capability('scalar_'+name,('integer','integer'),'integer',lambda a,b,ep=ep:ep.predict((a,b)),live,'scalar:'+name))
        return self
    def analogies(self,mind,operators):
        for name,ep in mind.episodes.items():
            def live(name=name,ep=ep):
                mind.reconcile_operators(operators)
                return mind.episodes.get(name) is ep and not ep.blocked and name not in mind.retracted and len(ep.audits)>=3 and all(ok for _,ok in ep.audits.values())
            self.register(Capability('rel_'+name,('relation','relation'),'relation',lambda a,b,ep=ep:ep.predict((a,b)),live,'analogy:'+name))
        return self

class Binding:
    """Frozen candidate family with disjoint training and audit inputs."""
    def __init__(self, workspace, inputs, output, compose=True):
        self.inputs=tuple(inputs);self.output=output
        self.models=workspace.programs(self.inputs,output,compose)
        self.training={}; self.audits={};self.blocked=False
    def args(self,args):
        args=tuple(args)
        if len(args)!=len(self.inputs) or not all(valid(t,x) for t,x in zip(self.inputs,args)):raise ValueError('invalid objective arguments')
        return args
    def key(self,args):
        # Current object adapters depend only on the value or relation field.
        # Relabeling or recoloring the same projected inputs cannot manufacture
        # independent dispatch evidence. This key is intentionally conservative.
        return tuple(("object",x.value if self.output=="integer" else x.relation) if isinstance(x,Object) else x for x in args)
    def _live(self):
        return not self.blocked and bool(self.models) and all(p.available() for p in self.models)
    def observe(self,args,y):
        args=self.args(args);key=self.key(args)
        if not valid(self.output,y) or not self._live() or self.audits or key in self.training or len(self.training)>=MAX_EVENTS:raise ValueError('invalid, duplicate or frozen training')
        self.models=tuple(p for p in self.models if p.call(args)==y)
        self.training[key]=y
        if not self.models:self.blocked=True
    def predict(self,args,validated=True):
        args=self.args(args)
        if not self._live() or len(self.training)<2 or (validated and sum(self.audits.values())<MIN_AUDITS):return None
        vals={p.call(args) for p in self.models}
        return next(iter(vals)) if len(vals)==1 and None not in vals else None
    def audit(self,args,y):
        args=self.args(args); key=self.key(args)
        if not valid(self.output,y) or not self._live() or key in self.training or key in self.audits or len(self.audits)>=MAX_EVENTS:raise ValueError('invalid, repeated or blocked audit')
        ok=self.predict(args,False)==y;self.audits[key]=ok
        if not ok:self.blocked=True
        return ok
    def query(self,pool):
        if len(pool)>256:raise ValueError('inquiry budget')
        if not self._live():return None
        scored=[]
        for a in pool:
            a=self.args(a);key=self.key(a)
            if key in self.training or key in self.audits:continue
            counts=Counter(p.call(a) for p in self.models)
            if None in counts:continue
            scored.append(((len(counts),-max(counts.values())),a))
        return max(scored,key=lambda x:x[0])[1] if scored else None
    def advance(self,pool=()):
        query=self.query(pool); facts=dict(blocked=not self._live(),validated=sum(self.audits.values())>=MIN_AUDITS,has_query=query is not None,frozen=bool(self.audits),enough=len(self.training)>=2,resolved=len(self.models)==1)
        src=files('ix_zero').joinpath('policies','acce.ix').read_text(encoding='utf-8')
        r=run_ix(parse_ix(src),agent='Composition',event='advance',inputs=facts)
        if r.approvals_required or len(r.replies)!=1:raise ValueError('invalid composition policy')
        a=r.replies[0]
        allowed=('revise',) if facts['blocked'] else ('complete',) if facts['validated'] else ('inquire','exhausted') if not facts['enough'] else ('inquire','audit','exhausted')
        if a not in allowed or a=='inquire' and (query is None or facts['frozen']) or a=='audit' and not facts['enough']:raise ValueError('composition policy rejected')
        return dict(action=a,request=query if a=='inquire' else None,facts=facts,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())
    def status(self):
        return dict(training=len(self.training),audits=sum(self.audits.values()),blocked=not self._live(),programs=[dict(name=p.name,origins=p.origins) for p in self.models])

class CognitiveDispatch:
    def __init__(self,workspace):self.workspace=workspace;self.bindings={}
    def new(self,intent,inputs,output,compose=True):
        symbol(intent)
        key=(intent,tuple(inputs))
        if key in self.bindings or len(self.bindings)>=32:raise ValueError('duplicate or full objective catalog')
        self.bindings[key]=Binding(self.workspace,inputs,output,compose)
        return self.bindings[key]
    def execute(self,intent,args):
        kinds=tuple('object' if isinstance(a,Object) else 'integer' if type(a) is int else 'relation' if isinstance(a,str) else 'invalid' for a in args)
        b=self.bindings.get((intent,kinds))
        if b is None:return dict(abstained=True,reason='no learned objective binding')
        y=b.predict(args)
        if y is None:return dict(abstained=True,reason='unvalidated, contradicted or ambiguous objective',evidence=b.status())
        return dict(value=y,evidence=b.status())
    def interpret(self,episode,text,rows,values,relations=None):
        intent=episode.predict(text,rows)
        if intent is None:return dict(abstained=True,reason='unresolved linguistic meaning')
        if not isinstance(values,dict) or (relations is not None and not isinstance(relations,dict)):raise ValueError("expected object value and relation mappings")
        byid={i:Object(i,k,a,values.get(i), (relations or {}).get(i)) for i,k,a in scene(rows)}
        return dict(intent=intent,**self.execute(intent[0],tuple(byid[i] for i in intent[1:])))
