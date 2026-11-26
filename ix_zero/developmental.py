"""Replayable, evidence-driven curriculum over existing bounded cognitive engines.

The host supplies unlabeled task descriptions and disjoint evidence pools. No
teacher, held-out labels, executable source or external tools enter the runtime.
"""
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
from .checkpoint import canonical
from ._ix_vendor import parse_ix, run_ix
from .operators import OperatorMind, Domain as WordDomain
from .strategy import Strategy, advance as strategy_advance
from .representation import Episode as Representation, Abstraction, domain_labels
from .contextual import Episode as Contextual, Domain as ContextDomain
from .language import LanguageEpisode, words, scene
from .compositional import Grammar, tokenize
from .acce import Workspace, Binding, Object, CognitiveDispatch, Capability, valid
from .analogical import AnalogicalMind, Graph
from .epistemic import Inquiry, widths_of

MAX_TASKS = 32
MAX_EVENTS = 4096
MAX_BYTES = 4_000_000
KINDS = {'operator','representation','context','lexicon','grammar','binding','analogy','epistemic'}

def normal(x):
    return json.loads(canonical(x))

def ident(s):
    if not isinstance(s,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9-]{0,39}',s):
        raise ValueError('invalid curriculum identifier')
    return s

def key(x): return canonical(x)

def args_of(spec,x):
    if not isinstance(x,(list,tuple)) or len(x)!=len(spec['inputs']): raise ValueError('invalid binding arguments')
    a=tuple(Object(**v) if t=='object' and isinstance(v,dict) else v for t,v in zip(spec['inputs'],x))
    if not all(valid(t,v) for t,v in zip(spec['inputs'],a)): raise ValueError('invalid typed argument')
    return a

@dataclass
class Job:
    name: str
    spec: dict
    status: str = 'queued'
    episode: object = None
    history: list = field(default_factory=list)
    asked: set = field(default_factory=set)
    audit_count: int = 0
    expanded: bool = False
    broad_check: bool = False
    template_used: bool = False
    attempts: int = 0
    reason: str = ''
    inquiry: dict = field(default_factory=dict)
    @property
    def kind(self): return self.spec['kind']
    @property
    def deps(self): return self.spec['deps']

class Development:
    def __init__(self, *, reuse=True):
        if type(reuse) is not bool: raise ValueError('invalid reuse mode')
        self.reuse=reuse; self.words=OperatorMind(); self.analogies=AnalogicalMind(); self.jobs={}; self.events=[]; self.pending=None
        # Exact-equivalent inquiry strategy from the earlier strategy evaluation.
        self.strategy=Strategy(kernel='packed')

    def _record(self,event):
        self.events.append(normal(event))

    def _room(self):
        if len(self.events)>=MAX_EVENTS: raise ValueError('curriculum journal full')

    def _spec(self,s):
        s=normal(s)
        if not isinstance(s,dict) or not {'kind','deps','train','audit'} <= set(s): raise ValueError('invalid task schema')
        k=s['kind']; deps=s['deps']; train=s['train']; audit=s['audit']
        if k not in KINDS or not isinstance(deps,list) or len(deps)>8 or len(set(deps))!=len(deps): raise ValueError('invalid dependencies or kind')
        for n in deps: ident(n)
        if not isinstance(train,list) or not isinstance(audit,list) or not 1<=len(train)<=256 or not 3<=len(audit)<=64: raise ValueError('invalid evidence pool size')
        fields={'epistemic':{'width','widths','budget'},'operator':{'width','budget'},'representation':{'labels'},'context':{'labels','contexts'},'lexicon':set(),'grammar':set(),'binding':{'intent','inputs','output','contexts'},'analogy':{'states','relations','source_states','source_relations'}}[k]
        if set(s)!={'kind','deps','train','audit'}|fields: raise ValueError('unexpected task fields')
        if k=='epistemic':
            ws=widths_of(s['widths'])
            if type(s['width']) is not int or s['width'] not in ws or type(s['budget']) is not int or not 1<=s['budget']<=5:raise ValueError('invalid epistemic scope or budget')
            Inquiry.validate_pool(s['width'],train,audit)
            check=WordDomain(s['width']).pair
        elif k=='operator':
            d=WordDomain(s['width']); check=d.pair
            if type(s['budget']) is not int or not 1<=s['budget']<=5: raise ValueError('invalid search budget')
        elif k=='representation':
            lab=domain_labels(s['labels'])
            def check(x):
                if not isinstance(x,list) or len(x)!=2 or any(type(v) is not int or v not in lab for v in x): raise ValueError('invalid representation input')
                return x
        elif k=='context': check=ContextDomain(s['labels'],s['contexts']).key
        elif k=='analogy':
            g=Graph.read(s['states'],s['relations']);Graph.read(s['source_states'],s['source_relations']);check=g.pair
        elif k in ('lexicon','grammar'):
            def check(x):
                if not isinstance(x,list) or len(x)!=2: raise ValueError('invalid language input')
                t=(words if k=='lexicon' else tokenize)(x[0]); rows=scene(x[1]); return (t,rows)
        else:
            ident(s['intent'])
            if not isinstance(s['inputs'],list) or not 1<=len(s['inputs'])<=2 or any(t not in ('integer','object','relation') for t in [*s['inputs'],s['output']]): raise ValueError('invalid binding signature')
            if not isinstance(s['contexts'],dict) or not set(s['contexts'])<=set(deps): raise ValueError('invalid context selection')
            def check(x):
                a=args_of(s,x)
                # Mirror the binding evidence projection, not object identifiers.
                return tuple(('object',v.value if s['output']=='integer' else v.relation) if isinstance(v,Object) else v for v in a)
        keys=[key(check(x)) for x in train+audit]
        if len(set(keys))!=len(keys): raise ValueError('duplicate or overlapping evidence inputs')
        return s

    def add(self,name,spec):
        self._room(); ident(name); s=self._spec(spec)
        if name in self.jobs or name in s['deps'] or len(self.jobs)>=MAX_TASKS: raise ValueError('duplicate, cyclic or excessive objective')
        # Forward references are permitted; reject cycles among known tasks.
        graph={n:j.deps for n,j in self.jobs.items()}; graph[name]=s['deps']
        def visit(n,stack):
            if n in stack: raise ValueError('cyclic curriculum')
            for d in graph.get(n,[]): visit(d,stack|{n})
        visit(name,set())
        self.jobs[name]=Job(name,s); self._record(['add',name,s]); return name

    def offer(self,name,labels,train,audit):
        """Infer the acquisition prerequisites of a fresh finite representation.

        The problem and unlabeled evidence pools are supplied by the evaluator;
        the operator and abstraction dependencies are not. If none are live,
        report a prerequisite rather than inventing evidence or an operator.
        """
        self._room();ident(name)
        if self.pending is not None or name in self.jobs:raise ValueError('invalid offered objective')
        labels=domain_labels(labels);width=len(labels).bit_length()-1
        s=self._spec(dict(kind='representation',deps=[],labels=labels,train=train,audit=audit))
        sources=sorted(n for n,j in self.jobs.items() if j.kind=='operator' and self.live(n) and j.episode.domain.width==width)
        parents=[]
        if sources and self.reuse:
            from itertools import combinations
            eligible=[n for n,j in sorted(self.jobs.items()) if j.kind=='representation' and self.live(n) and j.episode.n==len(labels) and j.episode.labels!=labels]
            options=[]
            for a,b in combinations(eligible,2):
                try:t=Abstraction.induce(self.jobs[a].episode,self.jobs[b].episode)
                except ValueError:continue
                if t.source in sources:options.append((sum(getattr(t,k) is not None for k in ('family','multiplier','shift')),a,b))
            if options:
                _,a,b=min(options,key=lambda p:(-p[0],p[1],p[2]));parents=[a,b]
        if not sources:out={'action':'request_prerequisite','kind':'operator','width':width,'objective':name}
        else:
            s['deps']=sources+parents
            self.add(name,s);self.events.pop()
            out={'action':'registered','objective':name,'dependencies':s['deps'],'abstraction_parents':parents}
        self._record(['offer',name,labels,train,audit,out]);return out

    def live(self,name):
        j=self.jobs.get(name)
        if j is None or j.status!='complete' or not all(self.live(n) for n in j.deps):return False
        e=j.episode
        if not all(key(x) in j.asked for x in j.spec['audit']) and j.kind!='epistemic':return False
        if j.kind=='epistemic':return e.finished and e.validated()
        if j.kind=='operator':return name in self.words.promoted and e.resolved() and len(e.audits)>=3 and all(ok for _,ok in e.audits.values())
        if j.kind in ('representation','context','recovery'):return e.live() and len(e.audits)>=3
        if j.kind=='analogy':
            self.analogies.reconcile_operators(self.words)
            return not e.blocked and name in self.analogies.sources and name not in self.analogies.retracted and len(e.audits)>=3 and all(ok for _,ok in e.audits.values())
        if j.kind=='lexicon':return not e.blocked and all(e.count(n)>=3 for n in (3,5))
        if j.kind=='grammar':return e.live() and bool(e.rules) and all(len(v['train'])>=2 and len(v['audit'])>=3 for v in e.rules.values())
        return e._live() and len(e.audits)>=3 and all(e.audits.values())

    def _start(self,j):
        s=j.spec;k=j.kind
        if k=='epistemic':
            library=[]
            for n in j.deps:
                p=self.jobs[n]
                if p.kind not in ('operator','epistemic'):raise ValueError('epistemic dependencies must be learned operators')
                if self.reuse:library.append((n,p.episode.models[0].expr))
            j.episode=Inquiry(s['widths'],s['budget'],s['width'],s['train'],s['audit'],library);j.attempts=j.episode.search.attempts
        elif k=='operator':
            self.words.apply('new',j.name,s['width'],s['budget'],self.reuse and bool(j.deps)); j.episode=self.words.episodes[j.name]
        elif k=='representation':
            reps=[self.jobs[n].episode for n in j.deps if self.jobs[n].kind=='representation']
            template=None
            if self.reuse and len(reps)>=2:
                try:template=Abstraction.induce(*reps[:2])
                except ValueError:pass
            j.episode=Representation(self.words,s['labels'],template);j.template_used=template is not None;j.attempts=j.episode.attempts
        elif k=='context':
            caps=[self.jobs[n].episode.capability(n) for n in j.deps]
            if not caps or any(self.jobs[n].kind!='representation' for n in j.deps):raise ValueError('context requires representation sources')
            j.episode=Contextual(ContextDomain(s['labels'],s['contexts']),caps[0],caps[1:])
        elif k=='analogy':
            if len(j.deps)!=1 or self.jobs[j.deps[0]].kind!='operator':raise ValueError('analogy requires a two-bit operator source')
            source=j.name+'-src'
            self.analogies.apply('operator_source',source,s['source_states'],s['source_relations'],self.words.events,j.deps[0])
            self.analogies.apply('new',j.name,s['states'],s['relations'],'structural',[source]);j.episode=self.analogies.episodes[j.name]
        elif k=='lexicon':j.episode=LanguageEpisode()
        elif k=='grammar':
            if len(j.deps)!=1 or self.jobs[j.deps[0]].kind!='lexicon':raise ValueError('grammar requires one lexicon')
            j.episode=Grammar(self.jobs[j.deps[0]].episode)
        elif k=='binding':
            ws=Workspace().primitives()
            for n in j.deps:
                p=self.jobs[n]
                if p.kind=='representation':ws.register(p.episode.capability(n))
                elif p.kind in ('context','recovery'):ws.register(p.episode.capability(n,s['contexts'][n]))
                elif p.kind=='operator':
                    ep=p.episode
                    ws.register(Capability(n,('integer','integer'),'integer',lambda a,b,ep=ep:ep.predict((a,b)),lambda n=n:self.live(n),'operator:'+n))
                elif p.kind=='analogy':
                    ep=p.episode
                    ws.register(Capability(n,('relation','relation'),'relation',lambda a,b,ep=ep:ep.predict((a,b)),lambda n=n:self.live(n),'analogy:'+n))
                else:raise ValueError('unsupported binding dependency')
            j.episode=Binding(ws,s['inputs'],s['output'])
        else:raise ValueError('unsupported objective kind')
        j.status='learning'

    def _fresh(self,j,phase):return [x for x in j.spec[phase] if key(x) not in j.asked]
    def _expand(self,j):
        if j.kind!='representation' or not j.template_used or j.expanded or j.audit_count:return False
        e=Representation(self.words,j.spec['labels']);j.expanded=True;j.attempts+=e.attempts
        for x,y in j.history:
            if not e.live():break
            e.observe(x,y)
        j.episode=e;return True

    def _finish(self,j,ok,reason):j.status='complete' if ok else 'failed';j.reason=reason

    def _request(self,j):
        """Return the next evidence request or mark the current task terminal."""
        e=j.episode;k=j.kind;train=self._fresh(j,'train');audits=self._fresh(j,'audit')
        if k=='epistemic':
            d=e.choose();a=d['action'];j.inquiry=d
            if a in ('complete','failed'):
                e.finished=a=='complete';e.blocked=a=='failed';e.reason=d['reason'];self._finish(j,e.finished,e.reason);return
            return a,normal(d['input'])
        if k=='operator':
            # Reserve the entire host-supplied audit pool, not merely the minimum.
            d=strategy_advance(self.words,j.name,train,self.strategy,audit_target=len(j.spec['audit']));a=d['action']
            if a in ('abstract','complete'):
                self._finish(j,j.name in self.words.promoted,'operator promotion' if j.name in self.words.promoted else 'not a reusable operator');return
            if a=='inquire':return 'train',normal(d['result'])
            if a=='audit' and audits:return 'audit',audits[0]
        elif k=='representation':
            if not e.live() or not train and not e.resolved():
                if self._expand(j):return self._request(j)
            if not e.live():self._finish(j,False,'representation exhausted');return
            if len(e.audits)>=3 and audits:return 'audit',audits[0]
            if len(e.audits)>=3:self._finish(j,e.resolved(),'validated representation' if e.resolved() else 'ambiguous representation');return
            if not e.audits and (not e.resolved() or len(e.training)<6) and train:
                x=e.query(train)
                if x is not None:return 'train',normal(x)
            if not e.resolved() and self._expand(j):return self._request(j)
            if e.resolved() and len(e.training)>=6 and audits:
                # A narrow abstraction can agree on observed inputs while a
                # broader hypothesis family predicts differently. Reopen the
                # full family before *any* audit labels are seen.
                if j.template_used and not j.expanded:
                    j.broad_check=True
                    if self._expand(j):return self._request(j)
                return 'audit',audits[0]
        elif k in ('context','recovery'):
            if not e.live():self._finish(j,False,e.diagnose());return
            if len(e.audits)>=3:
                if audits:return 'audit',audits[0]
                self._finish(j,True,'validated contextual model');return
            d=e.advance(train)
            if d['action']=='inquire':return 'train',normal(d['inquiry']['input'])
            if d['action']=='audit' and audits:
                # Audit every reserved item for recovery; do not select by outcome.
                return 'audit',audits[0]
        elif k=='analogy':
            if e.blocked or not e.models:self._finish(j,False,'structural hypotheses exhausted');return
            if len(e.audits)>=3:
                if audits:
                    if e._prediction(e.graph.pair(audits[0])) is not None:return 'audit',audits[0]
                    self._finish(j,False,'unresolved reserved audit');return
                if len(e.models)==1:
                    self.analogies.apply('promote',j.name);self._finish(j,True,'validated structural transfer')
                else:self._finish(j,False,'ambiguous structural transfer')
                return
            if train and not e.audits and (len(e.training)<1 or len(e.models)>1):
                x=e.query(train)
                if x is not None:return 'train',normal(x)
            if e.training and audits and e._prediction(e.graph.pair(audits[0])) is not None:return 'audit',audits[0]
        elif k=='lexicon':
            if e.blocked or not e.models:self._finish(j,False,'lexicon contradicted');return
            if all(e.count(n)>=3 for n in (3,5)):
                if audits:
                    if e.predict(*audits[0],validated=False) is not None:return 'audit',audits[0]
                    self._finish(j,False,'unresolved reserved audit');return
                self._finish(j,True,'validated lexical model');return
            if train and not e.audits:
                q=e.query(train)
                if q:
                    # query returns normalized scene; map back to the supplied row.
                    x=next(x for x in train if (words(x[0]),scene(x[1]))==q)
                    return 'train',x
            for x in audits:
                if e.count(len(words(x[0])))<3 and e.predict(*x,validated=False) is not None:return 'audit',x
        elif k=='grammar':
            if not e.live():self._finish(j,False,'grammar contradicted');return
            if train:return 'train',train[0]
            if audits:return 'audit',audits[0]
            self._finish(j,bool(e.rules) and all(len(v['train'])>=2 and len(v['audit'])>=3 for v in e.rules.values()),'grammar validation');return
        elif k=='binding':
            if not e._live():self._finish(j,False,'binding invalidated');return
            if len(e.audits)>=3:
                if audits:return 'audit',audits[0]
                self._finish(j,True,'validated dispatch');return
            if not e.audits and train and (len(e.training)<2 or len(e.models)>1):
                x=e.query([args_of(j.spec,x) for x in train])
                if x is not None:return 'train',next(v for v in train if args_of(j.spec,v)==x)
            if len(e.training)>=2 and audits:return 'audit',audits[0]
        self._finish(j,False,'insufficient supported evidence')

    def next(self):
        self._room()
        if self.pending is not None:raise ValueError('answer pending')
        # One evidence request per turn. Bounded internal transitions cannot loop.
        for _ in range(MAX_TASKS+1):
            for j in self.jobs.values():
                if j.status=='learning' and not all(self.live(n) for n in j.deps):self._finish(j,False,'dependency invalidated during learning')
            ready=[j for j in self.jobs.values() if j.status=='learning' or j.status=='queued' and all(self.live(n) for n in j.deps)]
            unfinished=any(j.status in ('queued','learning') for j in self.jobs.values())
            r=run_ix(parse_ix(files('ix_zero').joinpath('policies','developmental.ix').read_text(encoding='utf-8')),agent='Development',event='advance',inputs={'ready':bool(ready),'unfinished':unfinished})
            expected='learn' if ready else 'wait' if unfinished else 'complete'
            if r.approvals_required or r.replies!=[expected]:raise ValueError('development policy rejected')
            if not ready:
                out={'action':expected,'waiting':[j.name for j in self.jobs.values() if j.status=='queued'],'trace':r.trace_as_dicts()};break
            j=min(ready,key=lambda j:(j.status!='learning',-sum(j.name in p.deps for p in self.jobs.values() if p.status=='queued'),j.name))
            if j.status=='queued':
                try:self._start(j)
                except (ValueError,KeyError) as exc:self._finish(j,False,str(exc));continue
            req=self._request(j)
            if req is None:continue
            phase,x=req;self.pending={'job':j.name,'phase':phase,'input':normal(x)}
            out={'action':'request',**self.pending,'trace':r.trace_as_dicts()}
            if j.kind=='epistemic':out['inquiry']=normal(j.inquiry)
            break
        else:raise RuntimeError('curriculum transition budget')
        self._record(['next',out]);return out

    def answer(self,y):
        self._room()
        if self.pending is None:raise ValueError('no pending observation')
        y=normal(y); p=self.pending;j=self.jobs[p['job']];x=p['input'];phase=p['phase'];e=j.episode
        if j.kind=='epistemic':
            if phase=='environment':
                if not isinstance(y,dict) or set(y)!={'train','audit'}:raise ValueError('environment grant requires unlabeled train and audit pools')
                e.grant(x,y['train'],y['audit']);self.pending=None;self._record(['answer',y]);return True
            result=(e.observe if phase=='train' else e.audit)(x,y)
        elif j.kind=='operator':result=self.words.apply('observe' if phase=='train' else 'audit',j.name,x,y)
        elif j.kind=='analogy':result=self.analogies.apply('observe' if phase=='train' else 'audit',j.name,x,y)
        else:
            args=(*x,y) if j.kind in ('lexicon','grammar') else (args_of(j.spec,x),y) if j.kind=='binding' else (x,y)
            result=(e.observe if phase=='train' else e.audit)(*args)
        j.asked.add(key(x));self.pending=None
        if phase=='train':j.history.append((x,y))
        else:
            j.audit_count+=1
            if result is False:self._finish(j,False,'fresh audit contradicted model')
        self._record(['answer',y]);return result is not False

    def abandon(self):
        """An unavailable or unrepresentable observation never counts as evidence."""
        self._room()
        if self.pending is None:raise ValueError('no pending observation')
        j=self.jobs[self.pending['job']];self.pending=None
        if j.kind=='epistemic':j.episode.blocked=True;j.episode.reason='evaluator declined evidence or environment'
        self._finish(j,False,'evaluator declined or could not provide requested evidence')
        self._record(['abandon'])

    def recover(self,name,new,x,y,train,audit,*,changed_context=False):
        self._room();ident(new)
        if self.pending is not None or new in self.jobs or len(self.jobs)>=MAX_TASKS or type(changed_context) is not bool:raise ValueError('invalid recovery request')
        j=self.jobs[name];e=j.episode
        if j.kind not in ('context','recovery') or not self.live(name):raise ValueError('validated contextual objective required')
        s=self._spec(dict(kind='context',deps=list(j.spec['deps']),labels=list(e.domain.labels),contexts=list(e.domain.contexts),train=train,audit=audit))
        x=e.domain.key(x);y=e.domain.output(y)
        used=set(e.training)|set(e.audits)|{x}
        if any(tuple(v) in used for v in s['train']+s['audit']):raise ValueError('recovery evidence must be fresh')
        e.contradict(x,y);nxt=e.recover(changed_context=changed_context)
        j.status='failed';j.reason='contradicted prediction';s['kind']='recovery'
        self.jobs[new]=Job(new,s,status='learning',episode=nxt)
        self._record(['recover',name,new,x,y,train,audit,changed_context]);return new

    def challenge(self,name,x,y):
        """Submit fresh measured evidence against a live learned law."""
        self._room()
        if self.pending is not None or not self.live(name):raise ValueError('live law required without pending evidence')
        j=self.jobs[name];e=j.episode
        if j.kind=='epistemic':ok=e.challenge(x,y)
        elif j.kind=='operator':
            x=e.domain.pair(x)
            if key(x) in j.asked or any(key(x)==key(v) for v in j.spec['train']+j.spec['audit']):raise ValueError('challenge must be outside reserved pools')
            ok=self.words.apply('audit',name,x,y)
            j.asked.add(key(x));j.audit_count+=1
        else:raise ValueError('challenge requires a learned word law')
        if not ok:self._finish(j,False,'later observation contradicted learned law')
        self._record(['challenge',name,x,y]);return ok

    def retract(self,name):
        self._room()
        if self.pending is not None:raise ValueError('answer pending')
        j=self.jobs[name]
        if j.kind not in ('operator','epistemic') or not self.live(name):raise ValueError('live learned operator required')
        if j.kind=='operator':self.words.apply('retract',name)
        else:j.episode.blocked=True;j.episode.reason='retracted'
        j.status='failed';j.reason='retracted';self._record(['retract',name])

    def predict(self,name,x):
        if not self.live(name):return None
        j=self.jobs[name]
        if j.kind in ('lexicon','grammar'):return j.episode.predict(*x)
        if j.kind=='binding':return j.episode.predict(args_of(j.spec,x))
        return j.episode.predict(x)

    def execute(self,grammar,text,rows,values,relations=None):
        if not self.live(grammar) or self.jobs[grammar].kind!='grammar':return {'abstained':True,'reason':'unavailable grammar'}
        d=CognitiveDispatch(Workspace())
        for n,j in self.jobs.items():
            if j.kind=='binding' and self.live(n):
                k=(j.spec['intent'],tuple(j.spec['inputs']))
                if k in d.bindings:return {'abstained':True,'reason':'ambiguous objective catalog'}
                d.bindings[k]=j.episode
        return self.jobs[grammar].episode.execute(text,rows,values,d,relations)

    def status(self):
        return {n:{'kind':j.kind,'status':j.status,'live':self.live(n),'reason':j.reason,'training':len(j.history),'audits':j.audit_count,'template':j.template_used,'expanded':j.expanded,'broad_check':j.broad_check,'candidate_programs':j.attempts,**({'epistemic':j.episode.status()} if j.kind=='epistemic' and j.episode is not None else {})} for n,j in self.jobs.items()}

    def save(self,path):
        data={'schema':1,'reuse':self.reuse,'events':self.events};blob=canonical({'data':data,'sha256':hashlib.sha256(canonical(data)).hexdigest()})
        if len(blob)>MAX_BYTES:raise ValueError('checkpoint too large')
        path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.ix-development-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES:raise ValueError('checkpoint too large')
        obj=json.loads(blob);d=obj['data']
        if d['schema']!=1 or hashlib.sha256(canonical(d)).hexdigest()!=obj['sha256'] or len(d['events'])>MAX_EVENTS:raise ValueError('checkpoint integrity failure')
        m=cls(reuse=d['reuse'])
        for action,*a in d['events']:
            if action=='add':m.add(*a)
            elif action=='offer':
                if normal(m.offer(*a[:-1]))!=a[-1]:raise ValueError('offered objective replay mismatch')
            elif action=='next':
                if normal(m.next())!=a[0]:raise ValueError('curriculum decision replay mismatch')
            elif action=='answer':m.answer(*a)
            elif action=='recover':m.recover(*a[:-1],changed_context=a[-1])
            elif action=='retract':m.retract(*a)
            elif action=='challenge':m.challenge(*a)
            elif action=='abandon' and not a:m.abandon()
            else:raise ValueError('unknown replay action')
        if normal(m.events)!=d['events']:raise ValueError('journal replay mismatch')
        return m
