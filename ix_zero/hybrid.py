"""Evidence-gated neural and symbolic learning over finite interactive domains.

Neural predictions guide inquiry but cannot validate a program. Only reserved
external measurements authorize symbolic predictions. No external side effects.
"""
from collections import Counter, deque
from copy import deepcopy
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import tempfile
from .checkpoint import canonical
from .epistemic import Inquiry
from .neural import Dense

MAX_EVENTS = 1024
MAX_BYTES = 2_000_000
DIRECTIONS = ((1,0),(0,1),(-1,0),(0,-1))


def bits(n, width=5):
    return [(n >> i)&1 for i in range(width)]


def integer(b):
    return sum(int(v >= .5) << i for i,v in enumerate(b))


def grid_step(model, x):
    """Executable candidate program: rotation, reflection, stride, boundary."""
    rotation, reflection, stride, wrap = model
    n, a, b, action = x
    dx, dy = DIRECTIONS[(rotation + reflection*action)%4]
    if wrap: return ((a+dx*stride)%n, (b+dy*stride)%n)
    return (min(n-1,max(0,a+dx*stride)), min(n-1,max(0,b+dy*stride)))


class GridInquiry:
    """Finite spatial program family with the same inquiry/audit contract."""
    def __init__(self, widths, initial, train, audit):
        if not 1 <= len(widths) <= 4 or tuple(sorted(set(widths))) != tuple(widths) or any(type(w) is not int or not 3 <= w <= 8 for w in widths):
            raise ValueError('invalid spatial scope')
        self.widths=tuple(widths); self.models=tuple(product(range(4),(-1,1),(1,2),(False,True)))
        self.pools={}; self.training={}; self.audits={}; self.challenges={}; self.blocked=False; self.finished=False
        self.grant(initial,train,audit)

    def input(self,x):
        if len(x)!=4 or any(type(v) is not int for v in x): raise ValueError('invalid spatial input')
        n,a,b,d=x
        if n not in self.widths or not (0<=a<n and 0<=b<n and 0<=d<4): raise ValueError('spatial input outside scope')
        return tuple(x)

    def value(self,m,x): return grid_step(m,x)
    def partition(self,x): return Counter(self.value(m,x) for m in self.models)
    def rank(self,x):
        p=self.partition(x); return (len(p),-max(p.values()),-x[0],*x[1:])

    def grant(self,w,train,audit):
        if type(w) is not int or w not in self.widths or w in self.pools or self.blocked or self.finished or self.audits: raise ValueError('invalid spatial grant')
        t=tuple(self.input((w,*x))[1:] for x in train); a=tuple(self.input((w,*x))[1:] for x in audit)
        if not 1<=len(t)<=256 or not 3<=len(a)<=64 or len(set(t+a))!=len(t+a): raise ValueError('invalid spatial evidence pools')
        self.pools[w]=(t,a)

    def choose(self):
        if self.blocked or self.finished: return dict(action='failed' if self.blocked else 'complete')
        if not self.models: return dict(action='failed',reason='candidate family exhausted')
        if not self.audits:
            xs=[(w,*x) for w,(t,_) in self.pools.items() for x in t if (w,*x) not in self.training]
            if len(self.models)>1 and xs:
                x=max(xs,key=self.rank)
                if len(self.partition(x))>1: return dict(action='train',input=x)
            remaining=[w for w in self.widths if w not in self.pools]
            if len(self.models)>1:
                if remaining:
                    def score(w):
                        return max(len(self.partition((w,a,b,d))) for a,b,d in product(range(w),range(w),range(4)))
                    w=max(remaining,key=lambda w:(score(w),-w))
                    return dict(action='environment',input=w)
                return dict(action='failed',reason='unresolved spatial ambiguity')
            if len(self.training)<3:
                if xs:return dict(action='train',input=min(xs))
                return dict(action='failed',reason='insufficient training')
            if remaining:return dict(action='environment',input=min(remaining))
        if len(self.models)!=1:return dict(action='failed',reason='unresolved spatial ambiguity')
        fresh=[(w,*x) for w,(_,a) in self.pools.items() for x in a if (w,*x) not in self.audits]
        if fresh:return dict(action='audit',input=min(fresh))
        return dict(action='complete' if self.validated() else 'failed')

    def output(self,x,y):
        if not isinstance(y,(list,tuple)) or len(y)!=2 or any(type(v) is not int or not 0<=v<x[0] for v in y):raise ValueError('invalid spatial output')
        return tuple(y)

    def observe(self,x,y):
        x=self.input(x);y=self.output(x,y)
        if self.blocked or self.finished or self.audits or x[0] not in self.pools or x[1:] not in self.pools[x[0]][0] or x in self.training:raise ValueError('invalid spatial observation')
        self.training[x]=y;self.models=tuple(m for m in self.models if self.value(m,x)==y)

    def audit(self,x,y):
        x=self.input(x);y=self.output(x,y)
        if self.blocked or self.finished or len(self.models)!=1 or len(self.training)<3 or x[0] not in self.pools or x[1:] not in self.pools[x[0]][1] or x in self.audits:raise ValueError('invalid spatial audit')
        ok=self.value(self.models[0],x)==y;self.audits[x]=(y,ok);self.blocked=not ok;return ok

    def validated(self):
        return not self.blocked and len(self.models)==1 and len(self.training)>=3 and set(self.pools)==set(self.widths) and all(self.audits.get((w,*x),(None,False))[1] for w,(_,a) in self.pools.items() for x in a)

    def predict(self,x):
        x=self.input(x)
        return self.value(self.models[0],x) if self.finished and self.validated() else None

    def challenge(self,x,y):
        x=self.input(x);y=self.output(x,y)
        p=self.predict(x)
        if p is None or x in self.challenges or any(x[1:] in pool for pool in self.pools[x[0]]):raise ValueError('invalid fresh challenge')
        ok=p==y;self.challenges[x]=y;self.blocked=not ok;return ok


class Hybrid:
    """One persistent inquiry with trainable sensing, prediction and guidance.

    The host grants evidence and supplies independent audits. Checkpoints replay
    the complete learning history, including weight updates, rather than trusting
    stored parameters or claimed promotions. Guidance only breaks exact heuristic
    ties, never prunes hypotheses or bypasses audit requirements.
    """
    def __init__(self, family, widths, initial, train, audit, *, seed=0, budget=2, mode='hybrid', prior=None):
        if family not in ('word','grid') or mode not in ('hybrid','unguided','symbolic','frozen','oracle','centroid'):raise ValueError('invalid hybrid configuration')
        if type(seed) is not int or not 0<=seed<2**32-2:raise ValueError('invalid seed')
        self.config=dict(family=family,widths=list(widths),initial=initial,train=list(train),audit=list(audit),seed=seed,budget=budget,mode=mode)
        self.family=family;self.mode=mode
        self.inquiry=Inquiry(tuple(widths),budget,initial,train,audit) if family=='word' else GridInquiry(tuple(widths),initial,train,audit)
        self.recognition=Dense(3,6,1,seed); self.world=Dense(16,16,10,seed+1);self.strategy=Dense(18,10,1,seed+2)
        self.events=[];self.examples=[];self.pending=None;self.centroids=None
        if prior is not None:
            if mode!='hybrid' or not isinstance(prior,dict) or prior.get('config',{}).get('prior') is not None:
                raise ValueError('only one verified source inquiry may be transferred')
            if len(canonical(prior))>MAX_BYTES//2:raise ValueError('source checkpoint too large')
            source=self._restore(prior)
            if source.mode!='hybrid' or not source.inquiry.finished or not source.inquiry.validated() or source.family==family:
                raise ValueError('transfer requires a validated hybrid inquiry in another family')
            self.recognition=deepcopy(source.recognition);self.strategy=deepcopy(source.strategy)
            self.config['prior']=deepcopy(prior)

    def _room(self):
        if len(self.events)>=MAX_EVENTS:raise ValueError('journal limit')

    def _record(self,kind,*args):
        self.events.append(json.loads(canonical([kind,*args])))

    def calibrate(self,rows):
        self._room()
        if self.config.get('prior') is not None or self.pending or self.inquiry.training or self.inquiry.finished or self.inquiry.blocked or not 1<=len(rows)<=512 or any(e[0]=='calibrate' for e in self.events):raise ValueError('invalid calibration')
        for x,y in rows:
            Dense.vector(x,3);Dense.vector([y],1)
            if y not in (0,1):raise ValueError('calibration requires binary label')
        if self.mode=='centroid':
            groups=[[x for x,y in rows if y==v] for v in (0,1)]
            if not all(groups):raise ValueError('both classes required for centroid calibration')
            self.centroids=[[sum(p[i] for p in group)/len(group) for i in range(3)] for group in groups]
        if self.mode in ('hybrid','unguided'):
            for _ in range(12):
                for x,y in rows:self.recognition.train(x,[y])
        self._record('calibrate',rows)

    def read(self,pixels):
        Dense.vector(pixels,3)
        if self.mode=='centroid' and self.centroids is not None:
            return min(range(2),key=lambda j:sum((x-y)**2 for x,y in zip(pixels,self.centroids[j])))
        if self.mode in ('symbolic','centroid'):return int(sum(pixels)/3>=.5)
        return int(self.recognition.predict(pixels)[0]>=.5)

    def decode(self,w,raster):
        if self.family=='word':
            if len(raster)!=w:raise ValueError('wrong word raster size')
            return integer([self.read(p) for p in raster])
        if len(raster)!=w*w:raise ValueError('wrong spatial raster size')
        marks=[i for i,p in enumerate(raster) if self.read(p)]
        if len(marks)!=1:return None
        return (marks[0]%w,marks[0]//w)

    def features(self,x):
        w,a,b=x[:3];d=x[3] if len(x)==4 else 0
        return [int(self.family=='grid'),w/8,*bits(a),*bits(b),*bits(d,4)]

    def encode_output(self,y):return bits(y)+[0]*5 if self.family=='word' else bits(y[0])+bits(y[1])

    def neural_prediction(self,x):
        p=self.world.predict(self.features(x));return integer(p[:5]) if self.family=='word' else (integer(p[:5]),integer(p[5:]))

    def guidance_features(self,x):
        p=self.inquiry.partition(x);n=len(self.inquiry.models)
        return self.features(x)+[len(p)/n,max(p.values())/n]

    def next(self):
        if self.pending:return self.pending.copy()
        r=self.inquiry.choose()
        if r['action']=='train' and self.mode in ('hybrid','frozen') and len(self.inquiry.models)>1:
            xs=[(w,*x) for w,(t,_) in self.inquiry.pools.items() for x in t if (w,*x) not in self.inquiry.training]
            if xs:
                best=max(self.inquiry.rank(x)[:2] for x in xs)
                ties=[x for x in xs if self.inquiry.rank(x)[:2]==best]
                def score(x):
                    p=self.world.predict(self.features(x));uncertainty=sum(1-abs(2*v-1) for v in p)/len(p)
                    return self.strategy.predict(self.guidance_features(x))[0]+.1*uncertainty,self.inquiry.rank(x)
                r=dict(r,input=max(ties,key=score))
        if r['action']=='complete':self.inquiry.finished=True
        elif r['action']=='failed':self.inquiry.blocked=True
        else:self.pending=r.copy()
        return r.copy()

    def answer(self,value):
        self._room()
        r=self.pending
        if not r:raise ValueError('no pending request')
        action=r['action'];x=r['input'];e=self.inquiry
        if action=='environment':e.grant(x,value['train'],value['audit'])
        elif action=='audit':e.audit(x,value)
        else:
            y=(e.domains[x[0]].word(value) if self.family=='word' else e.output(x,value)) if self.mode=='oracle' else self.decode(x[0],value)
            if y is None:
                e.blocked=True
            else:
                f=self.guidance_features(x);n=len(e.models)
                e.observe(x,y)
                self.examples.append((self.features(x),self.encode_output(y)))
                if self.mode in ('hybrid','unguided'):
                    self.strategy.train(f,[1-len(e.models)/n])
                    # Deterministic replay over observed training records only.
                    for _ in range(8):
                        for features,target in self.examples[-32:]:self.world.train(features,target)
        self._record('answer',value);self.pending=None

    def deny(self):
        self._room()
        if not self.pending or self.pending['action']!='environment':raise ValueError('no environment request')
        self.inquiry.blocked=True;self.pending=None;self._record('deny')

    def challenge(self,x,y):
        self._room();ok=self.inquiry.challenge(x,y);self._record('challenge',x,y);return ok

    def predict(self,x):return self.inquiry.predict(x)

    def plan(self,start,goal,w):
        """Shortest-path heuristic search under the audited spatial program.

        Neural ordering resolves ties only. The returned plan is a proposal,
        not authorization to act on a physical system.
        """
        if self.family!='grid' or self.predict((w,*start,0)) is None:return None
        start=tuple(start);goal=tuple(goal);self.inquiry.input((w,*goal,0))
        queue=deque([(start,[])]);seen={start}
        while queue:
            s,path=queue.popleft()
            if s==goal:return path
            actions=list(range(4))
            if self.mode in ('hybrid','frozen'):
                actions.sort(key=lambda a:sum(abs(v-g) for v,g in zip(self.neural_prediction((w,*s,a)),goal)))
            for a in actions:
                nxt=self.predict((w,*s,a))
                if nxt not in seen:seen.add(nxt);queue.append((nxt,path+[a]))
        return None

    def weights(self):return {n:getattr(self,n).state() for n in ('recognition','world','strategy')}

    def payload(self):
        return dict(version=1,config=self.config,events=self.events,weights=self.weights(),pending=self.pending,finished=self.inquiry.finished,blocked=self.inquiry.blocked)

    def save(self,path):
        path=Path(path);data=self.payload();blob=canonical(dict(payload=data,sha256=hashlib.sha256(canonical(data)).hexdigest()))
        if len(blob)>MAX_BYTES:raise ValueError('hybrid checkpoint too large')
        fd,tmp=tempfile.mkstemp(prefix='.ix-hybrid-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    def transfer(self,family,widths,initial,train,audit,*,seed=0,budget=2):
        """Reuse validated recognition and guidance weights; reset world model.

        The source journal is included and replayed on restore. The target's
        knowledge must independently pass its own audits. No symbolic law is
        transferred between incompatible task families.
        """
        return type(self)(family,widths,initial,train,audit,seed=seed,budget=budget,prior=self.payload())

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES:raise ValueError('hybrid checkpoint too large')
        obj=json.loads(blob);d=obj['payload']
        if d['version']!=1 or hashlib.sha256(canonical(d)).hexdigest()!=obj['sha256'] or len(d['events'])>MAX_EVENTS:raise ValueError('hybrid checkpoint integrity')
        return cls._restore(d)

    @classmethod
    def _restore(cls,d):
        if d['version']!=1 or len(d['events'])>MAX_EVENTS:raise ValueError('invalid hybrid payload')
        m=cls(**d['config'])
        for kind,*args in d['events']:
            if kind=='calibrate':m.calibrate(*args)
            elif kind=='answer':m.next();m.answer(*args)
            elif kind=='challenge':m.next();m.challenge(*args)
            elif kind=='deny':m.next();m.deny()
            else:raise ValueError('unknown hybrid event')
        if d['pending'] or d['finished'] or (d['blocked'] and not m.inquiry.blocked):m.next()
        if canonical(m.payload())!=canonical(d):raise ValueError('hybrid checkpoint replay mismatch')
        return m
