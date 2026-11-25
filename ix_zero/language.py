"""Grounded finite-language induction. No pretrained vocabulary or parser model.

A hypothesis consists of a word-to-typed-symbol lexicon and role orders for
one- and two-referent utterances. Demonstrations supply scenes and observed
semantic outcomes, never token-level annotations or role orders.
"""
from dataclasses import dataclass
from collections import Counter
from itertools import permutations, product
from importlib.resources import files
from pathlib import Path
import hashlib, json, os, re, tempfile
from ._ix_vendor import parse_ix, run_ix

MIN_AUDITS = 3
MAX_EVENTS = 512
MAX_BYTES = 4_000_000
MAX_POOL = 128
ROLES = {3: ('verb','attr1','kind1'), 5: ('verb','attr1','kind1','attr2','kind2')}

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()

def symbol(x):
    if not isinstance(x,str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,39}',x):
        raise ValueError('invalid scene or semantic symbol')
    return x

def words(text):
    if not isinstance(text,str) or len(text)>240: raise ValueError('invalid utterance')
    result=tuple(text.lower().strip().removesuffix('.').removesuffix('?').removesuffix('!').split())
    if len(result) not in ROLES or any(not re.fullmatch(r'[a-z][a-z0-9-]{0,23}',w) for w in result):
        raise ValueError('expected three or five lexical tokens')
    return result

def scene(rows):
    if not isinstance(rows,(list,tuple)) or not 1<=len(rows)<=16: raise ValueError('invalid scene size')
    if any(not isinstance(r,(list,tuple)) or len(r)!=3 for r in rows): raise ValueError('expected [id, kind, attribute]')
    out=tuple(sorted(tuple(symbol(v) for v in r) for r in rows))
    if len({r[0] for r in out})!=len(out): raise ValueError('duplicate object identity')
    return out

def outcome(y, rows, n):
    if not isinstance(y,(list,tuple)) or len(y)!=(n-1)//2+1: raise ValueError('wrong outcome arity')
    y=tuple(symbol(v) for v in y)
    if any(v not in {r[0] for r in rows} for v in y[1:]): raise ValueError('unknown outcome referent')
    return y

def grounding(y,rows):
    by_id={i:(k,a) for i,k,a in rows}
    ret={'verb':('verb',y[0])}
    for j,i in enumerate(y[1:],1):
        k,a=by_id[i]; ret[f'kind{j}']=('kind',k); ret[f'attr{j}']=('attr',a)
    return ret

@dataclass
class Hypothesis:
    orders: tuple
    lexicon: dict

    def order(self,n): return self.orders[0 if n==3 else 1]

    def learn(self,tokens,meaning):
        new=dict(self.lexicon)
        for w,r in zip(tokens,self.order(len(tokens))):
            v=meaning[r]
            if w in new and new[w]!=v: return None
            new[w]=v
        return Hypothesis(self.orders,new)

    def interpret(self,tokens,rows):
        meanings={}
        for w,r in zip(tokens,self.order(len(tokens))):
            v=self.lexicon.get(w)
            typ=r.rstrip('12')
            if v is None or v[0]!=typ: return None
            meanings[r]=v[1]
        ids=[]
        for j in range(1,(len(tokens)-1)//2+1):
            options=[i for i,k,a in rows if (k,a)==(meanings[f'kind{j}'],meanings[f'attr{j}'])]
            if len(options)!=1: return None
            ids.append(options[0])
        return (meanings['verb'],*ids)

class LanguageEpisode:
    def __init__(self):
        self.models=[Hypothesis((a,b),{}) for a,b in product(permutations(ROLES[3]),permutations(ROLES[5]))]
        self.training={}; self.audits={}; self.blocked=False

    def observe(self,text,rows,y):
        t,s=words(text),scene(rows); y=outcome(y,s,len(t)); key=(t,s)
        if self.blocked or self.audits or key in self.training or len(self.training)>=MAX_EVENTS: raise ValueError('training blocked, frozen, repeated or full')
        meaning=grounding(y,s)
        self.models=[m for h in self.models if (m:=h.learn(t,meaning)) is not None]
        self.training[key]=y
        if not self.models: self.blocked=True

    def count(self,n): return sum(len(t)==n and passed for (t,_),(_,passed) in self.audits.items())

    def _predict(self,t,s):
        if self.blocked or not self.models: return None
        values={m.interpret(t,s) for m in self.models}
        return next(iter(values)) if len(values)==1 and None not in values else None

    def predict(self,text,rows,validated=True):
        t,s=words(text),scene(rows)
        if validated and self.count(len(t))<MIN_AUDITS: return None
        return self._predict(t,s)

    def audit(self,text,rows,y):
        t,s=words(text),scene(rows); y=outcome(y,s,len(t)); key=(t,s)
        if self.blocked or key in self.training or key in self.audits or len(self.audits)>=MAX_EVENTS: raise ValueError('audit requires fresh input and unblocked model')
        p=self._predict(t,s)
        if p is None: raise ValueError('no agreed interpretation to audit')
        passed=p==y; self.audits[key]=(y,passed)
        if not passed: self.blocked=True
        return passed

    def query(self,pool):
        if self.blocked or self.audits or not self.models: return None
        if not isinstance(pool,(list,tuple)) or len(pool)>MAX_POOL: raise ValueError('invalid inquiry pool')
        choices=[]
        for item in pool:
            if not isinstance(item,(tuple,list)) or len(item)!=2: raise ValueError('expected utterance and scene')
            t,s=words(item[0]),scene(item[1]); key=(t,s)
            if key in self.training: continue
            predictions=Counter(h.interpret(t,s) for h in self.models)
            # Prefer disagreement, then unknown lexical coverage, then diverse tokens.
            unknown=sum(any(w not in h.lexicon for h in self.models) for w in set(t))
            score=(len(predictions),-max(predictions.values()),unknown,len(set(t)))
            choices.append((score,key))
        return max(choices)[1] if choices else None

    def status(self):
        common={}
        if self.models:
            common={w:v for w,v in self.models[0].lexicon.items() if all(h.lexicon.get(w)==v for h in self.models)}
        return dict(models=len(self.models),training=len(self.training),audits={n:self.count(n) for n in ROLES},blocked=self.blocked,
                    lexicon=common,orders={n:sorted({h.order(n) for h in self.models}) for n in ROLES})

    def express(self,rows,y):
        """Generate a supported utterance using the consensus lexicon and grammar."""
        s=scene(rows)
        if not isinstance(y,(list,tuple)) or len(y) not in (2,3): raise ValueError('invalid intent')
        n=2*len(y)-1; y=outcome(y,s,n)
        if self.blocked or self.count(n)<MIN_AUDITS or not self.models: return None
        m=self.models[0]; g=grounding(y,s)
        options=[sorted(w for w,v in m.lexicon.items() if v==g[r]) for r in m.order(n)]
        # Fixed cap avoids unbounded synonym expansion.
        from itertools import islice
        for t in islice(product(*options),256):
            if self._predict(t,s)==y: return ' '.join(t)
        return None

class LanguageMind:
    def __init__(self): self.episodes={}; self.events=[]

    def apply(self,action,name,*args):
        symbol(name)
        if len(self.events)>=MAX_EVENTS: raise ValueError('language journal full')
        # Normalize before mutation, so an unsupported JSON object cannot corrupt journal state.
        event=json.loads(canonical([action,name,*args]))
        if action=='new' and not args:
            if name in self.episodes or len(self.episodes)>=16: raise ValueError('duplicate episode or capacity exceeded')
            self.episodes[name]=LanguageEpisode(); result=None
        else:
            ep=self.episodes[name]
            if action=='observe' and len(args)==3: result=ep.observe(*args)
            elif action=='audit' and len(args)==3: result=ep.audit(*args)
            elif action=='retract' and not args: ep.blocked=True; result=None
            else: raise ValueError('invalid language event')
        self.events.append(event); return result

    def advance(self,name,pool=()):
        ep=self.episodes[name]
        if not isinstance(pool,(list,tuple)) or len(pool)>MAX_POOL: raise ValueError('invalid inquiry pool')
        fresh=[]
        for text,rows in pool:
            t,s=words(text),scene(rows)
            if (t,s) not in ep.training and (t,s) not in ep.audits: fresh.append((t,s))
        audit_options=[(t,s) for t,s in fresh if ep.count(len(t))<MIN_AUDITS and ep._predict(t,s) is not None]
        q=ep.query(pool)
        facts=dict(blocked=ep.blocked,complete=all(ep.count(n)>=MIN_AUDITS for n in ROLES),
                   has_query=q is not None,has_audit=bool(audit_options),frozen=bool(ep.audits),resolved=len(ep.models)==1,
                   unknown=any(any(w not in h.lexicon for h in ep.models) for t,_ in fresh for w in t))
        src=files('ix_zero').joinpath('policies','language.ix').read_text(encoding='utf-8')
        r=run_ix(parse_ix(src),agent='Language',event='advance',inputs=facts)
        if r.approvals_required or len(r.replies)!=1: raise ValueError('invalid IX language decision')
        a=r.replies[0]
        if a=='revise' and ep.blocked: result=None
        elif a=='complete' and facts['complete'] and not ep.blocked: result=None
        elif a=='inquire' and q is not None and not ep.blocked: result=(' '.join(q[0]),q[1])
        elif a=='audit' and audit_options and not ep.blocked: t,s=audit_options[0]; result=(' '.join(t),s)
        elif a=='exhausted' and not ep.blocked: result=None
        else: raise ValueError('language policy action rejected')
        return dict(action=a,request=result,facts=facts,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())

    def compute(self,name,text,rows,values,operator_mind):
        """Interpret language into a validated learned word operator; no external action."""
        p=self.episodes[name].predict(text,rows)
        if p is None or len(p)!=3: return None
        a,x,y=p; ep=operator_mind.episodes.get(a)
        if ep is None or ep.blocked or a not in operator_mind.promoted: return None
        if not isinstance(values,dict) or x not in values or y not in values: return None
        result=ep.predict([values[x],values[y]])
        return dict(intent=p,value=result) if result is not None else None

    def transfer(self,name,text,rows,analogy_mind,operator_mind):
        analogy_mind.reconcile_operators(operator_mind)
        p=self.episodes[name].predict(text,rows)
        if p is None or len(p)!=3: return None
        a,x,y=p; ep=analogy_mind.episodes.get(a)
        if ep is None or ep.blocked or x not in ep.graph.states or y not in ep.graph.states: return None
        result=ep.predict([x,y]); return dict(intent=p,value=result) if result is not None else None

    def simulate(self,name,text,rows,facts,vector,unified_mind):
        p=self.episodes[name].predict(text,rows)
        if p is None or len(set(p[1:]))!=len(p[1:]): return None
        result=unified_mind.predict(unified_mind.joint(facts,vector),p)
        return dict(intent=p,outcome=result) if result is not None else None

    def save(self,path):
        data=dict(schema=1,events=self.events); blob=canonical(dict(data=data,sha256=hashlib.sha256(canonical(data)).hexdigest()))
        if len(blob)>MAX_BYTES: raise ValueError('language journal too large')
        path=Path(path); fd,tmp=tempfile.mkstemp(prefix='.ix-language-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f: blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES: raise ValueError('language journal too large')
        obj=json.loads(blob); data=obj['data']
        if data['schema']!=1 or hashlib.sha256(canonical(data)).hexdigest()!=obj['sha256']: raise ValueError('invalid language checkpoint')
        m=cls()
        for a,n,*args in data['events']: m.apply(a,n,*args)
        if canonical(m.events)!=canonical(data['events']): raise ValueError('language replay mismatch')
        return m
