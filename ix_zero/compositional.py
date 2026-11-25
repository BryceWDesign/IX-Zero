"""Evidence-supervised induction of recursive surface-to-objective rules.

Training labels are complete semantic trees, not token alignments. The grammar
is finite; recursion is bounded. Execution uses existing audited ACCE bindings.
"""
from dataclasses import dataclass
from functools import lru_cache
import re
from .language import scene, symbol
from .acce import Object

MAX_TOKENS=64
MAX_DEPTH=8
MAX_RULES=32
MAX_EVENTS=256
MAX_WORK=100000


def tokenize(text):
    if not isinstance(text,str) or len(text)>1536: raise ValueError('utterance budget')
    t=tuple(text.lower().strip().removesuffix('.').removesuffix('?').removesuffix('!').split())
    if not 1<=len(t)<=MAX_TOKENS or any(not re.fullmatch(r'[a-z][a-z0-9-]{0,23}',w) for w in t):
        raise ValueError('unsupported tokens or utterance budget')
    return t


def tree(y, ids, depth=0):
    if depth>MAX_DEPTH: raise ValueError('objective depth budget')
    if isinstance(y,str):
        if y not in ids: raise ValueError('unknown referent')
        return y
    if not isinstance(y,(list,tuple)) or len(y) not in (2,3): raise ValueError('invalid objective tree')
    return (symbol(y[0]),*(tree(c,ids,depth+1) for c in y[1:]))


def depth(y): return 0 if isinstance(y,str) else 1+max(map(depth,y[1:]))


@dataclass(frozen=True)
class Rule:
    intent: str
    pattern: tuple # lexical anchors and child slots (0, 1)


class BudgetExceeded(ValueError): pass


class Grammar:
    def __init__(self,source):
        self.source=source
        st=source.status()
        if st['blocked'] or not st['models'] or st['audits'][3]<3 or len(st['orders'][3])!=1:
            raise ValueError('validated unambiguous reference grammar required')
        self.lexicon={w:tuple(v) for w,v in st['lexicon'].items() if v[0] in ('kind','attr')}
        self.order=tuple(r.rstrip('1') for r in st['orders'][3][0] if r!='verb')
        self.rules={}; self.training={}; self.audits=set(); self.blocked=False

    def live(self):
        st=self.source.status()
        return not self.blocked and not st['blocked'] and st['audits'][3]>=3 and all(st['lexicon'].get(w)==v for w,v in self.lexicon.items()) and len(st['orders'][3])==1 and tuple(r.rstrip('1') for r in st['orders'][3][0] if r!='verb')==self.order

    def _key(self,t,s):
        # Scene identity renaming alone is not new evidence.
        return (t,tuple(sorted((k,a) for _,k,a in s)))

    def _descriptor(self,y,s):
        byid={i:(k,a) for i,k,a in s}
        def walk(x): return ('ref',*byid[x]) if isinstance(x,str) else (x[0],*(walk(c) for c in x[1:]))
        return walk(y)

    def _chart(self,t,s):
        steps=0
        def tick():
            nonlocal steps
            steps+=1
            if steps>MAX_WORK: raise BudgetExceeded('parse work budget')
        @lru_cache(None)
        def parse(lo,hi):
            tick(); out={}
            def add(y,used):
                if depth(y)>MAX_DEPTH: raise BudgetExceeded('objective depth budget')
                out.setdefault(y,set()).update(used)
                if len(out)>128: raise BudgetExceeded('ambiguity budget')
            if hi-lo==2:
                vals=[self.lexicon.get(w) for w in t[lo:hi]]
                if all(v is not None for v in vals) and tuple(v[0] for v in vals)==self.order:
                    m=dict(vals)
                    for i,k,a in s:
                        if (k,a)==(m['kind'],m['attr']):add(i,())
            for rule in self.rules:
                def match(p,pos,args,used):
                    tick()
                    if p==len(rule.pattern):
                        if pos==hi:
                            y=(rule.intent,*(args[i] for i in range(len(args))))
                            add(y,used|{rule})
                        return
                    token=rule.pattern[p]
                    if isinstance(token,str):
                        if pos<hi and t[pos]==token:match(p+1,pos+1,args,used)
                    else:
                        # Each slot must consume a proper subspan; a rule always has a literal.
                        remaining=len(rule.pattern)-p-1
                        for end in range(pos+1,min(hi-remaining, pos+hi-lo-1)+1):
                            for y,u in parse(pos,end).items():
                                match(p+1,end,{**args,token:y},used|u)
                match(0,lo,{},set())
            return out
        return parse,tick

    def observe(self,text,rows,meaning):
        t,s=tokenize(text),scene(rows); y=tree(meaning,{i for i,_,_ in s}); key=self._key(t,s)
        if not self.live() or self.audits or key in self.training or len(self.training)>=MAX_EVENTS or isinstance(y,str):raise ValueError('training unavailable or repeated')
        parse,tick=self._chart(t,s)
        children=y[1:]; spans=[]
        for child in children:
            spans.append([(i,j) for i in range(len(t)) for j in range(i+1,len(t)+1) if j-i<len(t) and set(parse(i,j))=={child}])
        candidates=set()
        from itertools import product
        for parts in product(*spans):
            tick(); starts=sorted((i,j,n) for n,(i,j) in enumerate(parts))
            if any(a[1]>b[0] for a,b in zip(starts,starts[1:])):continue
            pattern=[]; pos=0
            for i,j,n in starts: pattern.extend(t[pos:i]);pattern.append(n);pos=j
            pattern.extend(t[pos:]); literals=sum(isinstance(x,str) for x in pattern)
            if 1<=literals<=8:candidates.add(Rule(y[0],tuple(pattern)))
        if len(candidates)!=1: raise ValueError('no unique grounded construction')
        rule=next(iter(candidates))
        if rule not in self.rules and len(self.rules)>=MAX_RULES:raise ValueError('grammar capacity')
        old=self.rules.get(rule)
        descriptor=self._descriptor(y,s)
        if old and descriptor in old['train']:raise ValueError('repeated semantic training input')
        if old is None:self.rules[rule]={'train':set(),'audit':set()}
        # Refuse contradictory or ambiguous complete meanings, transactionally.
        try:
            p,_=self._chart(t,s)
            if set(p(0,len(t)))!={y}:raise ValueError('ambiguous or contradictory construction')
            if old is None:
                for prior_t,prior_s,prior_y in self.training.values():
                    prior,_=self._chart(prior_t,prior_s)
                    if set(prior(0,len(prior_t)))!={prior_y}:raise ValueError('construction conflicts with earlier evidence')
        except Exception:
            if old is None:del self.rules[rule]
            raise
        self.rules[rule]['train'].add(descriptor);self.training[key]=(t,s,y)
        return rule

    def _interpret(self,t,s):
        if not self.live():return None,set()
        p,_=self._chart(t,s); models=p(0,len(t))
        if len(models)!=1:return None,set()
        y,used=next(iter(models.items()))
        if isinstance(y,str):return None,set()
        return y,used

    def audit(self,text,rows,meaning):
        t,s=tokenize(text),scene(rows); y=tree(meaning,{i for i,_,_ in s}); key=self._key(t,s)
        if not self.live() or key in self.training or key in self.audits or len(self.audits)>=MAX_EVENTS:raise ValueError('audit unavailable or repeated')
        p,used=self._interpret(t,s)
        if p is None:raise ValueError('no agreed interpretation to audit')
        d=self._descriptor(p,s)
        if any(len(self.rules[r]['train'])<2 or d in self.rules[r]['train'] or d in self.rules[r]['audit'] for r in used):raise ValueError('audit requires distinct semantic evidence')
        self.audits.add(key)
        if p!=y:self.blocked=True;return False
        for r in used:self.rules[r]['audit'].add(d)
        return True

    def predict(self,text,rows):
        try:
            y,used=self._interpret(tokenize(text),scene(rows))
            if y is None or any(len(self.rules[r]['train'])<2 or len(self.rules[r]['audit'])<3 for r in used):return None
            return y
        except BudgetExceeded:return None

    def execute(self,text,rows,values,dispatch,relations=None):
        try:y=self.predict(text,rows)
        except ValueError as e:return {'abstained':True,'reason':str(e)}
        if y is None:return {'abstained':True,'reason':'unresolved or unaudited construction'}
        objects={i:Object(i,k,a,values.get(i),(relations or {}).get(i)) for i,k,a in scene(rows)}
        trace=[]
        def run(x):
            if isinstance(x,str):return objects[x]
            args=tuple(run(c) for c in x[1:])
            if any(a is None for a in args):return None
            r=dispatch.execute(x[0],args);trace.append({'intent':x[0],**r})
            return r.get('value')
        value=run(y)
        return {'meaning':y,'value':value,'trace':trace,**({'abstained':True,'reason':'unsupported or invalidated objective'} if value is None else {})}
