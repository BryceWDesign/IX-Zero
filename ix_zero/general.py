"""Local, bounded language, evidence, exact mathematics and program induction.

No external inference service, arbitrary code execution or implicit promotion.
The controlled-language front end is intentionally not a general English parser.
"""
from collections import deque
from fractions import Fraction
from itertools import product
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from .constraints import solve as solve_constraints
from .continual import Continual
from .continual_cli import Console as CoreConsole

MAX_FACTS = 256
MAX_SCOPE = 256
MAX_SKILLS = 32
MAX_EXPANDED = 32
MAX_EVENTS = 4096
MAX_FILE = 60_000_000
NAME = r'[a-z][a-z0-9_-]{0,39}'
RELATIONS = ('inside', 'before', 'kind_of', 'connected_to')
TRANSITIVE = ('inside', 'before', 'kind_of')


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def hash_of(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def name(x):
    if not isinstance(x, str) or not re.fullmatch(NAME, x):
        raise ValueError('invalid identifier')
    return x


def integer(x):
    if type(x) is not int or abs(x) > 1_000_000:
        raise ValueError('bounded integer required')
    return x


def sequence(x):
    if not isinstance(x, (list, tuple)) or len(x) > 16:
        raise ValueError('expected at most 16 integers')
    return tuple(integer(v) for v in x)


def bounded_fraction(x):
    if max(abs(x.numerator).bit_length(), x.denominator.bit_length()) > 256:
        raise ValueError('arithmetic resource limit')
    return x


def affine(text):
    """Parse an affine expression into exact (coefficient, constant) pairs."""
    if not isinstance(text, str) or len(text) > 256:
        raise ValueError('expression length limit')
    tree = ast.parse(text.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError('expression node limit')
    def visit(n):
        zero = Fraction(0)
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return zero, bounded_fraction(Fraction(integer(n.value)))
        if isinstance(n, ast.Name) and n.id == 'x':
            return Fraction(1), zero
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            a, b = visit(n.operand)
            return (a, b) if isinstance(n.op, ast.UAdd) else (-a, -b)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left); c, d = visit(n.right)
            if isinstance(n.op, ast.Add): r = (a+c, b+d)
            elif isinstance(n.op, ast.Sub): r = (a-c, b-d)
            elif isinstance(n.op, ast.Mult) and (not a or not c): r = (a*d+b*c, b*d)
            elif isinstance(n.op, ast.Div) and not c and d: r = (a/d, b/d)
            else: raise ValueError('unsupported nonlinear or undefined operation')
            return tuple(bounded_fraction(v) for v in r)
        raise ValueError('unsupported expression')
    return visit(tree.body)


def math_answer(text, equation=False):
    if not equation:
        a, b = affine(text)
        if a: raise ValueError('unbound variable')
        return {'status':'answer', 'value':str(b), 'method':'exact rational arithmetic'}
    if text.count('=') != 1: raise ValueError('one equality required')
    left, right = map(affine, text.split('=')); a, b = left[0]-right[0], right[1]-left[1]
    if not a:
        return {'status':'answer', 'solution':'all rationals' if not b else 'none', 'method':'affine equality'}
    x = bounded_fraction(b/a)
    l, r = (bounded_fraction(p*x+q) for p,q in (left,right))
    if l != r: raise ArithmeticError('equation verification failed')
    return {'status':'answer','solution':str(x),'left':str(l),'right':str(r),'method':'affine equality'}


class Evidence:
    """Explicit source-scoped assertions, conservative transitive inference."""
    def __init__(self): self.sources = {}

    def ingest(self, source, text):
        name(source)
        if source in self.sources or not isinstance(text,str) or len(text)>12000:
            raise ValueError('duplicate source or document size limit')
        rows=[]
        for line in text.strip().splitlines():
            m=re.fullmatch(rf'({NAME}) is (not )?({"|".join(RELATIONS)}) ({NAME})\.',line.strip().lower())
            if not m: raise ValueError('unsupported assertion: '+line[:100])
            a, neg, rel, b=m.groups(); rows.append((a,rel,b,not neg))
        if not rows or sum(map(len,self.sources.values()))+len(rows)>MAX_FACTS:
            raise ValueError('assertion count limit')
        self.sources[source]=rows

    def retract(self, source):
        if source not in self.sources: raise ValueError('unknown source')
        del self.sources[source]

    def ask(self, a, rel, b):
        name(a);name(b)
        if rel not in RELATIONS: raise ValueError('unsupported relation')
        positive={}; negative={}
        for src,rows in sorted(self.sources.items()):
            for s,r,o,p in rows:
                if r==rel: (positive if p else negative).setdefault((s,o),[]).append(src)
        target=(a,b); direct=positive.get(target,[]); neg=negative.get(target,[])
        if direct and neg: return {'status':'conflict','positive':direct,'negative':neg}
        proof=[]
        if direct: proof=[{'from':a,'to':b,'sources':direct}]
        elif rel in TRANSITIVE:
            # Do not use a directly disputed premise in a derived proof.
            queue=deque([(a,[])]); seen={a}
            while queue:
                s,path=queue.popleft()
                for (u,v),sources in sorted(positive.items()):
                    if u!=s or (u,v) in negative:continue
                    new=path+[{'from':u,'to':v,'sources':sources}]
                    if v==b: proof=new;queue.clear();break
                    if v not in seen:seen.add(v);queue.append((v,new))
        if proof and neg:return {'status':'conflict','proof':proof,'negative':neg}
        if proof:return {'status':'answer','value':True,'proof':proof}
        if neg:return {'status':'answer','value':False,'sources':neg}
        return {'status':'unknown','reason':'no uncontested proof'}


# The interpreter is a finite typed language, not Python eval/exec.
PRIMITIVES = ('reverse','sort','unique','even','odd','sum','length','negate',
              'add1','add2','sub1','double','triple','square','prefixsum')


def primitive(op, x):
    if op=='reverse': y=x[::-1]
    elif op=='sort': y=tuple(sorted(x))
    elif op=='unique': y=tuple(dict.fromkeys(x))
    elif op=='even': y=tuple(v for v in x if v%2==0)
    elif op=='odd': y=tuple(v for v in x if v%2)
    elif op=='sum': y=(sum(x),)
    elif op=='length': y=(len(x),)
    elif op=='negate': y=tuple(-v for v in x)
    elif op=='add1': y=tuple(v+1 for v in x)
    elif op=='add2': y=tuple(v+2 for v in x)
    elif op=='sub1': y=tuple(v-1 for v in x)
    elif op=='double': y=tuple(2*v for v in x)
    elif op=='triple': y=tuple(3*v for v in x)
    elif op=='square': y=tuple(v*v for v in x)
    elif op=='prefixsum': y=tuple(sum(x[:i+1]) for i in range(len(x)))
    else: raise ValueError('unknown primitive')
    return sequence(y)


class Programs:
    """Enumerative induction, disagreement queries, held-back audits, macro reuse.

    Scope contains unlabeled inputs, not oracle code. A program is promoted only
    after discrimination on scope and all separately supplied audit responses.
    """
    def __init__(self,reuse=True):
        if type(reuse) is not bool:raise ValueError('reuse must be boolean')
        self.reuse=reuse;self.skills={};self.active=None

    def expanded_size(self, code, depth=0):
        if depth > MAX_SKILLS: raise ValueError('macro depth limit')
        size=0
        for op in code:
            size += 1 if op in PRIMITIVES else self.expanded_size(self.skills[op]['code'],depth+1)
            if size>MAX_EXPANDED:raise ValueError('expanded program limit')
        return size

    def execute(self, code, x, depth=0):
        if depth>MAX_SKILLS:raise ValueError('macro depth limit')
        for op in code:
            if op in PRIMITIVES:x=primitive(op,x)
            else:
                r=self.skills[op]
                if r['state']!='validated':raise ValueError('invalid dependency')
                # Macro may be used outside its original scope only as a candidate
                # computation; fresh validation of the enclosing program is mandatory.
                x=self.execute(r['code'],x,depth+1)
        return x

    def begin(self,key,scope,train,audit,budget=2):
        name(key)
        if key in self.skills or len(self.skills)>=MAX_SKILLS or type(budget) is not int or not 1<=budget<=2:
            raise ValueError('invalid skill or search budget')
        if self.active is not None and self.active['state']=='learning':raise ValueError('finish current inquiry')
        if any(not isinstance(z,(list,tuple)) or len(z)>MAX_SCOPE for z in (scope,train,audit)):
            raise ValueError('scope or partition size limit')
        scope=tuple(map(sequence,scope));train=tuple(map(sequence,train));audit=tuple(map(sequence,audit))
        if not 4<=len(scope)<=MAX_SCOPE or len(set(scope))!=len(scope) or not train or len(audit)<2 or len(set(train+audit))!=len(train+audit) or not set(train+audit)<=set(scope) or len(train+audit)>=len(scope):
            raise ValueError('invalid distinct scope and evidence partitions')
        ops=PRIMITIVES+tuple(sorted(k for k,r in self.skills.items() if self.reuse and r['state']=='validated'))
        if key in PRIMITIVES:raise ValueError('reserved skill name')
        candidates={}
        for n in range(budget+1):
            for code in product(ops,repeat=n):
                try:
                    self.expanded_size(code)
                    sig=tuple(self.execute(code,x) for x in scope)
                except ValueError:continue
                candidates.setdefault(sig,code)
        self.active={'key':key,'scope':scope,'train':train,'audit':audit,'candidates':candidates,'initial_candidates':len(candidates),'observed':{},'audited':{},'pending':None,'state':'learning','budget':budget}
        return {'candidates':len(candidates)}

    def next(self):
        a=self.active
        if a is None: return {'action':'idle'}
        if a['state']!='learning':return {'action':a['state']}
        if a['pending'] is not None:return dict(a['pending'])
        c=a['candidates']; scope=a['scope']
        if not c:
            a['state']='failed';return {'action':'failed','reason':'no consistent program'}
        if len(c)>1:
            pool=[x for x in a['train'] if x not in a['observed']]
            if not pool:
                a['state']='failed';return {'action':'failed','reason':'unresolved hypotheses'}
            def score(x):
                i=scope.index(x);counts={}
                for sig in c: counts[sig[i]]=counts.get(sig[i],0)+1
                return len(c)**2-sum(v*v for v in counts.values())
            x=max(pool,key=score)
            if not score(x):
                a['state']='failed';return {'action':'failed','reason':'training pool cannot discriminate'}
            q={'action':'observe','input':x}
        else:
            pool=[x for x in a['audit'] if x not in a['audited']]
            if pool:q={'action':'audit','input':pool[0]}
            else:
                sig,code=next(iter(c.items()))
                self.skills[a['key']]={'code':code,'scope':scope,'state':'validated','dependencies':tuple(op for op in code if op not in PRIMITIVES),'observed':list(a['observed'].items()),'audited':list(a['audited'].items()),'initial_candidates':a['initial_candidates']}
                a['state']='validated';return {'action':'validated','skill':a['key'],'program':code}
        a['pending']=q;return dict(q)

    def answer(self,y):
        a=self.active
        if not a or a['state']!='learning' or a['pending'] is None:raise ValueError('no pending measurement')
        y=sequence(y);q=a['pending'];x=q['input'];i=a['scope'].index(x)
        a['observed' if q['action']=='observe' else 'audited'][x]=y
        a['candidates']={sig:code for sig,code in a['candidates'].items() if sig[i]==y};a['pending']=None

    def deny(self):
        a=self.active
        if not a or a['state']!='learning' or a['pending'] is None:raise ValueError('no pending measurement')
        a['pending']=None;a['state']='failed'

    def predict(self,key,x):
        x=sequence(x);r=self.skills.get(key)
        if not r or r['state']!='validated' or x not in r['scope']:return None
        return self.execute(r['code'],x)

    def plan(self,start,goal,max_steps=8):
        """Shortest proposal over scoped, validated sequence skills only."""
        start=sequence(start);goal=sequence(goal)
        if type(max_steps) is not int or not 0<=max_steps<=8:raise ValueError('planning depth limit')
        queue=deque([(start,[])]);seen={start}
        while queue:
            state,path=queue.popleft()
            if state==goal:return path
            if len(path)>=max_steps:continue
            for key in sorted(self.skills):
                nxt=self.predict(key,state)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt);queue.append((nxt,path+[{'skill':key,'before':state,'after':nxt}]))
        return None

    def challenge(self,key,x,y):
        pred=self.predict(key,x);y=sequence(y)
        if pred is None:raise ValueError('unvalidated or out of scope')
        if pred==y:return True
        invalid={key}
        while True:
            new=invalid|{k for k,r in self.skills.items() if invalid.intersection(r['dependencies'])}
            if new==invalid:break
            invalid=new
        for k in invalid:self.skills[k]['state']='invalid'
        if self.active and self.active['state']=='learning':self.active['state']='failed';self.active['pending']=None
        return False


class General:
    """Language-routed workspace with one replayable journal and retained core."""
    def __init__(self,reuse=True):
        self.reuse=reuse;self.evidence=Evidence();self.programs=Programs(reuse);self.core=Continual();self.events=[]

    def ask(self,text):
        if not isinstance(text,str) or len(text)>512:raise ValueError('query length limit')
        text=text.strip().lower().rstrip('?')
        m=re.fullmatch(rf'is ({NAME}) ({"|".join(RELATIONS)}) ({NAME})',text)
        if m:return self.evidence.ask(*m.groups())
        m=re.fullmatch(rf'which items are ({"|".join(RELATIONS)}) ({NAME})',text)
        if m:
            rel,target=m.groups();entities=sorted({v for rows in self.evidence.sources.values() for a,_,b,_ in rows for v in (a,b)})
            results=[(e,self.evidence.ask(e,rel,target)) for e in entities]
            return {'status':'answer','matches':[{'entity':e,'evidence':r} for e,r in results if r.get('value') is True],
                    'conflicts':[{'entity':e,'evidence':r} for e,r in results if r['status']=='conflict'],
                    'unknown_count':sum(r['status']=='unknown' for _,r in results),'scope':'named entities in loaded documents'}
        m=re.fullmatch(rf'if ({NAME}) is ({"|".join(RELATIONS)}) ({NAME}) then (.+)',text)
        if m:
            condition=self.evidence.ask(*m.groups()[:3])
            if condition.get('value') is not True or condition['status']!='answer':
                return {'status':'unknown','reason':'condition not established','condition':condition}
            # Bounded decomposition: no nested conditional chains.
            if m[4].startswith('if '):return {'status':'unsupported'}
            return {'status':'conditional','condition':condition,'result':self.ask(m[4])}
        m=re.fullmatch(rf'apply ({NAME}) to result of calculate (.+)',text)
        if m:
            value=Fraction(math_answer(m[2])['value'])
            if value.denominator!=1:return {'status':'unknown','reason':'integer input required'}
            y=self.programs.predict(m[1],[integer(value.numerator)])
            return {'status':'unknown'} if y is None else {'status':'answer','value':y,'intermediate':str(value),'skill':m[1]}
        if text.startswith('calculate '):return math_answer(text[10:])
        if text.startswith('solve '):return math_answer(text[6:],True)
        m=re.fullmatch(r'plan from (\[[^\]]*\]) to (\[[^\]]*\])',text)
        if m:
            path=self.programs.plan(json.loads(m[1]),json.loads(m[2]))
            return {'status':'unknown','reason':'no scoped plan'} if path is None else {'status':'proposal','steps':path}
        m=re.fullmatch(rf'apply ({NAME}) to (\[[^\]]*\])',text)
        if m:
            y=self.programs.predict(m[1],json.loads(m[2]))
            return {'status':'unknown'} if y is None else {'status':'answer','value':y,'skill':m[1]}
        m=re.fullmatch(rf'apply ({NAME}) to core ({NAME}) at width (\d+) with (\d+) and (\d+)',text)
        if m:
            skill,episode,w,a,b=m.groups();r=self.core.records.get(episode)
            if not r or r['family']!='word':return {'status':'unknown','reason':'no word skill'}
            from .operators import Domain
            d=Domain(int(w));x=(int(w),d.word(int(a)),d.word(int(b)))
            p=self.core.predict(episode,x)
            y=None if p is None else self.programs.predict(skill,[p])
            return {'status':'unknown'} if y is None else {'status':'answer','value':y,'core_episode':episode,'intermediate':p,'skill':skill}
        return {'status':'unsupported','reason':'outside controlled language grammar'}

    def execute(self,request):
        if not isinstance(request,dict):raise ValueError('expected command object')
        # Roundtrip protects the journal from caller mutation and rejects NaNs.
        req=json.loads(canonical(request)); a=dict(req);cmd=a.pop('command')
        if cmd=='ask':return self.ask(**a)
        if cmd=='solve_constraints':return solve_constraints(**a)
        if cmd=='status':return {'sources':sorted(self.evidence.sources),'skills':{k:r['state'] for k,r in self.programs.skills.items()},'core_phase':self.core.phase}
        if len(self.events)>=MAX_EVENTS:raise ValueError('workspace journal limit')
        if cmd=='ingest':self.evidence.ingest(**a);r={'accepted':True}
        elif cmd=='retract':self.evidence.retract(**a);r={'accepted':True}
        elif cmd=='begin':r=self.programs.begin(**a)
        elif cmd=='next':r=self.programs.next()
        elif cmd=='answer':self.programs.answer(**a);r={'accepted':True}
        elif cmd=='deny':self.programs.deny(**a);r={'denied':True}
        elif cmd=='challenge':r={'consistent':self.programs.challenge(**a)}
        elif cmd=='core':
            if set(a)!={'request'} or not isinstance(a['request'],dict) or a['request'].get('command') not in ('begin','next','answer','deny','challenge','predict','compose_plan','status'):
                raise ValueError('unsupported core command')
            r=CoreConsole(self.core).execute(a['request'])
            if a['request']['command'] in ('predict','compose_plan','status'):return r
        else:raise ValueError('unsupported command')
        self.events.append(req);return r

    def fingerprint(self):
        # Replay includes every candidate-discrimination transition and core event.
        a=self.programs.active
        active=None if a is None else {**{k:v for k,v in a.items() if k not in ('candidates','observed','audited')},'candidates':list(a['candidates'].items()),'observed':list(a['observed'].items()),'audited':list(a['audited'].items())}
        return hash_of({'sources':self.evidence.sources,'skills':self.programs.skills,'active':active,'core':self.core.fingerprint()})

    def save(self,path):
        payload={'version':1,'reuse':self.reuse,'events':self.events,'fingerprint':self.fingerprint()}
        blob=canonical({'payload':payload,'sha256':hash_of(payload)})
        if len(blob)>MAX_FILE:raise ValueError('checkpoint too large')
        path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.ix-general-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:blob=f.read(MAX_FILE+1)
        if len(blob)>MAX_FILE:raise ValueError('checkpoint too large')
        obj=json.loads(blob);p=obj['payload']
        if p['version']!=1 or hash_of(p)!=obj['sha256'] or len(p['events'])>MAX_EVENTS:raise ValueError('checkpoint integrity')
        m=cls(p['reuse'])
        for event in p['events']:m.execute(event)
        if m.fingerprint()!=p['fingerprint']:raise ValueError('checkpoint replay mismatch')
        return m
