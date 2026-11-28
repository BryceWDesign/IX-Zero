"""Finite, inspectable cognitive exploration. No target labels enter search except
through explicit observations. Programs are data, not arbitrary host code.
"""
from collections import Counter, deque
from itertools import product
import hashlib
import json
import math
import random


def freeze(x,depth=0):
    if depth>4:raise ValueError('observation nesting limit')
    if type(x) is int and abs(x) <= 10**9:
        return x
    if isinstance(x, str) and len(x) <= 80:
        return x
    if isinstance(x, (tuple, list)) and len(x) <= 900:
        return tuple(freeze(v,depth+1) for v in x)
    raise ValueError('unsupported observation')


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def grid(x):
    if not isinstance(x, tuple) or not x or len(x)>30 or not isinstance(x[0], tuple) or not x[0] or len(x[0])>30 or any(not isinstance(r,tuple) or len(r)!=len(x[0]) or any(type(v) is not int or not 0<=v<=9 for v in r) for r in x):
        raise ValueError('not a rectangular color grid')
    return x


def apply_atom(op, x, grammar):
    if op in grammar:
        return grammar[op][x]
    if op == 'id': return x
    if op in ('inc','dec','neg','double','square','abs','mod2'):
        if type(x) is not int: raise ValueError('not an integer')
        return freeze({'inc':lambda:x+1,'dec':lambda:x-1,'neg':lambda:-x,'double':lambda:2*x,'square':lambda:x*x,'abs':lambda:abs(x),'mod2':lambda:x%2}[op]())
    if op in ('reverse','sort','unique','length','sum','first','last'):
        if not isinstance(x,tuple) or not all(type(v) is int for v in x): raise ValueError('not an integer sequence')
        if op=='reverse': return x[::-1]
        if op=='sort': return tuple(sorted(x))
        if op=='unique': return tuple(dict.fromkeys(x))
        if op=='length': return len(x)
        if op=='sum': return freeze(sum(x))
        if not x: raise ValueError('empty sequence')
        return x[0] if op=='first' else x[-1]
    g=grid(x)
    if op=='hflip':return tuple(r[::-1] for r in g)
    if op=='vflip':return g[::-1]
    if op=='transpose':return tuple(zip(*g))
    if op=='rotate':return tuple(zip(*g[::-1]))
    if op=='crop':
        cells=[(i,j) for i,r in enumerate(g) for j,v in enumerate(r) if v]
        if not cells:raise ValueError('empty foreground')
        rows,cols=zip(*cells);return tuple(r[min(cols):max(cols)+1] for r in g[min(rows):max(rows)+1])
    if op=='tileh':return grid(tuple(r+r for r in g))
    if op=='tilev':return grid(g+g)
    if op.startswith('color:'):
        _,a,b=op.split(':');return tuple(tuple(int(b) if v==int(a) else v for v in r) for r in g)
    raise ValueError('unknown primitive')


ATOMS=('id','inc','dec','neg','double','square','abs','mod2','reverse','sort','unique','length','sum','first','last','hflip','vflip','transpose','rotate','crop','tileh','tilev')


def parse_grammar(spec):
    """Independent authors may supply named finite transition tables, not Python."""
    if not isinstance(spec,dict) or len(spec)>32:raise ValueError('grammar size')
    out={}
    for name,rows in spec.items():
        if not isinstance(name,str) or not name.startswith('external:') or len(name)>80 or not isinstance(rows,list) or len(rows)>1000:raise ValueError('invalid grammar')
        table={}
        for a,b in rows:
            a,b=freeze(a),freeze(b)
            if a in table:raise ValueError('duplicate transition')
            table[a]=b
        out[name]=table
    return out


def execute(code,x,grammar):
    try:
        for op in code:x=apply_atom(op,x,grammar)
        return freeze(x)
    except (ValueError,KeyError,TypeError,OverflowError):return None


def wilson(success,n,z=1.96):
    if not n:return 0.0
    p=success/n
    return (p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)


def candidates(atoms,macros,depth,budget):
    """Breadth-first candidate calls; equal budget counts attempted programs."""
    calls=0;seen=set(); alphabet=[tuple(m) for m in macros]+[(a,) for a in atoms]
    for n in range(1,depth+1):
        for chunks in product(alphabet,repeat=n):
            code=tuple(a for c in chunks for a in c)
            if code in seen or len(code)>12:continue
            if calls>=budget:return
            seen.add(code);calls+=1;yield code


class Learner:
    """Active version-space learner with developmental macro reuse and diagnostics."""
    def __init__(self):
        self.library={};self.history=[]

    def fit(self,train,pool,audit,*,grammar=None,depth=2,budget=500,query_budget=2,mode='active',reuse=True,seed=0,strict=True):
        # Pool and audit are host callbacks returning separately sourced labels.
        # Only pool INPUTS are visible during selection. Audit inputs are disjoint.
        grammar=parse_grammar(grammar or {})
        if mode not in ('active','random','enumerative') or not 1<=depth<=4 or not 1<=budget<=20000 or not 0<=query_budget<=100:raise ValueError('search limits')
        obs={}
        for a,b in train:
            a,b=freeze(a),freeze(b)
            if a in obs:raise ValueError('duplicate train input')
            obs[a]=b
        if not obs or len(obs)>256:raise ValueError('train size')
        points=tuple(map(freeze,pool.inputs)); aud=tuple(map(freeze,audit.inputs))
        if len(points)>256 or not 2<=len(aud)<=256 or len(set(points+aud))!=len(points+aud) or set(obs)&set(points+aud):raise ValueError('overlapping evidence partitions')
        # Infer representation from input values, not a supplied task-family tag.
        x=next(iter(obs));
        if grammar:atoms=tuple(sorted(grammar))
        elif type(x) is int:atoms=ATOMS[:8]
        elif isinstance(x,tuple) and all(type(v) is int for v in x):atoms=('id',)+ATOMS[8:15]
        elif isinstance(x,tuple) and x and isinstance(x[0],tuple):
            atoms=('id',)+ATOMS[15:]+tuple(f'color:{a}:{b}' for a in range(10) for b in range(10) if a!=b)
        else:atoms=('id',)
        scope=digest({'grammar':[(k,sorted(v.items(),key=lambda r:repr(r[0]))) for k,v in sorted(grammar.items())],'atoms':atoms})
        old=[tuple(r['code']) for r in self.library.values() if r['scope']==scope and r['valid']] if reuse else []
        # Prefer already acquired macros before raw atoms at each length; baseline uses none.
        raw=list(candidates(atoms,old,depth,budget))
        valid=[p for p in raw if all(execute(p,a,grammar)==b for a,b in obs.items())]
        rng=random.Random(seed);trace=[]; remaining=list(points)
        for _ in range(min(query_budget,len(remaining))):
            if not valid:break
            def disagreement(a):
                counts=Counter(execute(p,a,grammar) for p in valid)
                return len(valid)**2-sum(v*v for v in counts.values())
            point=max(remaining,key=disagreement) if mode=='active' else rng.choice(remaining) if mode=='random' else remaining[0]
            if mode=='active' and disagreement(point)==0:break
            before=len(valid);prediction_counts=Counter(execute(p,point,grammar) for p in valid)
            answer=freeze(pool.observe(point)); remaining.remove(point);obs[point]=answer
            valid=[p for p in valid if execute(p,point,grammar)==answer]
            trace.append({'input':point,'answer':answer,'before':before,'after':len(valid),'predictions':[(k,v) for k,v in prediction_counts.items()]})
        # Freeze candidate before reading audit labels. Never select using audit.
        code=valid[0] if valid else None
        ambiguous=bool(code and any(len({execute(p,a,grammar) for p in valid})>1 for a in remaining+list(aud)))
        predictions=[execute(code,a,grammar) for a in aud] if code else [None]*len(aud)
        labels=[]; audit_conflicts=[]
        if code:
            for a in aud:
                try:labels.append(freeze(audit.observe(a)))
                except ValueError as exc:
                    labels.append(None);audit_conflicts.append({'input':a,'reason':str(exc)})
        hits=sum(v is not None and p==v for p,v in zip(predictions,labels))
        promoted=bool(code and hits==len(aud) and (not strict or not ambiguous))
        reason='validated_on_audit' if promoted else 'representation_or_search_gap' if not code else 'audit_contradiction' if hits!=len(aud) else 'unresolved_ambiguity'
        rec={'scope':scope,'code':code,'attempted':len(raw),'remaining_candidates':len(valid),'trace':trace,'audit_inputs':aud,'audit_predictions':predictions,'audit_labels':labels,'audit_hits':hits,'audit_conflicts':audit_conflicts,'audit_n':len(labels),'audit_wilson_lower':wilson(hits,len(labels)),'promoted':promoted,'reason':reason,'ambiguous':ambiguous,'reuse':reuse,'macro_count':len(old)}
        if promoted:
            key=digest({'scope':scope,'code':code});self.library[key]={'scope':scope,'code':code,'valid':True,'atoms':atoms,'grammar_fingerprint':digest([(k,sorted(v.items(),key=lambda r:repr(r[0]))) for k,v in sorted(grammar.items())])};rec['skill']=key
        self.history.append(rec)
        return rec

    def challenge(self,key,x,y,grammar=None):
        r=self.library[key];g=parse_grammar(grammar or {})
        if r['grammar_fingerprint']!=digest([(k,sorted(v.items(),key=lambda r:repr(r[0]))) for k,v in sorted(g.items())]):raise ValueError('grammar mismatch')
        p=execute(r['code'],freeze(x),g)
        if p!=freeze(y):
            # Conservatively invalidate all macros containing the contradicted contiguous program.
            needle=tuple(r['code'])
            for other in self.library.values():
                code=tuple(other['code'])
                if other['scope']==r['scope'] and any(code[i:i+len(needle)]==needle for i in range(len(code)-len(needle)+1)):other['valid']=False
        return p==freeze(y)

    def predict(self,key,x,grammar=None):
        r=self.library[key];g=parse_grammar(grammar or {})
        if r['grammar_fingerprint']!=digest([(k,sorted(v.items(),key=lambda r:repr(r[0]))) for k,v in sorted(g.items())]):raise ValueError('grammar mismatch')
        return execute(r['code'],freeze(x),g) if r['valid'] else None

    def self_model(self):
        return {'inquiries':len(self.history),'validated':sum(r['promoted'] for r in self.history),'gaps':dict(Counter(r['reason'] for r in self.history if not r['promoted'])),'active_macros':sum(r['valid'] for r in self.library.values()),'limitation':'finite candidate space; audit is evidence, not independent certification'}


class Measurements:
    """Host measurement boundary. No labels supplied through .inputs."""
    def __init__(self,rows):
        if not isinstance(rows,(list,tuple)) or len(rows)>256:raise ValueError('measurement size')
        self._values={freeze(a):freeze(b) for a,b in rows};self.inputs=tuple(self._values);self.asked=[]
        if len(rows)!=len(self._values):raise ValueError('duplicate measurements')
    def observe(self,x):
        self.asked.append(x);return self._values[x]


def route(start,goal,edges,assumed=(),forbidden=()):
    """Search goal satisfaction, not prescribed action sequence. Host supplies
    actual accessible transitions; optional assumptions are tested, not enforced."""
    if len(edges)>10000:raise ValueError('graph too large')
    start,goal=freeze(start),freeze(goal);blocked=set(map(freeze,assumed));hard=set(map(freeze,forbidden));adj={}
    for a,action,b in edges:adj.setdefault(freeze(a),[]).append((freeze(action),freeze(b)))
    def bfs(restrict):
        q=deque([(start,[])]);seen={start}
        while q:
            node,path=q.popleft()
            if node==goal:return path
            for action,nxt in adj.get(node,[]):
                if action in restrict or nxt in seen:continue
                seen.add(nxt);q.append((nxt,path+[(node,action,nxt)]))
        return None
    expected=bfs(blocked|hard);alternative=bfs(hard)
    return {'status':'proposal' if alternative is not None else 'unreachable','path':alternative,'constrained_path':expected,'reframed':alternative is not None and (expected is None or len(alternative)<len(expected))}
