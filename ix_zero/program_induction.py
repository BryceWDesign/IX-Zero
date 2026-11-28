"""Typed finite synthesis with executable state-transition programs.

Candidate programs are immutable data, not host Python. The learner selects
between expressions and iterative register representations from observations.
No task name, reference implementation or final test labels enter the learner.
"""
from collections import Counter, defaultdict
import random
from itertools import product
from .exploration import freeze

OPS = ('add','sub','mul','mod','div','min','max','xor','and','or','eq','lt','le')
LIMIT = 10**9
MAX_SPACE = 30000


def integer(x):
    if type(x) is not int or abs(x)>LIMIT:raise ValueError('integer bound')
    return x


def expression(e,x):
    kind=e[0]
    if kind=='arg':return x[e[1]]
    if kind=='const':return integer(e[1])
    if kind=='call':return integer(run(e[2],x))
    if kind=='inline':return integer(run(e[1],x))
    a=expression(e[1],x)
    if kind=='abs':return abs(a)
    if kind=='neg':return integer(-a)
    b=expression(e[2],x)
    if kind=='add':r=a+b
    elif kind=='sub':r=a-b
    elif kind=='mul':r=a*b
    elif kind=='mod':r=a%b
    elif kind=='div':r=a//b
    elif kind=='min':r=min(a,b)
    elif kind=='max':r=max(a,b)
    elif kind=='xor':r=a^b
    elif kind=='and':r=a&b
    elif kind=='or':r=a|b
    elif kind=='eq':r=int(a==b)
    elif kind=='lt':r=int(a<b)
    elif kind=='le':r=int(a<=b)
    else:raise ValueError('unknown expression')
    return integer(r)


def run(program,x):
    try:
        x=freeze(x)
        if not isinstance(x,tuple) or not 1<=len(x)<=4: return None
        x=tuple(map(integer,x))
        if program[0]=='expr':return expression(program[1],x)
        if program[0]=='revision':
            _,base,transform,out,guard,fallback=program
            if len(x)!=2:return None
            a,b=x
            condition={'never':False,'any_zero':a==0 or b==0,'all_zero':a==0 and b==0,'equal':a==b,'negative':a<0 or b<0}[guard]
            if condition:return integer({'zero':0,'one':1,'first':a,'second':b,'abs_first':abs(a),'abs_second':abs(b)}[fallback])
            args={'identity':x,'abs':(abs(a),abs(b)),'neg':(-a,-b),'swap':(b,a)}[transform]
            y=run(base,args)
            if y is None:return None
            return integer({'identity':y,'abs':abs(y),'neg':-y}[out])
        if program[0]!='loop' or len(x)!=2:return None
        _,guard,update_a,update_b,result=program
        seen=set()
        for _ in range(65):
            a,b=x
            if (guard=='a0' and a==0) or (guard=='b0' and b==0) or (guard=='equal' and a==b):return x[result]
            if x in seen:return None
            seen.add(x)
            x=(expression(update_a,x),expression(update_b,x))
        return None
    except (ValueError,TypeError,IndexError,ZeroDivisionError,OverflowError,RecursionError,KeyError):return None


def expressions(arity,macros=()):
    leaves=list(macros)+[('arg',i) for i in range(arity)]+[('const',c) for c in (0,1,-1,2)]
    yield from leaves
    for op,e in product(('abs','neg'),leaves):yield (op,e)
    for op,a,b in product(OPS,leaves,leaves):
        if op in ('add','mul','min','max','xor','and','or','eq') and a>b:continue
        yield (op,a,b)


def candidates(arity,allow_loops=True,macros=()):
    base=list(expressions(arity,macros))
    for e in base:yield ('expr',e)
    if allow_loops and arity==2:
        # Register transitions are independently generated, not a library of
        # prewritten algorithms. Simultaneous update, finite execution and cycle
        # detection let the candidate discover useful iterative programs.
        a,b=('arg',0),('arg',1)
        updates=[a,b,('const',0),('abs',a),('abs',b)]
        updates += [(op,u,v) for op in OPS[:7] for u,v in ((a,b),(b,a))]
        for guard,u,v,r in product(('a0','b0','equal'),updates,updates,(0,1)):
            if u==a and v==b:continue
            yield ('loop',guard,u,v,r)
    # Compose expressions after trying shallow transition hypotheses.
    args=list(macros)+[('arg',i) for i in range(arity)]
    for op,e,v in product(OPS,base,args):
        yield ('expr',(op,e,v))
        if op not in ('add','mul','min','max','xor','and','or','eq'):
            yield ('expr',(op,v,e))



def search_stream(arity,allow_loops,macros,obs,mode,seed):
    """An independent search control over exactly the generated finite grammar.
    Tournament selection uses training fitness only; single-field mutations stay
    within the enumerated candidate space. Not a DreamCoder or PushGP port.
    """
    def unique_space():
        seen=set()
        for p in candidates(arity,allow_loops,macros):
            if p in seen:continue
            if len(seen)>=MAX_SPACE:return
            seen.add(p);yield p
    if mode=='enumerative':yield from unique_space();return
    if mode not in ('random','evolutionary'):raise ValueError('search mode')
    rng=random.Random(seed);space=list(unique_space());rng.shuffle(space)
    if mode=='random':yield from space;return
    # Each program is mapped to top-level syntax slots. Candidates sharing all
    # but one slot are legal point mutations of one another.
    def slots(p):return ('expr',*p[1]) if p[0]=='expr' else p
    def keys(p):
        s=slots(p)
        return [(len(s),i,s[:i]+s[i+1:]) for i in range(1,len(s))]
    index=defaultdict(list)
    for p in space:
        for key in keys(p):index[key].append(p)
    population=[];seen=set();cursor=0
    while len(seen)<len(space):
        chosen=None
        if len(population)>=12 and rng.random()<0.85:
            parent=max(rng.sample(population,min(4,len(population))),key=lambda z:z[0])[1]
            ks=keys(parent);rng.shuffle(ks)
            for k in ks:
                neighbours=index[k]
                for _ in range(min(10,len(neighbours))):
                    v=rng.choice(neighbours)
                    if v not in seen:chosen=v;break
                if chosen is not None:break
        if chosen is None:
            while cursor<len(space) and space[cursor] in seen:cursor+=1
            if cursor==len(space):break
            chosen=space[cursor];cursor+=1
        seen.add(chosen);yield chosen
        fitness=sum(run(chosen,x)==y for x,y in obs.items());population.append((fitness,chosen))
        # Keep a small tournament population; tie order is reproducibly random.
        if len(population)>48:
            rng.shuffle(population);population.sort(key=lambda t:-t[0]);del population[48:]

def inquire(train,pool,audit,*,budget=6000,query_budget=4,allow_loops=True,macros=(),search='enumerative',seed=0):
    if type(allow_loops) is not bool or search not in ('enumerative','random','evolutionary') or type(seed) is not int:raise ValueError('search options')
    if type(budget) is not int or not 1<=budget<=30000 or type(query_budget) is not int or not 0<=query_budget<=32:raise ValueError('search budget')
    obs={freeze(x):integer(y) for x,y in train}
    pts=tuple(map(freeze,pool.inputs));audit_pts=tuple(map(freeze,audit.inputs));all_pts=tuple(obs)+pts+audit_pts
    if len(obs)!=len(train) or not 2<=len(obs)<=128 or not 3<=len(audit_pts)<=128 or len(pts)>256 or len(set(all_pts))!=len(all_pts):raise ValueError('evidence partitions')
    arity=len(all_pts[0]) if isinstance(all_pts[0],tuple) else 0
    if not 1<=arity<=4 or any(not isinstance(x,tuple) or len(x)!=arity or any(type(v) is not int for v in x) for x in all_pts):raise ValueError('typed integer tuple required')
    viable=[];seen=set();attempts=0;kinds=Counter()
    for p in search_stream(arity,allow_loops,macros,obs,search,seed):
        if attempts>=budget:break
        attempts+=1;kinds[p[0]]+=1
        if any(run(p,x)!=y for x,y in obs.items()):continue
        sig=tuple(run(p,x) for x in all_pts)
        # equivalence only on exposed inputs, not universal equivalence.
        if sig not in seen:seen.add(sig);viable.append(p)
    remaining=list(pts);trace=[]
    for _ in range(min(query_budget,len(remaining))):
        if not viable:break
        def score(x):
            c=Counter(run(p,x) for p in viable);return len(viable)**2-sum(n*n for n in c.values())
        x=max(remaining,key=score);gain=score(x)
        if not gain:break
        y=integer(pool.observe(x));before=len(viable);viable=[p for p in viable if run(p,x)==y];remaining.remove(x)
        trace.append({'input':x,'observation':y,'disagreement':gain,'before':before,'after':len(viable)})
    candidate=viable[0] if viable else None
    ambiguity=bool(candidate and any(len({run(p,x) for p in viable})>1 for x in remaining+list(audit_pts)))
    predictions=[run(candidate,x) for x in audit_pts] if candidate else []
    labels=[];conflict=False
    if candidate:
        from .audit import AuditConflict
        for x in audit_pts:
            try:labels.append(integer(audit.observe(x)))
            except AuditConflict:conflict=True;break
    accepted=bool(candidate and not conflict and not ambiguity and predictions==labels and None not in predictions)
    return {'search':search,'seed':seed,'attempted':attempts,'attempted_by_kind':dict(kinds),'remaining':len(viable),'queries':trace,'audit_predictions':predictions,'audit_labels':labels,'promoted':accepted,'program':candidate if accepted else None,'reason':'audited' if accepted else 'representation_or_search_gap' if not candidate else 'audit_conflict' if conflict else 'ambiguity' if ambiguity else 'audit_contradiction'}


def dependencies(program):
    if not isinstance(program,(tuple,list)):return set()
    found={program[1]} if program and program[0]=='call' else set()
    for item in program:
        if isinstance(item,(list,tuple)):found.update(dependencies(item))
    return found

def macro_depth(program):
    if not isinstance(program,(tuple,list)):return 0
    child=max((macro_depth(item) for item in program if isinstance(item,(list,tuple))),default=0)
    return child+int(bool(program) and program[0] in ('call','inline','revision'))


class ProgramMemory:
    """A developmental search vocabulary; contradiction withdraws dependents.
    Only programs admitted through observation/audit can become callable leaves.
    The host evaluator supplies labels and retains the final holdout.
    """
    def __init__(self):self.skills={};self.history=[]
    def learn(self,key,train,pool,audit,*,reuse=True,**options):
        if not isinstance(key,str) or not key or len(key)>80 or key in self.skills or len(self.skills)>=128 or 'macros' in options:raise ValueError('program skill identity or options')
        arity=len(train[0][0]) if train else 0
        macros=tuple(('call',k,v['program']) for k,v in self.skills.items() if reuse and v['valid'] and v['arity']==arity and macro_depth(v['program'])<3)[:8]
        report=inquire(train,pool,audit,macros=macros,**options)
        program=report['program'];used=sorted(dependencies(program));report.update(key=key,dependencies=used,available_macros=[m[1] for m in macros])
        if hasattr(audit,'log'):report['audit_source_votes']=audit.log
        self.skills[key]={'program':program,'valid':report['promoted'],'arity':arity,'dependencies':used};self.history.append(report)
        return report
    def predict(self,key,x):
        entry=self.skills[key]
        return run(entry['program'],x) if entry['valid'] and isinstance(x,(tuple,list)) and len(x)==entry['arity'] else None
    def challenge(self,key,x,expected):
        integer(expected);entry=self.skills[key]
        point=freeze(x)
        if not isinstance(point,tuple) or len(point)!=entry['arity'] or any(type(v) is not int for v in point):raise ValueError('challenge scope')
        value=self.predict(key,point);withdrawn=[]
        if value!=expected:
            for k,v in self.skills.items():
                if v['valid'] and (k==key or key in v['dependencies']):
                    v['valid']=False;withdrawn.append(k)
        return {'consistent':value==expected,'prediction':value,'expected':expected,'withdrawn':withdrawn}


def inline_program(code):
    """Copy the previous executable body, preserving semantics but removing
    reliance on a revoked library key. The new body must be audited afresh.
    """
    if not isinstance(code,(tuple,list)):return code
    if code and code[0]=='call':return ('inline',inline_program(code[2]))
    return tuple(inline_program(item) if isinstance(item,(tuple,list)) else item for item in code)


def propose_revisions(base):
    base=inline_program(base)
    for transform,out in product(('identity','abs','neg','swap'),('identity','abs','neg')):
        yield ('revision',base,transform,out,'never','zero')
        for guard,fallback in product(('any_zero','all_zero','equal','negative'),('zero','one','first','second','abs_first','abs_second')):
            yield ('revision',base,transform,out,guard,fallback)


def revise(memory,base_key,key,train,pool,audit,*,query_budget=8):
    """Select an alternative executable implementation after new observations.
    Base code may be withdrawn. Independent fresh validation is required;
    final holdout remains external. The previous artifact is not edited.
    """
    if not isinstance(key,str) or not key or len(key)>80 or key in memory.skills or len(memory.skills)>=128 or type(query_budget) is not int or not 0<=query_budget<=32:raise ValueError('revision key or budget')
    entry=memory.skills[base_key];base=entry['program']
    if base is None or entry['arity']!=2 or macro_depth(base)>=3:raise ValueError('no supported base program')
    obs={freeze(x):integer(y) for x,y in train};pts=tuple(map(freeze,pool.inputs));aud=tuple(map(freeze,audit.inputs));all_pts=tuple(obs)+pts+aud
    if len(obs)!=len(train) or not 2<=len(obs)<=128 or not 3<=len(aud)<=128 or len(pts)>256 or len(set(all_pts))!=len(all_pts) or any(not isinstance(x,tuple) or len(x)!=2 or any(type(v) is not int for v in x) for x in all_pts):raise ValueError('revision evidence partitions')
    viable=[];seen=set();attempts=0
    for p in propose_revisions(base):
        attempts+=1
        if any(run(p,x)!=y for x,y in obs.items()):continue
        sig=tuple(run(p,x) for x in all_pts)
        if sig not in seen:seen.add(sig);viable.append(p)
    trace=[];remaining=list(pts)
    for _ in range(min(query_budget,len(remaining))):
        if not viable:break
        def score(x):
            c=Counter(run(p,x) for p in viable);return len(viable)**2-sum(n*n for n in c.values())
        x=max(remaining,key=score);gain=score(x)
        if not gain:break
        y=integer(pool.observe(x));remaining.remove(x);before=len(viable);viable=[p for p in viable if run(p,x)==y];trace.append({'input':x,'observed':y,'before':before,'after':len(viable),'disagreement':gain})
    candidate=viable[0] if viable else None
    ambiguous=bool(candidate and any(len({run(p,x) for p in viable})>1 for x in remaining+list(aud)))
    predictions=[run(candidate,x) for x in aud] if candidate else [];labels=[];conflict=False
    if candidate:
        from .audit import AuditConflict
        for x in aud:
            try:labels.append(integer(audit.observe(x)))
            except AuditConflict:conflict=True;break
    promoted=bool(candidate and not conflict and not ambiguous and predictions==labels and None not in predictions)
    r={'key':key,'derived_from':base_key,'revision':True,'attempted':attempts,'remaining':len(viable),'queries':trace,'audit_predictions':predictions,'audit_labels':labels,'promoted':promoted,'program':candidate if promoted else None,'dependencies':[],'available_macros':[],'reason':'audited_revision' if promoted else 'no_valid_revision' if not candidate else 'audit_conflict' if conflict else 'ambiguity' if ambiguous else 'audit_contradiction'}
    if hasattr(audit,'log'):r['audit_source_votes']=audit.log
    memory.skills[key]={'program':r['program'],'arity':2,'valid':promoted,'dependencies':[]};memory.history.append(r);return r
