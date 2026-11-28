"""Discover and compare independently authored numerical software tools.

Tool-assisted induction is reported separately from native synthesis. Only an
operator-selected, fixed subset of standard-library modules is supported;
function names and signatures come from the actual module at runtime.
"""
import importlib
import inspect
import math
from collections import Counter
from .exploration import freeze

# Pure numerical standard-library interfaces. Exclude general object mutation,
# arbitrary module loading and arbitrary code execution. These are capability
# catalogs, not a list of task solutions; the learner never receives a target name.
MODULES={'math':frozenset(('ceil','floor','trunc','isqrt','gcd','lcm','comb','perm','factorial','copysign','fmod','remainder','pow','hypot','dist','fabs','frexp','ldexp','log','log2','log10','sqrt','exp','expm1','log1p','sin','cos','tan','asin','acos','atan','atan2','sinh','cosh','tanh','degrees','radians','isfinite','isinf','isnan')),
         'operator':frozenset(('add','sub','mul','floordiv','truediv','mod','and_','or_','xor','lshift','rshift','neg','pos','abs','eq','ne','lt','le','gt','ge','not_','truth','invert'))}


def discover(modules,arity):
    if not modules or not set(modules)<=set(MODULES) or not 1<=arity<=4:raise ValueError('unsupported numerical tool scope')
    out=[]
    for name in sorted(set(modules)):
        m=importlib.import_module(name)
        for member,f in inspect.getmembers(m,inspect.isbuiltin):
            if member not in MODULES[name]:continue
            try:
                sig=inspect.signature(f);sig.bind(*([1]*arity))
            except (TypeError,ValueError):continue
            out.append((name+'.'+member,f,str(sig)))
    return out


def call(f,x):
    try:
        if not isinstance(x,(list,tuple)) or any(type(v) is not int or abs(v)>128 for v in x):return None
        # Shift width and factorial are bounded by the input limits; normalize
        # integer outputs only (bool is explicit int), never invoke external AI.
        y=f(*x)
        if type(y) in (int,bool) and abs(y)<=10**9:return int(y)
        if type(y) is float and math.isfinite(y) and y.is_integer() and abs(y)<=10**9:return int(y)
        return None
    except (ArithmeticError,TypeError,ValueError,OverflowError):return None


def inquire(train,pool,audit,*,modules=('math','operator'),query_budget=4):
    if not 2<=len(train)<=128 or not 3<=len(audit.inputs)<=128 or not 0<=query_budget<=32 or len(pool.inputs)>256:raise ValueError('evidence budget')
    obs={freeze(x):freeze(y) for x,y in train};pts=tuple(map(freeze,pool.inputs));aud=tuple(map(freeze,audit.inputs));inputs=tuple(obs)+pts+aud
    if len(obs)!=len(train) or len(set(inputs))!=len(inputs) or not all(type(y) is int for y in obs.values()):raise ValueError('invalid evidence partitions')
    arity=len(inputs[0]) if isinstance(inputs[0],tuple) else 0
    if any(not isinstance(x,tuple) or len(x)!=arity or any(type(v) is not int or abs(v)>128 for v in x) for x in inputs):raise ValueError('integer tool inputs required')
    tools=discover(modules,arity);viable=[(n,f,s) for n,f,s in tools if all(call(f,x)==y for x,y in obs.items())];trace=[];remaining=list(pts)
    for _ in range(min(query_budget,len(remaining))):
        if not viable:break
        def score(x):
            c=Counter(call(f,x) for n,f,s in viable);return len(viable)**2-sum(v*v for v in c.values())
        x=max(remaining,key=score);gain=score(x)
        if not gain:break
        y=pool.observe(x);remaining.remove(x);before=len(viable);viable=[t for t in viable if call(t[1],x)==y];trace.append({'input':x,'observation':y,'before':before,'after':len(viable),'disagreement':gain})
    candidate=viable[0] if viable else None
    ambiguity=bool(candidate and any(len({call(f,x) for n,f,s in viable})>1 for x in remaining+list(aud)))
    predictions=[call(candidate[1],x) for x in aud] if candidate else []
    labels=[];conflict=False
    if candidate:
        from .audit import AuditConflict
        for x in aud:
            try:labels.append(audit.observe(x))
            except AuditConflict:conflict=True;break
    promoted=bool(candidate and not conflict and not ambiguity and None not in predictions and predictions==labels)
    return {'tool_assisted':True,'discovered':[{'name':n,'signature':s} for n,f,s in tools],'remaining':[n for n,f,s in viable],'selected':candidate[0] if promoted else None,'promoted':promoted,'trace':trace,'audit_predictions':predictions,'audit_labels':labels,'reason':'audited' if promoted else 'no_tool' if not candidate else 'audit_conflict' if conflict else 'ambiguity' if ambiguity else 'audit_contradiction'}


def predict(name,x):
    if not isinstance(name,str) or '.' not in name or not isinstance(x,(list,tuple)):return None
    module,member=name.split('.',1)
    if module not in MODULES or member not in MODULES[module]:return None
    return call(getattr(importlib.import_module(module),member),x)


class ToolMemory:
    def __init__(self):self.skills={};self.history=[]
    def learn(self,key,train,pool,audit,*,modules,query_budget=4):
        if not isinstance(key,str) or not key or len(key)>80 or key in self.skills or len(self.skills)>=128:raise ValueError('tool key')
        report=inquire(train,pool,audit,modules=modules,query_budget=query_budget);report['key']=key
        if hasattr(audit,'log'):report['audit_source_votes']=audit.log
        self.skills[key]={'selected':report['selected'],'arity':len(train[0][0]),'valid':report['promoted']};self.history.append(report);return report
    def predict(self,key,x):
        r=self.skills[key]
        return predict(r['selected'],x) if r['valid'] and isinstance(x,(tuple,list)) and len(x)==r['arity'] else None
    def challenge(self,key,x,expected):
        if type(expected) is not int or not isinstance(x,(list,tuple)) or len(x)!=self.skills[key]['arity'] or any(type(v) is not int or abs(v)>128 for v in x):raise ValueError('integer observation within tool scope required')
        y=self.predict(key,x)
        if y!=expected:self.skills[key]['valid']=False
        return {'prediction':y,'expected':expected,'consistent':y==expected,'tool_assisted':True}
