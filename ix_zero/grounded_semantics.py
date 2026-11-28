"""Experimental grounded action-language induction and planning.

Learn executable, role-relative transition rules from observed world changes.
No host code generation, hidden goal oracle, or natural-language competence claim.
The representation is an authored relational object/attribute substrate; rule
expressions, lexical slot order, and role-relative writes are induced from data.
"""
from collections import Counter, deque
from itertools import product, permutations
import random

MAX_VALUE = 10**6
MAX_OBJECTS = 12
MAX_FIELDS = 4


def token(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 64 or any(c.isspace() for c in value):
        raise ValueError('single opaque token required')
    return value


def world(raw):
    if not isinstance(raw, dict) or not 2 <= len(raw) <= MAX_OBJECTS:
        raise ValueError('world requires 2–12 named objects')
    schema = None
    for name, fields in raw.items():
        token(name)
        if not isinstance(fields, dict) or not 1 <= len(fields) <= MAX_FIELDS:
            raise ValueError('integer attribute mapping required')
        for f, v in fields.items():
            token(f)
            if type(v) is not int or abs(v) > MAX_VALUE:
                raise ValueError('attribute integer outside domain')
        if schema is None:
            schema = set(fields)
        elif set(fields) != schema:
            raise ValueError('inconsistent attribute schema')
    return {n: dict(sorted(fs.items())) for n, fs in sorted(raw.items())}


def state_key(s):
    return tuple((n, tuple(sorted(fs.items()))) for n, fs in sorted(s.items()))


def utterance(s, words):
    if not isinstance(words, (list, tuple)) or not 2<=len(words)<=4:
        raise ValueError('one verb and one to three entity tokens required')
    words = tuple(map(token, words))
    outside = [i for i,w in enumerate(words) if w not in s]
    if len(outside) != 1:
        raise ValueError('one action token and grounded entity tokens required')
    slot = outside[0]
    roles = tuple(w for i,w in enumerate(words) if i != slot)
    if len(set(roles))!=len(roles):
        raise ValueError('distinct entity roles required')
    return words[slot], slot, roles


def input_case(raw):
    if not isinstance(raw, (list,tuple)) or len(raw)!=2:
        raise ValueError('expected [world, utterance]')
    s, words = world(raw[0]), raw[1]
    verb, slot, roles = utterance(s, words)
    return s, tuple(words), verb, slot, roles


def input_key(case):
    s, words, *_ = case
    return state_key(s), words


def labelled(raw):
    if not isinstance(raw,(list,tuple)) or len(raw)!=2:
        raise ValueError('expected [input, observed_world]')
    case, out = input_case(raw[0]), world(raw[1])
    if set(case[0])!=set(out) or any(set(case[0][n])!=set(out[n]) for n in out):
        raise ValueError('object creation or schema change not supported')
    return case, out


def eval_expr(expr, s, roles):
    op, *args = expr
    if op=='field': return s[roles[args[0]]][args[1]]
    if op=='const': return args[0]
    if op=='if':
        pred,a,b,yes,no=args
        x,y=eval_expr(a,s,roles),eval_expr(b,s,roles)
        return eval_expr(yes if (x==y if pred=='eq' else x<y) else no,s,roles)
    a,b=(eval_expr(x,s,roles) for x in args)
    if op=='add':return a+b
    if op=='sub':return a-b
    if op=='min':return min(a,b)
    if op=='max':return max(a,b)
    raise ValueError('unknown expression')


def expressions(refs, constants, target):
    leaves = refs+tuple(('const',c) for c in constants)
    yield target
    yield from leaves
    for op,a,b in product(('add','sub','min','max'), leaves, leaves):
        if op!='sub' and a>b:continue
        yield (op,a,b)
    # Conditional effects: a generated role-relative predicate controls a write.
    # This is one authored composition grammar, not unrestricted program search.
    for pred,a,b in product(('eq','lt'),refs,refs):
        if a==b or (pred=='eq' and a>b):continue
        for value in leaves:
            yield ('if',pred,a,b,value,target)


def model_arity(model):
    return len(model['writes'])//len(model['fields'])


def apply_model(model, raw, words):
    try:
        s=world(raw);verb,slot,roles=utterance(s,words)
        if verb!=model['verb'] or slot!=model['slot'] or len(roles)!=model_arity(model) or tuple(next(iter(s.values())))!=tuple(model['fields']):return None
        out={n:dict(fs) for n,fs in s.items()}
        for i,f,expr in model['writes']:
            value=eval_expr(expr,s,roles)
            if type(value) is not int or abs(value)>MAX_VALUE:return None
            out[roles[i]][f]=value
        return out
    except (ValueError,KeyError,TypeError,IndexError):return None


def fit(train, pool, audit, *, query_budget=16, selection='active', seed=0, conditional=True):
    """Pool and audit are measurement objects with inputs and observe(input).
    Train labels only are visible during rule generation. Audit labels are
    consumed only after candidate freeze. Finite observed-equivalence is not
    universal equivalence. Search factors writes to avoid Cartesian explosion.
    """
    if not 2<=len(train)<=128 or not 3<=len(audit.inputs)<=128 or len(pool.inputs)>256:
        raise ValueError('evidence partition size')
    if type(query_budget) is not int or not 0<=query_budget<=32 or selection not in ('active','random','fixed') or type(seed) is not int or type(conditional) is not bool:
        raise ValueError('inquiry options')
    rows=[labelled(row) for row in train]
    p=[input_case(x) for x in pool.inputs];a=[input_case(x) for x in audit.inputs]
    cases=[c for c,_ in rows]+p+a
    keys=[input_key(c) for c in cases]
    if len(set(keys))!=len(keys):raise ValueError('overlapping evidence partitions')
    s,_,verb,slot,roles=cases[0];arity=len(roles);fields=tuple(next(iter(s.values())))
    if any(c[2]!=verb or c[3]!=slot or len(c[4])!=arity or tuple(next(iter(c[0].values())))!=fields for c in cases):
        raise ValueError('one action lexeme, word order and field schema per fit')
    result={'verb':verb,'slot':slot,'fields':fields,'selection':selection,'conditional':conditional,'candidate_attempts':0,'queries':[],'audit':[],'promoted':False,'model':None}
    def target_value(out,case,target):return out[case[4][target[0]]][target[1]]
    def valid_out(case,out):
        if set(case[0])!=set(out) or any(set(case[0][n])!=set(out[n]) for n in out):return False
        return all(case[0][n]==out[n] for n in out if n not in case[4])
    if any(not valid_out(c,y) for c,y in rows):
        result['reason']='nonlocal_or_schema_effect';return result
    targets=tuple(product(range(arity),fields));refs=tuple(('field',i,f) for i,f in targets)
    constants={-1,0,1,2}
    # Constant values suggested by consistent training targets, not holdout labels.
    for t in targets:
        vals={target_value(y,c,t) for c,y in rows}
        if len(vals)==1:constants.update(vals)
    constants=tuple(sorted(constants));viable={};ntrain=len(rows)
    for t,ref in zip(targets,refs):
        candidates=[];seen_expr=set();seen_sig=set()
        for e in expressions(refs,constants,ref):
            if not conditional and e[0]=='if':continue
            if e in seen_expr:continue
            seen_expr.add(e);result['candidate_attempts']+=1
            if any(eval_expr(e,c[0],c[4])!=target_value(y,c,t) for c,y in rows):continue
            sig=tuple(eval_expr(e,c[0],c[4]) for c in cases[ntrain:])
            if sig not in seen_sig:seen_sig.add(sig);candidates.append((e,sig))
        viable[t]=candidates
    result['initial_alternatives']=[len(v) for v in viable.values()]
    if any(not v for v in viable.values()):result['reason']='representation_gap';return result
    remaining=list(range(len(p)));rng=random.Random(seed)
    def gain(j):
        total=0.0
        for v in viable.values():
            counts=Counter(sig[j] for _,sig in v);n=len(v)
            total+=1-sum((k/n)**2 for k in counts.values())
        return total
    for _ in range(min(query_budget,len(p))):
        if not remaining:break
        j=max(remaining,key=gain) if selection=='active' else rng.choice(remaining) if selection=='random' else remaining[0]
        g=gain(j)
        if selection=='active' and g==0:break
        c=p[j];out=world(pool.observe([c[0],list(c[1])]))
        result['queries'].append({'input':[c[0],c[1]],'observed':out,'disagreement':g})
        if not valid_out(c,out):result['reason']='nonlocal_or_schema_effect';return result
        for t in targets:viable[t]=[(e,sig) for e,sig in viable[t] if sig[j]==target_value(out,c,t)]
        remaining.remove(j)
        if any(not v for v in viable.values()):result['reason']='contradiction';return result
    result['remaining_alternatives']=[len(v) for v in viable.values()]
    if any(len({sig[j] for _,sig in v})>1 for v in viable.values() for j in remaining+list(range(len(p),len(p)+len(a)))):
        result['reason']='ambiguous';return result
    model={'verb':verb,'slot':slot,'fields':fields,'writes':[(i,f,viable[i,f][0][0]) for i,f in targets]}
    # Freeze before observing independent audit outcomes.
    result['candidate']=model
    for c in a:
        pred=apply_model(model,c[0],c[1]);out=world(audit.observe([c[0],list(c[1])]))
        result['audit'].append({'input':[c[0],c[1]],'predicted':pred,'observed':out})
    result['promoted']=all(r['predicted']==r['observed'] for r in result['audit'])
    result['reason']='audited' if result['promoted'] else 'audit_contradiction'
    if result['promoted']:result['model']=model
    return result


class RecordedMeasurements:
    """Offline replay adapter. The learner reads labels only through observe."""
    def __init__(self, rows):
        self.inputs=[];self._values={};self.log=[]
        for row in rows:
            c,y=labelled(row);k=input_key(c)
            if k in self._values:raise ValueError('duplicate measurement')
            self.inputs.append([c[0],list(c[1])]);self._values[k]=y
    def observe(self, raw):
        k=input_key(input_case(raw));self.log.append(k)
        return self._values[k]


class SemanticMemory:
    def __init__(self):self.models={};self.history=[];self.dependencies={}
    def learn(self,train,pool,audit,**options):
        self._check_new(train)
        r=fit(train,pool,audit,**options);verb=r['verb']
        if verb in self.models:raise ValueError('lexeme exists; use a fresh scope')
        self.models[verb]=r['model'];self.dependencies[verb]=[];self.history.append(r);return r
    def _check_new(self,train):
        if not train or len(self.models)>=128:raise ValueError('semantic memory capacity or empty training')
        if labelled(train[0])[0][2] in self.models:raise ValueError('lexeme exists; use a fresh scope')
    def transfer(self,train,pool,audit,**options):
        self._check_new(train)
        r=transfer_fit(self.models,train,pool,audit,**options);verb=r['verb']
        if verb in self.models:raise ValueError('lexeme exists; use a fresh scope')
        self.models[verb]=r['model'];self.dependencies[verb]=[r['source']] if r['promoted'] else []
        self.history.append(r);return r
    def predict(self,s,words):
        s=world(s);verb,_,_=utterance(s,words);m=self.models.get(verb)
        return apply_model(m,s,words) if m else None
    def challenge(self,s,words,observed):
        s=world(s);verb,_,_=utterance(s,words);pred=self.predict(s,words);observed=world(observed)
        consistent=pred is not None and pred==observed
        withdrawn=[]
        if not consistent and verb in self.models:
            pending=[verb]
            while pending:
                key=pending.pop()
                if key in withdrawn:continue
                self.models[key]=None;withdrawn.append(key)
                pending.extend(k for k,deps in self.dependencies.items() if key in deps and self.models.get(k) is not None)
        r={'verb':verb,'consistent':consistent,'prediction':pred,'observed':observed,'withdrawn':sorted(withdrawn)};self.history.append({'challenge':r});return r
    def plan(self,start,goal,*,max_depth=3,state_budget=10000):
        s=world(start)
        if type(max_depth) is not int or not 0<=max_depth<=8 or type(state_budget) is not int or not 1<=state_budget<=100000:raise ValueError('planning budget')
        if not isinstance(goal,dict) or not goal or any(n not in s or not isinstance(fs,dict) or not fs or any(f not in s[n] or type(v) is not int or abs(v)>MAX_VALUE for f,v in fs.items()) for n,fs in goal.items()):raise ValueError('invalid partial goal')
        def reached(x):return all(x[n][f]==v for n,fs in goal.items() for f,v in fs.items())
        queue=deque([(s,[])]);seen={state_key(s)};expanded=0;attempted=0
        actions=[]
        for verb,m in sorted(self.models.items()):
            if not m or verb in s:continue
            for roles in permutations(s,model_arity(m)):
                w=list(roles);w.insert(m['slot'],verb);actions.append((m,w))
        while queue:
            x,path=queue.popleft();expanded+=1
            if reached(x):return {'status':'predicted_goal','actions':path,'predicted':x,'expanded':expanded}
            if len(path)>=max_depth:continue
            for m,w in actions:
                if attempted>=200000:return {'status':'budget','actions':None,'expanded':expanded}
                attempted+=1
                y=apply_model(m,x,w)
                if y is None:continue
                k=state_key(y)
                if k in seen:continue
                if len(seen)>=state_budget:return {'status':'budget','actions':None,'expanded':expanded}
                seen.add(k);queue.append((y,path+[w]))
        return {'status':'no_model_route','actions':None,'expanded':expanded}


def generate_probes(train, *, limit=192):
    """Construct interventions from observed role-field bindings, without labels.

    Perturb an observed state by equating any two role-relative attributes, then
    explore zero, sign reversal and unit shifts. This is a generic authored
    perturbation grammar, not autonomous discovery of arbitrary experiments.
    The environment must independently measure these proposed inputs.
    """
    if type(limit) is not int or not 1<=limit<=256 or not 2<=len(train)<=128:raise ValueError('probe limits')
    rows=[labelled(x) for x in train];seen={input_key(c) for c,_ in rows};out=[]
    for c,_ in rows:
        s,words,_,_,roles=c
        refs=[(n,f) for n in roles for f in s[n]]
        interventions=[]
        for n,f in refs:
            v=s[n][f]
            interventions.extend((n,f,s[m][g]) for m,g in refs if (m,g)!=(n,f))
            interventions.extend((n,f,x) for x in (0,-v,v+1,v-1))
        for n,f,v in interventions:
            if abs(v)>MAX_VALUE or v==s[n][f]:continue
            x={k:dict(fs) for k,fs in s.items()};x[n][f]=v
            case=[x,list(words)];key=input_key(input_case(case))
            if key in seen:continue
            seen.add(key);out.append(case)
            if len(out)>=limit:return out
    return out


def rebind(expr,role_map,field_map):
    if expr[0]=='field':return ('field',role_map[expr[1]],field_map[expr[2]])
    if expr[0]=='const':return tuple(expr)
    if expr[0]=='if':return ('if',expr[1],*(rebind(e,role_map,field_map) for e in expr[2:]))
    return (expr[0],*(rebind(e,role_map,field_map) for e in expr[1:]))


def transfer_fit(models, train, pool, audit, *, budget=256, query_budget=4):
    """Search role and attribute bijections between previously acquired models
    and an unfamiliar lexeme, with separate observations and audit. This is
    relational analogical transfer, not a learned universal ontology.
    """
    if type(budget) is not int or not 1<=budget<=10000 or type(query_budget) is not int or not 0<=query_budget<=32:raise ValueError('transfer budget')
    if not 2<=len(train)<=128 or not 3<=len(audit.inputs)<=128 or len(pool.inputs)>256:raise ValueError('evidence partition size')
    rows=[labelled(row) for row in train];p=[input_case(x) for x in pool.inputs];a=[input_case(x) for x in audit.inputs];cases=[c for c,_ in rows]+p+a
    keys=[input_key(c) for c in cases]
    if len(set(keys))!=len(keys):raise ValueError('overlapping evidence partitions')
    s,_,verb,slot,roles=cases[0];fields=tuple(next(iter(s.values())));arity=len(roles)
    if any(c[2]!=verb or c[3]!=slot or len(c[4])!=arity or tuple(next(iter(c[0].values())))!=fields for c in cases):raise ValueError('one action lexeme, word order and field schema per fit')
    result={'verb':verb,'slot':slot,'fields':fields,'candidate_attempts':0,'queries':[],'audit':[],'promoted':False,'model':None,'source':None,'method':'relational_transfer'}
    def candidates():
        for source,m in sorted(models.items()):
            if not m or model_arity(m)!=arity or len(m['fields'])!=len(fields):continue
            for fm,rm in product(permutations(fields),permutations(range(arity))):
                fmap=dict(zip(m['fields'],fm));writes=sorted((rm[i],fmap[f],rebind(e,rm,fmap)) for i,f,e in m['writes'])
                yield source,{'verb':verb,'slot':slot,'fields':fields,'writes':writes},list(zip(m['fields'],fm)),rm
    viable=[];seen=set();ntrain=len(rows)
    for source,m,fmap,rmap in candidates():
        if result['candidate_attempts']>=budget:break
        result['candidate_attempts']+=1
        if any(apply_model(m,c[0],c[1])!=y for c,y in rows):continue
        outputs=[apply_model(m,c[0],c[1]) for c in cases[ntrain:]]
        sig=tuple(None if y is None else state_key(y) for y in outputs)
        if sig in seen:continue
        seen.add(sig);viable.append((source,m,fmap,rmap,sig))
    result['initial_alternatives']=len(viable);remaining=list(range(len(p)))
    for _ in range(min(query_budget,len(p))):
        if not viable or not remaining:break
        def gain(j):
            counts=Counter(v[-1][j] for v in viable);return len(viable)**2-sum(n*n for n in counts.values())
        j=max(remaining,key=gain);g=gain(j)
        if not g:break
        c=p[j];out=world(pool.observe([c[0],list(c[1])]))
        viable=[v for v in viable if v[-1][j]==state_key(out)];remaining.remove(j)
        result['queries'].append({'input':[c[0],c[1]],'observed':out,'disagreement':g})
    result['remaining_alternatives']=len(viable)
    if not viable:result['reason']='no_transfer_model';return result
    if any(len({v[-1][j] for v in viable})>1 for j in remaining+list(range(len(p),len(p)+len(a)))):result['reason']='ambiguous';return result
    source,m,fmap,rmap,_=viable[0];result.update(candidate=m,proposed_source=source,field_mapping=fmap,role_mapping=rmap)
    for c in a:
        pred=apply_model(m,c[0],c[1]);out=world(audit.observe([c[0],list(c[1])]))
        result['audit'].append({'input':[c[0],c[1]],'predicted':pred,'observed':out})
    result['promoted']=all(r['predicted']==r['observed'] for r in result['audit'])
    result['reason']='audited' if result['promoted'] else 'audit_contradiction'
    if result['promoted']:result.update(model=m,source=source)
    return result
