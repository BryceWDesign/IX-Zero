"""Cross-mechanism finite inquiry over the existing Sally-derived scalar
vocabulary and native NAND/shift synthesizer, plus generated order predicates.
A common lossless observation packet drives disagreement-based questions.
No solver is told which representation is correct.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import product
from .scalar import vocabulary
from .operators import Domain, synthesize, execute as word_execute
from .exploration import freeze


@dataclass(frozen=True)
class Hypothesis:
    mechanism:str
    syntax:object
    fn:object


def models(inputs,mechanisms=('scalar','word','predicate'),gate_budget=2):
    pts=tuple(map(freeze,inputs))
    if not pts or any(not isinstance(x,tuple) or len(x)!=2 or any(type(v) is not int or abs(v)>32 for v in x) for x in pts):raise ValueError('signed pair inputs required')
    if not set(mechanisms)<=set(('scalar','word','predicate')) or not 0<=gate_budget<=3:raise ValueError('mechanism or budget')
    out=[];counts={}
    if 'scalar' in mechanisms:
        vs=vocabulary();out.extend(Hypothesis('scalar',n,f) for _,n,f in vs);counts['scalar']=len(vs)
    if 'word' in mechanisms and all(0<=v<=31 for x in pts for v in x):
        width=max(2,max(v for x in pts for v in x).bit_length());d=Domain(width);search=synthesize(width,gate_budget)
        out.extend(Hypothesis('word',(width,c.expr),lambda x,c=c,d=d:d.value(c.table,x)) for c in search.models)
        counts['word']={'models':len(search.models),'generation_attempts':search.attempts,'complete_cost':search.complete_cost,'truncated':search.truncated}
    if 'predicate' in mechanisms:
        # Compose comparisons from interchangeable operands and comparison ops.
        ops={'eq':lambda a,b:a==b,'lt':lambda a,b:a<b,'le':lambda a,b:a<=b,'ne':lambda a,b:a!=b}
        operands=[('x0',lambda x:x[0]),('x1',lambda x:x[1]),('zero',lambda x:0),('one',lambda x:1)]
        for (an,a),(bn,b),(op,f) in product(operands,operands,ops.items()):
            out.append(Hypothesis('predicate',(op,an,bn),lambda x,a=a,b=b,f=f:int(f(a(x),b(x)))))
        counts['predicate']=len(operands)**2*len(ops)
    return out,counts


def inquire(train,pool,audit,*,mechanisms=('scalar','word','predicate'),gate_budget=2,query_budget=3,active=True):
    if not 0<=query_budget<=32 or not 2<=len(train)<=128 or not 3<=len(audit.inputs)<=128 or len(pool.inputs)>256:raise ValueError('evidence budget')
    obs={freeze(x):freeze(y) for x,y in train};p=tuple(map(freeze,pool.inputs));a=tuple(map(freeze,audit.inputs))
    if len(obs)!=len(train) or len(set(p+a))!=len(p+a) or set(obs)&set(p+a) or any(type(v) is not int for v in obs.values()):raise ValueError('bad evidence partitions')
    all_inputs=tuple(obs)+p+a;hs,counts=models(all_inputs,mechanisms,gate_budget)
    # Preserve distinct candidate semantics on exposed, unlabeled scope; no final test labels.
    viable=[];signatures=set()
    for h in hs:
        if not all(h.fn(x)==y for x,y in obs.items()):continue
        sig=tuple(h.fn(x) for x in all_inputs)
        if sig not in signatures:
            viable.append(h);signatures.add(sig)
    trace=[];remaining=list(p)
    for _ in range(min(query_budget,len(remaining))):
        if not viable:break
        def score(x):
            c=Counter(h.fn(x) for h in viable);return len(viable)**2-sum(v*v for v in c.values())
        x=max(remaining,key=score) if active else remaining[0];score_before=score(x)
        if active and score_before==0:break
        label=pool.observe(x);remaining.remove(x);before=Counter(h.mechanism for h in viable);viable=[h for h in viable if h.fn(x)==label];after=Counter(h.mechanism for h in viable)
        trace.append({'input':x,'label':label,'disagreement':score_before,'before':dict(before),'after':dict(after)})
    candidate=viable[0] if viable else None
    # Only consensus across surviving models on audit and unused pool may promote.
    ambiguity=bool(candidate and any(len({h.fn(x) for h in viable})>1 for x in remaining+list(a)))
    preds=[candidate.fn(x) for x in a] if candidate else []
    labels=[audit.observe(x) for x in a] if candidate else []
    promoted=bool(candidate and not ambiguity and preds==labels)
    report={'mechanisms':counts,'attempted_models':len(hs),'remaining_models':len(viable),'remaining_mechanisms':dict(Counter(h.mechanism for h in viable)),'selected':None if not candidate else {'mechanism':candidate.mechanism,'syntax':candidate.syntax},'trace':trace,'audit_predictions':preds,'audit_labels':labels,'promoted':promoted,'reason':'validated_on_audit' if promoted else 'representation_or_search_gap' if candidate is None else 'unresolved_ambiguity' if ambiguity else 'audit_contradiction'}
    return report,(candidate if promoted else None)


def predict(h,x):
    if h is None:return None
    try:
        x=freeze(x)
        if not isinstance(x,tuple) or len(x)!=2 or any(type(v) is not int or abs(v)>32 for v in x):return None
        if h.mechanism=='word':return word_execute(h.syntax[1],x,h.syntax[0])
        return h.fn(x)
    except (ValueError,TypeError,IndexError):return None
