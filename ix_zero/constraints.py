"""Finite-domain relational constraint search with explicit resource limits.

Constraints are declarative data. No source code is executed. An exhausted
budget is distinguished from unsatisfiability; all returned assignments are
checked against the original constraints.
"""
import re

MAX_VARIABLES = 10
MAX_DOMAIN = 16
MAX_CONSTRAINTS = 32
MAX_NODES = 50_000


def solve(domains, constraints, max_nodes=MAX_NODES):
    if (not isinstance(domains, dict) or not 1 <= len(domains) <= MAX_VARIABLES or
            type(max_nodes) is not int or not 1 <= max_nodes <= MAX_NODES or
            not isinstance(constraints, list) or len(constraints)>MAX_CONSTRAINTS):
        raise ValueError('constraint problem size limit')
    for k,v in domains.items():
        if not isinstance(k,str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,39}',k) or not isinstance(v,list) or not 1<=len(v)<=MAX_DOMAIN or any(type(x) is not int or abs(x)>1000 for x in v) or len(set(v))!=len(v):
            raise ValueError('invalid finite domain')
    domains={k:tuple(sorted(v)) for k,v in sorted(domains.items())}
    rules=[]
    for c in constraints:
        if not isinstance(c,dict) or not isinstance(c.get('vars'),list) or not c['vars'] or any(type(v) is not str or v not in domains for v in c['vars']) or len(set(c['vars']))!=len(c['vars']):
            raise ValueError('invalid constraint variables')
        op=c.get('op');vs=tuple(c['vars'])
        if op=='distinct' and set(c)=={'op','vars'} and len(vs)>=2:pass
        elif op in ('lt','eq','neq') and set(c)=={'op','vars'} and len(vs)==2:pass
        elif op in ('sum_eq','diff_eq') and set(c)=={'op','vars','value'} and type(c['value']) is int and abs(c['value'])<=10000 and (op!='diff_eq' or len(vs)==2):pass
        else:raise ValueError('unsupported constraint')
        rules.append((op,vs,c.get('value')))
    def feasible(rule,a):
        op,vs,value=rule
        vals=[a[v] for v in vs if v in a]
        if op=='distinct':return len(vals)==len(set(vals))
        choices=[(a[v],) if v in a else domains[v] for v in vs]
        if op=='sum_eq':
            lo=sum(min(d) for d in choices);hi=sum(max(d) for d in choices)
            return lo<=value<=hi
        x,y=choices
        if op=='lt':return min(x)<max(y)
        if op=='eq':return bool(set(x)&set(y))
        if op=='neq':return len(x)>1 or len(y)>1 or x[0]!=y[0]
        if op=='diff_eq':return any(v-value in y for v in x)
        raise RuntimeError('unknown constraint')
    def valid(a):return all(feasible(r,a) for r in rules)
    nodes=0;solutions=[];exhausted=False
    def search(a):
        nonlocal nodes,exhausted
        if len(solutions)>=2 or exhausted:return
        if nodes>=max_nodes:exhausted=True;return
        nodes+=1
        if not valid(a):return
        if len(a)==len(domains):
            if not valid(a):raise RuntimeError('solution verification failed')
            solutions.append(dict(sorted(a.items())));return
        # Forward-filter values then select the smallest viable domain.
        options=[]
        for v,ds in domains.items():
            if v not in a:
                viable=[x for x in ds if valid({**a,v:x})]
                if not viable:return
                options.append((len(viable),v,viable))
        _,v,viable=min(options,key=lambda q:(q[0],q[1]))
        for x in viable:
            search({**a,v:x})
            if exhausted or len(solutions)>=2:return
    search({})
    status='ambiguous' if len(solutions)>=2 else 'budget' if exhausted else 'unique' if solutions else 'unsatisfiable'
    return {'status':status,'solutions':solutions,'nodes':nodes,'node_limit':max_nodes}
