"""Finite symbolic transition learning. Ports selected BlackFox-AGI / IX-AGI ideas.

Facts and actions are inert tuples, not executable strings. States are closed-world
snapshots. No simulator callback or external action is accepted by the learner.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
import re

MAX_FACTS = 64
MAX_EVIDENCE = 512
MAX_SCHEMAS = 24
MAX_LITERALS = 48
MIN_SUPPORT = 2
MIN_AUDITS = 3
MAX_COMMANDS = 128
MAX_TREES = 128
MAX_EPOCHS = 4


def token(x):
    if not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,47}', x):
        raise ValueError('invalid symbolic token')
    return x


def fact(x):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 5:
        raise ValueError('fact must contain predicate and up to four terms')
    return tuple(token(v) for v in x)


def state(x):
    if not isinstance(x, (list, tuple, set, frozenset)) or len(x) > MAX_FACTS:
        raise ValueError('state exceeds fact budget')
    return frozenset(fact(v) for v in x)


def command(x):
    a = fact(x)
    if len(set(a[1:])) != len(a[1:]):
        raise ValueError('aliased arguments are unsupported')
    return a


def schema(a):
    return (a[0], len(a)-1)


def lift(f, a, full=False):
    args = a[1:]
    if not full and any(v not in args for v in f[1:]):
        return None
    return (f[0],) + tuple('?' + str(args.index(v)) if v in args else ':' + v for v in f[1:])


def localize(s, a, full=False):
    return frozenset(v for f in s if (v := lift(f, a, full)) is not None)


def ground(f, a):
    return (f[0],) + tuple(a[int(v[1:])+1] if v.startswith('?') else v[1:] for v in f[1:])


def effect(before, after, a, full):
    return localize(after-before,a,full), localize(before-after,a,full)


def apply_effect(s,a,e):
    add,remove=e
    return (s-frozenset(ground(f,a) for f in remove)) | frozenset(ground(f,a) for f in add)


@dataclass(frozen=True)
class Observation:
    before: frozenset
    action: tuple
    after: frozenset


def fit_contextual(training):
    """Enumerate minimal, depth-2 consistent decision trees over role-lifted
    full contexts. Return no usable prediction if the enumerator truncates.
    Support is based on distinct grounded observations; no target oracle.
    """
    from functools import lru_cache
    rows=[(localize(o.before,o.action,True),effect(o.before,o.after,o.action,True)) for o in training]
    features=tuple(sorted(set().union(*(s for s,_ in rows)))) if rows else ()
    if not rows or len(features)>MAX_LITERALS:return (),len(features)>MAX_LITERALS
    truncated=False
    @lru_cache(maxsize=None)
    def build(indices,depth):
        nonlocal truncated
        if len(indices)<MIN_SUPPORT:return None,()
        labels={rows[i][1] for i in indices}
        if len(labels)==1:return 0,((None,next(iter(labels))),)
        if not depth:return None,()
        best=None;trees=[]
        for f in features:
            yes=tuple(i for i in indices if f in rows[i][0]);no=tuple(i for i in indices if f not in rows[i][0])
            if not yes or not no:continue
            a,left=build(yes,depth-1);b,right=build(no,depth-1)
            if a is None or b is None:continue
            cost=1+a+b
            if best is None or cost<best:best=cost;trees=[]
            if cost!=best:continue
            if len(trees)+len(left)*len(right)>MAX_TREES:truncated=True
            for l in left:
                for r in right:
                    if len(trees)>=MAX_TREES:truncated=True;break
                    trees.append((f,l,r))
                if len(trees)>=MAX_TREES:break
        return best,tuple(trees)
    _,trees=build(tuple(range(len(rows))),2)
    return trees,truncated


def tree_effect(tree,s):
    while tree[0] is not None:tree=tree[1] if tree[0] in s else tree[2]
    return tree[1]


class ActionMemory:
    """Local rule lane + independently fitted full-context transition lane.

    Enumerates all separating conjunctions of minimum length <= 2 per effect.
    Requires agreement across the surviving clauses, not just a greedy winner.
    This is still a restricted hypothesis class, not a logical guarantee.
    """
    def __init__(self,representation="local",epoch=0):
        self.representation=representation; self.epoch=epoch
        self.trees=();self.tree_truncated=False
        self.training=[]
        self.audits=[]
        self.quarantined=False
        self.rules={}
        self.counter={}
        self.local_blocked=False
        self.context_conflicts=set()

    def fit(self):
        groups=defaultdict(list); exact=defaultdict(set)
        universe=set(); self.local_blocked=False
        for o in self.training:
            s=localize(o.before,o.action); e=effect(o.before,o.after,o.action,False)
            universe.update(s)
            if any(lift(f,o.action) is None for f in o.before ^ o.after):
                self.local_blocked=True
            groups[e].append((s,o.action,o.before))
            exact[s].add(e)
        self.context_conflicts={k for k,v in exact.items() if len(v)>1}
        self.counter={}
        for o in self.training:
            k=localize(o.before,o.action,True); e=effect(o.before,o.after,o.action,True)
            self.counter.setdefault(k,defaultdict(set))[e].add((o.before,o.action))
        self.rules={};self.trees=();self.tree_truncated=False
        if self.representation=='contextual':
            self.trees,self.tree_truncated=fit_contextual(self.training)
            return
        if self.local_blocked or len(universe)>MAX_LITERALS:
            return
        for e,pos in groups.items():
            if len({(s,a) for _,a,s in pos}) < MIN_SUPPORT:
                continue
            neg=[s for other,rows in groups.items() if other!=e for s,_,_ in rows]
            lits=[(f,f in pos[0][0]) for f in sorted(universe) if all((f in s)==(f in pos[0][0]) for s,_,_ in pos)]
            clauses=[]
            for n in range(3):
                for clause in combinations(lits,n):
                    if all(any((f in s)!=p for f,p in clause) for s in neg):
                        clauses.append(clause)
                if clauses: break
            if clauses:self.rules[e]=tuple(clauses)

    def lanes(self,s,a):
        local=localize(s,a); local_pred=None; counter_pred=None
        if self.representation=='contextual' and self.trees and not self.tree_truncated:
            values={tree_effect(t,localize(s,a,True)) for t in self.trees}
            if len(values)==1:local_pred=apply_effect(s,a,next(iter(values)))
        elif self.representation=='local' and not self.local_blocked and local not in self.context_conflicts:
            possible=[]; certain=[]
            for e,clauses in self.rules.items():
                matches=[all((f in local)==p for f,p in c) for c in clauses]
                if any(matches): possible.append(e)
                if all(matches): certain.append(e)
            if len(possible)==1 and possible==certain:
                local_pred=apply_effect(s,a,possible[0])
        options=self.counter.get(localize(s,a,True),{})
        if len(options)==1:
            e,examples=next(iter(options.items()))
            if len(examples)>=MIN_SUPPORT:counter_pred=apply_effect(s,a,e)
        return local_pred,counter_pred,len(options)>1

    def predict(self,s,a,mode='combined',validated=True):
        if self.quarantined or (validated and sum(ok for _,ok in self.audits)<MIN_AUDITS): return None
        l,c,conflict=self.lanes(s,a)
        if mode=='local':return l
        if mode=='counter':return c
        if mode!='combined':raise ValueError('invalid model lane')
        if conflict or (l is not None and c is not None and l!=c):return None
        return l if l is not None else c


class WorldMemory:
    def __init__(self,mode='combined'):
        if mode not in ('combined','local','counter'):raise ValueError('invalid model lane')
        self.mode=mode; self.models={}; self.procedures={}; self.history=defaultdict(list)

    def observe(self,before,a,after):
        s,a,t=state(before),command(a),state(after); k=schema(a)
        if k not in self.models and len(self.models)>=MAX_SCHEMAS:raise ValueError('schema budget')
        m=self.models.setdefault(k,ActionMemory()); o=Observation(s,a,t)
        if m.audits:raise ValueError('audit frozen: start a new world epoch')
        if o in m.training:return False
        if len(m.training)>=MAX_EVIDENCE:raise ValueError('training evidence budget')
        if any((q.before,q.action)==(s,a) and q.after!=t for q in m.training):m.quarantined=True
        m.training.append(o);m.fit();return True

    def predict(self,before,a,validated=True):
        s,a=state(before),command(a);m=self.models.get(schema(a))
        if m is None:return None
        t=m.predict(s,a,self.mode,validated)
        return t if t is None or len(t)<=MAX_FACTS else None

    def audit(self,before,a,after):
        s,a,t=state(before),command(a),state(after);m=self.models.get(schema(a));o=Observation(s,a,t)
        if m is None or not m.training:raise ValueError('no training evidence')
        if any((x.before,x.action)==(s,a) for x in m.training) or any((x.before,x.action)==(s,a) for x,_ in m.audits):raise ValueError('audit input is not fresh')
        if any((o.before,o.action)==(s,a) for old in self.history[schema(a)] for o in old.training+[x for x,_ in old.audits]):raise ValueError('audit input used in earlier epoch')
        if len(m.audits)>=MAX_EVIDENCE:raise ValueError('audit budget')
        p=self.predict(s,a,False);ok=p==t
        if p is not None and not ok:m.quarantined=True
        m.audits.append((o,ok));return ok

    def recover(self,a):
        """Explicit new epoch after falsification; old audits become training.

        Contextual representation is more expressive but still bounded. If the
        exact same grounded input has incompatible outcomes, start with no reused
        examples rather than fitting inconsistent evidence. Fresh audits remain
        mandatory and old audit inputs can never certify the new epoch.
        """
        k=schema(command(a));old=self.models.get(k)
        if old is None or not old.quarantined:raise ValueError('recovery requires a quarantined model')
        if len(self.history[k])>=MAX_EPOCHS:raise ValueError('epoch budget')
        data=list(dict.fromkeys(old.training+[o for o,_ in old.audits]));contexts=defaultdict(set)
        for o in data:contexts[o.before,o.action].add(o.after)
        reusable=len(data)<=MAX_EVIDENCE and all(len(v)==1 for v in contexts.values())
        m=ActionMemory('contextual',old.epoch+1);m.training=data if reusable else [];m.fit()
        self.history[k].append(old);self.models[k]=m
        return dict(epoch=m.epoch,reused=len(m.training),representation=m.representation,requires_fresh_audits=True)

    def remember(self,trace):
        """Store parameterized sequences only from measured, contiguous transitions.

        A macro requires two different object bindings before it can be proposed.
        All constituent predictions are rechecked when planning.
        """
        if not 2<=len(trace)<=16:raise ValueError('trace length')
        obs=[Observation(state(s),command(a),state(t)) for s,a,t in trace]
        if any(x.after!=y.before for x,y in zip(obs,obs[1:])):raise ValueError('noncontiguous trace')
        for o in obs:
            m=self.models.get(schema(o.action))
            if m is None or m.quarantined or o.before==o.after or (o not in m.training and (o,True) not in m.audits):raise ValueError('trace lacks measured successful transition')
        for n in range(2,min(4,len(obs))+1):
            for i in range(len(obs)-n+1):
                binding={};pattern=[]
                for o in obs[i:i+n]:
                    pattern.append((o.action[0],)+tuple(binding.setdefault(v,len(binding)) for v in o.action[1:]))
                key=tuple(pattern)
                if key not in self.procedures and len(self.procedures)>=128:continue
                stored=self.procedures.setdefault(key,set())
                if len(stored)<128:stored.add(tuple(binding))

    def macro_edges(self,s,actions,predict):
        emitted=0
        for pattern,bindings in sorted(self.procedures.items()):
            if len(bindings)<2:continue
            partial=[(s,(),{})]
            for name,*variables in pattern:
                nxt=[]
                for x,path,old in partial:
                    for a in actions:
                        if a[0]!=name or len(a)-1!=len(variables):continue
                        bound=dict(old);valid=True
                        for v,arg in zip(variables,a[1:]):
                            if (v in bound and bound[v]!=arg) or (v not in bound and arg in bound.values()):valid=False;break
                            bound[v]=arg
                        if not valid:continue
                        y=predict(x,a)
                        if y is not None and y!=x:nxt.append((y,path+(a,),bound))
                        if len(nxt)>=128:break
                    if len(nxt)>=128:break
                partial=nxt
                if not partial:break
            for x,path,_ in partial:
                yield x,path
                emitted+=1
                if emitted>=128:return

    def status(self):
        return {str(k):dict(epoch=m.epoch,representation=m.representation,contextual_trees=len(m.trees),tree_truncated=m.tree_truncated,training=len(m.training),audits=len(m.audits),passes=sum(ok for _,ok in m.audits),quarantined=m.quarantined,local_blocked=m.local_blocked,rules=len(m.rules),counter_contexts=len(m.counter)) for k,m in self.models.items()}


def catalog(actions):
    if not isinstance(actions,(tuple,list)) or not 1<=len(actions)<=MAX_COMMANDS:raise ValueError('catalog budget')
    return tuple(sorted(set(command(a) for a in actions)))


def plan(world,start,goal,actions,predict=None,max_expansions=1024,max_depth=16,macros=True):
    """Bounded forward search, shortest primitive path within explored graph.

    The optional predictor is an internal composition of learned models, not an
    environment oracle. Macro edges are ranked by primitive length.
    """
    import heapq
    from itertools import count
    if not 1<=max_expansions<=4096 or not 1<=max_depth<=32:raise ValueError('planning budget')
    actions=catalog(actions);predict=predict or world.predict
    q=[];serial=count();heapq.heappush(q,(0,next(serial),start,()));best={start:0};expanded=0
    while q and expanded<max_expansions:
        cost,_,s,path=heapq.heappop(q)
        if cost!=best[s]:continue
        if goal(s):return dict(path=path,expanded=expanded,final=s)
        expanded+=1
        if cost>=max_depth:continue
        edges=list(world.macro_edges(s,actions,predict)) if macros else []
        edges.extend((y,(a,)) for a in actions if (y:=predict(s,a)) is not None)
        for t,steps in edges:
            d=cost+len(steps)
            if d<=max_depth and d<best.get(t,10**9):
                best[t]=d;heapq.heappush(q,(d,next(serial),t,path+steps))
    return dict(path=None,expanded=expanded,final=None)
