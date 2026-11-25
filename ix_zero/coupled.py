"""Evidence-gated joint transition induction: inert role-lifted decision trees."""
from functools import lru_cache
from itertools import combinations
from .core import Space
from .relational import state, command, localize, effect, apply_effect

MAX_ROWS=128
MAX_FEATURES=96
MAX_TREES=64
MAX_SEARCH=120_000
MIN_AUDITS=3


def feature_value(f,s,v):
    if f[0]=='fact':return f[1] in s
    if f[0]=='eq':return v[f[1]]==f[2]
    if f[0]=='lt':return v[f[1]]<f[2]
    if f[0]=='pair_eq':return v[f[1]]==v[f[2]]
    if f[0]=='pair_lt':return v[f[1]]<v[f[2]]
    raise ValueError('unknown feature')


def fit(rows,space,kind):
    """All minimum-size consistent depth <=2 trees, or fail closed on truncation."""
    if not rows:return (),False
    contexts=[(localize(s,a,True),v) for s,v,a,t,y in rows]
    features=[('fact',f) for f in sorted(set().union(*(s for s,v in contexts)))]
    for i in range(space.width):
        features.extend(('eq',i,n) for n in range(space.modulus))
        features.extend(('lt',i,n) for n in range(1,space.modulus))
    for i,j in combinations(range(space.width),2):
        features.extend((('pair_eq',i,j),('pair_lt',i,j),('pair_lt',j,i)))
    if len(features)>MAX_FEATURES:return (),True
    labels=[(effect(s,t,a,True),tuple((b-c)%space.modulus for c,b in zip(v,y)) if kind=='delta' else y) for s,v,a,t,y in rows]
    masks=[(f,sum(1<<i for i,(s,v) in enumerate(contexts) if feature_value(f,s,v))) for f in features]
    attempts=0
    class BudgetExceeded(Exception):pass
    @lru_cache(maxsize=None)
    def search(mask,depth):
        nonlocal attempts
        indices=[i for i in range(len(rows)) if mask>>i&1]
        if len(indices)<2:return None,(),False
        outcomes={labels[i] for i in indices}
        if len(outcomes)==1:return 0,((None,next(iter(outcomes))),),False
        if not depth:return None,(),False
        best=None;trees=[];truncated=False
        for f,bits in masks:
            attempts+=1
            if attempts>MAX_SEARCH:raise BudgetExceeded
            yes=mask&bits;no=mask&~bits
            if not yes or not no:continue
            ca,aa,ta=search(yes,depth-1);cb,bb,tb=search(no,depth-1)
            if ca is None or cb is None:continue
            cost=1+ca+cb
            if best is None or cost<best:best=cost;trees=[];truncated=False
            if cost!=best:continue
            truncated|=ta or tb or len(trees)+len(aa)*len(bb)>MAX_TREES
            for left in aa:
                for right in bb:
                    if len(trees)>=MAX_TREES:break
                    trees.append((f,left,right))
                if len(trees)>=MAX_TREES:break
        return best,tuple(trees),truncated
    try:
        _,trees,truncated=search((1<<len(rows))-1,2)
        return trees,truncated
    except BudgetExceeded:return (),True


def leaf(tree,s,v):
    while tree[0] is not None:tree=tree[1] if feature_value(tree[0],s,v) else tree[2]
    return tree[1]


class CoupledEpisode:
    def __init__(self,width,modulus):
        self.space=Space(width,modulus);self.training=[];self.audits=[];self.blocked=False;self.models={}
    def normalize(self,s,v,a,t,y):
        return state(s),self.space.check(v),command(a),state(t),self.space.check(y)
    def observe(self,*row):
        row=self.normalize(*row)
        if self.audits or self.blocked:raise ValueError('coupled training frozen or blocked')
        if row in self.training:return False
        if len(self.training)>=MAX_ROWS:raise ValueError('coupled training budget')
        if any(r[:3]==row[:3] for r in self.training):
            self.blocked=True;self.training.append(row);self.models={};return False
        self.training.append(row);self.models={};return True
    def _fit(self):
        if not self.models:self.models={k:fit(self.training,self.space,k) for k in ('delta','absolute')}
    def predict(self,s,v,a,validated=True):
        if self.blocked or (validated and sum(ok for _,ok in self.audits)<MIN_AUDITS):return None
        s,v,a=state(s),self.space.check(v),command(a);self._fit();ctx=localize(s,a,True)
        outputs=set()
        for kind,(trees,truncated) in self.models.items():
            if truncated:return None
            for tree in trees:
                e,y=leaf(tree,ctx,v)
                if kind=='delta':y=tuple((x+d)%self.space.modulus for x,d in zip(v,y))
                outputs.add((apply_effect(s,a,e),y))
                if len(outputs)>1:return None
        if len(outputs)!=1:return None
        out=next(iter(outputs))
        return out if len(out[0])<=64 else None
    def audit(self,*row):
        row=self.normalize(*row)
        if self.blocked:raise ValueError('coupled epoch blocked; begin fresh epoch')
        if len(self.audits)>=MAX_ROWS or any(r[:3]==row[:3] for r in self.training+[r for r,_ in self.audits]):raise ValueError('coupled audit must be fresh and within budget')
        predicted=self.predict(*row[:3],validated=False);ok=predicted==row[3:]
        self.audits.append((row,ok))
        if not ok:self.blocked=True
        return ok
    def status(self):
        self._fit();return {'space':[self.space.width,self.space.modulus],'training':len(self.training),'audits':len(self.audits),'passes':sum(ok for _,ok in self.audits),'blocked':self.blocked,'models':{k:{'trees':len(ts),'truncated':tr} for k,(ts,tr) in self.models.items()}}
