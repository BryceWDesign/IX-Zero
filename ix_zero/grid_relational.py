"""Induce a compact pixel program over generated relational coordinates.

Candidate output geometry and cell representations are generated from input
and training dimensions, not task IDs. A learned program may combine two
coordinate frames through a conditional or fit a finite value remapping.
An unseen remapping key abstains. Minimum-complexity consistency is a prior,
not evidence of universal correctness; exact tied predictions must agree.
"""
from collections import Counter
from itertools import product
from .exploration import grid, freeze
from .object_induction import components


def geometry_specs(train):
    yield ('same',)
    yield ('transpose',)
    for a,b in product(range(1,5),repeat=2):
        if (a,b)!=(1,1):yield ('scale',a,b)
    for a,b in product(range(1,5),repeat=2):
        yield ('shrink',a,b)
    for a,b in product(('h','w'),repeat=2):yield ('square',a,b)
    shapes={(len(y),len(y[0])) for _,y in train}
    if len(shapes)==1:yield ('constant',*next(iter(shapes)))


def shape(spec,g):
    h,w=len(g),len(g[0]);k=spec[0]
    if k=='same':s=(h,w)
    elif k=='transpose':s=(w,h)
    elif k=='scale':s=(h*spec[1],w*spec[2])
    elif k=='shrink':
        if h%spec[1] or w%spec[2]:return None
        s=(h//spec[1],w//spec[2])
    elif k=='square':s=(h*({'h':h,'w':w}[spec[1]]),w*({'h':h,'w':w}[spec[2]]))
    else:s=spec[1:]
    return s if all(1<=x<=30 for x in s) else None


# Each coordinate frame defines a general map from output positions to input
# cells. These are generic geometry operations, not complete target solutions.
FRAMES=('local','block','scaled','reverse_local','reverse_block','reverse_scaled','transpose_local','transpose_block','transpose_scaled','row_first','row_last','col_first','col_last')


def coordinate(mode,i,j,h,w,oh,ow):
    if mode=='local':return i%h,j%w
    if mode=='block':return i//h,j//w
    if mode=='scaled':return i*h//oh,j*w//ow
    if mode=='reverse_local':return h-1-i%h,w-1-j%w
    if mode=='reverse_block':return h-1-i//h,w-1-j//w
    if mode=='reverse_scaled':return h-1-i*h//oh,w-1-j*w//ow
    if mode=='transpose_local':return j%h,i%w
    if mode=='transpose_block':return j//h,i//w
    if mode=='transpose_scaled':return j*h//ow,i*w//oh
    if mode=='row_first':return i%h,0
    if mode=='row_last':return i%h,w-1
    if mode=='col_first':return 0,j%w
    if mode=='col_last':return h-1,j%w
    raise ValueError('unknown coordinate frame')


def features(g,sh):
    h,w=len(g),len(g[0]);oh,ow=sh;bg=Counter(v for row in g for v in row).most_common(1)[0][0]
    objs=components(g,bg,4,True)
    sizes={pos:len(o) for o in objs for pos in o};ordered=sorted(set(sizes.values()));rank={n:i+1 for i,n in enumerate(ordered)}
    maps={('background',):[],('row_edge',):[],('col_edge',):[],('diagonal',):[]}
    for frame in FRAMES:
        for attr in ('color','area_rank','object_size'):
            maps[(attr,frame)]=[]
    for i,j in product(range(oh),range(ow)):
        maps[('background',)].append(bg);maps[('row_edge',)].append(int(i in (0,oh-1)));maps[('col_edge',)].append(int(j in (0,ow-1)));maps[('diagonal',)].append(int(i==j or i+j==ow-1))
        for frame in FRAMES:
            r,c=coordinate(frame,i,j,h,w,oh,ow);inside=0<=r<h and 0<=c<w
            size=sizes.get((r,c),0) if inside else 0
            maps[('color',frame)].append(g[r][c] if inside else bg)
            maps[('area_rank',frame)].append(rank.get(size,0))
            maps[('object_size',frame)].append(size)
    return {k:tuple(v) for k,v in maps.items()}


def infer(train,inputs,*,budget=20000):
    if type(budget) is not int or not 1<=budget<=100000 or not 1<=len(train)<=30 or not 1<=len(inputs)<=30:raise ValueError('relational grid search limits')
    train=[(grid(freeze(x)),grid(freeze(y))) for x,y in train];inputs=[grid(freeze(x)) for x in inputs]
    all_grids=[x for x,_ in train]+inputs;ntrain=len(train);target=tuple(v for _,y in train for row in y for v in row);nt=len(target)
    palette=sorted(set(target));attempted=0;best_cost=None;best=[];shape_seen=set();shape_count=0;exhausted=False;best_model_count=0
    # Shape hypotheses with identical dimensions on all exposed inputs share
    # an observational equivalence class, not a proof of symbolic equivalence.
    for spec in geometry_specs(train):
        shapes=tuple(shape(spec,g) for g in all_grids)
        if None in shapes or any(shapes[i]!=(len(y),len(y[0])) for i,(_,y) in enumerate(train)):continue
        if shapes in shape_seen:continue
        shape_seen.add(shapes);shape_count+=1
        matrices=[features(g,sh) for g,sh in zip(all_grids,shapes)];leaves=[];sig_seen=set()
        def add_leaf(key,values):
            if values not in sig_seen:sig_seen.add(values);leaves.append((key,values))
        for key in matrices[0]:add_leaf(key,tuple(v for m in matrices for v in m[key]))
        total=sum(h*w for h,w in shapes)
        for c in palette:add_leaf(('constant',c),(c,)*total)
        # Only leaves valid as colors can emit a pixel directly. Other integer
        # features may still be used as selectors or value-remapping keys.
        emit=[(k,v) for k,v in leaves if all(0<=x<=9 for x in v)]
        def evaluate(cost,program,out):
            nonlocal attempted,best_cost,best,best_model_count,exhausted
            if attempted>=budget:exhausted=True;return False
            attempted+=1
            if tuple(out[:nt])!=target:return True
            if best_cost is None or cost<best_cost:best_cost=cost;best=[];best_model_count=0
            if cost==best_cost:
                best_model_count+=1
                if not any(shapes==s and out[nt:]==v for s,v,_ in best):best.append((shapes,out[nt:],{'geometry':spec,'program':program,'complexity':cost}))
            return True
        for k,v in emit:
            if not evaluate(1,('direct',k),v):break
        if exhausted:break
        if best_cost==1:continue
        # Fit a discrete value mapping from training labels; holdout inputs
        # contribute no labels and a novel feature value remains unknown.
        for k,v in leaves:
            mapping={};ok=True
            for x,y in zip(v[:nt],target):
                if x in mapping and mapping[x]!=y:ok=False;break
                mapping[x]=y
            out=tuple(mapping.get(x,-1) for x in v) if ok else ()
            if not evaluate(2+len(mapping),('map',k,sorted(mapping.items())),out):break
        if exhausted:break
        if best_cost is not None and best_cost<3:continue
        # Generate predicates from training feature values. Factor the branch
        # search using exact consistency on training cells before combining.
        for k,v in leaves:
            vals=sorted(set(v[:nt]))
            if len(vals)>16:continue
            for value in vals:
                yes=tuple(i for i in range(nt) if v[i]==value);no=tuple(i for i in range(nt) if v[i]!=value)
                if not yes or not no:continue
                good_y=[(key,arr) for key,arr in emit if all(arr[i]==target[i] for i in yes)]
                good_n=[(key,arr) for key,arr in emit if all(arr[i]==target[i] for i in no)]
                # Count one predicate consideration even when no branch pair
                # survives; this cap does not measure wall-clock operations.
                if not evaluate(3,('predicate',k,value),()):break
                for (ka,a),(kb,b) in product(good_y,good_n):
                    out=tuple(x if z==value else y for z,x,y in zip(v,a,b))
                    if not evaluate(3,('if_eq',k,value,ka,kb),out):break
                if exhausted:break
            if exhausted:break
        if exhausted:break
    predictions=[]
    for j,g in enumerate(inputs):
        possible=set()
        for shapes,values,_ in best:
            sh=shapes[ntrain+j];offset=sum(h*w for h,w in shapes[ntrain:ntrain+j]);h,w=sh;v=values[offset:offset+h*w]
            possible.add(tuple(tuple(v[r*w:(r+1)*w]) for r in range(h)) if all(0<=x<=9 for x in v) else None)
        predictions.append(next(iter(possible)) if len(possible)==1 and not exhausted else None)
    return {'predictions':predictions,'attempted':attempted,'geometry_classes':shape_count,'minimum_cost':best_cost,'minimum_cost_models':best_model_count,'distinct_predictions':len(best),'selected_examples':[m for _,_,m in best[:8]],'budget_exhausted':exhausted,'claim':'minimum-complexity consistency within authored relational feature grammar; not universal equivalence'}
