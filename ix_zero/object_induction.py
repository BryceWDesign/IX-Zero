"""Object-centric grid program induction. A candidate may represent a grid as
connected components, projections, or raw pixels. Only training labels enter
search; each candidate is executable on a new grid. No task-ID exceptions.
"""
from collections import Counter, deque
from itertools import product
from .exploration import grid, freeze, execute


def background(g, mode):
    if mode=='zero':return 0
    return Counter(v for row in g for v in row).most_common(1)[0][0]


def components(g,bg,connectivity=4,by_color=True):
    h,w=len(g),len(g[0]);remaining={(i,j) for i in range(h) for j in range(w) if g[i][j]!=bg};out=[]
    steps=[(1,0),(-1,0),(0,1),(0,-1)]
    if connectivity==8:steps += [(1,1),(1,-1),(-1,1),(-1,-1)]
    while remaining:
        start=min(remaining);remaining.remove(start);q=deque([start]);points=[start]
        while q:
            i,j=q.popleft()
            for di,dj in steps:
                nxt=(i+di,j+dj)
                if nxt in remaining and (not by_color or g[nxt[0]][nxt[1]]==g[i][j]):
                    remaining.remove(nxt);q.append(nxt);points.append(nxt)
        out.append(tuple(sorted(points)))
    return out


def bounds(points):
    rows,cols=zip(*points);return min(rows),max(rows),min(cols),max(cols)


def select(objs,g,mode):
    if not objs:raise ValueError('no objects')
    if mode=='all':return tuple(p for o in objs for p in o)
    if mode=='largest':return max(objs,key=lambda o:(len(o),-o[0][0],-o[0][1]))
    if mode=='smallest':return min(objs,key=lambda o:(len(o),o[0]))
    if mode=='first':return objs[0]
    if mode=='last':return objs[-1]
    if mode=='widest':return max(objs,key=lambda o:bounds(o)[3]-bounds(o)[2])
    if mode=='tallest':return max(objs,key=lambda o:bounds(o)[1]-bounds(o)[0])
    if mode.startswith('color:'):
        chosen=tuple(p for o in objs for p in o if g[p[0]][p[1]]==int(mode.split(':')[1]))
        if not chosen:raise ValueError('color absent')
        return chosen
    raise ValueError('unknown selection')


def run(p,x):
    try:
        g=grid(freeze(x));kind=p[0]
        if kind=='raw':return execute(p[1],g,{})
        if kind=='scale':
            _,row,col=p;return grid(tuple(tuple(v for v in r for _ in range(col)) for r in g for _ in range(row)))
        if kind=='tile':
            _,row,col=p;return grid(tuple(r*col for r in g)*row)
        if kind=='projection':
            _,bgmode,axis,mode=p;bg=background(g,bgmode);rows=g if axis=='row' else tuple(zip(*g));out=[]
            for r in rows:
                if mode=='count':out.append(sum(v!=bg for v in r))
                elif mode=='first':out.append(next((v for v in r if v!=bg),bg))
                elif mode=='last':out.append(next((v for v in reversed(r) if v!=bg),bg))
                elif mode=='major':out.append(Counter(v for v in r if v!=bg).most_common(1)[0][0] if any(v!=bg for v in r) else bg)
            return grid((tuple(out),) if axis=='row' else tuple((v,) for v in out))
        if kind=='objects':
            _,bgmode,conn,by_color,selector,action,color=p;bg=background(g,bgmode);objs=components(g,bg,conn,by_color);points=select(objs,g,selector);s=set(points);t,b,l,r=bounds(points)
            if action=='crop':return tuple(row[l:r+1] for row in g[t:b+1])
            if action=='extract':return tuple(tuple(g[i][j] if (i,j) in s else bg for j in range(l,r+1)) for i in range(t,b+1))
            if action=='mask':return tuple(tuple(g[i][j] if (i,j) in s else bg for j in range(len(row))) for i,row in enumerate(g))
            if action=='erase':return tuple(tuple(bg if (i,j) in s else v for j,v in enumerate(row)) for i,row in enumerate(g))
            if action=='paint':return tuple(tuple(color if (i,j) in s else v for j,v in enumerate(row)) for i,row in enumerate(g))
            if action=='box':return tuple(tuple(color if t<=i<=b and l<=j<=r and (i in (t,b) or j in (l,r)) else v for j,v in enumerate(row)) for i,row in enumerate(g))
            if action=='fill':return tuple(tuple(color if t<=i<=b and l<=j<=r else v for j,v in enumerate(row)) for i,row in enumerate(g))
            if action=='count':return grid(((len(objs),),))
        if kind=='compose':
            v=run(p[1],g);return run(p[2],v) if v is not None else None
    except (ValueError,IndexError,TypeError,KeyError,OverflowError):return None
    return None


def programs():
    raw=('id','hflip','vflip','transpose','rotate','crop','tileh','tilev')
    for op in raw:yield ('raw',(op,))
    for n,m in product(range(1,5),repeat=2):
        if (n,m)!=(1,1):yield ('scale',n,m);yield ('tile',n,m)
    for bg,axis,mode in product(('zero','mode'),('row','col'),('count','first','last','major')):yield ('projection',bg,axis,mode)
    basic=[]
    for bg,c,split,sel in product(('zero','mode'),(4,8),(True,False),('all','largest','smallest','first','last','widest','tallest')+tuple(f'color:{i}' for i in range(10))):
        for act in ('crop','extract','mask','erase'):
            p=('objects',bg,c,split,sel,act,0);basic.append(p);yield p
    for bg,c,split,sel,act,color in product(('zero','mode'),(4,8),(True,False),('all','largest','smallest','first','last'),('paint','box','fill'),range(10)):
        yield ('objects',bg,c,split,sel,act,color)
    for a,b in product(range(10),repeat=2):
        if a!=b:yield ('raw',(f'color:{a}:{b}',))
    for p,op in product(basic,raw[1:5]):yield ('compose',p,('raw',(op,)))
    for a,b in product(raw,raw):yield ('raw',(a,b))


def solve(train,inputs,budget=5000,representation='all'):
    if type(budget) is not int or not 1<=budget<=20000 or representation not in ('all','pixels') or not 1<=len(train)<=100 or not 1<=len(inputs)<=100:raise ValueError('grid search parameters')
    train=[(grid(freeze(x)),grid(freeze(y))) for x,y in train];inputs=[grid(freeze(x)) for x in inputs];valid=[];attempts=0;counts=Counter()
    for p in programs():
        if representation=='pixels' and p[0] not in ('raw','scale','tile'):continue
        if attempts>=budget:break
        attempts+=1;counts[p[0]]+=1
        if all(run(p,x)==y for x,y in train):valid.append(p)
    output=[]
    for x in inputs:
        v={run(p,x) for p in valid};output.append(next(iter(v)) if len(v)==1 else None)
    return {'predictions':output,'candidate_count':len(valid),'attempted':attempts,'attempted_by_kind':dict(counts),'selected_examples':valid[:8],'representation':representation}
