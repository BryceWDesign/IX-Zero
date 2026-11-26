"""External synthetic stress tests, not a blinded AGI benchmark.
Run from original IX-Zero root: python -m tools.hybrid_stress --output stress.json
"""
import argparse, hashlib, json, random, tempfile, time
from itertools import product
from pathlib import Path
from ix_zero.hybrid import Hybrid
from ix_zero.checkpoint import canonical
SEEDS=range(78320,78326)
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def inputs(f,w): return list(product(range(2**w),repeat=2)) if f=='word' else list(product(range(w),range(w),range(4)))
def pools(f,ws,s,exclude=()):
    out={}
    for w in ws:
        xs=[x for x in inputs(f,w) if (w,*x) not in exclude];random.Random(s+w*97).shuffle(xs)
        n=min(len(xs)-7,96) if f=='word' else min(128,len(xs)//2)
        out[w]={'train':xs[:n],'audit':xs[n:n+4]}
    return out
def hold(f,ws,p): return [(w,*x) for w in ws for x in inputs(f,w) if x not in p[w]['train'] and x not in p[w]['audit']]
def word(x): w,a,b=x;return ((~(a&b))<<1)&((1<<w)-1)
def add(x): w,a,b=x;return (a+b)%2**w
def identity(x): return x[1]
def grid(x):
    w,a,b,d=x;dx,dy=((0,-2),(2,0),(0,2),(-2,0))[d];return ((a+dx)%w,(b+dy)%w)
def changed(x):
    w,a,b,d=x;dx,dy=((1,0),(0,1),(-1,0),(0,-1))[d];return (max(0,min(w-1,a+dx)),max(0,min(w-1,b+dy)))
def pixel(y,r,flip=False): return [round(.5+(-1 if flip else 1)*sign*(2*y-1)*r.uniform(.21,.33)+r.uniform(-.06,.06),6) for sign in (1,-1,1)]
def calibration(s,flip=False):
    r=random.Random(s);rows=[(pixel(y,r,flip),y) for y in (0,1) for _ in range(48)];r.shuffle(rows);return rows
def raster(f,w,y,s,flip):
    vals=[(y>>i)&1 for i in range(w)] if f=='word' else [int((i%w,i//w)==y) for i in range(w*w)]
    r=random.Random(s);return [pixel(v,r,flip) for v in vals]
def run(f,ws,p,s,truth,prior=None,flip=False):
    m=Hybrid(f,ws,ws[0],**p[ws[0]],seed=s,budget=2) if prior is None else prior.transfer(f,ws,ws[0],**p[ws[0]],seed=s,budget=2)
    test=calibration(s+129,flip);before=sum(m.read(x)==y for x,y in test)
    cal=calibration(s+17,flip) if prior is None else []
    if cal:m.calibrate(cal)
    after=sum(m.read(x)==y for x,y in test);trace=[]
    for _ in range(150):
        r=m.next();a=r['action']
        if a in ('complete','failed'):break
        x=r['input']
        if a=='environment':v=p[x]
        else:
            y=truth(x);v=y if a=='audit' else raster(f,x[0],y,s+int(digest(x)[:10],16),flip)
        trace.append({'request':r,'answer':v});m.answer(v)
    else:raise RuntimeError('request limit')
    rows=[{'input':x,'expected':truth(x),'predicted':m.predict(x)} for x in hold(f,ws,p)]
    result=dict(seed=s,family=f,scope=ws,flip=flip,calibration=cal,recognition_before=before,recognition_after=after,recognition_total=len(test),partitions=p,terminal=r,trace=trace,promoted=m.inquiry.finished and m.inquiry.validated(),correct=sum(z['predicted']==z['expected'] for z in rows),wrong=sum(z['predicted'] is not None and z['predicted']!=z['expected'] for z in rows),abstain=sum(z['predicted'] is None for z in rows),rows=rows,payload_digest=digest(m.payload()))
    return m,result
def attempt(fn):
    try:return {'returned':fn()}
    except (ValueError,TypeError,RuntimeError,AttributeError) as e:return {'error':type(e).__name__,'message':str(e)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();start=time.monotonic();out={k:[] for k in ('generalization','transfer_shift','regime_change','indistinguishable_worlds','three_generation_transfer')}
    for s in SEEDS:
        wp=pools('word',[3,5],s);gp=pools('grid',[6,8],s)
        wm,wr=run('word',[3,5],wp,s,word);_,ar=run('word',[3,5],wp,s,add);gm,gr=run('grid',[6,8],gp,s,grid)
        out['generalization'].append(dict(composed_word=wr,outside_grammar=ar,larger_grid=gr))
        target,same=run('grid',[6,8],gp,s,grid,prior=wm);_,flipped=run('grid',[6,8],gp,s,grid,prior=wm,flip=True);_,fresh=run('grid',[6,8],gp,s,grid,flip=True)
        out['transfer_shift'].append(dict(compatible=same,inverted=flipped,fresh_calibrated=fresh))
        x=next(x for x in hold('grid',[6,8],gp) if grid(x)!=changed(x));old_prediction=gm.predict(x);accepted=gm.challenge(x,changed(x));after=gm.predict(x);next_request=gm.next();plan=gm.plan((0,0),(1,1),6);recal=attempt(lambda:gm.calibrate(calibration(s+17)))
        _,new=run('grid',[6,8],gp,s,changed)
        out['regime_change'].append(dict(original_payload=gr['payload_digest'],contradictory_input=x,old_prediction=old_prediction,new_measurement=changed(x),challenge_accepted=accepted,prediction_after=after,next_request=next_request,plan_after=plan,recalibration_attempt=recal,fresh_instance=new))
        ws=[3,5];exceptions={(5,a,31) for a in range(32)};p=pools('word',ws,s+3,exceptions)
        def alternate(x):return identity(x)^int(x in exceptions)
        m1,r1=run('word',ws,p,s+3,identity);m2,r2=run('word',ws,p,s+3,alternate)
        h=sorted(exceptions)[0];v=m2.predict(h);ok=m2.challenge(h,alternate(h));out['indistinguishable_worlds'].append(dict(ordinary=r1,alternate=r2,identical_payload=r1['payload_digest']==r2['payload_digest'],hidden_slice=sorted(exceptions),hidden_prediction=v,hidden_truth=alternate(h),challenge_accepted=ok,prediction_after=m2.predict(h)))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.json';target.save(path);restored=Hybrid.load(path);replayed=digest(target.payload())==digest(restored.payload()); third=attempt(lambda:restored.transfer('word',[3,5],3,**wp[3],seed=s+1,budget=2))
        if 'returned' in third:third={'returned':'accepted'}
        _,control=run('word',[3,5],wp,s+1,identity)
        out['three_generation_transfer'].append(dict(source_promoted=wr['promoted'],second_promoted=same['promoted'],replayed=replayed,third_transfer=third,fresh_control=control))
        print('finished seed',s,flush=True)
        a.output.write_text(json.dumps(dict(protocol='external authored stress tests; no learner changes; six fixed fresh seeds; synthetic; not blinded',seeds=list(SEEDS),tests=out,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
