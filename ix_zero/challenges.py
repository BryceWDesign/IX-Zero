"""Out-of-family, shift, and planning stress tests. Independent teacher execution."""
import json
import random
from pathlib import Path
from .core import Space,Task,Mind
from .experiment import teacher,train,score

def run(seed):
    rng=random.Random(seed); s=Space(2,5); states=list(s.states)
    # Arbitrary random bijection. No assumption it is expressible by the grammar.
    outputs=states.copy(); rng.shuffle(outputs); truth=dict(zip(states,outputs))
    inputs=states.copy(); rng.shuffle(inputs); holdout,cal,pool=inputs[:8],inputs[8:11],inputs[11:]
    unknown=Task(s); train(unknown,truth.__getitem__,pool,5,True,rng); raw=score(unknown,truth.__getitem__,holdout)
    for x in cal:
        if unknown.blocked: break
        unknown.audit(x,truth[x])
    gated=[(x,unknown.validated_predict(x),truth[x]) for x in holdout]
    # Shift of an audited action. Observe an actual counterexample to block old model;
    # learn a new epoch rather than silently erase old observations.
    m=Mind(); p=('mix','rot','inc'); q=('unmix','rot','dec')
    def fit(t,program):
        oracle=lambda x:teacher(program,x,s.modulus); xs=states.copy(); rng.shuffle(xs); audits=xs[:3]; train(t,oracle,xs[3:],8,True,rng)
        if not t.resolved(): return False
        for x in audits:
            if not t.audit(x,oracle(x)): return False
        return t.concept() is not None
    old=m.new('old',s); old_fit=fit(old,p)
    if old_fit:
        m.promote('old'); counter=next((x for x in old.observations if teacher(p,x,s.modulus)!=teacher(q,x,s.modulus)),None)
        if counter is not None: old.observe(counter,teacher(q,counter,s.modulus))
    fresh=m.new('fresh',s); fresh_fit=fit(fresh,q)
    a=m.new('add',s,reuse=False); b=m.new('rotate',s,reuse=False)
    a_fit=fit(a,('inc',)); b_fit=fit(b,('rot',)); plans=[]
    if a_fit and b_fit:
        for _ in range(10):
            x,y=rng.choice(states),rng.choice(states); path=m.plan(s,x,y,['add','rotate']); reached=x
            if path is not None:
                for n,expected in path:
                    reached=teacher(('inc',) if n=='add' else ('rot',),reached,s.modulus)
                    if reached!=expected: break
            plans.append({'start':x,'goal':y,'steps':path,'teacher_end':reached,'success':path is not None and reached==y})
    return {'seed':seed,'unstructured_raw':raw,'unstructured_gated':gated,'unstructured_blocked':unknown.blocked,'old_fit':old_fit,'old_blocked_after_observed_shift':old.blocked,'old_program_removed_from_new_library':not m.library(),'fresh_fit':fresh_fit,'fresh_exhaustive_correct':sum(fresh.predict(x)==teacher(q,x,s.modulus) for x in states),'fresh_total':len(states),'plans':plans}

def main():
    rows=[run(s) for s in range(871,883)]
    summary={'trials':len(rows),'unstructured_raw_correct':sum(r['unstructured_raw']['correct'] for r in rows),'unstructured_raw_wrong':sum(r['unstructured_raw']['wrong'] for r in rows),'unstructured_gated_wrong':sum(p is not None and p!=y for r in rows for x,p,y in r['unstructured_gated']),'unstructured_gated_abstentions':sum(p is None for r in rows for x,p,y in r['unstructured_gated']),'old_fit':sum(r['old_fit'] for r in rows),'shift_blocked':sum(r['old_blocked_after_observed_shift'] for r in rows),'fresh_fit':sum(r['fresh_fit'] for r in rows),'fresh_exhaustive_correct':sum(r['fresh_exhaustive_correct'] for r in rows),'fresh_total':sum(r['fresh_total'] for r in rows),'plans_attempted':sum(len(r['plans']) for r in rows),'plans_successful':sum(p['success'] for r in rows for p in r['plans'])}
    Path('results/challenges.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
