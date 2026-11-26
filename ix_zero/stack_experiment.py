"""Three-level skill composition and independent finite planning evaluation."""
import argparse
from collections import deque
from itertools import product
import json
from pathlib import Path
from .continual import Continual
from .continual_experiment import Host, drive, replay

class CascadeHost(Host):
    def truth(self,x):
        if self.law=='lowbit':
            w,a,b=x
            return ((a<<4)&((1<<w)-1))>>4
        return super().truth(x)


def reference_distance(start,goal,depth=4):
    """Independent oracle over the two primitive learned shift-two skills."""
    q=deque([(start,0)]);seen={start}
    while q:
        s,d=q.popleft()
        if s==goal:return d
        if d==depth:continue
        for v in ((s*4)%32,s//4):
            if v not in seen:seen.add(v);q.append((v,d+1))
    return None


def trial(seed,reuse):
    m=Continual(max_budget=2,reuse_skills=reuse);rows=[]
    for i,law in enumerate(('left2','right2','left4','right4','lowbit')):
        rows.append(drive(m,CascadeHost('word',[3,5],seed+i,law,i%2),budget=2))
    m,ok=replay(m)
    keys=[r['episode'] for r in rows];plans=[]
    # The last goal task is separate from synthesis. Its actions are the first
    # two acquired skills, not the target answer or the later four-shift laws.
    if all(m.live(k) for k in keys[:2]):
        for a,b in product(range(32),repeat=2):
            p=m.compose_plan(5,a,b,keys[:2],max_steps=4);d=reference_distance(a,b)
            valid=(p is None and d is None)
            if p is not None:
                s=a;valid=len(p)==d
                for e in p:
                    v=(s*4)%32 if e['skill']==keys[0] else s//4 if e['skill']==keys[1] else None
                    valid=valid and e['before']==s and e['operand']==0 and e['after']==v;s=v
                valid=valid and s==b
            plans.append(dict(start=a,goal=b,oracle_distance=d,proposal=p,correct=valid))
    # Invalidate the first source in a three-level chain; test transitive quarantine.
    invalidation=None
    if reuse and all(m.live(k) for k in keys):
        ok_challenge=m.challenge(keys[0],(5,1,0),1)
        invalidation=dict(accepted=ok_challenge,source_live=m.live(keys[0]),second_level_live=m.live(keys[2]),third_level_live=m.live(keys[4]),third_prediction=m.predict(keys[4],(5,1,0)))
    return dict(seed=seed,reuse_skills=reuse,episodes=rows,replayed=ok,plans=plans,invalidation=invalidation)


def campaign(seed=710000,trials=6):
    if not 1<=trials<=6:raise ValueError('trial bound')
    rows=[trial(seed+i*100,reuse) for i in range(trials) for reuse in (True,False)]
    arms=[]
    for i in range(trials):
        a,b=rows[2*i:2*i+2]
        for x,y in zip(a['episodes'],b['episodes']):
            ax={tuple(z['input']):z for z in x['rows']};by={tuple(z['input']):z for z in y['rows']};common=set(ax)&set(by)
            arms.append(dict(seed=a['seed'],law=x['law'],total=len(common),full_correct=sum(ax[k]['predicted']==ax[k]['expected'] for k in common),control_correct=sum(by[k]['predicted']==by[k]['expected'] for k in common),full_wrong=sum(ax[k]['predicted'] is not None and ax[k]['predicted']!=ax[k]['expected'] for k in common),control_wrong=sum(by[k]['predicted'] is not None and by[k]['predicted']!=by[k]['expected'] for k in common)))
    full=[r for r in rows if r['reuse_skills']];control=[r for r in rows if not r['reuse_skills']]
    # Validate provenance rather than merely counting successful predictions.
    def chain(r):
        e=r['episodes'];return e[0]['episode'] in e[2]['dependencies'] and e[1]['episode'] in e[3]['dependencies'] and set(e[4]['dependencies'])=={e[2]['episode'],e[3]['episode']}
    plans=[p for r in full for p in r['plans']]
    s=dict(full_validated=sum(e['state']=='validated' for r in full for e in r['episodes']),control_validated=sum(e['state']=='validated' for r in control for e in r['episodes']),third_level_chains=sum(chain(r) for r in full),third_level_control_validated=sum(r['episodes'][-1]['state']=='validated' for r in control),common_total=sum(r['total'] for r in arms),full_correct=sum(r['full_correct'] for r in arms),control_correct=sum(r['control_correct'] for r in arms),full_wrong=sum(r['full_wrong'] for r in arms),control_wrong=sum(r['control_wrong'] for r in arms),plans_correct=sum(p['correct'] for p in plans),plans_total=len(plans),reachable=sum(p['oracle_distance'] is not None for p in plans),replays=sum(r['replayed'] for r in rows),invalidated=sum(r['invalidation']==dict(accepted=False,source_live=False,second_level_live=False,third_level_live=False,third_prediction=None) for r in full))
    return dict(protocol='authored finite three-level executable skill and planning evaluation',seed=seed,trials=trials,summary=s,matched_holdouts=arms,runs=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=710000);p.add_argument('--trials',type=int,default=6);p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();c=campaign(a.seed,a.trials);s=c['summary'];print(json.dumps(s,indent=2))
    if a.output:a.output.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    if a.check:
        n=a.trials
        # Controls are allowed to expose false promotions; their observed
        # errors remain in the evidence. Do not require controls to succeed.
        assert s['full_validated']==5*n and s['control_validated']<5*n and s['third_level_chains']==n and s['third_level_control_validated']==0,s
        assert s['full_correct']==s['common_total'] and s['full_wrong']==0,s
        assert s['plans_correct']==s['plans_total']==1024*n and s['replays']==2*n and s['invalidated']==n,s

if __name__=='__main__':main()
