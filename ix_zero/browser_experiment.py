"""Optional real Chromium integration experiment. The environment is an authored
local application, not a real-world website benchmark or Windows desktop.
The learner sees only visible observations, not the graph in the HTML.
"""
import argparse
import json
from pathlib import Path
import random
from .browser_learning import BrowserLearner,ChromiumEnvironment


def fixture(seed):
    rng=random.Random(seed);names=[f'panel-{n}' for n in rng.sample(range(100,999),5)];labels=[f'control-{n}' for n in rng.sample(range(100,999),9)]
    a,b,c,d,e=names;l=labels
    graph={a:{l[0]:c,l[1]:b},b:{l[2]:a,l[3]:d,l[4]:e},c:{l[5]:a},d:{l[6]:b},e:{l[7]:b,l[8]:a}}
    for state,edges in graph.items():
        items=list(edges.items());rng.shuffle(items);graph[state]=dict(items)
    html='''<!doctype html><html><body><p id="state"></p><div id="controls"></div><script>const graph='''+json.dumps(graph)+''';let state='''+json.dumps(a)+''';function draw(){document.querySelector('#state').textContent=state;const p=document.querySelector('#controls');p.replaceChildren();for(const [label,next] of Object.entries(graph[state])){const t=document.createElement('button');t.textContent=label;t.onclick=()=>{state=next;draw()};p.appendChild(t)}}draw()</script></body></html>'''
    return html,[d,e]


def run(trials,seed,executable=None):
    rows=[]
    for s in range(seed,seed+trials):
        html,goals=fixture(s)
        with ChromiumEnvironment(html,executable) as env:
            m=BrowserLearner();learn=m.learn(env,click_budget=100);snapshot=m.snapshot();restored=BrowserLearner.restore(json.loads(json.dumps(snapshot)));tests=[]
            for goal in goals:
                r=restored.achieve(env,goal,click_budget=2);actual=goal in env.observe().text.splitlines();control=BrowserLearner().achieve(env,goal,click_budget=2)
                # Blind random clicks with the same execution budget. No graph,
                # no access to target transitions, only visible button labels.
                rng=random.Random(s+len(tests)+1000);o=env.reset();blind=[]
                for _ in range(2):
                    if goal in o.text.splitlines() or not o.controls:break
                    a=rng.choice(o.controls);o=env.click(a);blind.append(a)
                tests.append({'goal':goal,'learned':r,'independent_visible_check':actual,'no_memory':control,'blind_random':{'achieved':goal in o.text.splitlines(),'actions':blind,'observed_text':o.text}})
            missing=restored.achieve(env,'unavailable-panel',click_budget=2)
        rows.append({'seed':s,'training':learn,'memory':snapshot,'tests':tests,'unreachable_control':missing})
    return {'trials':trials,'seed':seed,'summary':{'learned_goals':sum(t['independent_visible_check'] and t['learned']['status']=='achieved' for r in rows for t in r['tests']),'goal_trials':2*trials,'no_memory_goals':sum(t['no_memory']['status']=='achieved' for r in rows for t in r['tests']),'blind_random_goals':sum(t['blind_random']['achieved'] for r in rows for t in r['tests']),'training_clicks':sum(r['training']['clicks'] for r in rows),'learned_execution_clicks':sum(t['learned']['clicks'] for r in rows for t in r['tests'])},'rows':rows,'claim':'actual offline Chromium clicks on authored local apps; no general desktop or live website claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=2);p.add_argument('--seed',type=int,default=301100);p.add_argument('--browser-executable');p.add_argument('--output');a=p.parse_args()
    if not 1<=a.trials<=10:p.error('trials 1..10')
    r=run(a.trials,a.seed,a.browser_executable);print(json.dumps(r['summary'],indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    return 0

if __name__=='__main__':raise SystemExit(main())
