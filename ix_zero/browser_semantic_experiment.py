"""Optional actual Chromium experiment: acquire action models from a trusted
local application, generate additional interventions, then execute a learned
plan through the same visible form. This is a typed laboratory UI, not
arbitrary GUI comprehension or live network access.
"""
import argparse
import json
from pathlib import Path
from .browser_learning import ChromiumEnvironment
from .grounded_semantics import SemanticMemory, generate_probes
from .semantic_experiment import Environment, LiveMeasurements, plan_task


def app_html(env):
    # The application is the independent measurement provider. Its JavaScript
    # runs in Chromium. The candidate learner receives no source or target rule.
    config=json.dumps({'fields':env.fields,'verbs':env.verbs})
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Experimental world</title></head>
<body><h1>Transition laboratory</h1><label for="world">Observed world</label><textarea id="world" rows="8" cols="70"></textarea>
<label for="action">Action tokens</label><input id="action"><button id="run">Apply action</button><pre id="result" role="status"></pre>
<script>const config='''+config+''';
document.getElementById('run').addEventListener('click',()=>{
const result=document.getElementById('result');try{
const before=JSON.parse(document.getElementById('world').value);const words=JSON.parse(document.getElementById('action').value);
const verb=words.find(w=>config.verbs.includes(w));const kind=config.verbs.indexOf(verb);
const [a,b]=words.filter(w=>w!==verb);const [f,g]=config.fields;
const out=JSON.parse(JSON.stringify(before));
if(kind===0) out[b][f]=before[a][f];
else if(kind===1){out[a][g]=before[b][g];out[b][g]=before[a][g];}
else if(kind===2){if(before[a][g]===before[b][g])out[b][f]=before[a][f];}
else if(kind===3){out[a][f]=before[a][f]+before[b][f];out[b][f]=0;}
else if(kind===4)out[b][f]=before[a][f]*before[b][f];
else throw Error('unknown action');
result.innerText=JSON.stringify(out);
}catch(e){result.innerText='ERROR: '+String(e);}});</script></body></html>'''


class FormMeasurements:
    def __init__(self, browser, generator):
        self.browser=browser;self.generator=generator;self.fields=generator.fields;self.verbs=generator.verbs;self.log=[]
    @property
    def serial(self):return self.generator.serial
    @serial.setter
    def serial(self,v):self.generator.serial=v
    def state(self,*a,**kw):return self.generator.state(*a,**kw)
    def case(self,*a,**kw):return self.generator.case(*a,**kw)
    def observe(self,raw):
        s,words=raw;page=self.browser.page
        page.get_by_label('Observed world',exact=True).fill(json.dumps(s))
        page.get_by_label('Action tokens',exact=True).fill(json.dumps(words))
        page.get_by_role('button',name='Apply action',exact=True).click()
        text=page.get_by_role('status').inner_text();out=json.loads(text)
        self.log.append({'input':raw,'observed':out});return out


def trial(seed,executable_path=None):
    generator=Environment(seed);m=SemanticMemory();reports=[]
    with ChromiumEnvironment(app_html(generator),executable_path) as b:
        b.reset();e=FormMeasurements(b,generator)
        for k in range(4):
            inputs=[e.case(k,unequal=k==2) for _ in range(6)];tr=[[x,e.observe(x)] for x in inputs];p=generate_probes(tr,limit=144);a=[e.case(k,force_equal=k==2 and i%2==0) for i in range(12)]
            report=m.learn(tr,LiveMeasurements(e,p),LiveMeasurements(e,a),seed=seed)
            holdout=[]
            for i in range(8):
                x=e.case(k,size=5,spread=40,force_equal=k==2 and i%2==0);pred=m.predict(*x);observed=e.observe(x);holdout.append({'input':x,'predicted':pred,'observed':observed,'independent_reference_agrees':generator.observe(x)==observed})
            reports.append({'training':tr,'proposed_probes':p,'audit_inputs':a,'learning':report,'holdout':holdout})
        plan=plan_task(e,m);trace=e.log
    return {'seed':seed,'reports':reports,'planning':plan,'actual_button_clicks':len(trace),'visible_measurements':trace}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=371100);p.add_argument('--trials',type=int,default=2);p.add_argument('--browser-executable');p.add_argument('--output');a=p.parse_args()
    if not 1<=a.trials<=8:p.error('1–8 trials')
    runs=[trial(a.seed+i,a.browser_executable) for i in range(a.trials)]
    rows=[r for run in runs for rep in run['reports'] for r in rep['holdout']];summary={'trials':a.trials,'correct':sum(r['predicted']==r['observed'] for r in rows),'wrong':sum(r['predicted'] is not None and r['predicted']!=r['observed'] for r in rows),'abstained':sum(r['predicted'] is None for r in rows),'independent_reference_agreement':sum(r['independent_reference_agrees'] for r in rows),'plans_achieved':sum(r['planning']['success'] for r in runs),'actual_button_clicks':sum(r['actual_button_clicks'] for r in runs)}
    data={'summary':summary,'runs':runs,'claim':'Actual Chromium; authored, trusted local form with a predetermined typed interface. No arbitrary web or desktop competence.'}
    if a.output:Path(a.output).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0 if summary['correct']==len(rows) and summary['independent_reference_agreement']==len(rows) and summary['plans_achieved']==a.trials else 1

if __name__=='__main__':raise SystemExit(main())
