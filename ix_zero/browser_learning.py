"""Learn a deterministic visible-control transition graph by actual interaction.

The learner receives only visible text and button affordances. It does not
inspect application scripts or hidden state. The optional Playwright adapter
uses a fresh offline browser context with operator-provided TRUSTED HTML.
This is not a general desktop agent or a hardened untrusted-code sandbox.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json


@dataclass(frozen=True)
class Observation:
    text: str
    controls: tuple
    @property
    def key(self):
        return hashlib.sha256(json.dumps([self.text,self.controls],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


class BrowserLearner:
    def __init__(self):self.states={};self.transitions={};self.conflicts=set();self.trace=[]
    def observe(self,obs):
        if not isinstance(obs,Observation) or len(obs.text)>20000 or len(obs.controls)>64 or any(not isinstance(x,str) or not x or len(x)>200 for x in obs.controls) or len(set(obs.controls))!=len(obs.controls):raise ValueError('unsupported visible state')
        self.states[obs.key]=obs;return obs.key
    def record(self,source,control,obs):
        if source not in self.states or control not in self.states[source].controls:raise ValueError('unobserved transition')
        target=self.observe(obs);edge=(source,control)
        if edge in self.transitions and self.transitions[edge]!=target:
            self.conflicts.add(edge);self.transitions.pop(edge,None)
        elif edge not in self.conflicts:self.transitions[edge]=target
        self.trace.append({'source':source,'control':control,'target':target,'conflict':edge in self.conflicts});return target
    def path(self,start,goal):
        q=deque([(start,())]);seen={start}
        while q:
            s,p=q.popleft()
            if goal(self.states[s]):return p
            for a in self.states[s].controls:
                nxt=self.transitions.get((s,a))
                if nxt is not None and nxt not in seen:
                    seen.add(nxt);q.append((nxt,p+((s,a,nxt),)))
        return None
    def learn(self,env,*,click_budget=100,max_states=32):
        if not 1<=click_budget<=1000 or not 1<=max_states<=128:raise ValueError('exploration budget')
        start=self.observe(env.reset());clicks=0;done=set();reason='exhausted_visible_graph'
        while True:
            # Prefer unexplored controls reachable by the shortest observed path.
            path=self.path(start,lambda obs:any((obs.key,a) not in done for a in obs.controls))
            if path is None:break
            state=path[-1][2] if path else start
            control=next(a for a in self.states[state].controls if (state,a) not in done)
            if clicks+len(path)+1>click_budget or len(self.states)>=max_states:reason='budget';break
            cur=self.observe(env.reset())
            if cur!=start:reason='reset_changed';break
            stable=True
            for s,a,nxt in path:
                if cur!=s:stable=False;break
                cur=self.record(cur,a,env.click(a));clicks+=1
                if cur!=nxt:stable=False;break
            if not stable:reason='replay_contradiction';break
            self.record(cur,control,env.click(control));clicks+=1;done.add((cur,control))
        return {'status':reason,'states':len(self.states),'transitions':len(self.transitions),'conflicts':len(self.conflicts),'clicks':clicks,'trace':list(self.trace)}
    def achieve(self,env,goal,*,click_budget=16):
        if not isinstance(goal,str) or not goal.strip() or len(goal)>200 or not 0<=click_budget<=1000:raise ValueError('goal or budget')
        start=self.observe(env.reset());match=lambda obs:goal in obs.text.splitlines();path=self.path(start,match)
        if path is None:return {'status':'unknown_route','clicks':0,'goal':goal}
        if len(path)>click_budget:return {'status':'budget','clicks':0,'goal':goal,'planned_length':len(path)}
        cur=start;actions=[]
        for s,a,nxt in path:
            if cur!=s:return {'status':'replay_contradiction','clicks':len(actions),'actions':actions,'goal':goal}
            cur=self.record(cur,a,env.click(a));actions.append(a)
            if cur!=nxt:return {'status':'replay_contradiction','clicks':len(actions),'actions':actions,'goal':goal}
        return {'status':'achieved' if match(self.states[cur]) else 'not_achieved','clicks':len(actions),'actions':actions,'goal':goal,'observed_text':self.states[cur].text}

    def snapshot(self):
        payload={'version':1,'states':{k:{'text':v.text,'controls':list(v.controls)} for k,v in sorted(self.states.items())},'transitions':[[s,a,t] for (s,a),t in sorted(self.transitions.items())],'conflicts':sorted(self.conflicts)}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        return {'payload':payload,'sha256':digest}
    @classmethod
    def restore(cls,snapshot):
        p=snapshot['payload'];digest=hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if digest!=snapshot['sha256'] or p['version']!=1 or len(p['states'])>128 or len(p['transitions'])>8192 or len(p['conflicts'])>8192:raise ValueError('browser memory integrity or size')
        m=cls()
        for key,entry in p['states'].items():
            if m.observe(Observation(entry['text'],tuple(entry['controls'])))!=key:raise ValueError('state identity mismatch')
        for s,a,t in p['transitions']:
            if s not in m.states or t not in m.states or a not in m.states[s].controls or (s,a) in m.transitions:raise ValueError('invalid transition')
            m.transitions[s,a]=t
        for s,a in p['conflicts']:
            if s not in m.states or a not in m.states[s].controls or (s,a) in m.transitions:raise ValueError('invalid conflict')
            m.conflicts.add((s,a))
        return m


class ChromiumEnvironment:
    def __init__(self,html,executable_path=None):
        if not isinstance(html,str) or len(html)>1_000_000:raise ValueError('trusted local HTML required')
        self.html=html;self.executable_path=executable_path;self.manager=None;self.browser=None;self.context=None;self.page=None
    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self.manager=sync_playwright().start()
        try:
            opts={'headless':True}
            if self.executable_path:opts['executable_path']=self.executable_path
            self.browser=self.manager.chromium.launch(**opts)
            self.context=self.browser.new_context(offline=True,accept_downloads=False,service_workers='block')
            self.context.route('**/*',lambda route:route.abort())
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,*_):
        try:
            if self.context:self.context.close()
            if self.browser:self.browser.close()
        finally:
            if self.manager:self.manager.stop()
    def reset(self):
        if self.page:self.page.close()
        self.page=self.context.new_page();self.page.set_default_timeout(3000);self.page.set_content(self.html,wait_until='domcontentloaded',timeout=5000)
        return self.observe()
    def observe(self):
        controls=[]
        for loc in self.page.get_by_role('button').all():
            if loc.is_visible() and loc.is_enabled():controls.append((loc.get_attribute('aria-label') or loc.inner_text()).strip())
        text='\n'.join(s.strip() for s in self.page.locator('body').inner_text().splitlines() if s.strip())
        return Observation(text,tuple(controls))
    def click(self,label):
        if label not in self.observe().controls:raise ValueError('control not observed')
        self.page.get_by_role('button',name=label,exact=True).click();return self.observe()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('html',help='operator-provided trusted local HTML');p.add_argument('--goal',required=True,help='exact line of desired visible state');p.add_argument('--browser-executable');p.add_argument('--learn-budget',type=int,default=100);p.add_argument('--goal-budget',type=int,default=16);p.add_argument('--save');p.add_argument('--load');a=p.parse_args()
    html=Path(a.html).read_text(encoding='utf-8');m=BrowserLearner.restore(json.loads(Path(a.load).read_text(encoding='utf-8'))) if a.load else BrowserLearner()
    with ChromiumEnvironment(html,a.browser_executable) as e:
        training={'status':'restored','states':len(m.states)} if a.load else m.learn(e,click_budget=a.learn_budget)
        result=m.achieve(e,a.goal,click_budget=a.goal_budget)
    if a.save:Path(a.save).write_text(json.dumps(m.snapshot(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'learning':training,'result':result},indent=2))
    return 0 if result['status']=='achieved' else 2

if __name__=='__main__':raise SystemExit(main())
