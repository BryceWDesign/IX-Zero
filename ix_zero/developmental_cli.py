"""Interactive local developmental curriculum, with JSON data commands."""
import argparse
import json
from pathlib import Path
from .developmental import Development

HELP='''Commands (JSON data only; no executable input):
add {"name": "task", "spec": {...}}    Register an unlabeled objective
offer {"name":"domain","labels":[...],"train":[...],"audit":[...]}  Infer prerequisites
next                                   Select next evidence request
answer 3                               Supply actual observed outcome (any valid JSON value)
answer {"train":[...],"audit":[...]}  Grant a requested wider environment (unlabeled pools)
challenge {"name":"task","x":[...],"y":3}  Challenge a live learned word law
abandon                                Decline pending request without promotion
status                                 Inspect learning jobs and their evidence counts
predict {"name":"task","input":[1,2]}  Predict using a live acquired model
execute {"grammar":"grammar","text":"...","rows":[...],"values":{...},"relations":{...}}
recover {"name":"world","new":"revision","x":[...],"y":3,"train":[...],"audit":[...]}
retract "task"                         Invalidate an acquired word operator
save                                   Save current journal
load                                   Reload and replay journal
help / exit
See docs/DEVELOPMENTAL.md for task schemas and reproduction.'''

class Console:
    def __init__(self,path):self.path=Path(path);self.mind=Development()
    def command(self,line):
        op,_,raw=line.strip().partition(' ')
        payload=json.loads(raw) if raw else None
        if op=='help':return HELP
        if op=='status' and payload is None:return self.mind.status()
        if op=='next' and payload is None:return self.mind.next()
        if op=='add':return self.mind.add(**payload)
        if op=='offer':return self.mind.offer(**payload)
        if op=='answer':return self.mind.answer(payload)
        if op=='abandon' and payload is None:return self.mind.abandon()
        if op=='predict':return self.mind.predict(payload['name'],payload['input'])
        if op=='execute':return self.mind.execute(**payload)
        if op=='recover':return self.mind.recover(**payload)
        if op=='retract':return self.mind.retract(payload)
        if op=='challenge':return self.mind.challenge(**payload)
        if op=='save' and payload is None:self.mind.save(self.path);return {'saved':str(self.path)}
        if op=='load' and payload is None:self.mind=Development.load(self.path);return {'loaded':str(self.path)}
        raise ValueError('unknown or malformed command')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',default='ix-zero-developmental.json');p.add_argument('--check',action='store_true');p.add_argument('--demo',action='store_true');a=p.parse_args()
    if a.check:
        from itertools import product
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            c=Console(Path(tmp)/'state.json');xs=list(product(range(8),repeat=2));spec=dict(kind='operator',deps=[],width=3,budget=5,train=xs[:24],audit=xs[24:27]);c.command('add '+json.dumps(dict(name='sample',spec=spec)));r=c.command('next');c.command('save');c.command('load');assert c.mind.pending['input']==r['input'];c.command('answer '+str(r['input'][0]^r['input'][1]));assert c.command('predict {"name":"sample","input":[1,2]}') is None;print('Developmental console checkpoint smoke test passed.');return
    c=Console(a.state);print('IX-Zero | developmental curriculum');print(HELP)
    demo_scene=None
    if a.demo:
        from .developmental_experiment import trial
        c.mind,rows,values,relations=trial(208000,True,demo=True);demo_scene=(rows,values,relations)
        print('Synthetic demonstration: acquired curriculum restored from replay.')
        print('Scene:',json.dumps(rows));print('Values:',json.dumps(values));print('Relations:',json.dumps(relations))
        print('Try: compare inspect red sphere combine red cube with green cube')
        print('Try: join red cube with green cube')
    if c.path.exists():print('A journal exists; use load to replay it.')
    while True:
        try:
            text=input('You > ').strip()
            if text.lower() in ('quit','exit'):return
            if demo_scene and text.split(' ',1)[0] not in ('help','status','next','add','offer','answer','abandon','predict','execute','recover','retract','challenge','save','load'):
                rows,values,relations=demo_scene;r=c.mind.execute('grammar',text,rows,values,relations)
            else:r=c.command(text)
            print('IX-Zero > '+(r if isinstance(r,str) else json.dumps(r,ensure_ascii=False)))
        except (EOFError,KeyboardInterrupt):return
        except (ValueError,KeyError,TypeError,OSError) as e:print('IX-Zero > Refused: '+str(e))

if __name__=='__main__':main()
