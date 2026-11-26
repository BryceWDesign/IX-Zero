"""Local JSON-lines host interface for bounded continual inquiry."""
import argparse
import json
import sys
from .continual import Continual, MAX_BYTES

class Console:
    def __init__(self,mind=None):self.mind=mind or Continual()
    def execute(self,request):
        if not isinstance(request,dict):raise ValueError('expected command object')
        a=dict(request);c=a.pop('command');m=self.mind
        if c=='new':self.mind=Continual(**a);return {'created':True}
        if c=='load':self.mind=Continual.load(a['path']);return {'loaded':True}
        if c=='begin':return {'episode':m.begin(**a)}
        if c=='next':return m.next()
        if c=='answer':m.answer(a['value']);return {'accepted':True}
        if c=='deny':m.deny();return {'denied':True}
        if c=='predict':return {'prediction':m.predict(a['episode'],a['input'])}
        if c=='challenge':return {'consistent':m.challenge(a['episode'],a['input'],a['value'])}
        if c=='compose_plan':return {'proposal':m.compose_plan(**a)}
        if c=='status':return m.status()
        if c=='save':m.save(a['path']);return {'saved':True}
        raise ValueError('unknown command')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--load');a=p.parse_args();c=Console(Continual.load(a.load) if a.load else None)
    while True:
        line=sys.stdin.readline(MAX_BYTES+1)
        if not line:return 0
        if len(line)>MAX_BYTES:print(json.dumps({'error':'command exceeds size limit'}),flush=True);return 1
        try:print(json.dumps(c.execute(json.loads(line)),allow_nan=False),flush=True)
        except (ValueError,TypeError,KeyError,IndexError,OSError,UnicodeError) as e:print(json.dumps({'error':str(e)}),flush=True)

if __name__=='__main__':raise SystemExit(main())
