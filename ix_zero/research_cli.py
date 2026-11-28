"""JSON-lines interface for the v5 experimental research workspace."""
import argparse
import json
import sys
from .research_workspace import ResearchWorkspace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--load');p.add_argument('--workspace',help='optional authorized read-only documentation root');a=p.parse_args()
    w=ResearchWorkspace.load(a.load,workspace=a.workspace) if a.load else ResearchWorkspace(workspace=a.workspace)
    while True:
        line=sys.stdin.readline(2_000_001)
        if not line:return 0
        if len(line)>2_000_000:print(json.dumps({'status':'error','error':'command size'}),flush=True);return 1
        try:
            req=json.loads(line)
            if req.get('command')=='save':w.save(req['path']);r={'saved':True}
            else:r=w.execute(req)
            print(json.dumps(r,allow_nan=False),flush=True)
        except (ValueError,TypeError,KeyError,IndexError,OSError,AttributeError,ArithmeticError,RecursionError) as e:
            print(json.dumps({'status':'error','error':str(e)}),flush=True)

if __name__=='__main__':raise SystemExit(main())
