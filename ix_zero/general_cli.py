"""Local JSON-lines interface for the integrated reasoning workspace."""
import argparse
import json
import sys
from .general import General, MAX_FILE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--load');a=p.parse_args()
    m=General.load(a.load) if a.load else General()
    while True:
        line=sys.stdin.readline(MAX_FILE+1)
        if not line:return 0
        if len(line)>MAX_FILE:print(json.dumps({'error':'command size limit'}),flush=True);return 1
        try:
            req=json.loads(line)
            if isinstance(req,dict) and req.get('command')=='save':m.save(req['path']);result={'saved':True}
            else:result=m.execute(req)
            print(json.dumps(result,allow_nan=False),flush=True)
        except (ValueError,TypeError,KeyError,IndexError,OSError,SyntaxError,ArithmeticError,RecursionError,UnicodeError) as exc:
            print(json.dumps({'status':'error','error':str(exc)}),flush=True)


if __name__=='__main__':raise SystemExit(main())
