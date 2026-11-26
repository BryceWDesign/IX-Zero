"""Local JSON-lines interface for calibrated raster and symbolic inquiry."""
import argparse
import json
from pathlib import Path
import sys
from .hybrid import Hybrid, MAX_BYTES


def ppm(path):
    """Read a bounded ASCII PPM (P3), returning normalized RGB observations.

    This is a file adapter, not a camera driver or a general vision model.
    """
    with open(path,'rb') as f:blob=f.read(16385)
    if len(blob)>16384:raise ValueError('PPM file exceeds size limit')
    tokens=' '.join(line.split('#',1)[0] for line in blob.decode('ascii').splitlines()).split()
    if len(tokens)<4 or tokens[0]!='P3':raise ValueError('expected ASCII PPM P3')
    width,height,maximum=map(int,tokens[1:4]);values=list(map(int,tokens[4:]))
    if not (1<=width<=8 and 1<=height<=8 and 1<=maximum<=65535) or len(values)!=3*width*height or any(not 0<=v<=maximum for v in values):raise ValueError('invalid PPM dimensions or samples')
    return width,height,[[v/maximum for v in values[i:i+3]] for i in range(0,len(values),3)]


class Console:
    def __init__(self):self.mind=None

    def execute(self,r):
        if not isinstance(r,dict) or 'command' not in r:raise ValueError('expected command object')
        c=r['command'];a={k:v for k,v in r.items() if k!='command'}
        if c=='new':self.mind=Hybrid(**a);return {'created':True}
        if c=='load':self.mind=Hybrid.load(a['path']);return {'loaded':True}
        if self.mind is None:raise ValueError('create or load an inquiry first')
        m=self.mind
        if c=='next':return m.next()
        if c=='calibrate':m.calibrate(a['rows']);return {'calibrated':True}
        if c=='answer':m.answer(a['value']);return {'accepted':True}
        if c=='answer_ppm':
            req=m.pending
            if not req or req['action']!='train' or m.mode=='oracle':raise ValueError('a raster training request is required')
            w,h,p=ppm(a['path']);n=req['input'][0]
            if (w,h)!=((n,1) if m.family=='word' else (n,n)):raise ValueError('PPM dimensions do not match request')
            m.answer(p);return {'accepted':True}
        if c=='deny':m.deny();return {'denied':True}
        if c=='predict':return {'prediction':m.predict(a['input'])}
        if c=='neural_prediction':return {'unvalidated_proposal':m.neural_prediction(a['input'])}
        if c=='challenge':return {'consistent':m.challenge(a['input'],a['value'])}
        if c=='plan':return {'proposal':m.plan(a['start'],a['goal'],a['width'])}
        if c=='save':m.save(a['path']);return {'saved':True}
        if c=='transfer':self.mind=m.transfer(**a);return {'transferred':True}
        if c=='status':return dict(family=m.family,finished=m.inquiry.finished,blocked=m.inquiry.blocked,validated=m.inquiry.finished and m.inquiry.validated(),remaining=len(m.inquiry.models),training=len(m.inquiry.training),audits=len(m.inquiry.audits),updates={k:v['updates'] for k,v in m.weights().items()})
        raise ValueError('unknown command')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--load',type=Path);args=parser.parse_args();c=Console()
    if args.load:c.mind=Hybrid.load(args.load)
    while True:
        line=sys.stdin.readline(MAX_BYTES+1)
        if not line:return 0
        if len(line)>MAX_BYTES:print(json.dumps({'error':'command exceeds size limit'}),flush=True);return 1
        try:print(json.dumps(c.execute(json.loads(line)),allow_nan=False),flush=True)
        except (ValueError,TypeError,KeyError,IndexError,OSError,UnicodeError) as exc:print(json.dumps({'error':str(exc)}),flush=True)


if __name__=='__main__':raise SystemExit(main())
