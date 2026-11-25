"""Replay-validated atomic local snapshots. Digest detects accidental corruption,
not malicious forgery. Single-writer usage only; no credentials or executable code.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from .core import Mind, Space, Task

LIMIT=2_000_000

def canonical(data):
    return json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def payload(mind):
    return {'version':1,'tasks':[{'name':n,'space':[t.space.width,t.space.modulus], 'library':t.library,'depth':t.depth,'observations':list(t.observations.items()),'audits':[(x,p,y) for x,(p,y) in t.audits.items()], 'contradictions':t.contradictions} for n,t in mind.tasks.items()], 'concepts':list(mind.concepts.items())}

def save(mind,path):
    path=Path(path); data=payload(mind)
    blob=canonical({'payload':data,'sha256':hashlib.sha256(canonical(data)).hexdigest()})
    if len(blob)>LIMIT: raise ValueError('checkpoint too large')
    fd,tmp=tempfile.mkstemp(prefix='.ixzero-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load(path):
    with open(path,'rb') as f: blob=f.read(LIMIT+1)
    if len(blob)>LIMIT: raise ValueError('checkpoint too large')
    obj=json.loads(blob); data=obj['payload']
    if hashlib.sha256(canonical(data)).hexdigest()!=obj['sha256'] or data['version']!=1:
        raise ValueError('checkpoint digest or version mismatch')
    m=Mind()
    # Restore the exact grammar snapshot, not the current library.
    for row in data['tasks']:
        t=m.new(row['name'],Space(*row['space']),reuse=False,depth=row['depth'])
        t=Task(t.space,tuple(tuple(p) for p in row['library']),t.depth); m.tasks[row['name']]=t
        for x,y in row['observations']: t.observe(x,y)
        for x,p,y in row['audits']:
            t.audit(x,y)
            if t.audits[tuple(x)][0]!=(None if p is None else tuple(p)): raise ValueError('audit replay mismatch')
        for x,old,y in row['contradictions']:
            if t.observations.get(tuple(x))!=tuple(old) or tuple(old)==tuple(y): raise ValueError('invalid contradiction')
            t.observe(x,y)
    # Historical promotions may have since been invalidated. Validate by replay
    # of the pre-contradiction prefix, but do not enable invalidated concepts.
    for n,p in data['concepts']:
        t=m.tasks[n]
        copy=Task(t.space,t.library,t.depth)
        for x,y in t.observations.items(): copy.observe(x,y)
        # A source may have a later failing audit after promotion.
        for x,(pred,y) in t.audits.items():
            if pred is None or pred!=y: break
            copy.audit(x,y)
        if copy.concept()!=tuple(p): raise ValueError('concept replay mismatch')
        m.promote(n) if not t.blocked else m.concepts.__setitem__(n,tuple(p))
    if canonical(payload(m))!=canonical(data): raise ValueError('checkpoint replay mismatch')
    return m
