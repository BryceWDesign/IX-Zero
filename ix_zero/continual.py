"""Bounded lifelong neural-symbolic inquiry with replayable, flat provenance.

The host supplies measurements, calibration labels and access. The learner
chooses probes, sensor reuse, synthesis depth and recovery requests. Neither
neural scores nor prior task success can replace fresh symbolic verification.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import tempfile

from .checkpoint import canonical
from .epistemic import Inquiry
from .hybrid import Hybrid
from .neural import Dense
from .progressive import ProgressiveRecognition, MAX_DEPTH, MAX_PARENTS
from .operators import Domain, MAX_LIBRARY, nodes

MAX_EPISODES = 160
MAX_EVENTS = 24000
MAX_BYTES = 48_000_000
MAX_BANK = 8
MAX_RETRIES = 2
PROBES_PER_WIDTH = 2


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def rows_of(rows):
    if not isinstance(rows, (list, tuple)) or not 8 <= len(rows) <= 512:
        raise ValueError('invalid calibration sample count')
    result = []
    for x, y in rows:
        x = Dense.vector(x, 3)
        if type(y) is not int or y not in (0, 1):
            raise ValueError('binary label required')
        result.append((x, y))
    if set(y for _, y in result) != {0, 1} or len({x for x, _ in result}) != len(result):
        raise ValueError('distinct samples from both classes required')
    return result


def accuracy(net, rows):
    return sum(int(net.predict(x)[0] >= .5) == y for x, y in rows)


class Continual:
    """Single-writer journal; all models are reconstructed from observed events.

    Memory is bounded and separates executable laws from frozen recognition
    experts. Historical rules are predictions about their original scope, not
    universal facts. An invalidated source disables every recorded dependent.
    """
    def __init__(self, *, reuse=True, max_budget=3, probes=PROBES_PER_WIDTH, progressive=True, reuse_skills=True):
        if type(reuse) is not bool or type(progressive) is not bool or type(reuse_skills) is not bool or type(max_budget) is not int or not 1 <= max_budget <= 3 or type(probes) is not int or not 1 <= probes <= 8:
            raise ValueError('invalid continual configuration')
        self.config = dict(reuse=reuse, max_budget=max_budget, probes=probes, progressive=progressive, reuse_skills=reuse_skills)
        self.events = []
        self.records = {}
        self.bank = []
        self.active = None
        self.mind = None
        self.pending = None
        self.phase = 'idle'
        self.retry = 0
        self.calibration = []
        self.sensor_checks = []
        self.selected = None
        self.strategy_source = None
        self.probes = []
        self.recovery_reason = None

    def _room(self):
        if len(self.events) >= MAX_EVENTS:
            raise ValueError('continual journal limit')

    def _record(self, action, *args):
        self.events.append(json.loads(canonical([action, *args])))

    def live(self, key):
        return key in self.records and self.records[key]['state'] == 'validated'

    def _library(self, widths):
        if not self.config['reuse'] or not self.config['reuse_skills']:
            return ()
        result, seen = [], set()
        for key, r in reversed(list(self.records.items())):
            if not self.live(key) or r['family'] != 'word' or not set(widths) <= set(r['widths']):
                continue
            expr = r['expr']
            if expr in seen or nodes(expr) <= 1:
                continue
            seen.add(expr); result.append((key, expr))
            if len(result) == MAX_LIBRARY:
                break
        return tuple(reversed(result))

    def _start(self, family, widths, initial, train, audit, seed, budget, retry):
        if len(self.records) >= MAX_EPISODES:
            raise ValueError('continual episode limit')
        if type(budget) is not int or not 1 <= budget <= self.config['max_budget']:
            raise ValueError('invalid initial synthesis budget')
        key = 'e%04d' % (len(self.records) + 1)
        mind = Hybrid(family, widths, initial, train, audit, seed=seed, budget=budget)
        library = self._library(widths) if family == 'word' else ()
        if library:
            mind.inquiry = Inquiry(tuple(widths), budget, initial, train, audit, library)
        self.records[key] = dict(state='learning', family=family, widths=list(widths), initial=initial,
                                 seed=seed, budget=budget, retry=retry, library=[k for k, _ in library],
                                 dependencies=[], sensor_source=None, strategy_source=None, representation_sources=[],
                                 expr=None, model=None, training=[], audits=[], probes=[], challenges=[], reason=None)
        self.active, self.mind, self.retry = key, mind, retry
        self.phase, self.pending = 'sensor_check', None
        self.calibration, self.sensor_checks, self.probes = [], [], []
        self.selected = self.strategy_source = None
        self.recovery_reason = None
        # Transfer the most recent verified guidance model within this family.
        # Recognition experts are selected separately by independent sensor checks.
        if self.config['reuse']:
            for k, r in reversed(list(self.records.items())):
                if self.live(k) and r['family'] == family and 'strategy' in r:
                    mind.strategy = deepcopy(r['strategy']); self.strategy_source = k
                    break
        return key

    def begin(self, family, widths, initial, train, audit, *, seed=0, budget=1):
        self._room()
        if self.phase not in ('idle', 'complete', 'failed') or self.pending:
            raise ValueError('finish or deny the active inquiry first')
        key = self._start(family, widths, initial, train, audit, seed, budget, 0)
        self._record('begin', family, widths, initial, train, audit, seed, budget)
        return key

    def _fail(self, reason, *, deepen=False):
        r = self.records[self.active]
        r['state'], r['reason'] = 'invalid', reason
        self.pending = None
        self.recovery_reason = reason
        self.phase = 'recover' if self.retry < MAX_RETRIES and len(self.records) < MAX_EPISODES else 'failed'
        self.recovery_budget = min(self.config['max_budget'], r['budget'] + int(deepen and r['family'] == 'word'))

    def _boundary(self):
        e = self.mind.inquiry
        # Choose outside all host-supplied train/audit pools, not just unused
        # labels. Prefer boundary corners then farthest inputs from evidence.
        for w in e.widths:
            prior = [x[1:] for x, _ in self.probes if x[0] == w]
            if len(prior) >= self.config['probes']:
                continue
            t, a = e.pools[w]; excluded = set(t+a) | set(prior)
            if self.mind.family == 'word':
                high = (1 << w) - 1; xs = Domain(w).inputs
                def edge(x): return sum(v in (0, high) for v in x)
            else:
                high = w-1; xs = product(range(w), range(w), range(4))
                def edge(x): return sum(v in (0, high) for v in x[:2])
            available = [x for x in xs if x not in excluded]
            if not available:
                continue
            known = list(t+a)+prior
            def rank(x):
                return (edge(x), min(sum(abs(v-u) for v,u in zip(x,k)) for k in known), x)
            return (w, *max(available, key=rank))
        return None

    def _promote(self):
        m = self.mind; e = m.inquiry; r = self.records[self.active]
        if not e.finished or not e.validated() or any(not self.live(k) for k in r['library']):
            self._fail('source or independent validation failed'); return
        r['state'] = 'validated'
        if m.family == 'word':
            c = e.models[0]; r['expr'] = c.expr; r['dependencies'] = list(c.dependencies)
        else:
            r['model'] = e.models[0]
        r['sensor_source'] = self.selected
        r['recognition_depth'] = getattr(m.recognition,'depth',1)
        r['recognition_updates'] = m.recognition.updates
        r['strategy_source'] = self.strategy_source
        r['strategy'] = deepcopy(m.strategy)
        r['training'] = list(e.training.items())
        r['audits'] = list(e.audits.items())
        r['probes'] = deepcopy(self.probes)
        r['weights_digest'] = digest(m.weights())
        # Freeze independent recognition experts, preserving older capabilities.
        # Do not retain a duplicate expert if a previously validated one worked.
        if self.selected is None and self.config['reuse']:
            self.bank.append((self.active, deepcopy(m.recognition)))
            self.bank = self.bank[-MAX_BANK:]
        self.phase = 'complete'

    def next(self):
        self._room()
        if self.pending:
            return deepcopy(self.pending)
        r = None
        while r is None:
            if self.phase in ('idle', 'complete', 'failed'):
                r = dict(action=self.phase, episode=self.active)
            elif self.phase in ('sensor_check', 'calibrate', 'verify_sensor'):
                r = dict(action=self.phase, episode=self.active)
            elif self.phase == 'recover':
                r = dict(action='recover', episode=self.active, reason=self.recovery_reason, budget=self.recovery_budget,
                         widths=self.records[self.active]['widths'], initial=self.records[self.active]['initial'])
            elif self.phase == 'inquiry':
                q = self.mind.next()
                if q['action'] == 'complete': self.phase = 'boundary'
                elif q['action'] == 'failed': self._fail(q.get('reason') or 'inquiry failed', deepen=True)
                else: r = dict(q, episode=self.active)
            elif self.phase == 'boundary':
                x = self._boundary()
                if x is None: self._promote()
                else: r = dict(action='probe', input=x, episode=self.active)
            else: raise RuntimeError('invalid state')
        if r['action'] not in ('idle','complete','failed'):
            self.pending = deepcopy(r)
        self._record('next')
        return deepcopy(r)

    def answer(self, value):
        self._room()
        if not self.pending: raise ValueError('no pending request')
        q = self.pending; a = q['action']; m = self.mind
        if a == 'sensor_check':
            rows = rows_of(value)
            self.sensor_checks = rows
            for key, net in reversed(self.bank):
                if self.live(key) and accuracy(net, rows) == len(rows):
                    m.recognition = deepcopy(net); self.selected = key; break
            self.phase = 'inquiry' if self.selected else 'calibrate'
            if not self.selected and self.config['progressive']:
                parents=[(k,n) for k,n in self.bank if self.live(k) and getattr(n,'depth',1)<MAX_DEPTH][-MAX_PARENTS:]
                if parents:
                    m.recognition=ProgressiveRecognition([n for _,n in parents],m.config['seed'])
                    self.records[self.active]['representation_sources']=[k for k,_ in parents]
        elif a == 'calibrate':
            rows = rows_of(value)
            if {x for x,_ in rows} & {x for x,_ in self.sensor_checks}:
                raise ValueError('calibration overlaps sensor check')
            m.calibrate(rows); self.calibration = rows; self.phase = 'verify_sensor'
        elif a == 'verify_sensor':
            rows = rows_of(value)
            if {x for x,_ in rows} & {x for x,_ in self.calibration+self.sensor_checks}:
                raise ValueError('verification overlaps earlier sensor evidence')
            if accuracy(m.recognition, rows) == len(rows): self.phase = 'inquiry'
            else: self._fail('independent sensor verification failed')
        elif a == 'recover':
            old = self.records[self.active]
            # Host must explicitly grant a fresh investigation. No old labels
            # or old model are carried into the revalidated law.
            self._start(old['family'], old['widths'], old['initial'], value['train'], value['audit'],
                        (old['seed']+1) % (2**32-2), self.recovery_budget, self.retry+1)
        elif a == 'probe':
            x = q['input']; ok = m.challenge(x,value)
            self.probes.append((x,deepcopy(value)))
            if not ok: self._fail('self-selected boundary observation contradicted law')
        else:
            m.answer(value)
        self.pending = None
        self._record('answer', value)

    def deny(self):
        self._room()
        if not self.pending: raise ValueError('no pending request')
        r = self.records[self.active]; r['state'] = 'invalid'; r['reason'] = 'host denied access'
        self.phase, self.pending = 'failed', None
        self._record('deny')

    def _invalidate(self, key):
        invalid = {key}
        while True:
            extra = {k for k,r in self.records.items() if self.live(k) and invalid.intersection(r['dependencies'] + r['representation_sources'] + [r['sensor_source'],r['strategy_source']])}
            if extra <= invalid: break
            invalid |= extra
        for k in invalid:
            self.records[k]['state'] = 'invalid'
            self.records[k]['reason'] = 'contradicted source or dependent'
        self.bank = [(k,n) for k,n in self.bank if self.live(k)]
        return invalid

    def predict(self, key, x):
        if not self.live(key): return None
        r = self.records[key]; w = x[0]
        if type(w) is not int or w not in r['widths']: return None
        if r['family'] == 'word':
            from .operators import execute
            return execute(r['expr'], x[1:], w)
        from .hybrid import grid_step
        if len(x) != 4 or any(type(v) is not int for v in x) or not (0<=x[1]<w and 0<=x[2]<w and 0<=x[3]<4):
            raise ValueError('invalid grid input')
        return grid_step(r['model'], x)

    def compose_plan(self, width, start, goal, skills, operands=(0,), max_steps=8):
        """Bounded iterative-deepening search over previously audited word skills.

        Return only a proposal, with each transition evaluated by the retained
        executable law. No unrestricted code or external action is possible.
        """
        d=Domain(width);start=d.word(start);goal=d.word(goal)
        if type(max_steps) is not int or not 0<=max_steps<=8 or not 1<=len(skills)<=MAX_LIBRARY or len(set(skills))!=len(skills) or not 1<=len(operands)<=8:
            raise ValueError('invalid composition search bounds')
        operands=tuple(d.word(x) for x in operands)
        if any(not self.live(k) or self.records[k]['family']!='word' or width not in self.records[k]['widths'] for k in skills):return None
        actions=tuple((k,b) for k in sorted(skills) for b in sorted(set(operands)))
        # Build each finite transition once, then deepen without changing laws.
        edges={s:[(k,b,self.predict(k,(width,s,b))) for k,b in actions] for s in range(d.mask+1)}
        for limit in range(max_steps+1):
            best={};
            def search(state, remaining):
                if state==goal:return []
                if not remaining or best.get(state,-1)>=remaining:return None
                best[state]=remaining
                for k,b,nxt in edges[state]:
                    tail=search(nxt,remaining-1)
                    if tail is not None:return [dict(skill=k,operand=b,before=state,after=nxt)]+tail
                return None
            result=search(start,limit)
            if result is not None:return result
        return None

    def challenge(self, key, x, value):
        self._room()
        if self.phase != 'complete' or self.pending or not self.live(key):
            raise ValueError('challenge requires a completed inquiry')
        r = self.records[key]; x = tuple(x); pred = self.predict(key,x)
        if pred is None: raise ValueError('outside validated scope')
        if r['family']=='word': value=Domain(x[0]).word(value)
        else:
            if not isinstance(value,(tuple,list)) or len(value)!=2 or any(type(v) is not int or not 0<=v<x[0] for v in value): raise ValueError('invalid grid output')
            value=tuple(value)
        if x in [tuple(z[0]) for z in r['challenges']]: raise ValueError('duplicate challenge')
        # New temporal observations may revisit known inputs: the environment
        # can change. They are never silently mixed into the previous law.
        ok = pred == value; r['challenges'].append((x,value,ok))
        if not ok:
            invalid = self._invalidate(key)
            if self.active in invalid: self._fail('later measurement contradicted source')
        self._record('challenge',key,x,value)
        return ok

    def status(self):
        records = {k:{n:v for n,v in r.items() if n!='strategy'} for k,r in self.records.items()}
        return deepcopy(dict(active=self.active, phase=self.phase, records=records, sensor_bank=[k for k,_ in self.bank],
                             pending=self.pending, retries=self.retry, selected_sensor=self.selected,
                             strategy_source=self.strategy_source))

    def fingerprint(self):
        return digest(dict(status=self.status(), weights=None if self.mind is None else self.mind.weights(),
                           bank=[(k,n.state()) for k,n in self.bank],
                           strategies=[(k,r['strategy'].state()) for k,r in self.records.items() if 'strategy' in r],
                           current=None if self.mind is None else self.mind.payload(),
                           recovery=getattr(self,'recovery_budget',None), calibration=self.calibration,
                           checks=self.sensor_checks, probes=self.probes))

    def save(self,path):
        data=dict(version=1,config=self.config,events=self.events,fingerprint=self.fingerprint())
        blob=canonical(dict(payload=data,sha256=digest(data)))
        if len(blob)>MAX_BYTES: raise ValueError('continual checkpoint too large')
        path=Path(path); fd,tmp=tempfile.mkstemp(prefix='.ix-continual-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f: blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES:raise ValueError('continual checkpoint too large')
        obj=json.loads(blob); d=obj['payload']
        if d['version']!=1 or digest(d)!=obj['sha256'] or len(d['events'])>MAX_EVENTS:raise ValueError('continual checkpoint integrity')
        m=cls(**d['config'])
        for action,*args in d['events']:
            if action=='begin':m.begin(*args[:5],seed=args[5],budget=args[6])
            elif action=='next':m.next()
            elif action=='answer':m.answer(*args)
            elif action=='deny':m.deny()
            elif action=='challenge':m.challenge(*args)
            else:raise ValueError('unknown journal event')
        if canonical(m.events)!=canonical(d['events']) or m.fingerprint()!=d['fingerprint']:
            raise ValueError('continual checkpoint replay mismatch')
        return m
