"""Finite contextual hypothesis generation and counterexample-guided recovery.

The evaluator supplies observations and unlabeled contexts. No target function
or audit oracle is available to the learner. All candidate execution is through
bounded, live ACCE capabilities or finite constants, never arbitrary code.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import product
from importlib.resources import files
from .acce import Capability, valid
from ._ix_vendor import parse_ix, run_ix

MIN_TRAINING = 6
MIN_AUDITS = 3


@dataclass(frozen=True)
class Hypothesis:
    context: str | None
    condition: tuple
    alternative: str
    table: tuple

    def description(self):
        return {'context': self.context, 'condition': self.condition,
                'alternative': self.alternative}


class Domain:
    def __init__(self, labels, contexts):
        self.labels = tuple(labels); self.contexts = tuple(contexts)
        if not 3 <= len(self.labels) <= 8 or any(not valid('integer', x) for x in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError('expected 3–8 distinct bounded integer labels')
        if not 2 <= len(self.contexts) <= 4 or any(not valid('relation', c) for c in self.contexts) or len(set(self.contexts)) != len(self.contexts):
            raise ValueError('expected 2–4 distinct context identifiers')
        self.inputs = tuple(product(self.contexts, self.labels, self.labels))
        self.index = {x: i for i, x in enumerate(self.inputs)}

    def key(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 3 or type(x[0]) is not str or any(type(v) is not int for v in x[1:]) or tuple(x) not in self.index:
            raise ValueError('input outside declared context domain')
        return tuple(x)

    def output(self, y):
        if not valid('integer', y) or y not in self.labels:
            raise ValueError('outcome outside declared context domain')
        return y


def condition_holds(condition, a, b):
    op, *args = condition
    if op == 'always': return True
    if op == 'eq': return a == b
    if op == 'lt': return a < b
    if op == 'gt': return a > b
    if op == 'left': return a == args[0]
    if op == 'right': return b == args[0]
    if op == 'pair': return (a, b) == tuple(args)
    raise ValueError('unknown condition')


class Episode:
    """One frozen version space. A contradiction never modifies it in place."""
    def __init__(self, domain, base, alternatives=(), expanded=False):
        self.domain = domain; self.base = base; self.alternatives = tuple(alternatives)
        self.sources = (base, *self.alternatives)
        if len(self.sources) > 4 or len({c.name for c in self.sources}) != len(self.sources) or any(c.inputs != ('integer','integer') or c.output != 'integer' for c in self.sources):
            raise ValueError('invalid bounded capability portfolio')
        self.expanded = expanded; self.training = {}; self.audits = {}; self.feedback = None
        self.blocked = False; self.models = (); self.generated = 0
        if not all(c.available() for c in self.sources):
            self.blocked = True; return
        pairs = tuple(product(domain.labels, repeat=2))
        tables = {c.name: tuple(c.call(x) for x in pairs) for c in self.sources}
        if any(y not in domain.labels or type(y) is not int for t in tables.values() for y in t):
            raise ValueError('source capability does not cover finite domain')
        n = len(pairs); incumbent = tables[base.name] * len(domain.contexts)
        unique = {incumbent: Hypothesis(None, ('always',), base.name, incumbent)}
        self.generated = 1
        if expanded:
            conditions = [('always',), ('eq',), ('lt',), ('gt',)]
            conditions += [(op, v) for op in ('left','right') for v in domain.labels]
            conditions += [('pair', a, b) for a,b in pairs]
            choices = [(c.name, tables[c.name]) for c in self.alternatives]
            choices += [(f'constant:{v}', (v,)*n) for v in domain.labels]
            for ctx, cond, (name, table) in product(domain.contexts, conditions, choices):
                # Guard language is bounded. No arbitrary condition strings are evaluated.
                mask = tuple(c == ctx and condition_holds(cond,a,b) for c,a,b in domain.inputs)
                prediction = tuple(table[i % n] if hit else incumbent[i] for i,hit in enumerate(mask))
                self.generated += 1
                unique.setdefault(prediction, Hypothesis(ctx,cond,name,prediction))
        self.models = tuple(unique.values())

    def live(self):
        return not self.blocked and bool(self.models) and all(c.available() for c in self.sources)

    def diagnose(self):
        if not all(c.available() for c in self.sources): return 'source_invalidated'
        if self.feedback is not None: return 'contradicted_prediction'
        if not self.models: return 'representation_exhausted'
        if self.blocked: return 'audit_failed'
        if len(self.training) < MIN_TRAINING: return 'insufficient_observations'
        if len(self.models) > 1: return 'competing_hypotheses'
        if len(self.audits) < MIN_AUDITS: return 'insufficient_validation'
        return 'validated'

    def observe(self, x, y):
        x, y = self.domain.key(x), self.domain.output(y)
        if not self.live() or self.audits or x in self.training: raise ValueError('training blocked, frozen or repeated')
        self.training[x] = y; p = self.domain.index[x]
        self.models = tuple(m for m in self.models if m.table[p] == y)
        if not self.models: self.blocked = True

    def predict(self, x, validated=True):
        p = self.domain.index[self.domain.key(x)]
        if not self.live() or len(self.training) < MIN_TRAINING or (validated and len(self.audits) < MIN_AUDITS): return None
        vals = {m.table[p] for m in self.models}
        return next(iter(vals)) if len(vals) == 1 else None

    def audit(self, x, y):
        x, y = self.domain.key(x), self.domain.output(y)
        if not self.live() or len(self.training) < MIN_TRAINING or x in self.training or x in self.audits: raise ValueError('invalid or repeated audit')
        ok = self.predict(x, False) == y
        self.audits[x] = (y, ok)
        if not ok: self.blocked = True
        return ok

    def contradict(self, x, y):
        """Feedback can contradict a previously seen input, unlike fresh audits."""
        x,y = self.domain.key(x), self.domain.output(y)
        p = self.predict(x)
        if p is None or p == y: raise ValueError('no validated contradictory prediction')
        self.feedback = (x, y, p); self.blocked = True

    def inquiry(self, pool):
        """Construct a question whose possible answers distinguish hypotheses."""
        if not self.live() or self.audits: return None
        pool = tuple(dict.fromkeys(self.domain.key(x) for x in pool))
        if len(pool) > len(self.domain.inputs): raise ValueError('inquiry budget')
        best = None
        for x in pool:
            if x in self.training: continue
            counts = Counter(m.table[self.domain.index[x]] for m in self.models)
            # Expected pair disagreement; integer arithmetic is deterministic.
            score = len(self.models)**2 - sum(n*n for n in counts.values())
            rank = (score, len(counts), x)
            if best is None or rank > best[0]: best = (rank, x, counts)
        if best is None: return None
        rank, x, counts = best
        c,a,b = x
        return {'input': x, 'question': f'In context {c}, what outcome is observed for {a} and {b}?',
                'possible_outcomes': dict(sorted(counts.items())), 'disagreement': rank[0],
                'hypotheses': len(self.models)}

    def advance(self, pool=()):
        q = self.inquiry(pool)
        facts = dict(blocked=not self.live(), validated=len(self.audits)>=MIN_AUDITS,
                     has_query=q is not None, frozen=bool(self.audits), enough=len(self.training)>=MIN_TRAINING,
                     resolved=len(self.models)==1, informative=bool(q and q['disagreement']))
        src = files('ix_zero').joinpath('policies','contextual.ix').read_text(encoding='utf-8')
        r = run_ix(parse_ix(src), agent='Contextual', event='advance', inputs=facts)
        if r.approvals_required or len(r.replies)!=1: raise ValueError('invalid contextual IX policy')
        action = r.replies[0]
        allowed = ('revise',) if facts['blocked'] else ('complete',) if facts['validated'] else ('inquire','audit','exhausted')
        if action not in allowed or action=='inquire' and (q is None or facts['frozen']) or action=='audit' and not facts['enough']:
            raise ValueError('contextual IX action rejected')
        return {'action': action, 'inquiry': q if action=='inquire' else None,
                'diagnosis': self.diagnose(), 'trace': r.trace_as_dicts()}

    def hypotheses(self, limit=5):
        if not self.live(): return []
        if type(limit) is not int or not 1<=limit<=20: raise ValueError('hypothesis display budget')
        return [m.description() for m in self.models[:limit]]

    def recover(self, *, changed_context=False):
        """Start a new epoch; replay evidence as training, never as audits.

        For declared context drift discard earlier observations in that context;
        for a newly discovered exception retain noncontradictory history.
        The caller must still supply fresh validation and remaining observations.
        """
        if self.live() or self.feedback is None or not all(c.available() for c in self.sources):
            raise ValueError('recovery requires a quarantined contradiction and live sources')
        x,y,_ = self.feedback
        nxt = Episode(self.domain,self.base,self.alternatives,expanded=True)
        history = {**self.training, **{k:v for k,(v,ok) in self.audits.items() if ok}}
        for k,v in history.items():
            if k==x or changed_context and k[0]==x[0]: continue
            if not nxt.live(): break
            nxt.observe(k,v)
        if nxt.live(): nxt.observe(x,y)
        return nxt

    def capability(self, name, context):
        if context not in self.domain.contexts: raise ValueError('unknown context')
        return Capability(name, ('integer','integer'), 'integer', lambda a,b:self.predict((context,a,b)),
                          lambda:self.live() and len(self.audits)>=MIN_AUDITS, 'contextual:'+name)

    def status(self):
        return dict(expanded=self.expanded, generated=self.generated, remaining=len(self.models),
                    training=len(self.training), audits=sum(ok for _,ok in self.audits.values()),
                    blocked=not self.live(), diagnosis=self.diagnose(), hypotheses=self.hypotheses())
