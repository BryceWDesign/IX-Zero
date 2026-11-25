"""IX decision language bridged to bounded Python cognitive instructions.

The IX policy is trusted local configuration, not arbitrary user input. Its
outputs select named host operations; it never supplies training outcomes.
"""
from dataclasses import dataclass
from importlib.resources import files
import hashlib
from ._ix_vendor import parse_ix, run_ix, IXError

@dataclass(frozen=True)
class Decision:
    action: str
    message: str
    facts: dict
    trace: list
    policy_sha256: str


def policy_source():
    return files('ix_zero').joinpath('policies', 'cognition.ix').read_text(encoding='utf-8')


def advance(mind, name, *, source=None):
    """Execute one IX decision, enforce host invariants, and return its trace."""
    t = mind.tasks[name]
    resolved = t.resolved()
    concept = t.concept() if resolved else None
    unused = tuple(x for x in t.space.states if x not in t.observations and x not in t.audits)
    facts = dict(blocked=t.blocked, resolved=resolved, audits=len(t.audits),
                 observations=len(t.observations), unused=len(unused),
                 promotable=concept is not None and len(concept) >= 2 and name not in mind.concepts)
    src = policy_source() if source is None else source
    if not isinstance(src, str) or len(src) > 8192:
        raise ValueError('invalid IX policy size')
    try:
        result = run_ix(parse_ix(src), agent='Zero', event='advance', inputs=facts)
    except (IXError, ArithmeticError, TypeError, RecursionError) as e:
        raise ValueError(f'IX policy refused: {e}') from e
    if result.approvals_required or len(result.replies) != 1:
        raise ValueError('policy must return exactly one action without pending approvals')
    action = result.replies[0]
    # An edited policy cannot promote unverified knowledge or fabricate evidence.
    if action == 'revise' and t.blocked:
        message = 'Task contradicted or unrepresentable. Start a new task with new observations; this task remains blocked.'
    elif action == 'inquire' and not t.blocked and not resolved and not t.audits and len(unused) > 3:
        x = t.query(unused)
        if x is None:
            raise ValueError('no discriminating observation available')
        message = 'Request observation: ' + str(x) + '. Supply the measured output with teach INPUT -> OUTPUT.'
    elif action == 'audit' and not t.blocked and resolved and len(t.audits) < 3 and len(unused) >= 3-len(t.audits):
        message = f'Request independent audit: choose an unused input and measure its output. {3-len(t.audits)} passing audit(s) still required. Use audit INPUT -> OUTPUT.'
    elif action == 'promote' and facts['promotable'] and not t.blocked:
        message = 'Promoted validated composition: ' + str(mind.promote(name))
    elif action == 'complete' and not t.blocked and resolved and len(t.audits) >= 3:
        message = 'Bounded task resolved and audit gate passed. No further acquisition requested.'
    elif action == 'exhausted' and not t.blocked and ((not resolved and (t.audits or len(unused)<=3)) or (resolved and len(t.audits)<3 and len(unused)<3-len(t.audits))):
        message = 'Abstain: insufficient unused evidence or frozen acquisition. No concept promoted.'
    else:
        raise ValueError(f'IX action rejected by host evidence checks: {action}')
    return Decision(action, message, facts, result.trace_as_dicts(), hashlib.sha256(src.encode()).hexdigest())
