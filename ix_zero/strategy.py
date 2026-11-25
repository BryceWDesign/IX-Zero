"""Bounded strategy discovery: optimize inquiry ordering, not evaluator or host code."""
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import tempfile
from .checkpoint import canonical
from .operators import Domain, OperatorMind
from ._ix_vendor import parse_ix, run_ix

MAX_BYTES = 2_000_000
MAX_TASKS = 96

@dataclass(frozen=True)
class Strategy:
    metric: str = 'distinct'
    weighting: str = 'all'
    tie: str = 'high'
    kernel: str = 'checked'

    def __post_init__(self):
        if self.metric not in ('distinct','minimax','collision') or self.weighting not in ('all','cheap','frontier') or self.tie not in ('high','low') or self.kernel not in ('checked','packed'):
            raise ValueError('unsupported strategy')

BASELINE = Strategy()


def proposals(mode='heuristic'):
    if mode=='exact':return (BASELINE,Strategy(kernel='packed'))
    if mode!='heuristic':raise ValueError('unsupported proposal family')
    return tuple(Strategy(m,w,t,'packed') for m in ('distinct','minimax','collision')
                 for w in ('all','cheap','frontier') for t in ('high','low'))


def query(episode,pool,strategy,work=None):
    if episode.blocked or episode.audits or not episode.models:return None
    choices=set(episode.domain.pair(x) for x in pool)-episode.training.keys()-episode.audits.keys()
    if not choices:return None
    best=episode.models[0].cost;d=episode.domain
    models=episode.models
    if strategy.weighting=='frontier':models=tuple(c for c in models if c.cost<=best+1)
    def rank(models,weighting):
        # Each evaluated model/input pair is counted, independent of wall time.
        if work is not None:
            work[0]+=len(models)*len(choices)
            if strategy.kernel=='checked':work[1]+=len(models)*len(choices)
        def score(x):
            counts=Counter();shift=(x[0]*(d.mask+1)+x[1])*d.width
            for c in models:
                weight=1 if weighting!='cheap' else 1<<max(0,6-(c.cost-best))
                value=d.value(c.table,x) if strategy.kernel=='checked' else (c.table>>shift)&d.mask
                counts[value]+=weight
            distinct=len(counts);largest=max(counts.values());collision=sum(n*n for n in counts.values())
            order={'distinct':(distinct,-largest),'minimax':(-largest,distinct),'collision':(-collision,distinct)}[strategy.metric]
            return (*order,x if strategy.tie=='high' else tuple(-v for v in x)),distinct
        scores={x:score(x) for x in choices}
        return max(scores,key=lambda x:scores[x][0]),max(v[1] for v in scores.values())
    selected,diversity=rank(models,strategy.weighting)
    if strategy.weighting=='frontier' and diversity==1 and len(models)<len(episode.models):
        selected,_=rank(episode.models,'all')
    return selected


def advance(mind,name,pool,strategy,work=None,*,audit_target=3):
    """Original IX policy and host gate authorize all actions."""
    decision=mind.advance(name,pool,query_selector=lambda t,allowed:query(t,allowed,strategy,work),audit_target=audit_target)
    decision['strategy']=asdict(strategy)
    return decision


def validate_corpus(corpus):
    if not isinstance(corpus,dict) or set(corpus)!={'selection','confirmation','challenge','mode'} or corpus['mode'] not in ('heuristic','exact'):raise ValueError('expected three task partitions and a proposal mode')
    seen=set();groups={};fingerprints=set()
    for phase in ('selection','confirmation','challenge'):
        tasks=corpus[phase]
        if not isinstance(tasks,list) or not 2<=len(tasks)<=MAX_TASKS:raise ValueError('invalid task count')
        groups[phase]=set()
        phase_fingerprints=set()
        for task in tasks:
            if not isinstance(task,dict) or set(task)!={'id','family','width','budget','train','audit','hold'}:raise ValueError('invalid task schema')
            tid,family=task['id'],task['family']
            if any(not isinstance(v,str) or not 1<=len(v)<=80 for v in (tid,family)) or tid in seen:raise ValueError('invalid or repeated task identity')
            seen.add(tid);groups[phase].add(family);d=Domain(task['width'])
            if type(task['budget']) is not int or not 1<=task['budget']<=5:raise ValueError('strategy search budget must be 1..5')
            inputs=set();pairs=[]
            for part,lo,hi in [('train',6,18),('audit',3,6),('hold',4,256)]:
                rows=task[part]
                if not isinstance(rows,list) or not lo<=len(rows)<=hi:raise ValueError('invalid partition size')
                for row in rows:
                    if not isinstance(row,list) or len(row)!=2:raise ValueError('invalid labeled row')
                    x=d.pair(row[0]);y=d.word(row[1])
                    if x in inputs:raise ValueError('partition overlap')
                    inputs.add(x);pairs.append((x,y))
            fp=hashlib.sha256(canonical([d.width,sorted(pairs)])).hexdigest()
            if fp in fingerprints:raise ValueError('repeated cross-phase evidence')
            phase_fingerprints.add(fp)
        fingerprints.update(phase_fingerprints)
    if any(groups[a]&groups[b] for a,b in [('selection','confirmation'),('selection','challenge'),('confirmation','challenge')]):raise ValueError('task families overlap')
    return corpus


def episode(task,strategy,details=False):
    """Evaluator supplies only requested labels; held-out outcomes are scored last."""
    m=OperatorMind();m.apply('new','trial',task['width'],task['budget'],False)
    t=m.episodes['trial'];train={tuple(x):y for x,y in task['train']};pool=tuple(train)
    audits=iter(task['audit']);trace=[];work=[0,0]
    for _ in range(len(train)+len(task['audit'])+2):
        d=advance(m,'trial',pool,strategy,work,audit_target=len(task['audit']));action=d['action']
        if details:trace.append(d)
        if action=='inquire':
            x=d['result'];m.apply('observe','trial',x,train[x])
        elif action=='audit':
            x,y=next(audits);m.apply('audit','trial',x,y)
        else:break
    else:raise RuntimeError('strategy step limit')
    predictions=[[x,y,t.predict(x)] for x,y in task['hold']]
    out=dict(id=task['id'],family=task['family'],correct=sum(y==p for _,y,p in predictions),
             incorrect=sum(p is not None and p!=y for _,y,p in predictions),abstained=sum(p is None for _,y,p in predictions),
             labels=len(t.training)+len(t.audits),training=len(t.training),audits=len(t.audits),query_work=work[0],redundant_validation_work=work[1],promoted='trial' in m.promoted)
    if details:out.update(predictions=predictions,trace=trace,training_rows=list(t.training.items()),audit_rows=list(t.audits.items()))
    return out


def compare(candidate,incumbent):
    if not candidate or [r['id'] for r in candidate]!=[r['id'] for r in incumbent]:raise ValueError('unmatched evaluation tasks')
    totals=lambda rows:{k:sum(r[k] for r in rows) for k in ('correct','incorrect','abstained','labels','query_work','redundant_validation_work')}
    a,b=totals(candidate),totals(incumbent)
    nonregression=all(x['correct']>=y['correct'] and x['incorrect']<=y['incorrect'] for x,y in zip(candidate,incumbent))
    # Two explicit improvement routes: Pareto gain, or substantial inquiry-work
    # reduction with a small, bounded observation-cost tradeoff. No accuracy loss.
    pareto=a['labels']<=b['labels'] and a['query_work']<=b['query_work'] and a['redundant_validation_work']<=b['redundant_validation_work'] and (a['correct']>b['correct'] or a['incorrect']<b['incorrect'] or a['labels']<b['labels'] or a['query_work']<b['query_work'] or a['redundant_validation_work']<b['redundant_validation_work'])
    efficiency=(a['query_work']*2<=b['query_work'] and a['query_work']<b['query_work'] and a['labels']*100<=b['labels']*105 and all(x['labels']<=y['labels']+2 for x,y in zip(candidate,incumbent)))
    gain=nonregression and (pareto or efficiency)
    return dict(eligible=gain,nonregression=nonregression,pareto=pareto,efficiency_tradeoff=efficiency,candidate=a,incumbent=b)


def choose(corpus,incumbent=BASELINE,details=False):
    """Select one challenger on selection families; confirm once, or retain incumbent."""
    validate_corpus(corpus);selection=corpus['selection'];confirmation=corpus['confirmation']
    baseline=[episode(t,incumbent,details) for t in selection];records=[];eligible=[]
    for p in proposals(corpus['mode']):
        if p==incumbent:continue
        rows=[episode(t,p,details) for t in selection];verdict=compare(rows,baseline)
        records.append(dict(strategy=asdict(p),verdict=verdict,rows=rows))
        if verdict['eligible']:
            a=verdict['candidate'];eligible.append(((a['correct'],-a['incorrect'],-a['query_work'],-a['labels'],-a['redundant_validation_work']),p))
    challenger=max(eligible,key=lambda item:item[0])[1] if eligible else None
    base_confirm=[episode(t,incumbent,details) for t in confirmation]
    candidate_confirm=[episode(t,challenger,details) for t in confirmation] if challenger else []
    verdict=compare(candidate_confirm,base_confirm) if challenger else None
    challenge=corpus['challenge'];base_challenge=[episode(t,incumbent,details) for t in challenge]
    passed=bool(verdict and verdict['eligible'])
    candidate_challenge=[episode(t,challenger,details) for t in challenge] if passed else []
    challenge_verdict=compare(candidate_challenge,base_challenge) if passed else None
    facts={'has_candidate':challenger is not None,'confirmed':passed,'challenge_passed':bool(challenge_verdict and challenge_verdict['eligible'])}
    approved=facts['confirmed'] and facts['challenge_passed']
    src=files('ix_zero').joinpath('policies','strategy.ix').read_text(encoding='utf-8')
    r=run_ix(parse_ix(src),agent='Strategy',event='select',inputs=facts)
    if r.approvals_required or len(r.replies)!=1:raise ValueError('invalid strategy policy')
    decision=r.replies[0]
    if decision=='retain' and not approved:selected=incumbent
    elif decision=='adopt' and approved:selected=challenger
    else:raise ValueError('strategy adoption refused')
    return dict(incumbent=asdict(incumbent),selected=asdict(selected),challenger=asdict(challenger) if challenger else None,
                decision=decision,selection_baseline=baseline,selection_candidates=records,confirmation_baseline=base_confirm,
                confirmation_candidate=candidate_confirm,confirmation_verdict=verdict,challenge_baseline=base_challenge,challenge_candidate=candidate_challenge,challenge_verdict=challenge_verdict,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())


class StrategyMind:
    """Single bounded experiment; persistent state is replayed, never executed as code."""
    def __init__(self):self.current=BASELINE;self.corpus=None;self.report=None;self.rolled_back=False
    def fit(self,corpus):
        if self.report is not None:raise ValueError('strategy epoch already evaluated')
        frozen=json.loads(canonical(corpus));report=choose(frozen)
        self.current=Strategy(**report['selected']);self.corpus=frozen;self.report=report
        return report
    def rollback(self):
        if self.report is None:raise ValueError('no strategy experiment')
        self.current=BASELINE;self.rolled_back=True
    def save(self,path):
        if self.report is None:raise ValueError('no experiment to persist')
        payload={'schema':2,'corpus':self.corpus,'report':self.report,'rolled_back':self.rolled_back};blob=canonical({'payload':payload,'sha256':hashlib.sha256(canonical(payload)).hexdigest()})
        if len(blob)>MAX_BYTES:raise ValueError('strategy checkpoint exceeds budget')
        path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.ix-strategy-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES:raise ValueError('strategy checkpoint exceeds budget')
        obj=json.loads(blob);payload=obj['payload']
        if payload['schema']!=2 or hashlib.sha256(canonical(payload)).hexdigest()!=obj['sha256']:raise ValueError('invalid strategy checkpoint')
        mind=cls();report=mind.fit(payload['corpus'])
        if canonical(report)!=canonical(payload['report']):raise ValueError('strategy replay mismatch')
        if type(payload['rolled_back']) is not bool:raise ValueError('invalid rollback state')
        if payload['rolled_back']:mind.rollback()
        return mind
