"""Shared bounded developmental workspace: relational + numeric prediction,
Sally goal/analogy/meta-learning, and an executable IX policy.

No external actions. All outputs are proposals; feedback must be supplied by a
separate evaluator. A counter model is not independent external evidence.
"""
from collections import deque
from dataclasses import asdict
from importlib.resources import files
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .checkpoint import canonical
from .discovery import DiscoveryMind
from .coupled import CoupledEpisode
from .scalar import ScalarEpisode, checked as signed_pair, truth_value
from .relational import WorldMemory, state, fact, command, catalog, schema, plan, token
from ._ix_vendor import parse_ix, run_ix, IXError
from ._sally_vendor.cognition.open_goals import OpenGoalGenesis
from ._sally_vendor.cognition.open_choice import ActionPrimitive
from ._sally_vendor.cognition.relational_transfer import RelationalWorld, RelationEdge, RelationalTransferEngine
from ._sally_vendor.cognition.meta_learning import MetaLearningController, LearningStrategyTrial
from ._sally_vendor.cognition.goal_reasoning import GoalArbiter, GoalEvidence
from ._sally_vendor.cognition.goals import GoalSpec
from ._sally_vendor.cognition.world_model import FactPattern
from ._sally_vendor.cognition.values import CognitiveValue

LIMIT=2_000_000


def serial(s):
    return sorted(s)


class UnifiedMind:
    def __init__(self, discovery=None,mode='combined'):
        self.discovery=discovery if discovery is not None else DiscoveryMind()
        self.world=WorldMemory(mode); self.bindings={}; self.events=[]
        self.joint_audits={}; self.joint_history={}; self.joint_blocked=set()
        self.coupled={}; self.coupled_history={};self.scalar={}

    def apply(self,operation,*args):
        if len(self.events)>=4096:raise ValueError('unified journal budget')
        if operation in ('observe','audit'):
            s,a,t=args; s,a,t=serial(state(s)),command(a),serial(state(t))
            if operation=='audit' and any((x[0],ac)==(state(s),a) for x,ac,_,_ in self.joint_history.get(schema(a),[])+self.joint_audits.get(schema(a),[])):raise ValueError('symbolic audit input used for joint audit')
            result=getattr(self.world,operation)(s,a,t); args=(s,a,t)
        elif operation=='remember':
            (trace,)=args
            trace=[(serial(state(s)),command(a),serial(state(t))) for s,a,t in trace]
            self.world.remember(trace);args=(trace,);result=None
        elif operation=='bind':
            a,name=args;a=command(a);name=token(name)
            if name not in self.discovery.episodes:raise ValueError('unknown numeric episode')
            k=schema(a)
            if k in self.bindings and self.bindings[k]!=name:raise ValueError('binding immutable; start new workspace')
            if k not in self.bindings and len(self.bindings)>=24:raise ValueError('binding budget')
            self.bindings[k]=name;args=(a,name);result=None
        elif operation=='recover':
            (a,)=args;a=command(a);result=self.world.recover(a);k=schema(a);self.joint_history.setdefault(k,[]).extend(self.joint_audits.pop(k,[]));args=(a,)
        elif operation=='scalar_start':
            (name,)=args;name=token(name)
            if name in self.scalar or len(self.scalar)>=24:raise ValueError('scalar episode exists or budget exhausted')
            self.scalar[name]=ScalarEpisode();args=(name,);result=None
        elif operation in ('scalar_observe','scalar_audit'):
            name,x,y=args;name=token(name);x,y=signed_pair(x),truth_value(y)
            if name not in self.scalar:raise ValueError('unknown scalar episode')
            result=getattr(self.scalar[name],operation.split('_')[1])(x,y);args=(name,x,y)
        elif operation=='coupled_start':
            a,width,modulus=args;a=command(a);k=schema(a)
            prior=self.coupled.get(k)
            if prior is not None and not prior.blocked:raise ValueError('coupled epoch already active')
            if prior is None and len(self.coupled)>=24:raise ValueError('coupled schema budget')
            history=self.coupled_history.setdefault(k,[])
            if len(history)>=3:raise ValueError('coupled epoch budget')
            candidate=CoupledEpisode(width,modulus)
            if prior is not None:history.append(prior)
            self.coupled[k]=candidate;args=(a,width,modulus);result=None
        elif operation in ('coupled_observe','coupled_audit'):
            f,v,a,tf,tv=args;a=command(a);k=schema(a);e=self.coupled.get(k)
            if e is None:raise ValueError('start coupled epoch first')
            row=e.normalize(f,v,a,tf,tv)
            if operation=='coupled_audit':
                if any(r[:3]==row[:3] for old in self.coupled_history[k] for r in old.training+[r for r,_ in old.audits]):raise ValueError('prior coupled input cannot audit')
                if any(x==row[:2] and ac==a for x,ac,_,_ in self.joint_history.get(k,[])+self.joint_audits.get(k,[])):raise ValueError('prior joint input cannot audit')
            result=getattr(e,operation.split('_')[1])(*row);args=(serial(row[0]),row[1],a,serial(row[3]),row[4])
        elif operation=='joint_audit':
            f,v,a,tf,tv=args;a=command(a);cur=self.joint(f,v);target=self.joint(tf,tv)
            if cur[1] is None or target[1] is None:raise ValueError('joint audit requires two vectors')
            k=schema(a)
            if k in self.coupled:raise ValueError('active coupled schema requires coupled audit')
            name=self.bindings.get(k);e=self.discovery.episodes.get(name);m=self.world.models.get(k)
            if e is None or m is None or e.blocked or e.passes<3 or m.quarantined or sum(ok for _,ok in m.audits)<3:raise ValueError('joint audit requires both component gates')
            e.space.check(cur[1]);e.space.check(target[1]);bucket=self.joint_audits.setdefault(k,[])
            if len(bucket)>=512 or any((x[0],ac)==(cur[0],a) for x,ac,_,_ in bucket+self.joint_history.get(k,[])):raise ValueError('joint audit is not fresh or exceeds budget')
            if any((o.before,o.action)==(cur[0],a) for o in m.training) or any((o.before,o.action)==(cur[0],a) for o,_ in m.audits):raise ValueError('joint audit requires fresh grounded symbolic input')
            if any((o.before,o.action)==(cur[0],a) for old in self.world.history[k] for o in old.training+[x for x,_ in old.audits]):raise ValueError('joint audit input used in earlier symbolic epoch')
            prediction=self._component_predict(cur,a,True);result=prediction==target
            if prediction is not None and not result:self.joint_blocked.add(k)
            bucket.append((cur,a,target,result))
            # Bind evidence to the historical numeric journal, even if a later
            # retraction makes that numeric model unusable today.
            prefix=self.discovery.events
            proof=dict(count=len(prefix),sha256=hashlib.sha256(canonical(prefix)).hexdigest(),passed=result)
            args=(serial(cur[0]),cur[1],a,serial(target[0]),target[1],proof)
        else:raise ValueError('unknown unified operation')
        self.events.append(json.loads(canonical([operation,*args])));return result

    def joint(self,facts,vector=None):
        s=state(facts)
        if vector is None:return s,None
        if not isinstance(vector,(tuple,list)) or not 2<=len(vector)<=3 or any(type(x) is not int or not 0<=x<=6 for x in vector):raise ValueError('invalid numeric vector')
        return s,tuple(vector)

    def _component_predict(self,current,a,validated=True):
        s,v=current;a=command(a);t=self.world.predict(s,a,validated)
        if t is None:return None
        if v is None:return t,None
        name=self.bindings.get(schema(a));e=self.discovery.episodes.get(name)
        if e is None or e.blocked or (validated and e.passes<3):return None
        try:y=e.predict(e.space.check(v),validated)
        except ValueError:return None
        return (t,y) if y is not None else None

    def predict(self,current,a,validated=True):
        a=command(a);k=schema(a)
        if k in self.coupled:
            if current[1] is None:return None
            e=self.coupled[k]
            try:return e.predict(current[0],current[1],a,validated)
            except ValueError:return None
        if current[1] is not None and (k in self.joint_blocked or (validated and sum(ok for _,_,_,ok in self.joint_audits.get(k,[]))<3)):return None
        return self._component_predict(current,a,validated)

    def plan(self,current,actions,present=(),absent=(),vector=None,max_expansions=1024,macros=True):
        p,q=state(present),state(absent)
        if p&q:raise ValueError('contradictory goal')
        v=None if vector is None else self.joint((),vector)[1]
        if v is not None and current[1] is None:raise ValueError('numeric goal requires numeric start')
        return plan(self.world,current,lambda x:p<=x[0] and not q&x[0] and (v is None or x[1]==v),actions,self.predict,max_expansions=max_expansions,macros=macros)

    def lanes(self,s,a):
        s,a=state(s),command(a);m=self.world.models.get(schema(a))
        return (None,None,False) if m is None else m.lanes(s,a)

    def strategy(self):
        """Fit a curriculum preference from previous frozen audit outcomes.

        Correct prediction gets 1, abstention 0.25, error 0. This is a
        declared utility heuristic, not a probability or confidence calibration.
        The selected lane changes which context is examined first, never the
        evidence gate used in joint prediction or planning.
        """
        trials=[]
        for m in [old for epochs in self.world.history.values() for old in epochs]+list(self.world.models.values()):
            for o,_ in m.audits:
                l,c,conflict=m.lanes(o.before,o.action)
                combined=None if conflict or (l is not None and c is not None and l!=c) else (l if l is not None else c)
                for name,p in [('local',l),('counter',c),('combined',combined)]:
                    trials.append(LearningStrategyTrial(name,'symbolic-transition',1.0 if p==o.after else (0.25 if p is None else 0.0),1))
        if not trials:return dict(selected_strategy_id='combined',changed_strategy=False,observations=0)
        decision=MetaLearningController().select(trials,task_family='symbolic-transition',default_strategy_id='combined')
        return {**asdict(decision),'observations':len(trials)//3}

    def query(self,candidates):
        if not isinstance(candidates,(tuple,list)) or len(candidates)>256:raise ValueError('candidate budget')
        chosen=self.strategy()['selected_strategy_id'];rank=[]
        for raw,a in candidates:
            s,a=state(raw),command(a);m=self.world.models.get(schema(a))
            if schema(a) in self.coupled:continue
            if m is not None and (m.quarantined or any((o.before,o.action)==(s,a) for o in m.training) or any((o.before,o.action)==(s,a) for o,_ in m.audits)):continue
            if any((o.before,o.action)==(s,a) for old in self.world.history.get(schema(a),[]) for o in old.training+[x for x,_ in old.audits]):continue
            l,c,conflict=self.lanes(s,a)
            p={'local':l,'counter':c,'combined':self.world.predict(s,a,False)}[chosen]
            # Disagreement, unknown outcomes, and under-observed action schemas.
            score=(int(conflict or (l is not None and c is not None and l!=c)),int(p is None),-len(m.training) if m else 0)
            rank.append((score,tuple(sorted(s)),a))
        if not rank:return None
        score,s,a=max(rank,key=lambda r:(r[0],r[1],r[2]))
        m=self.world.models.get(schema(a));return dict(before=s,action=a,score=score,strategy=chosen,kind='audit' if m and m.audits else 'observe_or_independent_audit')

    def generate_goal(self,current,actions):
        """Actual Sally open-goal search over a graph learned by both engines.

        Map finite joint states to IDs. The donor's numeric risk term is only an
        ID-space tie-break heuristic; it has no real-world risk interpretation.
        Unsupported transitions are never returned as executable proposals.
        """
        actions=catalog(actions);ids={current:0};nodes=[current];edges={};q=deque([(current,0)])
        while q:
            s,d=q.popleft()
            if d>=4:continue
            for a in actions:
                t=self.predict(s,a)
                if t is None or t==s:continue
                if t not in ids:
                    if len(nodes)>=128:continue
                    ids[t]=len(nodes);nodes.append(t);q.append((t,d+1))
                edges[ids[s],a]=ids[t]
        if len(nodes)<2:return None
        # Sally's primitive interface is total. Unknown transitions become
        # identity for goal search only; returned path is independently checked.
        lookup={f'a{i}':a for i,a in enumerate(actions)}
        primitives=[ActionPrimitive(k,lambda x,a=a:edges.get((x,a),x)) for k,a in lookup.items()]
        observed={o.after for m in self.world.models.values() for o in m.training} | {o.after for m in self.world.models.values() for o,ok in m.audits if ok}
        known=[i for i,(s,_) in enumerate(nodes) if s in observed]
        g=OpenGoalGenesis().generate(initial_state=0,primitives=primitives,known_states=known,max_depth=4,max_programs=512,state_bound=256)
        target=nodes[g.target_state]
        p=plan(self.world,current,lambda x:x==target,actions,self.predict,max_expansions=1024,max_depth=4)
        if p['path'] is None or not p['path']:return None
        return dict(goal_id=g.goal.goal_id.value,novel=target[0] not in observed,score=g.intrinsic_score,target=target,path=p['path'],expanded=p['expanded'])

    def choose_goal(self,current,actions,alternatives):
        """Select among mutually exclusive sandbox outcomes, not external values.

        Each feasible plan is scored by Sally's actual goal arbiter. Priority and
        utility are explicit evaluator inputs, not inferred human preferences.
        """
        if not isinstance(alternatives,list) or not 1<=len(alternatives)<=8:raise ValueError('goal alternatives budget')
        plans={};specs=[];evidence=[];seen=set()
        for alt in alternatives:
            name=token(alt['id']);priority=alt.get('priority',0.5);utility=alt.get('utility',0.5)
            if name in seen or type(priority) not in (int,float) or type(utility) not in (int,float) or not 0<=priority<=1 or not 0<=utility<=1:raise ValueError('invalid goal alternatives')
            seen.add(name);goal=alt['goal'];p=self.plan(current,actions,**goal)
            if p['path'] is None:continue
            plans[name]=p
            spec=GoalSpec.create(goal_id=name,description='Alternative sandbox outcome '+name,
                desired_state=FactPattern.create(subject='sandbox',predicate='selected-outcome',value=CognitiveValue.from_python(name)),
                priority=priority,utility=utility,risk_limit=0.0)
            specs.append(spec);evidence.append(GoalEvidence(name,1.0,utility,1/(1+len(p['path']))))
        if not specs:return dict(selected=None,reason='No supported plan',plans={})
        result=GoalArbiter().resolve(specs,evidence=evidence)
        return dict(selected=result.selected_goal_id,reason=result.reason,scores=result.scores,plans=plans)

    def analogy_goal(self,source,target,relation,marker,effective):
        """Use actual Sally structure transfer to propose, not assert, a goal.

        The source marker must occur in a passing audit with the source graph.
        Target analogy is not evidence of a successful action.
        """
        source,target=state(source),state(target);relation,marker,effective=map(token,(relation,marker,effective))
        if (marker,effective) not in source:raise ValueError('source marker absent')
        if not any(not m.quarantined and ok and source<=o.after for m in self.world.models.values() for o,ok in m.audits):raise ValueError('source not backed by passing audit')
        def graph(name,s):return RelationalWorld(name,tuple(RelationEdge(f[1],relation,f[2]) for f in sorted(s) if f[0]==relation and len(f)==3))
        engine=RelationalTransferEngine();learned=engine.learn(world=graph('source',source),effective_node=effective)
        inferred=engine.transfer(learned,world=graph('target',target))
        return (marker,inferred.inferred_node)

    def advance(self,current,actions,candidates=(),goal=None,source=None):
        actions=catalog(actions);q=self.query(candidates)
        coupled_recovery={schema(a) for a in actions if schema(a) in self.coupled and self.coupled[schema(a)].blocked}
        facts=dict(has_goal=goal is not None,has_query=q is not None,has_recovery=bool(coupled_recovery) or any(self.world.models.get(schema(a)) is not None and self.world.models[schema(a)].quarantined for a in actions),has_move=any((y:=self.predict(current,a)) is not None and y!=current for a in actions))
        src=files('ix_zero').joinpath('policies','unified.ix').read_text(encoding='utf-8') if source is None else source
        if not isinstance(src,str) or len(src)>8192:raise ValueError('invalid IX policy')
        try:r=run_ix(parse_ix(src),agent='Unified',event='advance',inputs=facts)
        except (IXError,ArithmeticError,TypeError,RecursionError) as e:raise ValueError(f'IX policy refused: {e}') from e
        if r.approvals_required or len(r.replies)!=1:raise ValueError('invalid IX reply')
        action=r.replies[0]
        if action=='plan' and goal is not None:result=self.plan(current,actions,**goal)
        elif action=='inquire' and q is not None:result=q
        elif action=='generate' and facts['has_move']:result=self.generate_goal(current,actions)
        elif action=='revise' and facts['has_recovery']:result=dict(proposed_recovery=sorted({schema(a) for a in actions if self.world.models.get(schema(a)) is not None and self.world.models[schema(a)].quarantined}),coupled_recovery=sorted(coupled_recovery),requires_explicit_recovery=True)
        elif action=='exhausted' and not any(facts.values()):result=None
        else:raise ValueError('IX action denied by host')
        return dict(action=action,result=result,facts=facts,trace=r.trace_as_dicts(),policy_sha256=hashlib.sha256(src.encode()).hexdigest())

    def status(self):
        return dict(models=self.world.status(),procedures=sum(len(b)>=2 for b in self.world.procedures.values()),bindings={str(k):v for k,v in self.bindings.items()},joint_audits={str(k):dict(total=len(v),passes=sum(ok for _,_,_,ok in v),blocked=k in self.joint_blocked) for k,v in self.joint_audits.items()},strategy=self.strategy(),coupled={str(k):e.status() for k,e in self.coupled.items()},scalar={k:e.status() for k,e in self.scalar.items()},events=len(self.events))

    def save(self,path):
        data=dict(version=1,events=self.events,mode=self.world.mode);blob=canonical(dict(data=data,sha256=hashlib.sha256(canonical(data)).hexdigest()))
        if len(blob)>LIMIT:raise ValueError('checkpoint too large')
        path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.ix-unified-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @classmethod
    def load(cls,path,discovery=None):
        with open(path,'rb') as f:blob=f.read(LIMIT+1)
        if len(blob)>LIMIT:raise ValueError('checkpoint too large')
        obj=json.loads(blob);data=obj['data']
        if data['version']!=1 or hashlib.sha256(canonical(data)).hexdigest()!=obj['sha256']:raise ValueError('checkpoint invalid')
        m=cls(discovery,data['mode'])
        historical=DiscoveryMind();current=m.discovery;offset=0
        for op,*args in data['events']:
            if op!='joint_audit':m.apply(op,*args);continue
            *inputs,proof=args;n=proof['count']
            if type(n) is not int or not offset<=n<=len(current.events) or hashlib.sha256(canonical(current.events[:n])).hexdigest()!=proof['sha256']:raise ValueError('numeric dependency journal mismatch')
            for event in current.events[offset:n]:historical.apply(*event)
            offset=n;m.discovery=historical
            try:passed=m.apply(op,*inputs)
            finally:m.discovery=current
            if type(proof['passed']) is not bool or passed!=proof['passed']:raise ValueError('joint evidence replay mismatch')
        if canonical(m.events)!=canonical(data['events']):raise ValueError('replay mismatch')
        return m


def jsonable(x):
    if isinstance(x,(set,frozenset)):return sorted(x)
    raise TypeError(type(x).__name__)
