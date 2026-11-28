"""Unified experimental command interface: retained v4 workspace, new active
learner, goal search, evidence retrieval and measured cognitive self-model.
The external environment remains a measurement provider, not an evaluator
that can be rewritten by the candidate.
"""
import json
from pathlib import Path
import tempfile
import os
from .general import General, canonical, hash_of
from .exploration import Learner, Measurements, route
from .web_research import Research
from .audit import AuditConsensus
from .pair_inquiry import inquire as pair_inquire, predict as pair_predict
from .program_induction import ProgramMemory, revise
from .object_induction import solve as grid_solve
from .grid_relational import infer as grid_relational
from .tool_induction import ToolMemory
from .grounded_semantics import SemanticMemory, RecordedMeasurements, generate_probes
from ._sally_vendor.cognition.metacognition import CapabilityMeasure, SelfModel
from ._sally_vendor.digest import DigestRecord


class ResearchWorkspace:
    def __init__(self, transport=None, workspace=None):
        self.general=General();self.learner=Learner();self.research=Research(transport,workspace);self.events=[];self.pair_skills={};self.pair_reports=[];self.pair_challenges=[];self.programs=ProgramMemory();self.tools_memory=ToolMemory();self.semantics=SemanticMemory()

    def self_model(self):
        model=SelfModel(); groups={}
        for r in self.learner.history:groups.setdefault(r['scope'],[]).append(r)
        for i, (scope,rows) in enumerate(sorted(groups.items())):
            model=model.update(CapabilityMeasure.create(capability_id=f'bounded_scope_{i}',score=sum(r['promoted'] for r in rows)/len(rows),evidence_digests=[DigestRecord.from_payload(json.loads(canonical(r))) for r in rows],limitation='Promotion fraction on host-provided observations; not a general accuracy measure. Scope '+scope))
        return {'exploration':self.learner.self_model(),'sally_native':model.to_payload(),'cross_mechanism_inquiries':len(self.pair_reports),'cross_mechanism_promotions':sum(r['promoted'] for r in self.pair_reports),'program_induction_attempts':len(self.programs.history),'program_promotions':sum(r['promoted'] for r in self.programs.history),'program_active_skills':sum(v['valid'] for v in self.programs.skills.values()),'tool_induction_attempts':len(self.tools_memory.history),'tool_promotions':sum(r['promoted'] for r in self.tools_memory.history),'web_research_sessions':len(self.research.records),'web_evidence_sessions':sum(r['status']=='retrieved_evidence' for r in self.research.records),'semantic_lexemes':len(self.semantics.models),'semantic_promotions':sum(m is not None for m in self.semantics.models.values()),'weakest_scope':None if model.weakest() is None else model.weakest().to_payload()}

    def execute(self,request):
        req=json.loads(canonical(request));a=dict(req);cmd=a.pop('command')
        if cmd=='self_model':return self.self_model()
        if cmd=='tools':return {'local_documents_enabled':self.research.local is not None,'tools':[{'name':'general','accepts':'retained controlled reasoning tasks'},{'name':'explore','accepts':'finite labelled examples and measurement pools'},{'name':'pair_inquire','accepts':'signed pair observations; native scalar/word/predicate candidates'},{'name':'program_inquire','accepts':'fixed arity 1–4 integer tuples and measured outputs; expressions, loops and learned calls'},{'name':'program_revise','accepts':'previous program key, fresh observations, pool and audit; finite wrapper synthesis'},{'name':'grid_relational','accepts':'grid examples and unlabelled inputs; learned coordinate-feature rules'},{'name':'grid_induce','accepts':'grid train pairs and unlabelled inputs'},{'name':'tool_inquire','accepts':'integer observations and explicitly authorized numerical standard-library modules'},{'name':'semantic_inquire','accepts':'observed object states, opaque two- to four-token actions and measurements'},{'name':'semantic_transfer','accepts':'fresh transition observations for role/field analogy with acquired lexemes'},{'name':'semantic_plan','accepts':'partial object-state objective and learned action models'},{'name':'semantic_probes','accepts':'observed transitions; proposes new unlabeled interventions'},{'name':'route','accepts':'transition graph and goal'},{'name':'research','accepts':'subject and information need, read-only public HTTPS'}],'arbitrary_host_execution':False,'automatic_typed_dispatch':'task'}
        if cmd=='semantic_probes':return {'inputs':generate_probes(**a)}
        if cmd=='semantic_predict':return {'value':self.semantics.predict(**a),'claim':'model prediction, not an actual intervention'}
        if cmd=='semantic_plan':return self.semantics.plan(**a)
        if cmd=='research':return self.research.investigate(**a)
        if cmd=='task':return self.task(a)
        if cmd=='route':return route(**a)
        if cmd=='grid_induce':return grid_solve(**a)
        if cmd=='grid_relational':return grid_relational(**a)
        if cmd=='tool_predict':return {'value':self.tools_memory.predict(a['key'],a['input']),'tool_assisted':True,'claim':'experimental extrapolation; not independently verified on this input'}
        if cmd=='program_predict':return {'value':self.programs.predict(a['key'],a['input']),'claim':'experimental extrapolation; not independently verified on this input'}
        if cmd=='predict':return {'value':self.learner.predict(**a),'claim':'experimental extrapolation; not independently verified on this input'}
        if cmd=='pair_predict':return {'value':pair_predict(self.pair_skills[a['key']],a['input']),'claim':'experimental extrapolation; not independently verified on this input'}
        if cmd=='general':return self.general.execute(a['request'])
        if cmd=='resolve':return self.resolve(**a)
        if len(self.events)>=2048:raise ValueError('research journal capacity')
        if cmd in ('semantic_inquire','semantic_transfer'):
            train=a.pop('train');pool=RecordedMeasurements(a.pop('pool'));audit=RecordedMeasurements(a.pop('audit'))
            result=(self.semantics.learn if cmd=='semantic_inquire' else self.semantics.transfer)(train,pool,audit,**a)
        elif cmd=='semantic_challenge':result=self.semantics.challenge(**a)
        elif cmd=='explore':
            train=a.pop('train');pool=Measurements(a.pop('pool'))
            if 'audit_sources' in a:
                if 'audit' in a:raise ValueError('choose one audit protocol')
                audit=AuditConsensus(a.pop('audit_sources'),policy=a.pop('audit_policy','majority'))
            else:audit=Measurements(a.pop('audit'))
            result=self.learner.fit(train,pool,audit,**a)
            if isinstance(audit,AuditConsensus):result['audit_source_votes']=audit.log
        elif cmd in ('program_inquire','tool_inquire','program_revise'):
            train=a.pop('train');pool=Measurements(a.pop('pool'));key=a.pop('key')
            if 'audit_sources' in a:
                if 'audit' in a:raise ValueError('choose one audit protocol')
                audit=AuditConsensus(a.pop('audit_sources'),policy=a.pop('audit_policy','majority'))
            else:audit=Measurements(a.pop('audit'))
            if cmd=='program_revise':result=revise(self.programs,a.pop('base_key'),key,train,pool,audit,**a)
            else:result=(self.programs if cmd=='program_inquire' else self.tools_memory).learn(key,train,pool,audit,**a)
        elif cmd=='tool_challenge':result=self.tools_memory.challenge(a['key'],a['input'],a['expected'])
        elif cmd=='program_challenge':result=self.programs.challenge(a['key'],a['input'],a['expected'])
        elif cmd=='pair_inquire':
            train=a.pop('train');pool=Measurements(a.pop('pool'));audit=Measurements(a.pop('audit'));key=a.pop('key')
            if not isinstance(key,str) or not key or len(key)>80 or key in self.pair_skills:raise ValueError('invalid pair skill key')
            result,h=pair_inquire(train,pool,audit,**a);result['key']=key;self.pair_skills[key]=h;self.pair_reports.append(result)
        elif cmd=='pair_challenge':
            h=self.pair_skills[a['key']];prediction=pair_predict(h,a['input']);expected=a['expected']
            if type(expected) is not int:raise ValueError('integer outcome required')
            consistent=prediction is not None and prediction==expected
            if not consistent:self.pair_skills[a['key']]=None
            result={'consistent':consistent,'prediction':prediction,'expected':expected};self.pair_challenges.append(result)
        elif cmd=='challenge':result={'consistent':self.learner.challenge(**a)}
        else:raise ValueError('unsupported research command')
        self.events.append(req);return result

    def task(self,goal):
        """Choose a compatible mechanism from the structure of an objective.
        This is typed dispatch, not a general natural-language task parser.
        The selected native command journals its own evidence and replay.
        """
        if 'command' in goal:raise ValueError('nested command not accepted')
        if {'start','goal'}<=set(goal) and 'edges' not in goal and isinstance(goal['start'],dict):return {'chosen':'semantic_plan','result':self.execute({'command':'semantic_plan',**goal})}
        if 'train' in goal:
            rows=goal['train']
            if not rows:raise ValueError('empty task')
            if isinstance(rows[0][0],(list,tuple)) and len(rows[0][0])==2 and isinstance(rows[0][0][0],dict):return {'chosen':'semantic_inquire','result':self.execute({'command':'semantic_inquire',**goal})}
            if 'inputs' in goal:
                return {'chosen':'grid_induce','result':self.execute({'command':'grid_induce',**goal})}
            arity=len(rows[0][0]) if isinstance(rows[0][0],(tuple,list)) else 0
            fixed=not goal.get('grammar') and 1<=arity<=4 and all(isinstance(x,(tuple,list)) and len(x)==arity and all(type(v) is int for v in x) and type(y) is int for x,y in rows)
            observed=rows+goal.get('pool',[])+goal.get('audit',[])
            pair=fixed and 'audit_sources' not in goal and arity==2 and all(isinstance(x,(tuple,list)) and len(x)==2 and all(type(v) is int and abs(v)<=32 for v in x) for x,y in observed)
            request=dict(goal);program_budget=request.pop('program_budget',12000);tool_modules=request.pop('tool_modules',None)
            if tool_modules is not None and not fixed:raise ValueError('tool inquiry needs integer tuple observations')
            if not fixed:
                request['command']='explore';return {'chosen':'explore','result':self.execute(request)}
            key=goal.get('key',hash_of(goal)[:24]);program_request={k:v for k,v in goal.items() if k in ('train','pool','audit','audit_sources','audit_policy','query_budget','reuse','allow_loops','search','seed','budget')}
            program_request.setdefault('budget',program_budget);program_request.update(command='program_inquire',key=key)
            first=None
            # Pair-specific and expression-specific tuning options are not
            # silently forwarded to a mechanism that cannot interpret them.
            if pair and not any(k in goal for k in ('reuse','allow_loops','search','seed','budget')):
                pair_request={k:v for k,v in goal.items() if k in ('train','pool','audit','gate_budget','query_budget','active','mechanisms')}
                first=self.execute({'command':'pair_inquire','key':key,**pair_request})
                if first['promoted']:return {'chosen':'pair_inquire','result':first}
                if first['audit_labels']:
                    return {'chosen':'pair_inquire','result':first,'next':'fresh_audit_required','reason':'an audit has already been consumed; escalation must use new evidence'}
                program_request['key']=key[:64]+'-program'
            result=self.execute(program_request);response={'chosen':'program_inquire','result':result}
            if first is not None:response['prior_attempt']=first
            if result['promoted'] or tool_modules is None:return response
            if result['audit_labels'] or result['reason']=='audit_conflict':
                response.update(next='fresh_audit_required',reason='an audit has already been consumed; tool escalation needs new evidence');return response
            tool_request={k:v for k,v in goal.items() if k in ('train','pool','audit','audit_sources','audit_policy','query_budget')}
            tool_request.update(command='tool_inquire',key=key[:64]+'-tool',modules=tool_modules)
            return {'chosen':'tool_inquire','prior_attempts':response,'result':self.execute(tool_request)}
        if {'start','goal','edges'}<=set(goal):return {'chosen':'route','result':self.execute({'command':'route',**goal})}
        if {'subject','relation','target'}<=set(goal):return {'chosen':'resolve','result':self.execute({'command':'resolve',**goal})}
        if {'subject','property_name'}<=set(goal):return {'chosen':'research','result':self.execute({'command':'research',**goal})}
        if set(goal)=={'text'}:return {'chosen':'general','result':self.general.ask(goal['text'])}
        return {'chosen':None,'status':'unsupported','reason':'unrecognized task representation'}

    def resolve(self,subject,relation,target):
        # Existing proof is preferred; missing evidence generates a search goal.
        from .general import Evidence, name, RELATIONS
        name(subject);name(target)
        if relation not in RELATIONS:raise ValueError('unsupported relation')
        local=self.general.evidence.ask(subject,relation,target)
        if local['status']!='unknown':return {'method':'local_evidence','result':local}
        remote=self.research.investigate(subject,relation.replace('_',' ')+' '+target)
        e=Evidence();source_map={}
        # A source excerpt is not independently verified truth. Parse only exact
        # controlled statements, preserve disagreement rather than choose a winner.
        for i,source in enumerate(remote['sources']):
            lines=[]
            for excerpt in source['excerpts']:
                x=excerpt.lower().strip()
                if x in (f'{subject} is {relation} {target}.',f'{subject} is not {relation} {target}.'):lines.append(x)
            if lines:
                key=f'web{i}';e.ingest(key,'\n'.join(lines));source_map[key]=source['url']
        result=e.ask(subject,relation,target)
        return {'method':'source_claims','result':result,'source_map':source_map,'research':remote,'truth_verified':False}

    def save(self,path):
        # The deterministic learning journal is replayed on load. Web snapshots
        # are evidence records, not replay-certified observations or live network.
        p={'version':4,'general_events':self.general.events,'general_fingerprint':self.general.fingerprint(),'events':self.events,'learning_fingerprint':hash_of({'library':self.learner.library,'history':self.learner.history,'pair_reports':self.pair_reports,'pair_challenges':self.pair_challenges,'program_skills':self.programs.skills,'program_history':self.programs.history,'tool_skills':self.tools_memory.skills,'tool_history':self.tools_memory.history,'semantic_models':self.semantics.models,'semantic_history':self.semantics.history,'semantic_dependencies':self.semantics.dependencies}),'web_records':self.research.records}
        blob=canonical({'payload':p,'sha256':hash_of(p)})
        if len(blob)>60_000_000:raise ValueError('checkpoint size')
        path=Path(path);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.ix-research-')
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @classmethod
    def load(cls,path,transport=None,workspace=None):
        with open(path,'rb') as f:blob=f.read(60_000_001)
        if len(blob)>60_000_000:raise ValueError('checkpoint size')
        o=json.loads(blob);p=o['payload']
        if p['version'] not in (1,2,3,4) or hash_of(p)!=o['sha256'] or len(p['events'])>2048:raise ValueError('checkpoint integrity')
        m=cls(transport,workspace)
        for e in p['general_events']:m.general.execute(e)
        if m.general.fingerprint()!=p['general_fingerprint']:raise ValueError('retained workspace replay mismatch')
        for e in p['events']:m.execute(e)
        fingerprint={'library':m.learner.library,'history':m.learner.history,'pair_reports':m.pair_reports,'pair_challenges':m.pair_challenges}
        if p['version']>=2:fingerprint.update(program_skills=m.programs.skills,program_history=m.programs.history)
        if p['version']>=3:fingerprint.update(tool_skills=m.tools_memory.skills,tool_history=m.tools_memory.history)
        if p['version']>=4:fingerprint.update(semantic_models=m.semantics.models,semantic_history=m.semantics.history,semantic_dependencies=m.semantics.dependencies)
        if hash_of(fingerprint)!=p['learning_fingerprint']:raise ValueError('exploration replay mismatch')
        m.research.records=p['web_records']
        # Recalled results are visibly historical and never re-fetched silently.
        for r in m.research.records:
            key=(r['goal']['subject'].casefold().strip(),r['goal']['property'].casefold().strip())
            if r['status']=='retrieved_evidence':m.research.cache[key]=r
            else:m.research.cache.pop(key,None)
        return m
