"""Local conversational command interface. No language model or simulated dialogue."""
import argparse
import json
import shlex
from pathlib import Path
from .core import Mind,Space
from .checkpoint import save,load
from .instructions import advance
from .discovery import DiscoveryMind
from .unified import UnifiedMind, jsonable
from .operators import OperatorMind
from .analogical import AnalogicalMind
from .language import LanguageMind, scene
from .acce import Workspace, CognitiveDispatch, Object
from .compositional import Grammar
from .strategy import StrategyMind, advance as strategy_advance, MAX_BYTES as STRATEGY_MAX_BYTES
from dataclasses import asdict

HELP='''Commands (vectors are space-separated integers):
  new NAME WIDTH MODULUS       create and select a task
  use NAME                     select existing task
  teach INPUT -> OUTPUT        add example; e.g. teach 1 2 -> 3 2
  ask                          suggest most informative next input
  predict INPUT                provisional prediction or abstain
  validated INPUT              predict only after three unseen audits
  audit INPUT -> OUTPUT        test on unseen example; freezes teaching
  promote                      store a resolved multi-step concept
  plan START -> GOAL           propose path with audited same-space tasks
  advance                      execute the IX cognitive policy for one step
  policytrace                  show latest IX execution trace
  status                       current evidence, hypotheses, learned program
  tasks                        list tasks and status
  save                         save local checkpoint
  help                         show commands
  exit                         quit (save explicitly first)
Discovery commands (separate expanded representation engine):
  dnew NAME WIDTH MODULUS [strict|adaptive|symbolic|projection] [cold]
  duse NAME                    select discovery episode
  dteach INPUT -> OUTPUT       supply training observation
  dask                         request discriminating observation
  dadvance                     run discovery IX policy
  dexpand                      expand adaptive search from cost 3 to 5
  dpredict INPUT               provisional prediction
  dvalidated INPUT             audit-gated prediction
  daudit INPUT -> OUTPUT       evaluator-supplied fresh evidence
  dlearn                       retain supported expression modules
  dplan START -> GOAL          propose path using audit-gated discovery models
  dretract                     quarantine current episode and descendants
  dstatus                      show actual model counts and evidence
  dtrace                       last discovery IX execution trace
  dsave                        save discovery journal separately
Unified workspace commands (arguments are JSON; see docs/UNIFIED.md):
  uobserve [facts,action,outcome]  observe symbolic transition
  uaudit [facts,action,outcome]    test fresh transition, freeze training
  ubind [action,episode]          bind numeric discovery episode to action schema
  ujaudit [facts,vector,action,outcome_facts,outcome_vector]  test joint model
  usstart [name]                  start signed scalar task
  usobserve [name,[x,y],outcome]   teach signed relation
  usaudit [name,[x,y],outcome]     test signed relation
  uspredict [name,[x,y]]          audit-gated scalar prediction
  usask [name,[[x,y],...]]        request informative scalar input
  ucstart [action,width,modulus]  start independent coupled epoch
  ucobserve [facts,vector,action,outcome_facts,outcome_vector]  learn joint effects
  ucaudit [facts,vector,action,outcome_facts,outcome_vector]  validate joint effects
  uremember [transitions]         remember measured contiguous procedure
  urecover [action]              new contextual epoch for quarantined symbolic schema
  upredict [facts,vector,action]  audit-gated joint prediction (vector or null)
  uplan {"facts":[],"vector":null,"actions":[],"goal":{}}  joint planning
  uadvance {"facts":[],"vector":null,"actions":[],"candidates":[]}  IX policy
  ugoal {"facts":[],"vector":null,"actions":[]}  self-generated goal
  uchoose {"facts":[],"actions":[],"alternatives":[]}   goal conflict selection
  uanalogy [source,target,relation,marker,effective]  propose analogical goal
  ustatus                         show shared workspace status
  usave                           save numeric discovery and unified journal
Operator discovery commands (JSON arguments; see docs/OPERATORS.md):
  onew [name,width,budget,reuse]   create bounded word-operator episode
  oobserve [name,[a,b],outcome]    supply measured example
  oaudit [name,[a,b],outcome]      supply fresh independent check
  opredict [name,[a,b]]            audit-gated prediction
  oadvance [name]                  execute IX acquisition policy
  ostatus [name]                   inspect expression and search evidence
  oretract [name]                  invalidate episode and dependent knowledge
  osave                            save operator journal
Strategy improvement commands (JSON arguments; see docs/STRATEGY.md):
  mfit "PATH"                     evaluate supplied strategy corpus once
  mstatus                         selected strategy and adoption evidence
  madvance [name,[[a,b],...]]      apply selected strategy to operator inquiry
  mrollback                       restore original strategy
  msave                           save strategy experiment for replay
Cross-domain relational transfer commands (JSON arguments; docs/ANALOGY.md):
  asource [name,states,relations,observations]  register measured source table
  aimport [name,states,relations,operator]     import promoted two-bit operator
  anew [name,states,relations,mode,optional_sources]  start transfer episode
  aobserve [name,[a,b],outcome]                teach target observation
  aaudit [name,[a,b],outcome]                  test fresh target observation
  apredict [name,[a,b]]                        audit-gated prediction
  aadvance [name]                              run IX transfer policy
  astatus [name]                               inspect mapping and evidence counts
  aretract [name]                              quarantine source and descendants
  asave                                        save relational transfer journal
Grounded language commands (JSON arguments; docs/LANGUAGE.md):
  lnew [name]                         create a language learning episode
  lobserve [name,text,scene,intent]     teach from grounded demonstration
  laudit [name,text,scene,intent]       validate frozen interpretation
  linterpret [name,text,scene]         audit-gated semantic interpretation
  lexpress [name,scene,intent]         generate supported utterance
  ladvance [name,[[text,scene],...]]   IX-guided inquiry and validation
  lcontext [name,scene]                select scene for subsequent say requests
  say SENTENCE                        interpret a learned sentence in context
  lcompute [name,text,scene,values]    evaluate acquired numeric operator
  ltransfer [name,text,scene]          evaluate acquired relational operation
  lsimulate [name,text,scene,facts,vector]  predict audited simulated transition
  lstatus [name]                       inspect lexical and grammar hypotheses
  lretract [name]                      quarantine contradicted language episode
  lsave                               save language journal
Cognitive composition commands (JSON arguments; docs/ACCE.md):
  cworkspace                              create fresh catalog from live engines (clears dispatch bindings)
  ccatalog                                list typed capabilities and availability
  cnew [intent,[input_types],output_type]  start a binding from outcome examples
  cobserve [intent,[input_types],args,y]   supply measured example
  caudit [intent,[input_types],args,y]     supply fresh independent outcome
  cadvance [intent,[input_types],pool]     IX-guided inquiry or validation
  cstatus [intent,[input_types]]          inspect evidence and candidate programs
  cdispatch [intent,args]                 execute a validated typed binding
  csay [language_episode,text,scene,values,optional_relations]  interpret and dispatch
Object arguments use {"identity":"A","kind":"cube","attribute":"red","value":2,"relation":"s0"}.
Dispatch bindings are session-local; existing engine checkpoints are unchanged.
Recursive grounded language commands (JSON arguments; docs/COMPOSITIONAL.md):
  rnew [language_episode]              initialize grammar from audited references
  robserve [text,scene,meaning_tree]    learn a surface construction
  raudit [text,scene,meaning_tree]      audit fresh meaning; freezes training
  rinterpret [text,scene]              interpret a recursive objective
  rexecute [text,scene,values,optional_relations]  execute via live ACCE bindings
  rstatus                             inspect grammar and evidence
Recursive grammar and dispatch bindings are session-local. No general free-form chat.'''

def vector(tokens): return tuple(int(x) for x in tokens)
def pair(tokens):
    if tokens.count('->')!=1: raise ValueError('use INPUT -> OUTPUT')
    k=tokens.index('->'); return vector(tokens[:k]),vector(tokens[k+1:])

class Session:
    def __init__(self,path):
        self.path=Path(path); self.mind=load(path) if self.path.exists() else Mind(); self.current=None; self.last_decision=None
        self.dpath=self.path.with_name(self.path.stem+"-discovery.json"); self.discovery=DiscoveryMind.load(self.dpath) if self.dpath.exists() else DiscoveryMind(); self.dcurrent=None; self.dtrace=None
        self.upath=self.path.with_name(self.path.stem+"-unified.json"); self.unified=UnifiedMind.load(self.upath,self.discovery) if self.upath.exists() else UnifiedMind(self.discovery)
        self.opath=self.path.with_name(self.path.stem+"-operators.json"); self.operators=OperatorMind.load(self.opath) if self.opath.exists() else OperatorMind()
        self.mpath=self.path.with_name(self.path.stem+"-strategy.json"); self.strategy=StrategyMind.load(self.mpath) if self.mpath.exists() else StrategyMind()
        self.apath=self.path.with_name(self.path.stem+"-analogy.json"); self.analogy=AnalogicalMind.load(self.apath) if self.apath.exists() else AnalogicalMind(); self.analogy.reconcile_operators(self.operators)
        self.lpath=self.path.with_name(self.path.stem+"-language.json"); self.language=LanguageMind.load(self.lpath) if self.lpath.exists() else LanguageMind(); self.lcontext=None; self.composition=None; self.grammar=None
    def handle(self,line):
        if len(line)>32768: raise ValueError('input too long')
        raw=line.strip().split(maxsplit=1)
        if raw and raw[0] in {"rnew","robserve","raudit","rinterpret","rexecute","rstatus"}:
            return self.handle_recursive(raw[0],raw[1] if len(raw)>1 else None)
        if raw and raw[0] in {"cworkspace","ccatalog","cnew","cobserve","caudit","cadvance","cstatus","cdispatch","csay"}:
            return self.handle_composition(raw[0],raw[1] if len(raw)>1 else None)
        if raw and raw[0] in {"lnew","lobserve","laudit","linterpret","lexpress","ladvance","lcontext","say","lcompute","ltransfer","lsimulate","lstatus","lretract","lsave"}:
            return self.handle_language(raw[0],raw[1] if len(raw)>1 else None)
        if raw and raw[0] in {"asource","aimport","anew","aobserve","aaudit","apredict","aadvance","astatus","aretract","asave"}:
            return self.handle_analogy(raw[0],raw[1] if len(raw)>1 else None)
        if raw and raw[0] in {"mfit","mstatus","madvance","mrollback","msave"}:
            return self.handle_strategy(raw[0],raw[1] if len(raw)>1 else None)
        if raw and raw[0] in {"onew","oobserve","oaudit","opredict","oadvance","ostatus","oretract","osave"}:
            return self.handle_operator(raw[0],raw[1] if len(raw)>1 else None)
        if raw and raw[0] in {"uobserve","uaudit","ubind","ujaudit","ucstart","ucobserve","ucaudit","usstart","usobserve","usaudit","uspredict","usask","uremember","urecover","upredict","uplan","uadvance","ugoal","uchoose","uanalogy","ustatus","usave"}:
            return self.handle_unified(raw[0],raw[1] if len(raw)>1 else None)
        args=shlex.split(line)
        if not args: return ''
        cmd,*a=args
        if cmd=='help' and not a: return HELP
        if cmd in {'dnew','duse','dteach','dask','dadvance','dexpand','dpredict','dvalidated','daudit','dlearn','dplan','dretract','dstatus','dtrace','dsave'}: return self.handle_discovery(cmd,a)
        if cmd=='policytrace' and not a: return json.dumps(asdict(self.last_decision)) if self.last_decision else 'No IX policy executed in this session.'
        if cmd=='new' and len(a)==3:
            self.mind.new(a[0],Space(int(a[1]),int(a[2]))); self.current=a[0]; return 'Task created: '+a[0]
        if cmd=='use' and len(a)==1:
            if a[0] not in self.mind.tasks: raise ValueError('unknown task')
            self.current=a[0]; return 'Selected '+a[0]
        if cmd=='tasks' and not a: return json.dumps({n:t.summary() for n,t in self.mind.tasks.items()})
        if cmd=='save' and not a: save(self.mind,self.path); return 'Checkpoint saved.'
        if cmd not in {'teach','ask','predict','validated','audit','promote','plan','status','advance'}: return 'Unsupported request. Type help for the available commands.'
        if self.current is None: return 'Create or select a task first.'
        t=self.mind.tasks[self.current]
        if cmd=='advance' and not a:
            self.last_decision=advance(self.mind,self.current); return self.last_decision.message
        if cmd=='teach': t.observe(*pair(a)); return json.dumps(t.summary())
        if cmd=='ask' and not a:
            x=t.query(); return 'Suggested observation: '+str(x) if x is not None else 'No discriminating training query available (resolved, blocked or audit frozen).'
        if cmd in {'predict','validated'}:
            y=(t.validated_predict if cmd=='validated' else t.predict)(vector(a))
            if y is None: return 'Abstain: insufficient or contradictory evidence.'
            label='Audit-gated prediction' if cmd=='validated' else 'Prediction (provisional)'
            return label+': '+str(y)
        if cmd=='audit': return 'Audit passed.' if t.audit(*pair(a)) else 'Audit failed; task blocked.'
        if cmd=='promote' and not a: return 'Learned program: '+str(self.mind.promote(self.current))
        if cmd=='plan':
            x,y=pair(a); actions=[n for n,v in self.mind.tasks.items() if v.space==t.space and v.concept() is not None]; path=self.mind.plan(t.space,x,y,actions); return 'Simulation proposal: '+str(path) if path is not None else 'No supported path.'
        if cmd=='status' and not a: return json.dumps(t.summary())
        raise ValueError('incorrect command arguments')

    def handle_recursive(self,cmd,raw):
        try:
            if cmd=='rstatus' and raw is None:
                if self.grammar is None:raise ValueError('create a grammar first')
                g=self.grammar
                return json.dumps(dict(live=g.live(),rules=[dict(intent=r.intent,pattern=r.pattern,training=len(e['train']),audits=len(e['audit'])) for r,e in g.rules.items()]))
            x=json.loads(raw) if raw is not None else None
            if not isinstance(x,list):raise ValueError('expected JSON argument array')
            if cmd=='rnew' and len(x)==1:
                self.grammar=Grammar(self.language.episodes[x[0]]);return 'Recursive grammar created.'
            g=self.grammar
            if g is None:raise ValueError('create a grammar first')
            if cmd=='robserve' and len(x)==3:
                r=g.observe(*x);return json.dumps(dict(intent=r.intent,pattern=r.pattern))
            if cmd=='raudit' and len(x)==3:return json.dumps(dict(passed=g.audit(*x)))
            if cmd=='rinterpret' and len(x)==2:return json.dumps(dict(meaning=g.predict(*x)))
            if cmd=='rexecute' and len(x) in (3,4):
                if self.composition is None:raise ValueError('create and teach cognitive workspace first')
                return json.dumps(g.execute(*x[:3],self.composition,*(x[3:])))
            raise ValueError('invalid recursive language arguments')
        except (TypeError,IndexError,AttributeError,RecursionError) as e:raise ValueError('invalid recursive language arguments') from e

    def handle_composition(self,cmd,raw):
        if cmd=='cworkspace' and raw is None:
            ws=Workspace().primitives().operators(self.operators).scalars(self.unified).analogies(self.analogy,self.operators)
            self.composition=CognitiveDispatch(ws)
            return 'Fresh composition workspace created; previous dispatch bindings cleared.'
        d=self.composition
        if d is None:raise ValueError('create a composition workspace first')
        if cmd=='ccatalog' and raw is None:return json.dumps(d.workspace.catalog())
        if raw is None:raise ValueError('JSON arguments required')
        def args(x):
            if not isinstance(x,list) or not 1<=len(x)<=2:raise ValueError('expected one or two typed arguments')
            return tuple(Object(**a) if isinstance(a,dict) else a for a in x)
        try:
            x=json.loads(raw)
            if not isinstance(x,list):raise ValueError('expected argument list')
            if cmd=='cnew' and len(x)==3:r=d.new(*x).status()
            elif cmd=='cdispatch' and len(x)==2:r=d.execute(x[0],args(x[1]))
            elif cmd=='csay' and len(x) in (4,5):r=d.interpret(self.language.episodes[x[0]],*x[1:])
            else:
                b=d.bindings[(x[0],tuple(x[1]))]
                if cmd=='cobserve' and len(x)==4:b.observe(args(x[2]),x[3]);r=b.status()
                elif cmd=='caudit' and len(x)==4:r={'passed':b.audit(args(x[2]),x[3]),'status':b.status()}
                elif cmd=='cadvance' and len(x)==3:r=b.advance([args(a) for a in x[2]])
                elif cmd=='cstatus' and len(x)==2:r=b.status()
                else:raise ValueError('incorrect composition arguments')
        except (TypeError,IndexError,AttributeError,RecursionError) as e:raise ValueError('invalid composition arguments') from e
        return json.dumps(r,default=lambda v:asdict(v) if isinstance(v,Object) else jsonable(v))

    def handle_language(self,cmd,raw):
        m=self.language
        if cmd=='lsave' and raw is None:
            m.save(self.lpath);return 'Language checkpoint saved.'
        if cmd=='say':
            if raw is None or self.lcontext is None:raise ValueError('select a grounded language context first')
            name,rows=self.lcontext
            r=m.episodes[name].predict(raw,rows)
            return json.dumps({'intent':r}) if r is not None else 'Abstain: unsupported or ambiguous interpretation.'
        if raw is None:raise ValueError('JSON arguments required')
        try:
            x=json.loads(raw)
            if not isinstance(x,list) or not x or not isinstance(x[0],str):raise ValueError('expected name and arguments')
            name,*rest=x
            if cmd in ('lnew','lobserve','laudit','lretract'):r=m.apply(cmd[1:],name,*rest)
            elif cmd=='linterpret' and len(rest)==2:r=m.episodes[name].predict(*rest)
            elif cmd=='lexpress' and len(rest)==2:r=m.episodes[name].express(*rest)
            elif cmd=='ladvance' and len(rest)==1:r=m.advance(name,*rest)
            elif cmd=='lcontext' and len(rest)==1:
                if name not in m.episodes:raise ValueError('unknown language episode')
                self.lcontext=(name,scene(rest[0]));r='context selected'
            elif cmd=='lcompute' and len(rest)==3:r=m.compute(name,*rest,self.operators)
            elif cmd=='ltransfer' and len(rest)==2:r=m.transfer(name,*rest,self.analogy,self.operators)
            elif cmd=='lsimulate' and len(rest)==4:r=m.simulate(name,*rest,self.unified)
            elif cmd=='lstatus' and not rest:r=m.episodes[name].status()
            else:raise ValueError('incorrect language arguments')
        except (TypeError,IndexError,AttributeError,RecursionError) as e:raise ValueError('invalid language arguments') from e
        return json.dumps(r,default=jsonable)

    def handle_analogy(self,cmd,raw):
        m=self.analogy
        m.reconcile_operators(self.operators)
        if cmd=='asave' and raw is None:
            m.save(self.apath);return 'Relational transfer checkpoint saved.'
        if raw is None:raise ValueError('JSON arguments required')
        try:
            x=json.loads(raw)
            if not isinstance(x,list) or not x:raise ValueError('expected name and arguments')
            name,*rest=x
            if cmd in ('asource','anew','aobserve','aaudit','aretract'):result=m.apply(cmd[1:],name,*rest)
            elif cmd=='aimport' and len(rest)==3:
                states,relations,episode=rest;result=m.apply('operator_source',name,states,relations,self.operators.events,episode)
            elif cmd=='apredict' and len(rest)==1:result=m.episodes[name].predict(rest[0])
            elif cmd=='aadvance' and not rest:result=m.advance(name)
            elif cmd=='astatus' and not rest:result=m.episodes[name].status()
            else:raise ValueError('incorrect analogy arguments')
        except (TypeError,IndexError,AttributeError,RecursionError) as e:raise ValueError('invalid analogy arguments') from e
        return json.dumps(result)

    def handle_operator(self,cmd,raw):
        m=self.operators
        if cmd=='osave' and raw is None:
            m.save(self.opath); return 'Operator checkpoint saved.'
        if raw is None: raise ValueError('JSON arguments required')
        try:
            args=json.loads(raw)
            if not isinstance(args,list) or not args or not isinstance(args[0],str): raise ValueError('expected episode name and arguments')
            name,*rest=args
            if cmd in ('onew','oobserve','oaudit','oretract'): result=m.apply(cmd[1:],name,*rest)
            elif cmd=='opredict' and len(rest)==1: result=m.episodes[name].predict(rest[0])
            elif cmd=='oadvance' and len(rest)<=1: result=m.advance(name,*rest)
            elif cmd=='ostatus' and not rest: result=m.episodes[name].status()
            else: raise ValueError('incorrect operator arguments')
        except (TypeError,IndexError,AttributeError,RecursionError) as e: raise ValueError('invalid operator JSON arguments') from e
        return json.dumps(result)

    def handle_strategy(self,cmd,raw):
        m=self.strategy
        if cmd=='mstatus' and raw is None:
            r=m.report
            return json.dumps(dict(current=asdict(m.current),rolled_back=m.rolled_back,decision=r['decision'] if r else None,confirmation=r['confirmation_verdict'] if r else None,challenge=r['challenge_verdict'] if r else None))
        if cmd=='msave' and raw is None:m.save(self.mpath);return 'Strategy checkpoint saved.'
        if cmd=='mrollback' and raw is None:m.rollback();return 'Original inquiry strategy restored.'
        if raw is None:raise ValueError('JSON arguments required')
        arg=json.loads(raw)
        if cmd=='mfit' and isinstance(arg,str):
            with open(arg,'rb') as f:blob=f.read(STRATEGY_MAX_BYTES+1)
            if len(blob)>STRATEGY_MAX_BYTES:raise ValueError('strategy corpus too large')
            r=m.fit(json.loads(blob));return json.dumps(dict(decision=r['decision'],selected=r['selected'],confirmation=r['confirmation_verdict'],challenge=r['challenge_verdict']))
        if cmd=='madvance' and isinstance(arg,list) and len(arg)==2 and isinstance(arg[0],str) and isinstance(arg[1],list):
            return json.dumps(strategy_advance(self.operators,arg[0],arg[1],m.current))
        raise ValueError('incorrect strategy arguments')

    def handle_unified(self,cmd,raw):
        m=self.unified
        if raw is None:
            if cmd=='ustatus':return json.dumps(m.status())
            if cmd=='usave':self.discovery.save(self.dpath);m.save(self.upath);return 'Discovery and unified checkpoints saved.'
            raise ValueError('JSON arguments required')
        try:
            x=json.loads(raw)
            if cmd in ('uobserve','uaudit','ubind'):result=m.apply(cmd[1:],*x)
            elif cmd=='ujaudit':result=m.apply('joint_audit',*x)
            elif cmd in ('ucstart','ucobserve','ucaudit'):result=m.apply('coupled_'+cmd[2:],*x)
            elif cmd in ('usstart','usobserve','usaudit'):result=m.apply('scalar_'+cmd[2:],*x)
            elif cmd in ('uspredict','usask'):
                name,arg=x;result=getattr(m.scalar[name], 'predict' if cmd=='uspredict' else 'query')(arg)
            elif cmd=='uremember':result=m.apply('remember',x)
            elif cmd=='urecover':result=m.apply('recover',x)
            elif cmd=='upredict':f,v,a=x;result=m.predict(m.joint(f,v),a)
            elif cmd=='uanalogy':result=m.analogy_goal(*x)
            elif cmd in ('uplan','uadvance','ugoal','uchoose'):
                current=m.joint(x['facts'],x.get('vector'));actions=x['actions']
                if cmd=='uplan':result=m.plan(current,actions,**x['goal'])
                elif cmd=='uadvance':result=m.advance(current,actions,x.get('candidates',[]),x.get('goal'))
                elif cmd=='uchoose':result=m.choose_goal(current,actions,x['alternatives'])
                else:result=m.generate_goal(current,actions)
            else:raise ValueError('incorrect unified arguments')
        except (TypeError,IndexError,AttributeError,RecursionError) as e:raise ValueError('invalid unified JSON arguments') from e
        return json.dumps(result,default=jsonable)

    def handle_discovery(self,cmd,a):
        m=self.discovery
        if cmd=='dnew' and 3<=len(a)<=5:
            extra=a[3:]; cold='cold' in extra; modes=[v for v in extra if v!='cold']
            if len(modes)>1 or extra.count('cold')>1: raise ValueError('invalid discovery options')
            m.apply('new',a[0],int(a[1]),int(a[2]),not cold,modes[0] if modes else 'strict'); self.dcurrent=a[0]; return 'Discovery episode created: '+a[0]
        if cmd=='duse' and len(a)==1:
            if a[0] not in m.episodes: raise ValueError('unknown discovery episode')
            self.dcurrent=a[0]; return 'Selected discovery episode '+a[0]
        if cmd=='dsave' and not a: m.save(self.dpath); return 'Discovery checkpoint saved.'
        if cmd=='dtrace' and not a: return json.dumps(self.dtrace) if self.dtrace else 'No discovery policy executed.'
        if self.dcurrent is None: return 'Create or select a discovery episode first.'
        n=self.dcurrent; t=m.episodes[n]
        if cmd=='dteach': m.apply('observe',n,*pair(a)); return json.dumps(t.status())
        if cmd=='dask' and not a: return 'Suggested observation: '+str(t.query())
        if cmd=='dadvance' and not a: self.dtrace=m.advance(n); return self.dtrace['result']
        if cmd=='dexpand' and not a: m.apply('expand',n); return 'Search expanded.'
        if cmd in ('dpredict','dvalidated'):
            y=t.predict(vector(a),validated=cmd=='dvalidated'); return 'Abstain: insufficient or contradictory evidence.' if y is None else ('Audit-gated prediction: ' if cmd=='dvalidated' else 'Provisional prediction: ')+str(y)
        if cmd=='daudit': return 'Audit passed.' if m.apply('audit',n,*pair(a)) else 'Audit did not pass; see dstatus.'
        if cmd=='dlearn' and not a: return 'Learned expressions: '+str(m.apply('promote',n))
        if cmd=='dplan':
            x,y=pair(a); names=[k for k,v in m.episodes.items() if v.space==t.space and not v.blocked and v.passes>=3]; result=m.plan(t.space,x,y,names); return 'Simulation proposal: '+str(result) if result is not None else 'No supported path.'
        if cmd=='dretract' and not a: m.apply('retract',n); return 'Episode and dependent descendants quarantined.'
        if cmd=='dstatus' and not a: return json.dumps(t.status())
        raise ValueError('incorrect discovery arguments')

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--state',default='ix-zero-state.json'); a=parser.parse_args()
    try: s=Session(a.state)
    except (ValueError,OSError,KeyError,TypeError) as e: parser.exit(2,f'Checkpoint refused: {e}\n')
    print('IX-Zero 2.0.0 | local experimental learner | type help')
    while True:
        try:
            line=input('You > ')
            if line.strip()=='exit': break
            result=s.handle(line)
            if result: print('IX-Zero > '+result)
        except (EOFError,KeyboardInterrupt): break
        except (ValueError,KeyError,OSError) as e: print('IX-Zero > Request rejected:',e)

if __name__=='__main__': main()
