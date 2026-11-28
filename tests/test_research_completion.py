import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from ix_zero.audit import AuditConsensus, AuditConflict
from ix_zero.exploration import Measurements
from ix_zero.program_induction import ProgramMemory, inquire, run, candidates, search_stream, revise
from ix_zero.tool_induction import ToolMemory, discover
from ix_zero.object_induction import solve as grid_solve, components
from ix_zero.external_eval import evaluate
from ix_zero.browser_learning import BrowserLearner, Observation
from ix_zero.research_workspace import ResearchWorkspace


def packet(seed,f,offset=0):
    pts=[(a,b) for a in range(1,49) for b in range(1,49)];random.Random(seed).shuffle(pts);pts=pts[offset:offset+180]
    rows=[(x,f(*x)) for x in pts]
    return rows[:14],rows[14:44],rows[44:64],rows[64:]


def learn(memory,key,p,**opts):
    return memory.learn(key,p[0],Measurements(p[1]),Measurements(p[2]),budget=12000,**opts)


class ProgramTests(unittest.TestCase):
    def test_iteration_and_expression_control(self):
        p=packet(271100,math.gcd);a=ProgramMemory();r=learn(a,'x',p)
        self.assertTrue(r['promoted']);self.assertEqual(r['program'][0],'loop')
        self.assertTrue(all(a.predict('x',x)==y for x,y in p[3]))
        b=ProgramMemory();s=learn(b,'x',p,allow_loops=False)
        self.assertFalse(s['promoted']);self.assertEqual(s['attempted'],r['attempted'])
        self.assertIsNone(b.predict('x',(20,10)))
    def test_composition_and_counterfactual(self):
        p=packet(271101,math.gcd);q=packet(271101,math.lcm,180)
        for reuse in (True,False):
            a=ProgramMemory();self.assertTrue(learn(a,'a',p,reuse=reuse)['promoted']);r=learn(a,'b',q,reuse=reuse)
            self.assertEqual(r['promoted'],reuse)
            self.assertEqual(r['attempted'],12000)
            if reuse:
                self.assertEqual(r['dependencies'],['a']);self.assertTrue(all(a.predict('b',x)==y for x,y in q[3]));self.assertEqual(a.predict('b',(192,144)),576)
                self.assertEqual(a.challenge('a',(12,8),999)['withdrawn'],['a','b']);self.assertIsNone(a.predict('b',(12,8)))
            else:self.assertIsNone(a.predict('b',(12,8)))
    def test_nontermination_and_arithmetic_errors(self):
        a,b=('arg',0),('arg',1)
        self.assertIsNone(run(('loop','a0',a,b,1),(1,2)))
        self.assertIsNone(run(('expr',('div',a,b)),(1,0)))
        self.assertIsNone(run(('expr',('mul',a,b)),(10**9,10**9)))
        self.assertIsNone(run(('expr',a),[True]))
    def test_revision_on_fresh_signed_observations(self):
        p=packet(271100,math.gcd);m=ProgramMemory();learn(m,'a',p)
        rng=random.Random(43);xs=[(a,b) for a in range(-20,21) for b in range(-20,21) if a<=0 or b<=0];rng.shuffle(xs);xs.remove((0,0));xs=[(0,0)]+xs
        train=xs[:18]+[(60,42),(27,51)];pool=xs[18:58]+[(-30,-30),(70,70),(0,35),(-35,0),(35,0),(0,-35),(35,-35),(-35,35)];audit=xs[58:88]+[(63,77),(48,49)]
        f=lambda x:math.gcd(*x);rows=lambda p:[(x,f(x)) for x in p]
        report=revise(m,'a','r',rows(train),Measurements(rows(pool)),Measurements(rows(audit)))
        self.assertTrue(report['promoted']);self.assertEqual(report['derived_from'],'a');self.assertEqual(m.predict('r',(-70,-70)),70);self.assertEqual(m.predict('r',(0,75)),75)
        self.assertIsNone(m.predict('r',(1,2,3)))
    def test_scope_and_partition_rejection(self):
        p=packet(271102,math.gcd);m=ProgramMemory();learn(m,'a',p)
        self.assertIsNone(m.predict('a',(2,3,4)))
        with self.assertRaises(ValueError):m.challenge('a',(2,3,4),0)
        with self.assertRaises(ValueError):inquire(p[0],Measurements(p[0]),Measurements(p[2]))
    def test_unseen_ternary_expression(self):
        points=[(a,b,c) for a in range(1,6) for b in range(1,6) for c in range(1,6)];random.Random(31).shuffle(points)
        rows=[(x,x[0]+x[1]*x[2]) for x in points];m=ProgramMemory();r=m.learn('q',rows[:12],Measurements(rows[12:30]),Measurements(rows[30:42]),budget=12000)
        self.assertTrue(r['promoted']);self.assertTrue(all(m.predict('q',x)==y for x,y in rows[42:]))
    def test_search_control_unique_same_grammar(self):
        obs={(2,3):1,(6,8):2};space=set(candidates(2,True));
        for mode in ('enumerative','random','evolutionary'):
            stream=search_stream(2,True,(),obs,mode,41);first=[next(stream) for _ in range(100)]
            self.assertEqual(len(set(first)),100);self.assertTrue(set(first)<=space)
    def test_unanimity_and_correlated_failure(self):
        clean=[(3,4),(4,5),(5,6)];bad=[(3,99),(4,5),(5,6)]
        majority=AuditConsensus({'a':bad,'b':bad,'c':clean})
        self.assertEqual(majority.observe(3),99)
        strict=AuditConsensus({'a':bad,'b':bad,'c':clean},policy='unanimous')
        with self.assertRaises(AuditConflict):strict.observe(3)
        self.assertEqual(AuditConsensus({'a':bad,'b':bad,'c':bad},policy='unanimous').observe(3),99)
        with self.assertRaises(ValueError):AuditConsensus({'a':bad,'b':bad,'c':bad},policy='anything')
    def test_program_audit_conflict_blocks(self):
        p=packet(271103,math.gcd);bad=list(p[2]);bad[0]=(bad[0][0],bad[0][1]+1)
        audit=AuditConsensus({'a':p[2],'b':p[2],'c':bad},policy='unanimous');m=ProgramMemory();r=m.learn('a',p[0],Measurements(p[1]),audit,budget=12000)
        self.assertFalse(r['promoted']);self.assertEqual(r['reason'],'audit_conflict')


class ToolTests(unittest.TestCase):
    def test_real_signature_discovery(self):
        tools=discover(['math','operator'],2);self.assertIn('math.comb',[t[0] for t in tools]);self.assertNotIn('math.isqrt',[t[0] for t in tools])
        with self.assertRaises(ValueError):discover(['os'],2)
    def test_actual_tool_selection(self):
        xs=[(a,b) for a in range(1,17) for b in range(1,17)];random.Random(45).shuffle(xs);r=[(x,math.comb(*x)) for x in xs]
        m=ToolMemory();report=m.learn('a',r[:14],Measurements(r[14:44]),Measurements(r[44:64]),modules=['math','operator'])
        self.assertTrue(report['tool_assisted']);self.assertEqual(report['selected'],'math.comb');self.assertTrue(all(m.predict('a',x)==y for x,y in r[64:]))
        self.assertIsNone(m.predict('a',(2,3,4)));self.assertFalse(m.challenge('a',(4,2),1)['consistent']);self.assertIsNone(m.predict('a',(4,2)))


class GridTests(unittest.TestCase):
    def test_object_representation(self):
        train=[([[0,2,2,0,0],[0,2,2,0,1],[0,0,0,0,0]],[[2,2],[2,2]]),([[3,0,0,0],[0,0,4,4],[0,0,4,4]],[[4,4],[4,4]])]
        x=[[0,6,0,3,3],[0,0,0,3,3]];expected=((3,3),(3,3))
        self.assertEqual(grid_solve(train,[x])['predictions'],[expected]);self.assertEqual(grid_solve(train,[x],representation='pixels')['predictions'],[None])
    def test_components_modes(self):
        g=((1,0),(0,1));self.assertEqual(len(components(g,0,4)),2);self.assertEqual(len(components(g,0,8)),1)
        g=((1,2),);self.assertEqual(len(components(g,0,4,True)),2);self.assertEqual(len(components(g,0,4,False)),1)
    def test_ambiguous_and_invalid(self):
        r=grid_solve([([[1]],[[1]])],[[[1,2],[3,4]]]);self.assertIsNone(r['predictions'][0])
        with self.assertRaises(ValueError):grid_solve([([[1,2],[3]],[[1]])],[[[1]]])
    def test_external_worker_and_scoring(self):
        train=[{'input':[[1,2,3],[4,5,6]],'output':[[3,2,1],[6,5,4]]},{'input':[[0,4,7],[2,5,1]],'output':[[7,4,0],[1,5,2]]}]
        with tempfile.TemporaryDirectory() as d:
            Path(d,'fixture.json').write_text(json.dumps({'train':train,'test':[{'input':[[2,3,4],[5,6,7]],'output':[[4,3,2],[7,6,5]]}]}),encoding='utf-8')
            r=evaluate(d,budget=100,modes=('original',));self.assertEqual(r['summary']['original']['exact_tasks'],1);self.assertEqual(r['summary']['original']['execution_errors'],0)


class FixtureEnvironment:
    # Protocol-only test double; real Chromium execution is separate evidence.
    graph={'s':{'wrong':'d','a':'h'},'d':{'back':'s'},'h':{'b':'g','back':'s'},'g':{'back':'h'}}
    def reset(self):self.state='s';return self.observe()
    def observe(self):return Observation(self.state,tuple(self.graph[self.state]))
    def click(self,a):self.state=self.graph[self.state][a];return self.observe()


class BrowserTests(unittest.TestCase):
    def test_discovery_replay_and_restore(self):
        e=FixtureEnvironment();m=BrowserLearner();r=m.learn(e,click_budget=60);self.assertEqual(r['states'],4);self.assertEqual(r['conflicts'],0)
        self.assertEqual(m.achieve(e,'g',click_budget=2)['status'],'achieved');self.assertEqual(m.achieve(e,'g',click_budget=1)['status'],'budget')
        self.assertEqual(BrowserLearner().achieve(e,'g',click_budget=2)['status'],'unknown_route')
        restored=BrowserLearner.restore(json.loads(json.dumps(m.snapshot())));self.assertEqual(restored.achieve(e,'g',click_budget=2)['actions'],['a','b'])
        tamper=m.snapshot();tamper['payload']['transitions']=[]
        with self.assertRaises(ValueError):BrowserLearner.restore(tamper)
    def test_contradictory_transition(self):
        m=BrowserLearner();s=m.observe(Observation('s',('a',)));m.record(s,'a',Observation('t',()));m.record(s,'a',Observation('u',()));self.assertIn((s,'a'),m.conflicts);self.assertNotIn((s,'a'),m.transitions)
        with self.assertRaises(ValueError):m.record(s,'unseen',Observation('t',()))
    def test_invalid_controls(self):
        with self.assertRaises(ValueError):BrowserLearner().observe(Observation('s',('x','x')))


class IntegratedTests(unittest.TestCase):
    def test_program_checkpoint(self):
        w=ResearchWorkspace();p=packet(271104,math.gcd)
        r=w.execute({'command':'task','key':'a','train':p[0],'pool':p[1],'audit':p[2]});self.assertEqual(r['chosen'],'program_inquire');self.assertTrue(r['result']['promoted']);key=r['result']['key']
        with tempfile.TemporaryDirectory() as d:
            f=Path(d,'state.json');w.save(f);m=ResearchWorkspace.load(f);self.assertEqual(m.execute({'command':'program_predict','key':key,'input':[36,24]})['value'],12)
    def test_tool_fallback_and_replay(self):
        xs=[(a,b) for a in range(1,15) for b in range(1,15)];random.Random(5).shuffle(xs);r=[(x,math.comb(*x)) for x in xs];w=ResearchWorkspace()
        result=w.execute({'command':'task','key':'a','train':r[:10],'pool':r[10:40],'audit':r[40:55],'tool_modules':['math','operator']})
        self.assertEqual(result['chosen'],'tool_inquire');self.assertEqual(result['result']['selected'],'math.comb');k=result['result']['key']
        with tempfile.TemporaryDirectory() as d:
            f=Path(d,'state.json');w.save(f);m=ResearchWorkspace.load(f);self.assertEqual(m.execute({'command':'tool_predict','key':k,'input':[5,2]})['value'],10)
    def test_do_not_reuse_audit_after_failure(self):
        w=ResearchWorkspace();p=packet(271105,math.gcd);bad=list(p[2]);bad[0]=(bad[0][0],bad[0][1]+1)
        r=w.execute({'command':'task','key':'a','train':p[0],'pool':p[1],'audit':bad,'tool_modules':['math','operator']});self.assertEqual(r['next'],'fresh_audit_required');self.assertFalse(w.tools_memory.history)

if __name__=='__main__':unittest.main()
