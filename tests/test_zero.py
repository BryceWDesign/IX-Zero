import json
import tempfile
import unittest
from pathlib import Path
from ix_zero.core import Mind,Space,Task,execute,grammar,PRIMITIVES,MAX_EXPANDED
from ix_zero.checkpoint import save,load,canonical
from ix_zero.cli import Session
from ix_zero.experiment import teacher,train,score
from ix_zero.validation import run
import random

S=Space(2,5)
def fit(t,p):
    pool=list(t.space.states); audit=pool[-3:]; oracle=lambda x:teacher(p,x,t.space.modulus)
    train(t,oracle,pool[:-3],12,True,random.Random(0))
    assert t.resolved(),p
    for x in audit: assert t.audit(x,oracle(x))
    return t

class EngineTests(unittest.TestCase):
    def test_primitive_teacher_agreement(self):
        for w in (2,3):
            for m in range(2,8):
                s=Space(w,m)
                for x in s.states:
                    for op in PRIMITIVES: self.assertEqual(execute((op,),x,s),teacher((op,),x,m))
    def test_validation(self):
        for w,m in [(1,5),(4,5),(2,8),(True,5),(2,2.0)]:
            with self.assertRaises(ValueError): Space(w,m)
        for x in [(True,1),(-1,2),(1,5),(1,),('1',2)]:
            with self.assertRaises(ValueError): S.check(x)
    def test_invalid_op(self):
        with self.assertRaises(ValueError): execute(('shell',),(0,1),S)
    def test_execution_cap(self):
        with self.assertRaises(ValueError): execute(('inc',)*(MAX_EXPANDED+1),(0,1),S)
    def test_grammar_cap(self):
        with self.assertRaises(ValueError): grammar((),4)
        with self.assertRaises(ValueError): grammar((('inc','shell'),),3)
    def test_grammar_reuse(self):
        p=('mix','rot','inc'); self.assertNotIn(p+p,grammar((),3)); self.assertIn(p+p,grammar((p,),3))
    def test_cold_abstention(self): self.assertIsNone(Task(S).predict((0,0)))
    def test_disagreement(self):
        t=Task(S); t.observe((0,0),(0,0)); self.assertIsNone(t.predict((1,2)))
    def test_active_query_no_leak(self):
        t=Task(S); self.assertIn(t.query([(0,0),(2,3)]),[(0,0),(2,3)])
        x=t.query(); t.observe(x,teacher(('mix',),x,5)); self.assertNotEqual(t.query(),x)
    def test_fit_and_audit(self):
        t=fit(Task(S),('mix','rot','inc'))
        self.assertTrue(t.resolved()); self.assertIsNotNone(t.concept())
        for x in S.states: self.assertEqual(t.predict(x),teacher(('mix','rot','inc'),x,5))
    def test_audit_not_training(self):
        t=Task(S); t.observe((0,0),(0,0)); before=list(t.candidates)
        t.audit((1,2),(4,4)); self.assertEqual(before,t.candidates); self.assertTrue(t.blocked)
    def test_audit_duplicate(self):
        t=Task(S); t.observe((0,0),(0,0))
        with self.assertRaises(ValueError): t.audit((0,0),(0,0))
    def test_audit_freezes(self):
        t=fit(Task(S),('inc',)); x=next(x for x in S.states if x not in t.observations and x not in t.audits)
        with self.assertRaises(ValueError): t.observe(x,teacher(('inc',),x,5))
    def test_drift(self):
        t=fit(Task(S),('inc',)); x,y=next(iter(t.observations.items())); t.observe(x,((y[0]+1)%5,y[1])); self.assertTrue(t.blocked); self.assertIsNone(t.predict(x)); self.assertIsNone(t.concept())
    def test_unrepresentable(self):
        t=Task(S,depth=1)
        for x in S.states: t.observe(x,teacher(('mix','rot','inc'),x,5))
        self.assertFalse(t.candidates); self.assertTrue(t.blocked)
    def test_validated_prediction_gate(self):
        t=Task(S); t.observe((0,0),(1,0)); self.assertIsNone(t.validated_predict((0,0)))
        t=fit(Task(S),('inc',)); self.assertEqual(t.validated_predict((0,0)),(1,0))
    def test_checkpoint_failed_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state'; m=Mind(); t=m.new('a',S); t.observe((0,0),(0,0)); t.audit((1,2),(4,4)); save(m,p); self.assertTrue(load(p).tasks['a'].blocked)
    def test_promotion_gate(self):
        m=Mind(); m.new('a',S)
        with self.assertRaises(ValueError): m.promote('a')
    def test_promotion_and_drift(self):
        m=Mind(); t=fit(m.new('a',S),('mix','rot','inc')); p=m.promote('a'); self.assertIn(p,m.new('b',S).library)
        x,y=next(iter(t.observations.items())); t.observe(x,((y[0]+1)%5,y[1])); self.assertNotIn(p,m.new('c',S).library)
    def test_duplicate_task(self):
        m=Mind(); m.new('a',S)
        with self.assertRaises(ValueError): m.new('a',S)
    def test_invalid_name(self):
        with self.assertRaises(ValueError): Mind().new('../bad',S)
    def test_plan(self):
        m=Mind(); fit(m.new('a',S),('inc',)); fit(m.new('b',S),('rot',)); path=m.plan(S,(0,0),(1,2),['a','b']); x=(0,0)
        for n,y in path:
            x=teacher({'a':('inc',),'b':('rot',)}[n],x,5); self.assertEqual(x,y)
        self.assertEqual(x,(1,2)); self.assertEqual(len(path),4)
    def test_plan_unreachable(self):
        m=Mind(); fit(m.new('a',S),('inc',)); self.assertIsNone(m.plan(S,(0,0),(0,1),['a']))
    def test_plan_denies_unaudited(self):
        m=Mind(); m.new('a',S)
        with self.assertRaises(ValueError): m.plan(S,(0,0),(0,1),['a'])
    def test_plan_noop(self): self.assertEqual(Mind().plan(S,(0,0),(0,0),[]),())
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state'; m=Mind(); t=fit(m.new('a',S),('mix','rot','inc')); m.promote('a'); m.new('b',S); save(m,p); restored=load(p); self.assertEqual(m.library(),restored.library()); self.assertEqual(restored.tasks['a'].predict((3,4)),t.predict((3,4)))
    def test_checkpoint_drift_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state'; m=Mind(); t=fit(m.new('a',S),('mix','rot','inc')); m.promote('a'); x,y=next(iter(t.observations.items())); t.observe(x,((y[0]+1)%5,y[1])); save(m,p); self.assertFalse(load(p).library())
    def test_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state'; save(Mind(),p); obj=json.loads(p.read_bytes()); obj['payload']['version']=2; p.write_bytes(canonical(obj))
            with self.assertRaises(ValueError): load(p)
    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state'; s=Session(p); self.assertIn('Unsupported',s.handle('what is AGI?')); s.handle('new test 2 5'); self.assertIn('Abstain',s.handle('predict 1 1')); s.handle('teach 1 1 -> 2 1'); self.assertIn('Prediction',s.handle('predict 1 1')); s.handle('save'); self.assertIn('test',Session(p).handle('tasks'))
    def test_holdout_separation(self):
        r=run(20261201)
        h={tuple(x) for x in r['holdout']}; c={tuple(x) for x in r['calibration_pool']}; self.assertFalse(h&c)
        for a in r['arms'].values():
            self.assertFalse({x for x,y in a['training']} & (h|c)); self.assertLessEqual(len(a['training']),5)
            for mode in ('raw','guarded'):
                z=a[mode]; self.assertEqual(z['correct']+z['wrong']+z['abstained'],z['total'])

if __name__=='__main__': unittest.main()
