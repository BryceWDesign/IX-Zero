import unittest
from ix_zero.acce import Capability
from ix_zero.contextual import Domain, Episode
from ix_zero.contextual_demo import build
from ix_zero.contextual_experiment import sources


class ContextualTests(unittest.TestCase):
    def setUp(self):
        self.d=Domain(range(4),('ordinary','changed','other'))
        self.available=True
        self.base=Capability('base',('integer','integer'),'integer',lambda a,b:a^b,lambda:self.available,'test source')
        self.alt=Capability('alt',('integer','integer'),'integer',lambda a,b:a&b,lambda:True,'test source')
        self.normal=[x for x in self.d.inputs if x[0]=='ordinary']
        self.changed=[x for x in self.d.inputs if x[0]=='changed']
    def truth(self,x):return (x[1]&x[2]) if x[0]=='changed' else (x[1]^x[2])
    def incumbent(self):
        e=Episode(self.d,self.base,(self.alt,))
        for x in self.normal[:6]:e.observe(x,self.truth(x))
        for x in self.normal[6:9]:self.assertTrue(e.audit(x,self.truth(x)))
        return e
    def recovered(self):
        old=self.incumbent();x=next(x for x in self.changed if old.predict(x)!=self.truth(x));old.contradict(x,self.truth(x))
        rec=old.recover();pool=[x for x in self.d.inputs if x not in self.normal[9:12] and x not in self.changed[-3:]]
        for _ in range(20):
            q=rec.advance(pool)
            if q['action']!='inquire':break
            x=q['inquiry']['input'];rec.observe(x,self.truth(x))
        for x in self.normal[9:12]+self.changed[-3:]:self.assertTrue(rec.audit(x,self.truth(x)))
        return old,rec
    def test_no_unaudited_execution(self):
        e=Episode(self.d,self.base,(self.alt,))
        for x in self.normal[:6]:e.observe(x,self.truth(x))
        self.assertIsNone(e.predict(self.normal[7]));self.assertEqual(e.diagnose(),'insufficient_validation')
    def test_duplicate_training_rejected(self):
        e=Episode(self.d,self.base);e.observe(self.normal[0],0)
        with self.assertRaises(ValueError):e.observe(self.normal[0],0)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):self.d.key(('ordinary',True,0))
        with self.assertRaises(ValueError):self.d.output(99)
        with self.assertRaises(ValueError):Domain(range(9),('a','b'))
    def test_training_frozen_after_audit(self):
        e=self.incumbent()
        with self.assertRaises(ValueError):e.observe(self.normal[10],self.truth(self.normal[10]))
        with self.assertRaises(ValueError):e.audit(self.normal[0],0)
    def test_contradiction_quarantines(self):
        e=self.incumbent();x=('changed',0,1);self.assertEqual(e.predict(x),1);e.contradict(x,0)
        self.assertIsNone(e.predict(self.normal[0]));self.assertEqual(e.diagnose(),'contradicted_prediction');self.assertEqual(e.advance()['action'],'revise')
    def test_recovery_replays_audits_as_training(self):
        e=self.incumbent();e.contradict(('changed',0,1),0);r=e.recover()
        self.assertEqual(len(r.training),10);self.assertEqual(len(r.audits),0);self.assertIsNone(r.predict(('changed',0,1)))
        self.assertEqual(len(e.audits),3);self.assertFalse(e.live())
    def test_recovery_requires_contradiction(self):
        with self.assertRaises(ValueError):self.incumbent().recover()
    def test_question_discriminates(self):
        e=self.incumbent();e.contradict(('changed',0,1),0);r=e.recover();q=r.advance(self.changed)
        self.assertEqual(q['action'],'inquire');self.assertGreater(q['inquiry']['disagreement'],0);self.assertIn('what outcome',q['inquiry']['question']);self.assertGreater(len(q['inquiry']['possible_outcomes']),1)
    def test_recovered_predictions(self):
        old,r=self.recovered();self.assertFalse(old.live());self.assertTrue(r.live())
        for x in self.d.inputs:self.assertEqual(r.predict(x),self.truth(x))
    def test_context_drift_removes_stale_history(self):
        e=self.incumbent();x=self.normal[1];e.contradict(x,(self.truth(x)+1)%4);r=e.recover(changed_context=True)
        self.assertEqual(r.training,{x:(self.truth(x)+1)%4});self.assertEqual(len(r.audits),0)
    def test_unsupported_context_pair_abstains(self):
        old=self.incumbent();old.contradict(('changed',0,1),0);r=old.recover();r.observe(('other',0,1),0)
        self.assertEqual(r.diagnose(),'representation_exhausted');self.assertIsNone(r.predict(('changed',0,1)))
    def test_source_invalidation(self):
        _,r=self.recovered();cap=r.capability('ctx','changed');self.assertEqual(cap.call((3,1)),1)
        self.available=False;self.assertIsNone(cap.call((3,1)));self.assertEqual(r.diagnose(),'source_invalidated')
    def test_actual_operator_retraction(self):
        m,base,alts=sources(196200);d=Domain(range(8),('a','b'));e=Episode(d,base,alts)
        for x in d.inputs[:6]:e.observe(x,x[1]^x[2])
        for x in d.inputs[6:9]:self.assertTrue(e.audit(x,x[1]^x[2]))
        self.assertIsNotNone(e.predict(('b',2,3)));m.apply('retract','word_a');self.assertIsNone(e.predict(('b',2,3)))
    def test_recursive_language_recovered_representation(self):
        g,d,rows,v,old,rec,questions,feedback,truth=build()
        y=truth(('changed',v['B'],v['D']));r=g.execute('rule green cube blue cone',rows,v,d)
        self.assertEqual(r['value'],y);self.assertTrue(questions);self.assertFalse(old.live())
        self.assertEqual(g.execute('compare inspect red sphere rule green cube blue cone',rows,v,d)['value'],(v['C']>y)-(v['C']<y))

if __name__=='__main__':unittest.main()
