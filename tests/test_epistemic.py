import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from ix_zero.developmental import Development
from ix_zero.epistemic import Inquiry,joint_search
from ix_zero.operators import Domain,execute
from ix_zero.epistemic_experiment import late_audit,environment_trial


def pool(w):
    xs=list(Domain(w).inputs);audit=[(0,0),(1,0),(Domain(w).mask,0),(0,1)];train=[x for x in xs if x not in audit][:min(128,len(xs)-7)]
    return dict(train=train,audit=audit)


def add(m,name,budget=2,deps=()):
    m.add(name,dict(kind='epistemic',deps=list(deps),width=2,widths=[2,3,4,5],budget=budget,**pool(2)))


def drive(m,shift):
    for _ in range(100):
        r=m.next()
        if r['action']!='request':return r
        if r['phase']=='environment':m.answer(pool(r['input']))
        else:
            w,a,b=r['input'];m.answer((a<<shift)&((1<<w)-1))
    raise AssertionError('step limit')


class EpistemicTests(unittest.TestCase):
    def test_joint_pruning_retains_local_collision(self):
        s=joint_search((2,3,4,5),2);zero=next(c for c in s.models if c.expr==('zero',));shift=next(c for c in s.models if c.expr==('shl',('shl',('x',))))
        self.assertEqual(zero.table[0],shift.table[0]);self.assertNotEqual(zero.table[1:],shift.table[1:]);self.assertFalse(s.truncated)

    def test_packed_against_scalar(self):
        s=joint_search((2,3,4,5),4);rng=random.Random(44)
        for c in rng.sample(list(s.models),40):
            for i,w in enumerate((2,3,4,5)):
                d=Domain(w)
                for x in rng.sample(list(d.inputs),min(16,len(d.inputs))):self.assertEqual(d.value(c.table[i],x),execute(c.expr,x,w))

    def test_incomplete_search_never_validates(self):
        joint_search.cache_clear()
        try:
            with patch('ix_zero.epistemic.MAX_ATTEMPTS',10):
                e=Inquiry([2,3,4,5],4,2,**pool(2));self.assertTrue(e.search.truncated);self.assertEqual(e.choose()['action'],'failed');self.assertFalse(e.validated())
        finally:joint_search.cache_clear()

    def test_late_exception_caught(self):
        for mode in ('control','early_exception','late_exception'):
            r=late_audit(310001,mode);self.assertEqual(r['status']['live'],mode=='control');self.assertEqual(r['incorrect'],0)
            if mode!='control':self.assertTrue(r['exception_audited']);self.assertIsNone(r['exception_prediction'])

    def test_scope_extension_and_checkpoint(self):
        for mode in ('shift2','shift4','zero'):
            r=environment_trial(311002,mode);self.assertTrue(r['replay']);self.assertTrue(r['status_before']['live']);self.assertGreater(r['correct'],0);self.assertEqual(r['incorrect'],0);self.assertTrue(any(q['phase']=='environment' for q in r['requests']));self.assertTrue(all(v['predicted'] is None for v in r['ungranted']))
            if mode=='shift4':self.assertIn(5,r['status_before']['epistemic']['granted'])

    def test_denied_environment_and_bad_grant(self):
        m=Development();add(m,'law')
        for _ in range(30):
            r=m.next();self.assertEqual(r['action'],'request')
            if r['phase']=='environment':break
            m.answer(0)
        else:self.fail('no environment request')
        self.assertEqual(r['inquiry']['local_distinctions'],{'2':1});self.assertGreater(r['inquiry']['requested_distinctions'],1)
        with self.assertRaises(ValueError):m.answer({'train':[[0,0]],'audit':[[0,0],[0,1],[0,2]]})
        with self.assertRaises(ValueError):m.answer({'train':[[0,0]],'audit':[[0,1],[0,2],[0,3]],'labels':[0]})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state';m.save(p);m=Development.load(p);self.assertEqual(m.pending['phase'],'environment')
        m.abandon();self.assertFalse(m.live('law'));self.assertIsNone(m.predict('law',[2,0,0]));self.assertEqual(m.next()['action'],'complete')

    def test_contradiction_and_later_retraction(self):
        r=environment_trial(311002,'audit_contradiction');self.assertFalse(r['status_before']['live']);self.assertEqual(r['incorrect'],0);self.assertTrue(r['replay'])
        r=environment_trial(311002,'later_contradiction');self.assertTrue(r['status_before']['live']);self.assertFalse(r['status_after']['live']);self.assertIsNone(r['later']['prediction_after']);self.assertTrue(r['replay'])

    def test_acquired_law_reused_and_dependency_invalidated(self):
        m=Development();add(m,'source');drive(m,2);self.assertTrue(m.live('source'))
        add(m,'target',2,['source']);drive(m,4);self.assertTrue(m.live('target'));self.assertEqual(m.predict('target',[5,3,4]),16)
        self.assertIn('source',m.jobs['target'].episode.models[0].dependencies)
        self.assertEqual(m.jobs['target'].episode.models[0].cost,2)
        # Reuse-disabled control has same bounded grammar/budget and cannot learn shift-four.
        n=Development(reuse=False);add(n,'source');drive(n,2);add(n,'target',2,['source']);drive(n,4);self.assertFalse(n.live('target'));self.assertIsNone(n.predict('target',[5,3,4]))
        # Later contradiction to the prerequisite invalidates the dependent law.
        e=m.jobs['source'].episode;w=e.expansions[-1];x=next(x for x in Domain(w).inputs if not any(x in p for p in e.pools[w]));p=m.predict('source',[w,*x]);self.assertFalse(m.challenge('source',[w,*x],p^1));self.assertFalse(m.live('target'));self.assertIsNone(m.predict('target',[5,3,4]))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'state';m.save(path);z=Development.load(path);self.assertEqual(m.status(),z.status());self.assertFalse(z.live('target'))

    def test_unapproved_environment_abstains(self):
        m=Development();add(m,'law');drive(m,2);self.assertTrue(m.live('law'));self.assertEqual(m.predict('law',[5,1,0]),4)
        with self.assertRaises(ValueError):m.challenge('law',[6,1,0],4)
        with self.assertRaises(ValueError):m.challenge('law',[2,0,0],0)

    def test_width_bounds(self):
        self.assertEqual(Domain(5).mask,31);self.assertEqual(execute(('shl',('x',)),(17,0),5),2)
        with self.assertRaises(ValueError):Domain(6)

if __name__=='__main__':unittest.main()
