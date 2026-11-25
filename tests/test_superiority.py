import json
import tempfile
import unittest
from pathlib import Path
from itertools import product
from ix_zero.unified import UnifiedMind
from ix_zero.coupled import CoupledEpisode
from ix_zero.scalar import ScalarEpisode
from ix_zero.superiority_experiment import learn_coupled, coupled_truth, row, sparse_campaign, recovery_campaign
from ix_zero.cli import Session


class CoupledTests(unittest.TestCase):
    def test_learns_both_cross_channel_directions(self):
        m,combos=learn_coupled(2,'gate',5)
        for i,(tag,v) in enumerate(combos):
            s,v,a,t,y=row(2,'gate',5,'unseen',i,tag,v)
            self.assertEqual(m.predict((s,v),a),(t,y))
        self.assertIsNone(m.predict(m.joint([('raw','x')]),('prep','x')))

    def test_sparse_combinations(self):
        for r in sparse_campaign([905]):
            self.assertEqual((r['correct'],r['incorrect'],r['abstained']),(12,0,0))

    def test_previous_joint_failure_recovers(self):
        r=recovery_campaign([900])[0]
        self.assertTrue(r['wrong_before'] and r['blocked_before_recovery'])
        self.assertEqual((r['correct'],r['incorrect'],r['abstained']),(50,0,0))

    def test_coupled_missing_audit_and_bad_space(self):
        m=UnifiedMind();m.apply('coupled_start',('prep','a'),2,5)
        s=m.joint([('raw','a')],[0,0]);self.assertIsNone(m.predict(s,('prep','a')))
        with self.assertRaises(ValueError):m.apply('coupled_observe',s[0],[5,0],('prep','a'),[],[0,0])
        with self.assertRaises(ValueError):m.apply('coupled_start',('prep','a'),2,5)

    def test_duplicate_training_and_audit_rejected(self):
        m=CoupledEpisode(2,5);r=row(1,'gate',5,'train',0,'raw',(0,0))
        self.assertTrue(m.observe(*r));self.assertFalse(m.observe(*r))
        with self.assertRaises(ValueError):m.audit(*r)

    def test_same_input_conflict(self):
        m=CoupledEpisode(2,5);s,v,a,t,y=row(1,'gate',5,'train',0,'raw',(0,0))
        m.observe(s,v,a,t,y);self.assertFalse(m.observe(s,v,a,s,v));self.assertTrue(m.blocked);self.assertIsNone(m.predict(s,v,a,False))

    def test_falsification_quarantine_and_new_epoch(self):
        m,_=learn_coupled(2,'gate',5);a=('prep','new');s=frozenset({('raw','new')});v=(0,0)
        self.assertFalse(m.apply('coupled_audit',s,v,a,s,v));self.assertIsNone(m.predict((s,v),a))
        self.assertEqual(m.advance((s,v),[a])['result']['coupled_recovery'],[('prep',1)])
        m.apply('coupled_start',a,2,5)
        with self.assertRaises(ValueError):m.apply('coupled_audit',s,v,a,s,v)
        self.assertIsNone(m.predict((s,v),a))

    def test_audits_freeze_and_checkpoints_replay(self):
        m,_=learn_coupled(2,'threshold',5);r=row(2,'threshold',5,'new',0,'raw',(2,2))
        with self.assertRaises(ValueError):m.apply('coupled_observe',*r)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'u.json';m.save(p);n=UnifiedMind.load(p)
            self.assertEqual(n.status(),m.status());self.assertEqual(n.predict(r[:2],r[2]),r[3:])

    def test_planner_replays_joint_effect(self):
        m,_=learn_coupled(2,'gate',5);a=('prep','new');s=m.joint([('raw','new')],[3,3]);t,y=coupled_truth(*s,a,5,'gate')
        p=m.plan(s,[a],present=t,absent=s[0]-t,vector=y)['path'];self.assertEqual(p,(a,))
        self.assertEqual(coupled_truth(*s,p[0],5,'gate'),(t,y))

    def test_unsupported_arbitrary_transform_abstains(self):
        m=CoupledEpisode(2,5)
        for i,(x,y) in enumerate(product(range(5),repeat=2)):
            a=('step',f'n{i}');s=[('raw',a[1])];m.observe(s,(x,y),a,s,((x*y+2*x+y)%5,(x*y)%5))
        self.assertIsNone(m.predict([('raw','z')],(3,2),('step','z'),False))


class ScalarTests(unittest.TestCase):
    def fit(self,m):
        pool=list(product(range(-3,4),repeat=2));used=[]
        for _ in range(7):
            q=m.query(pool);used.append(q);m.observe(q,q[0]*q[1]+2)
        for q in [x for x in pool if x not in used][:3]:self.assertTrue(m.audit(q,q[0]*q[1]+2))
        return m

    def test_signed_extrapolation(self):
        m=self.fit(ScalarEpisode());self.assertEqual(m.predict((-8,7)),-54)

    def test_gate_and_conflicting_feedback(self):
        m=ScalarEpisode();self.assertIsNone(m.predict((0,1)))
        m.observe((1,2),2);self.assertFalse(m.observe((1,2),3));self.assertIsNone(m.predict((1,2),False))

    def test_scalar_duplicate_frozen_and_retraction(self):
        m=self.fit(ScalarEpisode());q=next(p for p in [(8,8),(8,7)] if p not in m.training and p not in m.audits)
        self.assertFalse(m.audit(q,0));self.assertIsNone(m.predict((7,8)))
        with self.assertRaises(ValueError):m.observe((7,8),58)

    def test_scalar_replay_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';s=Session(p);self.assertEqual(s.handle('usstart ["calc"]'),'null')
            pool=list(product(range(-3,4),repeat=2));e=s.unified.scalar['calc'];used=[]
            for _ in range(7):
                q=e.query(pool);used.append(q);s.handle('usobserve '+json.dumps(['calc',q,q[0]*q[1]+2]))
            for q in [x for x in pool if x not in used][:3]:self.assertEqual(s.handle('usaudit '+json.dumps(['calc',q,q[0]*q[1]+2])),'true')
            s.handle('usave');n=Session(p);self.assertEqual(n.handle('uspredict ["calc",[-8,7]]'),'-54')
            self.assertEqual(n.unified.status(),s.unified.status())

    def test_reject_malformed(self):
        m=ScalarEpisode()
        for x in [(True,1),(33,1),(1,2,3)]:
            with self.assertRaises(ValueError):m.observe(x,1)
        with self.assertRaises(ValueError):m.observe((1,2),1.2)


if __name__=='__main__':unittest.main()
