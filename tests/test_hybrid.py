import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from ix_zero.checkpoint import canonical
from ix_zero.neural import Dense
from ix_zero.hybrid import Hybrid, GridInquiry
from ix_zero.hybrid_experiment import trial, campaign, check
from ix_zero.hybrid_cli import Console, ppm


class NeuralTests(unittest.TestCase):
    def test_training_learns_xor(self):
        n=Dense(2,8,1,55); rows=[([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
        before=n.state()
        for _ in range(700):
            for x,y in rows:n.train(x,y,rate=.2)
        self.assertNotEqual(before,n.state())
        self.assertEqual([int(n.predict(x)[0]>=.5) for x,_ in rows],[0,1,1,0])

    def test_gradient_matches_finite_difference(self):
        n=Dense(2,3,1,55);x=[.2,.8];y=[1];w=n.w1[0][0];eps=1e-5
        n.w1[0][0]=w+eps;plus=-math.log(n.predict(x)[0]);n.w1[0][0]=w-eps;minus=-math.log(n.predict(x)[0]);n.w1[0][0]=w
        n.train(x,y,rate=.01);self.assertAlmostEqual((w-n.w1[0][0])/.01,(plus-minus)/(2*eps),places=6)

    def test_invalid_values(self):
        n=Dense(2,3,1,55)
        for x in ([float('nan'),0],[2,0],[0],[True,0]):
            with self.assertRaises(ValueError):n.predict(x)
        with self.assertRaises(ValueError):n.train([0,1],[2])


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=campaign(1,410000)

    def test_campaign_gate(self):self.assertTrue(check(self.result))

    def test_no_false_predictions(self):
        self.assertTrue(all(r['wrong']==0 for r in self.result['rows']))

    def test_independent_sensor_training(self):
        for r in self.result['rows']:
            if r['mode']=='hybrid':
                self.assertEqual(r['recognition_after'],96)
                self.assertFalse(set(tuple(x) for x,_ in r['calibration'])&set(tuple(x) for x,_ in r['calibration_test']))
                self.assertTrue(all(r['updates'][k]>0 and r['initial_weights'][k]!=r['final_weights'][k] for k in r['updates']))

    def test_calibrated_non_neural_control(self):
        for r in self.result['rows']:
            if r['mode']=='centroid':self.assertTrue(r['promoted']);self.assertTrue(all(v==0 for v in r['updates'].values()))

    def test_heldout_disjoint(self):
        for r in self.result['rows']:
            for w,p in r['partitions'].items():
                self.assertEqual(sum(len(p[k]) for k in p),len(set(tuple(x) for v in p.values() for x in v)))
                held={tuple(x) for x in p['hold']}
                self.assertFalse(any(tuple(q['input'][1:]) in held for q in r['requests'] if q['action']!='environment' and q['input'][0]==w))

    def test_transfer_weights(self):
        for t in self.result['transfers']:
            r=t['target_result'];self.assertEqual(r['calibration'],[]);self.assertTrue(r['promoted']);self.assertEqual(t['source_final']['recognition'],r['initial_weights']['recognition']);self.assertEqual(t['source_final']['strategy'],r['initial_weights']['strategy']);self.assertNotEqual(t['source_final']['world'],r['initial_weights']['world'])

    def test_audit_failure_and_denial(self):
        for r in self.result['negative_controls']:
            if r['negative'] in ('audit','deny'):self.assertFalse(r['promoted']);self.assertEqual(r['abstain'],len(r['rows']))

    def test_later_invalidation(self):
        self.assertTrue(all(r['invalidated'] for r in self.result['negative_controls'] if r['negative']=='later'))

    def test_spatial_plans(self):
        for r in self.result['rows']:
            if r['family']=='grid' and r['mode']=='hybrid':self.assertEqual(len(r['plans']),12);self.assertTrue(all(p['valid'] for p in r['plans']))

    def test_checkpoint_tamper_rejected_even_with_recomputed_digest(self):
        import hashlib
        _,m=trial(410000,'grid','hybrid',return_mind=True)
        with TemporaryDirectory() as d:
            p=Path(d)/'state';m.save(p);self.assertEqual(canonical(Hybrid.load(p).payload()),canonical(m.payload()))
            x=json.loads(p.read_text());x['payload']['weights']['world']['w1'][0][0]+=1;x['sha256']=hashlib.sha256(canonical(x['payload'])).hexdigest();p.write_bytes(canonical(x))
            with self.assertRaises(ValueError):Hybrid.load(p)

    def test_no_prediction_before_audit(self):
        m=Hybrid('word',[2],2,[(0,0),(1,0),(2,0)],[(0,1),(1,1),(2,1)])
        self.assertIsNone(m.predict((2,1,0)))
        with self.assertRaises(ValueError):m.answer(0)
        with self.assertRaises(ValueError):m.transfer('grid',[3],3,[(0,0,0)],[(0,0,1),(0,0,2),(0,0,3)])

    def test_invalid_spatial_partitions(self):
        with self.assertRaises(ValueError):GridInquiry((3,),3,[(0,0,0)],[(0,0,0),(0,0,1),(0,0,2)])

    def test_ambiguous_raster_not_a_measurement(self):
        m=Hybrid('grid',[3],3,[(0,0,0)],[(0,0,1),(0,0,2),(0,0,3)],mode='symbolic')
        self.assertIsNone(m.decode(3,[[0,0,0]]*9));self.assertIsNone(m.decode(3,[[1,1,1]]*9))

    def test_ppm_file_adapter(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'pixels.ppm';p.write_text('P3\n# marker\n2 1\n255\n255 255 255 0 0 0\n')
            self.assertEqual(ppm(p),(2,1,[[1.,1.,1.],[0.,0.,0.]]))
            c=Console();c.execute(dict(command='new',family='word',widths=[2],initial=2,train=[[0,0],[1,0],[2,0]],audit=[[0,1],[1,1],[2,1]],mode='symbolic'))
            self.assertEqual(c.execute(dict(command='next'))['action'],'train')
            self.assertTrue(c.execute(dict(command='answer_ppm',path=str(p)))['accepted'])
            self.assertEqual(c.execute(dict(command='status'))['training'],1)
            p.write_text('P3 9 1 255 0 0 0');self.assertRaises(ValueError,ppm,p)

    def test_pending_checkpoint(self):
        m=Hybrid('word',[2],2,[(0,0),(1,0),(2,0)],[(0,1),(1,1),(2,1)])
        req=m.next()
        with TemporaryDirectory() as d:
            p=Path(d)/'state';m.save(p);r=Hybrid.load(p);self.assertEqual(req,r.next())


if __name__=='__main__':unittest.main()
