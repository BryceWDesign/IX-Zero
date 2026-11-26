import json
import tempfile
import unittest
from pathlib import Path
from ix_zero.continual import Continual, digest, rows_of
from ix_zero.continual_experiment import Host, drive, replay, depth_control
from ix_zero.continual_cli import Console
from ix_zero.neural import Dense
from ix_zero.progressive import ProgressiveRecognition

class ContinualTests(unittest.TestCase):
    def setUp(self):self.seed=81310
    def host(self,law='left2',polarity=0,seed=None):return Host('word',[3,5],self.seed if seed is None else seed,law,polarity)
    def test_progressive_training_preserves_parent(self):
        h=self.host();rows=h.sensor.calibration(self.seed);p=Dense(3,6,1,7)
        for _ in range(12):
            for x,y in rows:p.train(x,[y])
        original=p.state();n=ProgressiveRecognition([p],8);before=n.state()['head'];other=self.host(polarity=1)
        for _ in range(12):
            for x,y in other.sensor.calibration(self.seed+1):n.train(x,[y])
        self.assertEqual(p.state(),original);self.assertEqual(n.parents[0].state(),original)
        self.assertNotEqual(n.state()['head'],before);self.assertEqual(n.depth,2)
        self.assertTrue(all(int(n.predict(x)[0]>=.5)==y for x,y in other.sensor.calibration(self.seed+2)))
    def test_lateral_features_really_enter_head(self):
        p=Dense(3,6,1,7);n=ProgressiveRecognition([p],8)
        self.assertEqual(n.head.shape,(9,8,1));x=(.1,.2,.3)
        y=n.predict(x);n.parents[0].w1[0][-1]+=10
        self.assertNotEqual(y,n.predict(x))
    def test_multi_generation_and_macro(self):
        m=Continual(max_budget=2);ids=[]
        for i,law in enumerate(('left2','left4','right2','left_right')):
            r=drive(m,self.host(law,i%2,self.seed+i),budget=2);self.assertEqual(r['state'],'validated');ids.append(r['episode'])
            self.assertEqual(r['wrong'],0)
        self.assertIn(ids[0],m.records[ids[1]]['dependencies'])
        self.assertEqual(m.records[ids[1]]['recognition_depth'],2)
        self.assertTrue(m.records[ids[2]]['sensor_source']);self.assertEqual(m.predict(ids[0],(5,3,0)),12)
        r,ok=replay(m);self.assertTrue(ok);self.assertEqual(r.predict(ids[1],(5,1,0)),16)
    def test_synthesis_macro_ablation(self):
        m=Continual(max_budget=2,reuse_skills=False)
        self.assertEqual(drive(m,self.host(),budget=2)['state'],'validated')
        self.assertNotEqual(drive(m,self.host('left4',seed=self.seed+1),budget=2)['state'],'validated')
    def test_automatic_depth_escalation(self):
        r=depth_control(self.seed);self.assertEqual(r['adaptive']['budget'],2)
        self.assertEqual(r['adaptive']['state'],'validated');self.assertNotEqual(r['fixed']['state'],'validated')
        self.assertNotEqual(r['outside']['state'],'validated');self.assertEqual(r['outside']['wrong'],0)
    def test_boundary_probe_blocks_hidden_exception(self):
        m=Continual(max_budget=1);h=Host('word',[3,5],self.seed,'identity',exception=(3,7,7));r=drive(m,h)
        self.assertNotEqual(r['state'],'validated');self.assertTrue(any(e['request']['action']=='probe' and tuple(e['request']['input'])==h.exception for e in r['trace']))
    def test_recovery_and_invalidation(self):
        m=Continual(max_budget=2);r=drive(m,self.host(),budget=2);key=r['episode']
        self.assertFalse(m.challenge(key,(5,3,0),3));self.assertIsNone(m.predict(key,(5,3,0)))
        changed=drive(m,self.host('identity',1,self.seed+1),start=False);self.assertEqual(changed['state'],'validated');self.assertFalse(m.live(key))
    def test_dependent_invalidation_and_composed_plan(self):
        m=Continual(max_budget=2);k=drive(m,self.host(),budget=2)['episode'];k2=drive(m,self.host('left4',seed=self.seed+1),budget=2)['episode']
        self.assertIsNone(m.compose_plan(5,1,16,[k],max_steps=1));p=m.compose_plan(5,1,16,[k]);self.assertEqual(len(p),2);self.assertEqual(p[-1]['after'],16)
        self.assertFalse(m.challenge(k,(5,1,0),1));self.assertFalse(m.live(k2));self.assertIsNone(m.compose_plan(5,1,16,[k2]))
    def test_checkpoint_rejects_rehashed_false_fingerprint(self):
        m=Continual();h=self.host();m.begin('word',[3,5],3,**h.pool(3,0));m.next()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json';m.save(p);self.assertEqual(Continual.load(p).fingerprint(),m.fingerprint())
            obj=json.loads(p.read_text());obj['payload']['fingerprint']='0'*64;obj['sha256']=digest(obj['payload']);p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):Continual.load(p)
    def test_denial_and_protocol(self):
        c=Console();h=self.host();k=c.execute(dict(command='begin',family='word',widths=[3,5],initial=3,**h.pool(3,0)))['episode']
        self.assertIsNone(c.execute(dict(command='predict',episode=k,input=[3,1,0]))['prediction'])
        self.assertEqual(c.execute(dict(command='next'))['action'],'sensor_check');c.execute(dict(command='deny'));self.assertEqual(c.execute(dict(command='next'))['action'],'failed')
        with self.assertRaises(ValueError):rows_of([([.5,.5,.5],0)]*8)
    def test_calibration_partition_overlap_rejected(self):
        m=Continual();h=self.host();m.begin('word',[3,5],3,**h.pool(3,0));m.next();rows=h.sensor.calibration(self.seed);m.answer(rows);m.next()
        with self.assertRaises(ValueError):m.answer(rows)
        self.assertEqual(m.next()['action'],'calibrate')

if __name__=='__main__':unittest.main()
