import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from ix_zero.operators import Domain, OperatorEpisode, OperatorMind, execute, synthesize
from ix_zero.cli import Session


def train(m,name,width=3,budget=5,reuse=False,fn=lambda a,b:a^b):
    m.apply('new',name,width,budget,reuse)
    t=m.episodes[name];xs=list(t.domain.inputs);random.Random(4).shuffle(xs)
    for x in xs[:14]:m.apply('observe',name,x,fn(*x))
    for x in xs[14:20]:
        if not m.apply('audit',name,x,fn(*x)):raise AssertionError('fixture failed')
    return t,xs[20:]


class OperatorsTest(unittest.TestCase):
    def test_packed_matches_independent_scalar(self):
        for width in (2,3,4):
            d=Domain(width);search=synthesize(width,4)
            for c in search.models:
                for x in d.inputs[::7]:self.assertEqual(d.value(c.table,x),execute(c.expr,x,width))

    def test_xor_discovered_without_xor_primitive(self):
        m=OperatorMind();t,hold=train(m,'rule');expr=t.simplest()[0].expr
        self.assertNotIn('xor',str(expr));self.assertEqual(t.simplest()[0].cost,5)
        self.assertTrue(all(t.predict(x)==x[0]^x[1] for x in hold))
        self.assertEqual(m.advance('rule')['action'],'abstract');self.assertEqual(m.promoted['rule'],expr)

    def test_transfer_reuses_operator_with_substituted_inputs(self):
        m=OperatorMind();train(m,'source');m.advance('source')
        t,hold=train(m,'target',4,3,True,lambda a,b:a^(b>>1))
        self.assertEqual(t.simplest()[0].dependencies,('source',))
        self.assertEqual(t.simplest()[0].cost,2)
        self.assertTrue(all(t.predict(x)==x[0]^(x[1]>>1) for x in hold))
        cold=OperatorEpisode(4,3)
        for x,y in t.training.items():cold.observe(x,y)
        self.assertFalse(cold.resolved())
        self.assertIsNone(cold.predict(hold[0]))

    def test_three_generation_composition(self):
        m=OperatorMind();train(m,'source');m.advance('source')
        train(m,'second',4,2,True,lambda a,b:a^(b>>1));m.advance('second')
        t,hold=train(m,'third',4,2,True,lambda a,b:(a^(b>>1))^(a>>1))
        self.assertIn('second',t.simplest()[0].dependencies)
        self.assertTrue(all(t.predict(x)==(x[0]^(x[1]>>1))^(x[0]>>1) for x in hold))
        self.assertEqual(m.advance('third')['action'],'abstract')

    def test_multilevel_dependency_retraction(self):
        m=OperatorMind();source,hold=train(m,'source');m.advance('source')
        target,_=train(m,'target',4,3,True,lambda a,b:a^(b>>1));m.advance('target')
        m.apply('new','descendant',4,2,True)
        self.assertFalse(m.apply('audit','source',hold[0],source.predict(hold[0])^1))
        self.assertTrue(all(t.blocked for t in m.episodes.values()))
        self.assertIsNone(target.predict((3,4)))
        m.apply('new','fresh',4,3,True);self.assertEqual(m.episodes['fresh'].library,())

    def test_no_premature_promotion(self):
        m=OperatorMind();m.apply('new','rule',3,5,False)
        with self.assertRaises(ValueError):m.apply('promote','rule')
        t=m.episodes['rule'];x=(1,2);m.apply('observe','rule',x,3)
        self.assertIsNone(t.predict(x,False));self.assertEqual(m.advance('rule')['action'],'inquire')
        with self.assertRaises(ValueError):m.apply('audit','rule',(3,4),7)

    def test_primitive_not_promotable(self):
        m=OperatorMind();train(m,'primitive',fn=lambda a,b:7^(a&b))
        self.assertEqual(m.advance('primitive')['action'],'complete')
        with self.assertRaises(ValueError):m.apply('promote','primitive')

    def test_audit_freezes_observations_and_disallows_duplicates(self):
        m=OperatorMind();t,hold=train(m,'rule')
        with self.assertRaises(ValueError):t.audit(next(iter(t.training)),0)
        with self.assertRaises(ValueError):t.audit(next(iter(t.audits)),0)
        with self.assertRaises(ValueError):t.observe(hold[0],0)

    def test_domain_bounds_and_types(self):
        self.assertEqual(len(Domain(5).inputs),1024)
        for bad in (1,6,True,3.0):
            with self.assertRaises(ValueError):Domain(bad)
        for bad in ((-1,0),(0,8),(True,0),(0,), '10'):
            with self.assertRaises(ValueError):OperatorEpisode(3,1).observe(bad,0)
        with self.assertRaises(ValueError):OperatorEpisode(3,7)

    def test_cap_discards_incomplete_layer(self):
        synthesize.cache_clear()
        try:
            with patch('ix_zero.operators.MAX_ATTEMPTS',30):
                s=synthesize(3,6)
                self.assertTrue(s.truncated);self.assertLess(s.complete_cost,6)
                self.assertTrue(all(c.cost<=s.complete_cost for c in s.models))
        finally:synthesize.cache_clear()

    def test_replay_and_corruption(self):
        m=OperatorMind();train(m,'source');m.advance('source');train(m,'target',4,3,True,lambda a,b:a^(b>>1));m.advance('target')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ops.json';m.save(p);loaded=OperatorMind.load(p)
            self.assertEqual(loaded.promoted,m.promoted);self.assertEqual(loaded.episodes['target'].predict((8,7)),11)
            obj=json.loads(p.read_text());obj['data']['events'][0][2]=4;p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):OperatorMind.load(p)

    def test_policy_training_pool_restriction(self):
        m=OperatorMind();m.apply('new','rule',3,5,False)
        allowed=[(0,0),(1,2),(3,4)];d=m.advance('rule',allowed)
        self.assertIn(d['result'],allowed);self.assertTrue(d['trace'])
        self.assertEqual(m.advance('rule',[])['action'],'exhausted')

    def test_cli_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';s=Session(p)
            s.handle('onew ["rule",3,5,false]')
            self.assertEqual(json.loads(s.handle('ostatus ["rule"]'))['training'],0)
            self.assertEqual(json.loads(s.handle('oadvance ["rule"]'))['action'],'inquire')
            s.handle('osave');self.assertIn('rule',Session(p).operators.episodes)
            with self.assertRaises(ValueError):s.handle('onew []')

    def test_unrepresentable_random_mapping_abstains(self):
        t=OperatorEpisode(4,3);rng=random.Random(100)
        for x in t.domain.inputs[:14]:t.observe(x,rng.randrange(16))
        self.assertFalse(t.resolved());self.assertIsNone(t.predict((15,15)))

if __name__=='__main__':unittest.main()
