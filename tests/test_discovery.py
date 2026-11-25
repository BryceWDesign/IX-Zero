import json
import tempfile
import unittest
from pathlib import Path
from ix_zero.core import Space
from ix_zero.discovery import DiscoveryMind,Episode,enumerate_models,evaluate,size
from ix_zero.cli import Session

class DiscoveryTests(unittest.TestCase):
    def fit(self,m,name='source',mod=5,reuse=False,mode='adaptive'):
        m.apply('new',name,2,mod,reuse,mode);t=m.episodes[name];f=lambda x:((x[0]+x[1])*x[0]%mod,x[1]);train=[(a,(a+k)%mod) for a in range(mod) for k in (0,1)]
        for x in train:m.apply('observe',name,x,f(x))
        if t.stage==3 and any(not h for h in t.symbolic):m.apply('expand',name)
        rest=[x for x in t.space.states if x not in train]
        for x in rest[:4]:self.assertTrue(m.apply('audit',name,x,f(x)))
        return t,f,rest

    def test_full_domain_semantic_dedup(self):
        s=Space(2,5);rows,truncated,_=enumerate_models(s,(),5);self.assertFalse(truncated);self.assertEqual(len(rows),len(set(c.table for c in rows)))
        for c in rows:self.assertEqual(c.table,bytes(evaluate(c.expr,x,s.modulus) for x in s.states))

    def test_learning_expression(self):
        m=DiscoveryMind();t,f,rest=self.fit(m);self.assertTrue(all(t.predict(x,True)==f(x) for x in rest[4:]));e=m.apply('promote','source');self.assertEqual(size(e[0]),5)

    def test_transfer_and_invalidation(self):
        m=DiscoveryMind();self.fit(m);m.apply('promote','source');m.apply('new','target',2,7,True,'adaptive');t=m.episodes['target'];f=lambda x:(((x[0]+x[1])*x[0]+1)%7,x[1]);train=[(a,(a+k)%7) for a in range(7) for k in (0,1)]
        for x in train:m.apply('observe','target',x,f(x))
        rest=[x for x in t.space.states if x not in train]
        for x in rest[:4]:self.assertTrue(m.apply('audit','target',x,f(x)))
        self.assertTrue(all(t.predict(x,True)==f(x) for x in rest[4:]));m.apply('retract','source');self.assertTrue(t.blocked);self.assertIsNone(t.predict(rest[-1],True));m.apply('new','future',2,7,True,'adaptive');self.assertFalse(m.episodes['future'].macros)

    def test_projection_without_known_expression(self):
        s=Space(2,5);t=Episode(s,mode='projection');p=[4,1,3,0,2];f=lambda x:(p[x[1]],p[x[0]]);train=[(a,(a+k)%5) for a in range(5) for k in (0,1)]
        for x in train:t.observe(x,f(x))
        rest=[x for x in s.states if x not in train]
        for x in rest[:3]:self.assertTrue(t.audit(x,f(x)))
        self.assertTrue(all(t.predict(x,True)==f(x) for x in rest[3:]));self.assertEqual(t.learned(),())

    def test_no_prediction_before_evidence(self):
        t=Episode(Space(2,5),mode='strict');self.assertIsNone(t.predict((0,0)));self.assertIsNone(t.predict((0,0),True))

    def test_audit_no_reuse_or_refit(self):
        m=DiscoveryMind();t,_,rest=self.fit(m)
        with self.assertRaises(ValueError):t.observe(rest[-1],(0,0))
        with self.assertRaises(ValueError):t.audit(rest[0],(0,0))
        with self.assertRaises(ValueError):t.expand()

    def test_wrong_audit_quarantines(self):
        m=DiscoveryMind();t,f,rest=self.fit(m);m.apply('promote','source');m.apply('new','child',2,5,True,'strict');x=rest[-1];y=f(x);self.assertFalse(m.apply('audit','source',x,((y[0]+1)%5,y[1])));self.assertTrue(m.episodes['child'].blocked)

    def test_replay_and_tamper(self):
        m=DiscoveryMind();self.fit(m);m.apply('promote','source');m.apply('new','child',2,7,True,'strict');m.apply('retract','source')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'d.json';m.save(p);n=DiscoveryMind.load(p);self.assertEqual(n.events,m.events);self.assertTrue(n.episodes['child'].blocked);o=json.loads(p.read_text());o['data']['events'].pop();p.write_text(json.dumps(o));self.assertRaises(ValueError,DiscoveryMind.load,p)

    def test_policy_is_executed_and_cannot_self_authorize(self):
        m=DiscoveryMind();m.apply('new','test',2,5,False,'adaptive');r=m.advance('test');self.assertEqual(r['action'],'inquire');self.assertEqual(r['trace'][-1]['kind'],'run.complete')
        with self.assertRaises(ValueError):m.advance('test','agent Discovery { on advance { reply "abstract" } }')
        with self.assertRaises(ValueError):m.advance('test','agent {')

    def test_cli_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';s=Session(p);self.assertIn('created',s.handle('dnew puzzle 2 5'));self.assertIn('Request',s.handle('dadvance'));s.handle('dteach 0 0 -> 0 0');s.handle('dsave');s=Session(p);s.handle('duse puzzle');self.assertEqual(json.loads(s.handle('dstatus'))['observations'],1)

    def test_ix_discovery_full_cycle(self):
        m=DiscoveryMind();m.apply('new','test',2,5,False,'adaptive');t=m.episodes['test'];f=lambda x:((x[0]+x[1])*x[0]%5,x[1]);train=[(a,(a+k)%5) for a in range(5) for k in (0,1)]
        for x in train:m.apply('observe','test',x,f(x))
        self.assertEqual(m.advance('test')['action'],'discover');self.assertEqual(m.advance('test')['action'],'audit')
        for x in [x for x in t.space.states if x not in train][:3]:m.apply('audit','test',x,f(x))
        self.assertEqual(m.advance('test')['action'],'abstract');self.assertEqual(m.advance('test')['action'],'complete')

    def test_discovery_planning_and_quarantine(self):
        m=DiscoveryMind();t,f,_=self.fit(m);start=(1,2);goal=f(start);self.assertEqual(m.plan(t.space,start,goal,['source']),(('source',goal),));m.apply('retract','source')
        with self.assertRaises(ValueError):m.plan(t.space,start,goal,['source'])

    def test_strict_keeps_longer_rivals(self):
        s=Space(2,5);a=Episode(s,mode='adaptive');a.expand();b=Episode(s,mode='strict')
        for t in (a,b):t.observe((0,0),(0,0));t.observe((1,1),(1,1))
        self.assertGreaterEqual(len(b.symbolic[0]),len(a.symbolic[0]));self.assertIsNone(b.predict((2,3),True))

if __name__=='__main__':unittest.main()
