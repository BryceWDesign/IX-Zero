import unittest
from itertools import product
from ix_zero.representation import Episode,Abstraction,Syntax,fit_with_abstraction,domain_labels
from ix_zero.representation_experiment import source,teacher,split,train,campaign
from ix_zero.representation_demo import build


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.m=source(184000);self.labels=tuple(range(-16,0,2));self.fn=teacher(self.labels,'ordinal',1,0,lambda a,b:a^b)
        self.p,self.a,self.h=split(self.labels,184001)

    def learn(self,labels=None,shift=0,op=None,mult=1):
        lab=self.labels if labels is None else labels
        fn=teacher(lab,'ordinal',mult,shift,op or (lambda a,b:a^b));p,a,h=split(lab,184001)
        e=Episode(self.m,lab);train(e,fn,p,a);return e,fn,h

    def abstract(self):
        a,_,_=self.learn();b,_,_=self.learn(tuple(range(-32,-16,2)),1)
        return Abstraction.induce(a,b)

    def test_domain_validation(self):
        for bad in ((1,2,3),(1,1,2,3),(False,1,2,3),(0,1,2,1000001)):
            with self.assertRaises(ValueError):domain_labels(bad)

    def test_bijection(self):
        for fam,m,c in product(('ordinal','gray'),range(1,8,2),range(8)):
            self.assertEqual(set(Syntax('word_a',fam,m,c).mapping(8)),set(range(8)))
        with self.assertRaises(ValueError):Syntax('word_a','ordinal',2,0).mapping(8)

    def test_no_unvalidated_prediction(self):
        e=Episode(self.m,self.labels)
        self.assertIsNone(e.predict(self.h[0]));self.assertEqual(e.advance(self.p)['action'],'inquire')
        for x in self.p[:6]:e.observe(x,self.fn(x))
        self.assertIsNone(e.predict(self.h[0]))

    def test_correct_heldout(self):
        e,fn,h=self.learn();self.assertTrue(e.resolved());self.assertEqual(len(e.audits),3)
        self.assertTrue(all(e.predict(x)==fn(x) for x in h));self.assertEqual(e.advance(self.p)['action'],'complete')

    def test_raw_unavailable(self):
        e=Episode(self.m,self.labels,raw=True);self.assertFalse(e.live());self.assertIsNone(e.predict(self.h[0]))

    def test_fresh_evidence_only(self):
        e,fn,h=self.learn();x=next(iter(e.training));a=next(iter(e.audits))
        for k in (x,a):
            with self.assertRaises(ValueError):e.audit(k,fn(k))
        with self.assertRaises(ValueError):e.observe(h[0],fn(h[0]))

    def test_contradiction_blocks(self):
        e,fn,h=self.learn();x=h[0];wrong=next(v for v in self.labels if v!=fn(x));self.assertFalse(e.audit(x,wrong));self.assertIsNone(e.predict(h[1]));self.assertEqual(e.advance(self.p)['action'],'revise')

    def test_retraction_blocks_capability(self):
        e,fn,h=self.learn();c=e.capability('r');self.assertEqual(c.call(h[0]),fn(h[0]));self.m.apply('retract','word_a');self.assertIsNone(c.call(h[0]))

    def test_abstraction_requires_distinct_validated_parents(self):
        a,_,_=self.learn();b=Episode(self.m,tuple(range(-32,-16,2)))
        with self.assertRaises(ValueError):Abstraction.induce(a,a)
        with self.assertRaises(ValueError):Abstraction.induce(a,b)

    def test_abstraction_novel_parameter(self):
        t=self.abstract();self.assertIsNone(t.shift);lab=tuple(range(-15,0,2));fn=teacher(lab,'ordinal',1,7,lambda a,b:a^b);p,a,h=split(lab,184001)
        e=Episode(self.m,lab,t);train(e,fn,p,a);self.assertTrue(all(e.predict(x)==fn(x) for x in h));self.assertLess(e.attempts,Episode(self.m,lab).attempts)

    def test_template_mind_identity(self):
        t=self.abstract();other=source(184001);self.assertFalse(Episode(other,self.labels,t).live())

    def test_template_dependency_invalidation(self):
        t=self.abstract();e=Episode(self.m,tuple(range(-15,0,2)),t);self.assertTrue(e.live());p=t.parents[0];x=next(x for x in product(p.labels,repeat=2) if x not in p.training and x not in p.audits);wrong=next(v for v in p.labels if v!=p.predict(x));p.audit(x,wrong);self.assertFalse(t.live());self.assertFalse(e.live())

    def test_adaptive_fallback_needs_fresh_audits(self):
        t=self.abstract();lab=tuple(range(-15,0,2));fn=teacher(lab,'ordinal',5,3,lambda a,b:a|b);p,a,h=split(lab,184001)
        full=Episode(self.m,lab);chosen=train(full,fn,p,a);e,cost=fit_with_abstraction(self.m,lab,t,[(x,fn(x)) for x in chosen]);self.assertTrue(cost['expanded']);self.assertIsNone(e.predict(h[0]));self.assertFalse(e.audits)
        for x in a:self.assertTrue(e.audit(x,fn(x)))
        self.assertTrue(all(e.predict(x)==fn(x) for x in h))

    def test_adaptive_rejects_duplicate_labels(self):
        t=self.abstract();x=self.p[0]
        with self.assertRaises(ValueError):fit_with_abstraction(self.m,self.labels,t,[(x,self.fn(x)),(x,self.fn(x))])

    def test_recursive_language_dispatch(self):
        g,d,rows,v,e,fn=build();r=g.execute('rule green cube blue cone',rows,v,d);self.assertEqual(r['value'],fn((v['B'],v['D'])));self.assertIn('representation7',r['trace'][0]['evidence']['programs'][0]['name']);e.mind.apply('retract','word_a');self.assertTrue(g.execute('rule green cube blue cone',rows,v,d)['abstained'])

    def test_campaign(self):
        s=campaign(185000,1,True)['summary'];self.assertEqual(s['adaptive']['correct'],205);self.assertEqual(s['negative']['abstained'],41);self.assertEqual(s['hidden_exception'],dict(errors=1,quarantined=1))
