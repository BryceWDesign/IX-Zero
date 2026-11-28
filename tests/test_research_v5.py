import json
from itertools import product
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from ix_zero.exploration import Learner, Measurements, parse_grammar, route, execute, candidates, freeze, wilson
from ix_zero.audit import AuditConsensus, AuditConflict
from ix_zero.pair_inquiry import inquire, predict
from ix_zero.research_workspace import ResearchWorkspace
from ix_zero.web_research import Page, Research, PublicHTTPS, validate_url, extract_links
from ix_zero.arc_eval import solve, evaluate
from ix_zero.research_experiment import run


def learn(m,f,**opts):
    return m.fit([(x,f(x)) for x in (2,3,4)],Measurements([(x,f(x)) for x in (5,6)]),Measurements([(x,f(x)) for x in (7,8,9)]),**opts)


class ExplorationTests(unittest.TestCase):
    def test_active_query(self):
        m=Learner();pool=Measurements([(2,2),(-1,1),(-2,2)]);r=m.fit([(0,0),(1,1)],pool,Measurements([(-4,4),(5,5)]),depth=1,budget=8,query_budget=1)
        self.assertTrue(r['promoted']);self.assertLess(pool.asked[0],0);self.assertEqual(m.predict(r['skill'],-10),10)

    def test_unresolvable(self):
        m=Learner();r=learn(m,lambda x:100+x,depth=1,budget=8)
        self.assertFalse(r['promoted']);self.assertEqual(r['reason'],'representation_or_search_gap')

    def test_macro_and_invalidation(self):
        m=Learner();a=learn(m,lambda x:2*x+1,depth=2,budget=72);b=learn(m,lambda x:(2*x+1)**2,depth=2,budget=72)
        self.assertTrue(a['promoted'] and b['promoted']);self.assertEqual(m.predict(b['skill'],10),441)
        self.assertFalse(m.challenge(a['skill'],10,22));self.assertIsNone(m.predict(b['skill'],10))

    def test_control_budget(self):
        m=Learner();learn(m,lambda x:2*x+1,depth=2,budget=72);b=learn(m,lambda x:(2*x+1)**2,depth=2,budget=72,reuse=False)
        self.assertFalse(b['promoted']);self.assertEqual(b['attempted'],72)

    def test_grammar(self):
        spec={'external:a':[[str(i),str((i+1)%8)] for i in range(8)]};f=lambda x:str((int(x)+2)%8);m=Learner()
        r=m.fit([(str(i),f(str(i))) for i in range(2)],Measurements([(str(i),f(str(i))) for i in range(2,4)]),Measurements([(str(i),f(str(i))) for i in range(4,6)]),grammar=spec,depth=2)
        self.assertTrue(r['promoted']);self.assertEqual(m.predict(r['skill'],'7',spec),'1')
        with self.assertRaises(ValueError):m.predict(r['skill'],'7',{})

    def test_grammar_duplicate(self):
        with self.assertRaises(ValueError):parse_grammar({'external:a':[[1,2],[1,3]]})

    def test_no_overlapping_partitions(self):
        with self.assertRaises(ValueError):Learner().fit([(1,2)],Measurements([(1,2)]),Measurements([(3,4),(4,5)]))

    def test_no_undefined_promotion(self):
        m=Learner();r=m.fit([(1,2),(2,3)],Measurements([]),Measurements([(3,99),(4,5)]),depth=1)
        self.assertFalse(r['promoted'])

    def test_observation_limits(self):
        with self.assertRaises(ValueError):freeze(True)
        with self.assertRaises(ValueError):freeze([[[[[[1]]]]]])
        self.assertLess(wilson(2,2),0.5)

    def test_budget(self):self.assertEqual(len(list(candidates(('id','inc'),(),3,5))),5)

    def test_route(self):
        edges=[('a','walk','b'),('b','walk','c')]
        self.assertTrue(route('a','c',edges,['walk'])['reframed']);self.assertEqual(route('a','c',edges,forbidden=['walk'])['status'],'unreachable')

    def test_consensus(self):
        a=AuditConsensus({'a':[(1,2),(2,3)],'b':[(1,2),(2,3)],'c':[(1,7),(2,3)]});self.assertEqual(a.observe(1),2);self.assertEqual(len(a.log[0]['votes']),3)

    def test_consensus_conflict(self):
        a=AuditConsensus({'a':[(1,2)],'b':[(1,3)],'c':[(1,4)]});self.assertRaises(AuditConflict,a.observe,1)

    def test_consensus_mismatched_scope(self):
        with self.assertRaises(ValueError):AuditConsensus({'a':[(1,2)],'b':[(1,2)],'c':[(2,2)]})

    def test_pair_routing(self):
        xs=list(product(range(8),repeat=2));Random(23).shuffle(xs)
        for target,mechanism in [(lambda x:x[0]*x[1]+2,'scalar'),(lambda x:7^(x[0]&x[1]),'word'),(lambda x:int(x[0]<=x[1]),'predicate')]:
            r,h=inquire([(x,target(x)) for x in xs[:6]],Measurements([(x,target(x)) for x in xs[6:30]]),Measurements([(x,target(x)) for x in xs[30:36]]),query_budget=4)
            self.assertTrue(r['promoted']);self.assertEqual(r['selected']['mechanism'],mechanism);self.assertTrue(all(predict(h,x)==target(x) for x in xs[36:]))

    def test_pair_unsupported(self):
        from math import gcd
        xs=list(product(range(8),repeat=2));Random(23).shuffle(xs);f=lambda x:gcd(*x)
        r,h=inquire([(x,f(x)) for x in xs[:6]],Measurements([(x,f(x)) for x in xs[6:30]]),Measurements([(x,f(x)) for x in xs[30:36]]),query_budget=4)
        self.assertFalse(r['promoted']);self.assertIsNone(predict(h,xs[-1]))

    def test_experiment_negative_controls(self):
        s=run(2,950000)['summary'];self.assertEqual(s['active']['active']['correct'],36)
        self.assertEqual(s['macro_second']['False']['correct'],0)
        self.assertEqual(s['consensus'][0]['false_promotions'],0);self.assertGreater(s['consensus'][1]['false_promotions'],0)


class FixtureTransport:
    """Explicit transport fixture, never represented as live Internet."""
    def __init__(self,conflict=False):self.calls=[];self.conflict=conflict
    def get(self,url):
        self.calls.append(url)
        if 'duckduckgo.com' in url:return Page(url,b'<a href="https://one.example/a">alpha inside beta</a><a href="https://two.example/b">alpha beta</a>')
        if 'two.example' in url and self.conflict:return Page(url,b'<p>alpha is not inside beta.</p>')
        return Page(url,b'<script>malicious text</script><p>alpha is inside beta.</p>')


class ResearchTests(unittest.TestCase):
    def test_query_fetch_and_cache(self):
        t=FixtureTransport();m=Research(t);r=m.investigate('alpha','inside beta')
        self.assertEqual(r['status'],'retrieved_evidence');self.assertEqual(r['distinct_hosts'],2);self.assertFalse(r['truth_verified']);self.assertTrue('q=alpha+inside+beta' in t.calls[0]);self.assertTrue(all('malicious' not in x for s in r['sources'] for x in s['excerpts']))
        n=len(t.calls);self.assertTrue(m.investigate('alpha','inside beta')['recalled']);self.assertEqual(len(t.calls),n)

    def test_unknown_triggers_research(self):
        t=FixtureTransport();m=ResearchWorkspace(t);r=m.resolve('alpha','inside','beta');self.assertEqual(r['method'],'source_claims');self.assertTrue(r['result']['value']);self.assertFalse(r['truth_verified'])

    def test_known_prefers_memory(self):
        t=FixtureTransport();m=ResearchWorkspace(t);m.general.evidence.ingest('local','alpha is inside beta.');r=m.resolve('alpha','inside','beta');self.assertEqual(r['method'],'local_evidence');self.assertEqual(t.calls,[])

    def test_disagreement(self):
        t=FixtureTransport(True);r=ResearchWorkspace(t).resolve('alpha','inside','beta');self.assertEqual(r['result']['status'],'conflict')

    def test_failed_network(self):
        class Fail:
            def get(self,url):raise OSError('unavailable')
        self.assertEqual(Research(Fail()).investigate('alpha','beta')['status'],'unavailable')

    def test_local_documents(self):
        with TemporaryDirectory() as d:
            Path(d,'guide.txt').write_text('alpha is inside beta.',encoding='utf-8');t=FixtureTransport();m=ResearchWorkspace(t,d)
            r=m.resolve('alpha','inside','beta');self.assertEqual(r['research']['sources'][0]['kind'],'local_document');self.assertEqual(t.calls,[])

    def test_reject_urls(self):
        for u in ('file:///etc/passwd','http://example.com','https://user@example.com','https://example.com:8080'):
            with self.assertRaises(ValueError):validate_url(u)

    def test_private_dns(self):
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('127.0.0.1',443))]):
            with self.assertRaises(ValueError):PublicHTTPS().get('https://example.com')

    def test_search_link_decode(self):
        p=Page('https://html.duckduckgo.com/html/',b'<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Doc</a><a href="http://example.com">Unsafe</a>')
        self.assertEqual(extract_links(p),[('https://example.com/doc','Doc')])

    def test_workspace_checkpoint(self):
        t=FixtureTransport();w=ResearchWorkspace(t);req={'command':'explore','train':[[1,2],[2,3]],'pool':[[3,4]],'audit':[[4,5],[5,6]],'depth':1}
        r=w.execute(req);w.execute({'command':'general','request':{'command':'ingest','source':'alpha','text':'alpha is inside beta.'}});w.research.investigate('alpha','inside beta')
        with TemporaryDirectory() as d:
            p=Path(d,'state.json');w.save(p);n=ResearchWorkspace.load(p,t);self.assertEqual(n.execute({'command':'predict','key':r['skill'],'x':6})['value'],7);self.assertEqual(n.self_model()['sally_native'],w.self_model()['sally_native']);self.assertTrue(n.research.investigate('alpha','inside beta')['recalled']);self.assertTrue(n.general.ask('is alpha inside beta')['value'])
            obj=json.loads(p.read_text());obj['payload']['events']=[];p.write_text(json.dumps(obj));self.assertRaises(ValueError,ResearchWorkspace.load,p)

    def test_typed_task_and_pair_checkpoint(self):
        xs=list(product(range(8),repeat=2));Random(23).shuffle(xs);f=lambda x:7^(x[0]&x[1])
        w=ResearchWorkspace(FixtureTransport());r=w.execute({'command':'task','train':[(x,f(x)) for x in xs[:6]],'pool':[(x,f(x)) for x in xs[6:30]],'audit':[(x,f(x)) for x in xs[30:36]],'query_budget':4})
        self.assertEqual(r['chosen'],'pair_inquire');key=r['result']['key'];self.assertEqual(w.execute({'command':'pair_predict','key':key,'input':[2,3]})['value'],5)
        self.assertFalse(w.execute({'command':'pair_challenge','key':key,'input':[2,3],'expected':4})['consistent'])
        with TemporaryDirectory() as d:
            p=Path(d,'state.json');w.save(p);n=ResearchWorkspace.load(p);self.assertIsNone(n.execute({'command':'pair_predict','key':key,'input':[2,3]})['value'])

    def test_refresh_failure_invalidates_cache(self):
        t=FixtureTransport();m=Research(t);m.investigate('alpha','inside beta')
        class Fail:
            def get(self,url):raise OSError('offline')
        m.transport=Fail();self.assertEqual(m.investigate('alpha','inside beta',refresh=True)['status'],'unavailable');self.assertEqual(m.investigate('alpha','inside beta')['status'],'unavailable')

    def test_audit_no_majority_blocks(self):
        m=Learner();a=AuditConsensus({'a':[(4,5),(5,6)],'b':[(4,6),(5,6)],'c':[(4,7),(5,6)]});r=m.fit([(1,2),(2,3)],Measurements([]),a,depth=1)
        self.assertFalse(r['promoted']);self.assertEqual(len(r['audit_conflicts']),1)

    def test_typed_route_and_real_constraint(self):
        w=ResearchWorkspace();r=w.execute({'command':'task','start':'a','goal':'c','edges':[['a','walk','b'],['b','walk','c']],'forbidden':['walk']})
        self.assertEqual(r['result']['status'],'unreachable')

    def test_arc_format_and_negative(self):
        # An authored format test, not a benchmark result. A simple transform.
        train=[([[1,2,3],[4,5,6]],[[3,2,1],[6,5,4]]),([[0,4,7],[2,5,1]],[[7,4,0],[1,5,2]])]
        r=solve(train,[[[2,3,4],[5,6,7]]],budget=100,depth=1);self.assertEqual(r['predictions'][0],((4,3,2),(7,6,5)))
        with TemporaryDirectory() as d:
            task={'train':[{'input':x,'output':y} for x,y in train],'test':[{'input':[[2,3,4],[5,6,7]],'output':[[4,3,2],[7,6,5]]}]};Path(d,'task.json').write_text(json.dumps(task));self.assertEqual(evaluate(d,100,1)['exact_tasks'],1)

if __name__=='__main__':unittest.main()
