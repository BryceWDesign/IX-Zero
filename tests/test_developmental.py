import hashlib
import json
from itertools import product
from pathlib import Path
import tempfile
import unittest
from ix_zero.checkpoint import canonical
from ix_zero.developmental import Development, normal
from ix_zero.developmental_experiment import trial


def spec(deps=(),width=3,budget=5):
    xs=list(product(range(1<<width),repeat=2));return dict(kind='operator',deps=list(deps),width=width,budget=budget,train=xs[:24],audit=xs[24:27])

class DevelopmentalTests(unittest.TestCase):
    def test_cycle_rejected(self):
        d=Development();d.add('first',spec(['second']))
        with self.assertRaises(ValueError):d.add('second',spec(['first']))
        self.assertEqual(list(d.jobs),['first'])

    def test_wait_missing_dependency(self):
        d=Development();d.add('first',spec(['missing']));self.assertEqual(d.next()['action'],'wait');self.assertIsNone(d.predict('first',[2,3]))

    def test_duplicate_partition_rejected(self):
        d=Development();s=spec();s['audit'][0]=s['train'][0]
        with self.assertRaises(ValueError):d.add('first',s)

    def test_teacher_not_in_schema(self):
        d=Development();s=spec();s['expected']=5
        with self.assertRaises(ValueError):d.add('first',s)

    def test_no_pending_answer(self):
        with self.assertRaises(ValueError):Development().answer(1)

    def test_pending_and_unvalidated(self):
        d=Development();d.add('first',spec());r=d.next();self.assertEqual(r['phase'],'train');self.assertIsNone(d.predict('first',[2,3]))
        with self.assertRaises(ValueError):d.next()
        with self.assertRaises(ValueError):d.answer(99)
        self.assertEqual(d.pending['input'],r['input']);d.answer(r['input'][0]^r['input'][1]);self.assertIsNone(d.predict('first',[2,3]))

    def test_checkpoint_pending(self):
        d=Development();d.add('first',spec());r=d.next()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state';d.save(p);e=Development.load(p);self.assertEqual(d.pending,e.pending);self.assertEqual(d.events,e.events)
            y=r['input'][0]^r['input'][1];d.answer(y);e.answer(y);self.assertEqual(normal(d.next()),normal(e.next()))

    def test_checkpoint_corruption(self):
        d=Development();d.add('first',spec())
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state';d.save(p);obj=json.loads(p.read_text());obj['data']['reuse']=False;p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):Development.load(p)

    def test_replay_decision_mismatch_even_with_digest(self):
        d=Development();d.add('first',spec());d.next()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state';d.save(p);obj=json.loads(p.read_text());obj['data']['events'][-1][1]['input']=[0,0];obj['sha256']=hashlib.sha256(canonical(obj['data'])).hexdigest();p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):Development.load(p)

    def test_abandon_does_not_promote(self):
        d=Development();d.add('first',spec());d.next();d.abandon();self.assertEqual(d.next()['action'],'complete');self.assertFalse(d.live('first'));self.assertEqual(d.status()['first']['training'],0)

    def test_audit_failure_not_promoted(self):
        # A full unlabeled pool avoids resolving from a single first-coordinate value.
        d=Development();s=spec();xs=list(product(range(8),repeat=2));s['train']=xs[::2];s['audit']=xs[1::2][:3];d.add('first',s)
        for _ in range(40):
            r=d.next()
            if r['action']!='request':break
            a,b=r['input'];d.answer((a^b)^1 if r['phase']=='audit' else a^b)
            if r['phase']=='audit':break
        self.assertFalse(d.live('first'));self.assertNotIn('first',d.words.promoted)

    def test_unsupported_job_does_not_create_capability(self):
        d=Development();d.add('first',dict(kind='grammar',deps=[],train=[['inspect red cube',[['a','cube','red']]]],audit=[['inspect '+v+' cube',[['a','cube',v]]] for v in ('green','blue','white')]))
        self.assertEqual(d.next()['action'],'complete');self.assertFalse(d.live('first'))

    def test_offer_reports_missing_prerequisite(self):
        d=Development();xs=list(product(range(4),repeat=2));r=d.offer("new-domain",list(range(4)),xs[:6],xs[6:9]);self.assertEqual(r["action"],"request_prerequisite");self.assertEqual(r["width"],2);self.assertEqual(d.jobs,{})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"state";d.save(p);e=Development.load(p);self.assertEqual(e.events,d.events)

    def test_replay_preserves_invalidation_and_independent_graph(self):
        d,rows,values,relations=trial(208000,True,demo=True);numeric="compare inspect red sphere combine red cube with green cube";graph="join red cube with green cube"
        g=d.execute("grammar",graph,rows,values,relations)["value"];self.assertIsNotNone(d.execute("grammar",numeric,rows,values,relations).get("value"))
        d.retract("word-a");self.assertIsNone(d.execute("grammar",numeric,rows,values,relations).get("value"));self.assertEqual(d.execute("grammar",graph,rows,values,relations)["value"],g)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"state";d.save(p);e=Development.load(p);self.assertIsNone(e.execute("grammar",numeric,rows,values,relations).get("value"));self.assertEqual(e.execute("grammar",graph,rows,values,relations)["value"],g)
            e.retract("word-d");self.assertIsNone(e.execute("grammar",graph,rows,values,relations).get("value"))

    def test_complete_curriculum_and_replay(self):
        r=trial(208000,True)
        self.assertTrue(r['replay']);self.assertTrue(r['quarantined']);self.assertTrue(r['dependency_blocked']);self.assertEqual(r['before']['word-c']['correct'],37);self.assertEqual(r['before']['graph']['correct'],8);self.assertEqual(r['before']['derived-domain']['correct'],37);self.assertEqual(r['before']['offered-domain']['correct'],37);self.assertEqual(r['offered']['action'],'registered')
        self.assertEqual(r['recovery']['correct'],49);self.assertTrue(all(x['prediction']==x['expected'] for x in r['language']+r['graph_language']))
        self.assertTrue(r['status_before']['target-base']['template']);self.assertTrue(r['status_before']['target-alt']['expanded'])

    def test_broader_check_regression(self):
        r=trial(209010,True);self.assertEqual(r["before"]["offered-domain"]["correct"],37);self.assertTrue(r["status_before"]["offered-domain"]["broad_check"]);self.assertTrue(r["replay"])

    def test_reuse_ablation(self):
        r=trial(208000,False);self.assertEqual(r['before']['word-c']['correct'],0);self.assertEqual(r['before']['word-c']['abstained'],37);self.assertTrue(r['replay'])

if __name__=='__main__':unittest.main()
