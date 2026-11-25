import json
from pathlib import Path
import random
import tempfile
import unittest
from ix_zero.analogical import AnalogicalMind, Graph
from ix_zero.analogy_experiment import fixtures, run_case, run_autonomous
from ix_zero.operators import OperatorMind
from ix_zero.cli import Session


def setup(family='rotation_phase',seed=88000):
    s,rs,labels,t,rt,targets=fixtures(family,seed)
    m=AnalogicalMind();m.apply('source','source',s,rs,labels);m.apply('new','target',t,rt,'structural')
    return m,targets


def validate(m,targets,name='target'):
    for x,y in targets[:2]:m.apply('observe',name,x,y)
    for x,y in targets[2:5]:
        if not m.apply('audit',name,x,y):raise AssertionError('fixture contradicted')


class AnalogicalTest(unittest.TestCase):
    def test_four_independent_structures(self):
        for f in ('rotation_phase','resource_signal','switch_membership','permutation_tiles'):
            r=run_case(f,88000,'structural')
            self.assertGreater(r['correct'],0);self.assertEqual(r['incorrect'],0)
            self.assertEqual(r['abstained'],0)

    def test_unstructured_ablation(self):
        for mode in ('structural','unstructured','local'):
            r=run_case('rotation_phase',88000,mode)
            self.assertEqual(r['correct'],44 if mode=='structural' else 0)

    def test_relation_names_and_orders_ignored(self):
        m,targets=setup();t=m.episodes['target'];self.assertEqual(t.mappings,7)
        self.assertIsNone(t.predict(targets[0][0]));validate(m,targets)
        self.assertTrue(all(t.predict(x)==y for x,y in targets[5:]))

    def test_audits_are_not_training_and_are_fresh(self):
        m,targets=setup();t=m.episodes['target']
        with self.assertRaises(ValueError):m.apply('audit','target',*targets[2])
        for x,y in targets[:2]:m.apply('observe','target',x,y)
        before=dict(t.models)
        m.apply('audit','target',*targets[2]);self.assertEqual(t.models,before)
        self.assertIsNone(t.predict(targets[5][0]))
        with self.assertRaises(ValueError):m.apply('audit','target',*targets[2])
        with self.assertRaises(ValueError):m.apply('observe','target',*targets[3])

    def test_mismatched_structure(self):
        m,_=setup('wrong_graph');self.assertEqual(m.episodes['target'].initial_models,0)
        self.assertEqual(m.advance('target')['action'],'exhausted')

    def test_wrong_operation_not_assumed(self):
        r=run_case('wrong_operation',88000,'structural');self.assertEqual(r['correct'],0);self.assertEqual(r['abstained'],44)

    def test_ambiguous_predictions_abstain(self):
        m,targets=setup();t=m.episodes['target'];self.assertGreater(len(t.models),1)
        self.assertIsNone(t._prediction(t.graph.pair(targets[0][0])))
        with self.assertRaises(ValueError):m.apply('promote','target')

    def test_policy_driven_acquisition(self):
        for f in ('rotation_phase','resource_signal','switch_membership','permutation_tiles'):
            r=run_autonomous(f,98000);self.assertTrue(r['promoted']);self.assertTrue(all(y==p for _,y,p in r['predictions']))
            self.assertTrue(all(s['trace'] for s in r['steps']))

    def test_promotion_and_multilevel_quarantine(self):
        m,targets=setup();validate(m,targets);self.assertEqual(m.advance('target')['action'],'abstract')
        s,rs,labels,t,rt,other=fixtures('rotation_phase',88002)
        m.apply('new','third',t,rt,'structural');validate(m,other,'third');m.apply('promote','third')
        self.assertTrue(all(m.episodes['third'].predict(x)==y for x,y in other[5:]))
        m.apply('retract','source');self.assertTrue(m.episodes['target'].blocked);self.assertTrue(m.episodes['third'].blocked)
        m.apply('new','fresh',t,rt,'structural');self.assertEqual(m.episodes['fresh'].initial_models,0)

    def test_contradiction_quarantines_promoted_source(self):
        m,targets=setup();validate(m,targets);m.apply('promote','target')
        x,y=targets[5];bad=next(k for k in m.episodes['target'].graph.states if k!=y)
        self.assertFalse(m.apply('audit','target',x,bad));self.assertIsNone(m.episodes['target'].predict(x));self.assertIn('target',m.retracted)

    def test_checkpoint_replays_and_detects_corruption(self):
        m,targets=setup();validate(m,targets);m.apply('promote','target');m.apply('retract','source')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';m.save(p);r=AnalogicalMind.load(p);self.assertEqual(r.events,m.events);self.assertEqual(r.retracted,m.retracted)
            obj=json.loads(p.read_text());obj['data']['events'].pop();p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):AnalogicalMind.load(p)

    def test_operator_bridge_and_invalidation(self):
        om=OperatorMind();om.apply('new','rule',2,5,False);t=om.episodes['rule'];xs=list(t.domain.inputs);random.Random(4).shuffle(xs)
        for x in xs[:10]:om.apply('observe','rule',x,x[0]^x[1])
        for x in xs[10:13]:self.assertTrue(om.apply('audit','rule',x,x[0]^x[1]))
        om.apply('promote','rule')
        m=AnalogicalMind();s=['offoff','onoff','offon','onon'];r={'flipA':[s[i^1] for i in range(4)],'flipB':[s[i^2] for i in range(4)]}
        m.apply('operator_source','bit_source',s,r,om.events,'rule');_,_,_,t,rt,targets=fixtures('switch_membership',88000)
        m.apply('new','target',t,rt,'structural');validate(m,targets);self.assertTrue(all(m.episodes['target'].predict(x)==y for x,y in targets[5:]))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';m.save(p);loaded=AnalogicalMind.load(p);loaded.reconcile_operators(om)
            self.assertFalse(loaded.episodes['target'].blocked)
            om.apply('retract','rule');loaded.reconcile_operators(om);self.assertTrue(loaded.episodes['target'].blocked)

    def test_source_and_graph_validation(self):
        for states,rs in [(['a','a','b'],{'r':['a','b','a']}),(['a','b','c'],{'r':['a','b','d']}),(['a','b','c'],{}),(['a','b'],{'r':['a','b']})]:
            with self.assertRaises(ValueError):Graph.read(states,rs)
        s,rs,labels,*_=fixtures('rotation_phase',88000);m=AnalogicalMind()
        with self.assertRaises(ValueError):m.apply('source','source',s,rs,labels[:-1])
        with self.assertRaises(ValueError):m.apply('source','source',s,rs,labels[:-1]+labels[:1])
        self.assertFalse(m.sources)

    def test_competing_sources_selected_by_target_evidence(self):
        s,rs,labels,t,rt,targets=fixtures('rotation_phase',88000)
        m=AnalogicalMind();m.apply('source','addition',s,rs,labels)
        # Constant operation is structurally plausible but inconsistent with
        # target evidence. The learner is not told which source is appropriate.
        m.apply('source','constant',s,rs,[[x,s[0]] for x,_ in labels]);m.apply('new','target',t,rt,'structural')
        e=m.episodes['target'];self.assertEqual(set(e.dependencies),{'addition','constant'})
        for x,y in targets[:3]:m.apply('observe','target',x,y)
        self.assertEqual(e.status()['sources'],['addition'])
        for x,y in targets[3:6]:self.assertTrue(m.apply('audit','target',x,y))
        self.assertTrue(all(e.predict(x)==y for x,y in targets[6:]))

    def test_hidden_exception_feedback(self):
        rows=[run_case('hidden_exception',88000+i,'structural') for i in range(4)]
        failures=[r for r in rows if r['incorrect']]
        self.assertTrue(failures)
        for r in failures:
            self.assertTrue(r['feedback']['quarantined']);self.assertEqual(r['feedback']['remaining_predictions'],0)

    def test_console(self):
        s,rs,labels,t,rt,targets=fixtures('rotation_phase',88000)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';a=Session(p)
            def cmd(c,args):return json.loads(a.handle(c+' '+json.dumps(args)))
            cmd('asource',['source',s,rs,labels]);cmd('anew',['target',t,rt,'structural']);self.assertEqual(cmd('aadvance',['target'])['action'],'inquire')
            for x,y in targets[:2]:cmd('aobserve',['target',x,y])
            for x,y in targets[2:5]:cmd('aaudit',['target',x,y])
            self.assertEqual(cmd('apredict',['target',targets[5][0]]),targets[5][1]);a.handle('asave');self.assertIn('target',Session(p).analogy.episodes)

if __name__=='__main__':unittest.main()
