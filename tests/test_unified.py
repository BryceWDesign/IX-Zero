import json
import tempfile
import unittest
from pathlib import Path
from ix_zero.relational import WorldMemory, state, command
from ix_zero.unified import UnifiedMind
from ix_zero.unified_experiment import build_joint, symbolic_fixture, symbolic_truth, stage_truth, numeric_truth
from ix_zero.cli import Session


class UnifiedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=build_joint(100)

    def fixture(self,kind='local',mode='combined'):
        w=WorldMemory(mode);training,audits,held=symbolic_fixture(1,kind)
        for o in training:w.observe(*o)
        for o in audits:w.audit(*o)
        return w,held

    def test_untrained_abstention(self):
        self.assertIsNone(WorldMemory().predict([],('step','a')))

    def test_local_generalizes_external_distractor(self):
        w,held=self.fixture();self.assertTrue(all(w.predict(s,a)==t for s,a,t in held))

    def test_nonlocal_counter(self):
        w,held=self.fixture('nonlocal');self.assertTrue(all(w.predict(s,a)==t for s,a,t in held));self.assertTrue(w.models[('step',1)].local_blocked)

    def test_nonlocal_local_only_abstains(self):
        w,held=self.fixture('nonlocal','local');self.assertTrue(all(w.predict(s,a) is None for s,a,t in held))

    def test_duplicate_not_support_and_freeze(self):
        w=WorldMemory();s=[('ready','a')];a=('step','a');t=[('done','a')]
        self.assertTrue(w.observe(s,a,t));self.assertFalse(w.observe(s,a,t));self.assertIsNone(w.predict(s,a,False))
        with self.assertRaises(ValueError):w.audit(s,a,t)
        w.observe([('ready','b')],('step','b'),[('done','b')]);w.audit([('ready','c')],('step','c'),[('done','c')])
        with self.assertRaises(ValueError):w.observe([],('step','z'),[])

    def test_exact_conflict_quarantines(self):
        w=WorldMemory();s=[('ready','a')];a=('step','a');w.observe(s,a,[('done','a')]);w.observe(s,a,s)
        self.assertTrue(w.models[('step',1)].quarantined);self.assertIsNone(w.predict(s,a,False))

    def test_new_contradiction_blocks_audited_model(self):
        w,_=self.fixture();s=[('ready','new')];a=('step','new');self.assertIsNotNone(w.predict(s,a))
        self.assertFalse(w.audit(s,a,s));self.assertIsNone(w.predict(s,a));self.assertTrue(w.models[('step',1)].quarantined)

    def test_unseen_nonlocal_context_abstention(self):
        w,_=self.fixture('nonlocal');s=[('ready','new'),('other','unknown')];self.assertIsNone(w.predict(s,('step','new')))

    def test_inert_validation(self):
        with self.assertRaises(ValueError):command(('x','a','a'))
        with self.assertRaises(ValueError):state([('shell',';rm')])
        with self.assertRaises(ValueError):state([('x',str(i)) for i in range(65)])

    def test_joint_plan_wraparound(self):
        m=self.model;s=m.joint([('raw','z')],[0,0]);actions=[('prep','z'),('finish','z')]
        p=m.plan(s,actions,present=[('done','z')],vector=[0,0]);self.assertIsNotNone(p['path']);self.assertEqual(len(p['path']),10)
        f,v=s
        for a in p['path']:f=stage_truth(f,a);v=numeric_truth(v,a[0])
        self.assertEqual((f,v),(frozenset({('done','z')}),(0,0)))

    def test_joint_requires_numeric_binding_and_gate(self):
        m=UnifiedMind();training,audits,_=symbolic_fixture(1,'local')
        for o in training:m.apply('observe',*o)
        for o in audits:m.apply('audit',*o)
        s=m.joint([('ready','z')],[1,2]);self.assertIsNone(m.predict(s,('step','z')))
        m.discovery.apply('new','untrained',2,5,False,'strict');m.apply('bind',('step','z'),'untrained');self.assertIsNone(m.predict(s,('step','z')))

    def test_retraction_propagates_to_joint_plans(self):
        m=build_joint(101);m.discovery.apply('retract','prep');s=m.joint([('raw','z')],[0,0]);self.assertIsNone(m.predict(s,('prep','z')))
        self.assertIsNone(m.plan(s,[('prep','z'),('finish','z')],present=[('done','z')])['path'])

    def test_actual_sally_generated_goal_and_ix_policy(self):
        m=self.model;s=m.joint([('raw','z')],[0,0]);actions=[('prep','z'),('finish','z')];d=m.advance(s,actions)
        self.assertEqual(d['action'],'generate');self.assertTrue(d['result']['goal_id'].startswith('self-goal-'));self.assertTrue(d['trace'])
        g=d['result'];f,v=s
        for a in g['path']:f=stage_truth(f,a);v=numeric_truth(v,a[0])
        self.assertEqual((f,v),g['target'])
        d=m.advance(s,actions,goal={'present':[('done','z')],'vector':[1,1]});self.assertEqual(d['action'],'plan');self.assertIsNotNone(d['result']['path'])

    def test_ix_policy_cannot_force_unsupported_goal(self):
        m=UnifiedMind();s=m.joint([]);a=[('step','z')]
        src='agent Unified { on advance { reply "generate" } }'
        with self.assertRaises(ValueError):m.advance(s,a,source=src)
        self.assertEqual(m.advance(s,a)['action'],'exhausted');d=m.advance(s,a,candidates=[([],('step','z'))]);self.assertEqual(d['action'],'inquire')

    def test_meta_learning_changes_curriculum_on_failure(self):
        m=UnifiedMind();fit,audits,_=symbolic_fixture(1,'local')
        for o in fit:m.apply('observe',*o)
        # A hidden rule reverses behavior in a novel external context. The local
        # model is confidently wrong; counter abstains, earning higher utility.
        for i in range(3):
            a=('step',f'new{i}');s=[('ready',a[1]),('blocked','external')];m.apply('audit',s,a,s)
        self.assertEqual(m.strategy()['selected_strategy_id'],'counter');self.assertTrue(m.strategy()['changed_strategy'])
        q=m.query([([],('new_action','z'))]);self.assertEqual(q['strategy'],'counter')

    def test_procedure_requires_evidence_and_two_bindings(self):
        m=self.model;self.assertGreaterEqual(m.status()['procedures'],1)
        with self.assertRaises(ValueError):m.apply('remember',[([('raw','z')],('prep','z'),[('ready','z')]),([('ready','z')],('finish','z'),[('done','z')])])

    def test_analogy_grounded_in_source_audit(self):
        m=UnifiedMind()
        for i in range(4):
            x=f't{i}';m.apply('observe',[('ready',x)],('mark',x),[('marked',x)]);m.apply('observe',[],('mark',x),[])
        source=[('edge','a','b'),('edge','b','c'),('marked','b')];before=[('edge','a','b'),('edge','b','c'),('ready','b')]
        self.assertTrue(m.apply('audit',before,('mark','b'),source))
        for x in ('audit1','audit2'):self.assertTrue(m.apply('audit',[('ready',x)],('mark',x),[('marked',x)]))
        target=[('edge','r','s'),('edge','s','t'),('ready','s')]
        g=m.analogy_goal(source,target,'edge','marked','b');self.assertEqual(g,('marked','s'))
        self.assertEqual(m.plan(m.joint(target),[('mark','r'),('mark','s'),('mark','t')],present=[g])['path'],(('mark','s'),))
        with self.assertRaises(ValueError):m.analogy_goal(source,target,'edge','marked','a')
        with self.assertRaises(ValueError):m.analogy_goal(source,[('edge','a','b'),('edge','a','c')],'edge','marked','b')

    def test_goal_arbitration(self):
        m=self.model;s=m.joint([('raw','z')],[0,0]);a=[('prep','z'),('finish','z')]
        alt=[{'id':'done','goal':{'present':[('done','z')]},'utility':0.9},{'id':'ready','goal':{'present':[('ready','z')]},'utility':0.1}]
        self.assertEqual(m.choose_goal(s,a,alt)['selected'],'done')
        alt=[{'id':'one','goal':{'present':[('done','z')]}},{'id':'two','goal':{'present':[('done','z')]}}]
        self.assertIsNone(m.choose_goal(s,a,alt)['selected'])

    def test_checkpoint_replay_and_tampering(self):
        m=self.model
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'d.json';p=Path(td)/'u.json';m.discovery.save(d);m.save(p)
            from ix_zero.discovery import DiscoveryMind
            restored=UnifiedMind.load(p,DiscoveryMind.load(d));self.assertEqual(m.status(),restored.status());s=m.joint([('raw','z')],[0,0]);self.assertEqual(m.predict(s,('prep','z')),restored.predict(s,('prep','z')))
            obj=json.loads(p.read_text());obj['data']['events'].pop();p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):UnifiedMind.load(p,m.discovery)

    def test_symbolic_recovery_requires_new_evidence_and_audits(self):
        m=UnifiedMind();fit,audits,_=symbolic_fixture(1,'local')
        for o in fit:m.apply('observe',*o)
        for o in audits:m.apply('audit',*o)
        a=('step','shift');s=[('ready','shift'),('blocked','external')];self.assertFalse(m.apply('audit',s,a,s));self.assertEqual(m.advance(m.joint(s),[a])['action'],'revise')
        r=m.apply('recover',a);self.assertEqual(r['representation'],'contextual');self.assertIsNone(m.world.predict(s,a));self.assertEqual(r['epoch'],1)
        with self.assertRaises(ValueError):m.apply('audit',*audits[0])
        for i in range(2):
            a=('step',f'relearn{i}');s=[('ready',a[1]),('blocked','external')];m.apply('observe',s,a,s)
        for i in range(4):
            a=('step',f'val{i}');s=[('ready',a[1])]+([('blocked','external')] if i%2 else []);t=s if i%2 else [('done',a[1])];self.assertTrue(m.apply('audit',s,a,t))
        for i in range(4):
            a=('step',f'held{i}');s=[('ready',a[1]),('noise','other')]+([('blocked','external')] if i%2 else []);t=s if i%2 else [('done',a[1]),('noise','other')];self.assertEqual(m.world.predict(s,a),frozenset(t))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.json';m.save(p);restored=UnifiedMind.load(p);self.assertEqual(restored.status(),m.status());self.assertEqual(restored.world.predict(s,a),frozenset(t))

    def test_conflicting_grounded_observations_start_clean_epoch(self):
        m=UnifiedMind();a=('step','a');s=[('ready','a')];m.apply('observe',s,a,[('done','a')]);m.apply('observe',s,a,s)
        r=m.apply('recover',a);self.assertEqual(r['reused'],0);self.assertIsNone(m.world.predict(s,a));self.assertEqual(len(m.world.history[('step',1)]),1)

    def test_joint_cross_modal_contradiction_quarantines_binding(self):
        m=build_joint(102);a=('prep','z');s=m.joint([('raw','z')],[0,0]);self.assertIsNotNone(m.predict(s,a))
        self.assertFalse(m.apply('joint_audit',s[0],s[1],a,[('ready','z')],[2,0]))
        self.assertIsNone(m.predict(s,a));self.assertIsNotNone(m.predict(m.joint(s[0]),a))
        self.assertIn(('prep',1),m.joint_blocked)

    def test_joint_checkpoint_replays_historical_audits_after_retraction(self):
        from ix_zero.discovery import DiscoveryMind
        m=build_joint(103);m.discovery.apply('retract','prep')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'d.json';p=Path(td)/'u.json';m.discovery.save(d);m.save(p)
            restored=UnifiedMind.load(p,DiscoveryMind.load(d));self.assertEqual(m.status(),restored.status())
            self.assertIsNone(restored.predict(restored.joint([('raw','z')],[0,0]),('prep','z')))
            with self.assertRaises(ValueError):UnifiedMind.load(p,build_joint(104).discovery)

    def test_cli_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            s=Session(Path(td)/'base.json');self.assertEqual(s.handle('uobserve [[ ["ready","a"] ], ["step","a"], [["done","a"]]]'),'true');self.assertIn('saved',s.handle('usave'))
            s2=Session(Path(td)/'base.json');self.assertEqual(s2.unified.status()['events'],1)
            with self.assertRaises(ValueError):s2.handle('uobserve {}')

if __name__=='__main__':unittest.main()
