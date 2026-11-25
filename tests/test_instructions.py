import tempfile
import unittest
from pathlib import Path
from ix_zero.core import Mind, Space
from ix_zero.instructions import advance
from ix_zero.cli import Session
from ix_zero.experiment import teacher


def source(action):
    return f'agent Zero {{\n on advance {{\n reply "{action}"\n }}\n}}'


class InstructionTests(unittest.TestCase):
    def setUp(self):
        self.m = Mind()
        self.t = self.m.new('puzzle', Space(2,5))

    def fit(self):
        oracle = lambda x: teacher(('rev','inc'),x,5)
        for _ in range(22):
            d = advance(self.m,'puzzle')
            if d.action != 'inquire':
                self.assertEqual(d.action,'audit')
                return oracle
            x = self.t.query()
            self.t.observe(x,oracle(x))
        self.fail('did not reach audit stage')

    def test_actual_ix_trace(self):
        d = advance(self.m,'puzzle')
        self.assertEqual(d.action,'inquire')
        self.assertIn('branch.evaluate',[e['kind'] for e in d.trace])
        self.assertEqual(d.trace[-1]['kind'],'run.complete')
        self.assertEqual(len(d.policy_sha256),64)
        self.assertEqual(len(self.t.observations),0)

    def test_full_cycle(self):
        oracle = self.fit()
        for n in range(3):
            self.assertEqual(advance(self.m,'puzzle').action,'audit')
            self.assertEqual(len(self.t.audits),n)
            x = next(x for x in self.t.space.states if x not in self.t.observations and x not in self.t.audits)
            self.assertTrue(self.t.audit(x,oracle(x)))
        self.assertEqual(advance(self.m,'puzzle').action,'promote')
        self.assertIn('puzzle',self.m.concepts)
        self.assertEqual(advance(self.m,'puzzle').action,'complete')
        for x in self.t.space.states:
            self.assertEqual(self.t.validated_predict(x),oracle(x))

    def test_failed_audit_routes_to_revision(self):
        oracle = self.fit()
        x = next(x for x in self.t.space.states if x not in self.t.observations)
        y = oracle(x)
        self.assertFalse(self.t.audit(x,((y[0]+1)%5,y[1])))
        self.assertEqual(advance(self.m,'puzzle').action,'revise')
        self.assertNotIn('puzzle',self.m.concepts)

    def test_policy_cannot_promote_unverified(self):
        with self.assertRaisesRegex(ValueError,'host evidence checks'):
            advance(self.m,'puzzle',source=source('promote'))
        self.assertFalse(self.m.concepts)

    def test_policy_unknown_action(self):
        with self.assertRaises(ValueError): advance(self.m,'puzzle',source=source('invent'))

    def test_policy_malformed(self):
        with self.assertRaises(ValueError): advance(self.m,'puzzle',source='agent {')

    def test_policy_multiple_actions(self):
        with self.assertRaises(ValueError):
            advance(self.m,'puzzle',source='agent Zero {\n on advance {\n reply "inquire"\n reply "inquire"\n }\n}')

    def test_policy_approval_not_self_granted(self):
        with self.assertRaises(ValueError):
            advance(self.m,'puzzle',source='agent Zero {\n on advance {\n require human_approval reason "review"\n reply "inquire"\n }\n}')

    def test_policy_edit_changes_trace(self):
        a=advance(self.m,'puzzle')
        b=advance(self.m,'puzzle',source=source('inquire'))
        self.assertNotEqual(a.policy_sha256,b.policy_sha256)
        self.assertNotEqual(a.trace,b.trace)
        self.assertEqual(a.action,b.action)

    def test_policy_edit_changes_action_and_memory(self):
        oracle = self.fit()
        for _ in range(3):
            x = next(x for x in self.t.space.states if x not in self.t.observations and x not in self.t.audits)
            self.assertTrue(self.t.audit(x,oracle(x)))
        self.assertEqual(advance(self.m,'puzzle',source=source('complete')).action,'complete')
        self.assertNotIn('puzzle',self.m.concepts)
        self.assertEqual(advance(self.m,'puzzle').action,'promote')
        self.assertIn('puzzle',self.m.concepts)

    def test_small_space_reserves_audits(self):
        self.m.new('tiny',Space(2,2))
        t=self.m.tasks['tiny']; t.observe((0,0),(0,0))
        self.assertEqual(advance(self.m,'tiny').action,'exhausted')
        self.assertFalse(t.audits)

    def test_console_trace(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session(Path(d)/'state.json')
            s.handle('new test 2 5')
            self.assertIn('Request observation',s.handle('advance'))
            self.assertIn('branch.evaluate',s.handle('policytrace'))
