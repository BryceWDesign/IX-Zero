import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ix_zero.acce import Workspace,CognitiveDispatch,Capability,Object
from ix_zero.acce_demo import build,objects,teach,reply
from ix_zero.cli import Session


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm,cls.om,cls.um,cls.am,cls.d,cls.rows,cls.values=build()

    def test_comparison_regression(self):
        p=reply('compare red sphere green cube',self.lm,self.d,self.rows,self.values)
        self.assertEqual(p['value'],1)
        self.assertEqual(p['evidence']['programs'][0]['name'],'ordering(value,value)')

    def test_renamed_operator(self):
        p=reply('rule blue cone green cube',self.lm,self.d,self.rows,self.values)
        self.assertEqual(p['value'],2)
        self.assertIn('operator:word7',p['evidence']['programs'][0]['origins'])

    def test_inspect_zero_is_not_abstention(self):
        self.assertEqual(reply('inspect red cube',self.lm,self.d,self.rows,self.values)['value'],0)

    def test_unrecognized_language(self):
        for text in ('inspect orange sphere','What would you ask me about the world?','please compare the red sphere with the green cube'):
            self.assertTrue(reply(text,self.lm,self.d,self.rows,self.values)['abstained'])

    def test_scalar_and_relational(self):
        self.assertEqual(self.d.execute('measure',objects(-7,4))['value'],-28)
        self.assertEqual(self.d.execute('combine',objects(0,3,True))['value'],'s3')

    def test_invalid_missing_projection(self):
        self.assertTrue(self.d.execute('compare',(Object('x','cube','red'),Object('y','cone','green',2)))['abstained'])
        with self.assertRaises(ValueError):self.d.execute('compare',(Object('x','cube','red',True),Object('y','cone','green',2)))

    def test_audit_required_and_no_duplicate(self):
        b=CognitiveDispatch(Workspace().primitives()).new('x',('integer','integer'),'integer')
        b.observe((0,3),-1);b.observe((3,0),1)
        self.assertIsNone(b.predict((2,1)))
        with self.assertRaises(ValueError):b.audit((0,3),-1)
        self.assertTrue(b.audit((1,1),0));self.assertIsNone(b.predict((2,1)))
        self.assertTrue(b.audit((1,3),-1));self.assertTrue(b.audit((2,0),1));self.assertEqual(b.predict((2,1)),1)
        with self.assertRaises(ValueError):b.observe((4,0),1)

    def test_label_rename_cannot_reuse_evidence(self):
        b=CognitiveDispatch(Workspace().primitives()).new('x',('object',),'integer')
        b.observe((Object('first','cube','red',1),),1)
        with self.assertRaises(ValueError):b.observe((Object('second','cone','blue',1),),1)

    def test_ambiguous_programs_abstain(self):
        b=CognitiveDispatch(Workspace().primitives()).new('x',('integer','integer'),'integer')
        for a in (1,2):b.observe((a,0),a)
        for a in (3,4,5):self.assertTrue(b.audit((a,0),a))
        self.assertIsNone(b.predict((2,3)))

    def test_contradiction_blocks_binding(self):
        b=CognitiveDispatch(Workspace().primitives()).new('x',('integer','integer'),'integer')
        teach(b,[((0,3),-1),((3,0),1)],[((1,1),0),((1,3),-1),((2,0),1)])
        self.assertFalse(b.audit((2,1),0));self.assertIsNone(b.predict((3,2)));self.assertEqual(b.advance()['action'],'revise')

    def test_source_retraction_blocks_numeric_and_relational(self):
        _,om,_,am,d,_,_=build()
        om.apply('retract','word7')
        self.assertTrue(d.execute('rule',objects(3,1))['abstained']);self.assertTrue(d.execute('combine',objects(0,3,True))['abstained'])
        self.assertEqual(d.execute('compare',objects(3,1))['value'],1)
        self.assertIn('structure9',am.retracted)

    def test_source_replacement_blocks(self):
        _,om,_,_,d,_,_=build();om.episodes['word7']=object()
        self.assertTrue(d.execute('rule',objects(3,1))['abstained'])

    def test_catalog_freezes(self):
        ws=Workspace().primitives();CognitiveDispatch(ws).new('a',('integer','integer'),'integer')
        with self.assertRaises(ValueError):ws.register(Capability('late',('integer',),'integer',lambda a:a,lambda:True,'host'))

    def test_policy_cannot_complete_unvalidated(self):
        b=CognitiveDispatch(Workspace().primitives()).new('x',('integer','integer'),'integer')
        class BadResult:
            approvals_required=[];replies=['complete']
        with patch('ix_zero.acce.run_ix',return_value=BadResult()):
            with self.assertRaises(ValueError):b.advance([(1,2)])

    def test_cli_live_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            s=Session(Path(td)/'state.json')
            s.handle('cworkspace');s.handle('cnew ["unknown_alias",["integer","integer"],"integer"]')
            for a,b,y in [(0,3,-1),(3,0,1)]:s.handle('cobserve '+json.dumps(['unknown_alias',['integer','integer'],[a,b],y]))
            for a,b,y in [(1,1,0),(1,3,-1),(2,0,1)]:self.assertTrue(json.loads(s.handle('caudit '+json.dumps(['unknown_alias',['integer','integer'],[a,b],y])))['passed'])
            self.assertEqual(json.loads(s.handle('cdispatch ["unknown_alias",[3,2]]'))['value'],1)

if __name__=='__main__':unittest.main()
