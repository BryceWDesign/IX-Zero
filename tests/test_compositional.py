import unittest
from ix_zero.compositional import Grammar, tokenize, tree, MAX_DEPTH
from ix_zero.compositional_demo import build
from ix_zero.compositional_experiment import run


class RecursiveLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g,cls.d,cls.rows,cls.values=build()

    def answer(self,t,rows=None):
        return self.g.execute(t,rows or self.rows,self.values,self.d)

    def test_longer_sentence(self):
        self.assertEqual(self.answer('please compare the red sphere with the green cube')['value'],1)

    def test_infix(self):
        self.assertEqual(self.answer('red sphere versus green cube')['value'],1)

    def test_nested(self):
        r=self.answer('compare inspect red sphere rule green cube blue cone')
        self.assertEqual(r['value'],0)
        self.assertEqual(r['meaning'],('compare',('inspect','C'),('rule','B','D')))
        self.assertEqual(len(r['trace']),3)
        self.assertIn('operator:word7',r['trace'][1]['evidence']['programs'][0]['origins'])

    def test_mixed_signature(self):
        self.assertEqual(self.answer('compare inspect red sphere green cube')['value'],1)
        self.assertEqual(self.answer('compare green cube inspect red sphere')['value'],-1)

    def test_nested_operator(self):
        self.assertEqual(self.answer('combine inspect red sphere with rule green cube blue cone')['value'],0)

    def test_unknown_and_ambiguous(self):
        self.assertTrue(self.answer('inspect orange cube')['abstained'])
        self.assertTrue(self.answer('inspect red cube',self.rows+[['E','cube','red']])['abstained'])
        self.assertTrue(self.answer('red cube versus green cube versus red sphere')['abstained'])

    def test_unlearned_signature(self):
        self.assertTrue(self.answer('inspect inspect red cube')['abstained'])

    def test_budget(self):
        with self.assertRaises(ValueError):tokenize('red '*65)
        y='A'
        for _ in range(MAX_DEPTH+1):y=('inspect',y)
        with self.assertRaises(ValueError):tree(y,{'A'})
        self.assertTrue(self.answer('inspect '*10+'red cube')['abstained'])

    def test_no_execution_before_audit(self):
        g=Grammar(self.g.source)
        g.observe('look red cube',self.rows,('inspect','A'))
        g.observe('look green cube',self.rows,('inspect','B'))
        self.assertIsNone(g.predict('look red sphere',self.rows))
        self.assertTrue(g.audit('look red sphere',self.rows,('inspect','C')))
        with self.assertRaises(ValueError):g.observe('look blue cone',self.rows,('inspect','D'))
        with self.assertRaises(ValueError):g.audit('look red sphere',self.rows,('inspect','C'))
        self.assertIsNone(g.predict('look blue cone',self.rows))

    def test_conflicting_training_transaction(self):
        g=Grammar(self.g.source)
        g.observe('seek red cube',self.rows,('inspect','A'))
        with self.assertRaises(ValueError):g.observe('seek green cube',self.rows,('other','B'))
        self.assertEqual(len(g.rules),1)
        self.assertEqual(len(g.training),1)

    def test_relabel_does_not_manufacture_audit(self):
        g=Grammar(self.g.source)
        g.observe('look red cube',self.rows,('inspect','A'))
        g.observe('look green cube',self.rows,('inspect','B'))
        renamed=[['X'+i,k,a] for i,k,a in self.rows]
        with self.assertRaises(ValueError):g.audit('look red cube',renamed,('inspect','XA'))

    def test_source_invalidation(self):
        g,d,rows,values=build();g.source.blocked=True
        self.assertTrue(g.execute('inspect red cube',rows,values,d)['abstained'])

    def test_dispatch_invalidation(self):
        g,d,rows,values=build();d.bindings['rule',('object','object')].blocked=True
        self.assertTrue(g.execute('compare inspect red sphere rule green cube blue cone',rows,values,d)['abstained'])

    def test_console_integration(self):
        import json,tempfile
        from pathlib import Path
        from ix_zero.cli import Session
        with tempfile.TemporaryDirectory() as tmp:
            s=Session(Path(tmp)/'state.json');s.language.episodes['demo']=self.g.source;s.composition=self.d
            self.assertIn('created',s.handle('rnew ["demo"]'))
            rows=[*self.rows,['E','sphere','blue']]
            def cmd(name,*args):return s.handle(name+' '+json.dumps(args))
            for word,i in [('red cube','A'),('green cube','B')]:cmd('robserve','look '+word,rows,('inspect',i))
            for word,i in [('red sphere','C'),('blue cone','D'),('blue sphere','E')]:self.assertTrue(json.loads(cmd('raudit','look '+word,rows,('inspect',i)))['passed'])
            self.assertEqual(json.loads(cmd('rexecute','look red sphere',rows,self.values))['value'],2)
            self.assertTrue(json.loads(s.handle('rstatus'))['live'])

    def test_shift_and_random_grammar(self):
        r=run(1,180070)
        self.assertEqual(r['counts']['nested']['correct'],96)
        self.assertEqual(r['negative']['quarantined'],1)


if __name__=='__main__':unittest.main()
