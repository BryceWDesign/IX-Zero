import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ix_zero.general import General, Evidence, Programs, math_answer, primitive, hash_of
from ix_zero.general_experiment import call, learn, program_chain, integrated


class GeneralTests(unittest.TestCase):
    def test_exact_arithmetic(self):
        self.assertEqual(math_answer('-(2/3)+7/9')['value'],'1/9')
        self.assertEqual(math_answer('3*(2*x-1)=x+4',True)['solution'],'7/5')
        self.assertEqual(math_answer('x=x',True)['solution'],'all rationals')
        self.assertEqual(math_answer('x=x+1',True)['solution'],'none')

    def test_no_arbitrary_execution(self):
        for expr in ('__import__("os")','(1).__class__','2**100','[1][0]','x*x','1/0','1000001'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):math_answer(expr)
        with self.assertRaises(ValueError):math_answer('x+1')

    def test_evidence_scope_and_proof(self):
        e=Evidence();e.ingest('one','a is inside b.');e.ingest('two','b is inside c.')
        proof=e.ask('a','inside','c')
        self.assertEqual([p['sources'] for p in proof['proof']],[['one'],['two']])
        self.assertEqual(e.ask('c','inside','a')['status'],'unknown')
        e.retract('two');self.assertEqual(e.ask('a','inside','c')['status'],'unknown')

    def test_conflict_and_nontransitive_relation(self):
        e=Evidence();e.ingest('one','a is inside b.\nb is inside c.\na is connected_to b.\nb is connected_to c.')
        e.ingest('two','b is not inside c.')
        self.assertEqual(e.ask('b','inside','c')['status'],'conflict')
        self.assertEqual(e.ask('a','inside','c')['status'],'unknown')
        self.assertEqual(e.ask('a','connected_to','c')['status'],'unknown')
        e.retract('two');e.ingest('three','a is not inside c.')
        self.assertEqual(e.ask('a','inside','c')['status'],'conflict')

    def test_bad_document_is_atomic(self):
        e=Evidence()
        with self.assertRaises(ValueError):e.ingest('one','a is inside b.\nUnknown text.')
        self.assertEqual(e.sources,{})

    def test_unknown_language_and_condition(self):
        m=General();self.assertEqual(m.ask('tell me about mars')['status'],'unsupported')
        self.assertEqual(m.ask('if a is inside b then calculate 1+2')['status'],'unknown')
        call(m,'ingest',source='s',text='a is inside b.')
        self.assertEqual(m.ask('if a is inside b then calculate 1+2')['result']['value'],'3')
        self.assertEqual(m.ask('if a is inside b then if a is inside b then calculate 1')['status'],'unsupported')

    def test_primitive_types_and_bounds(self):
        self.assertEqual(primitive('unique',(2,1,2)),(2,1))
        with self.assertRaises(ValueError):primitive('square',(10000,))
        with self.assertRaises(ValueError):primitive('not_defined',(1,))

    def test_partition_validation(self):
        p=Programs()
        with self.assertRaises(ValueError):p.begin('a',[[1],[2],[3],[4]],[[1]],[[1],[2]])
        with self.assertRaises(ValueError):p.begin('reverse',[[1],[2],[3],[4]],[[1]],[[2],[3]])
        with self.assertRaises(ValueError):p.begin('a',[[True],[2],[3],[4]],[[True]],[[2],[3]])

    def test_unresolved_evidence_abstains(self):
        p=Programs();p.begin('a',[[],[0],[1],[2]], [[]], [[1],[2]])
        q=p.next();p.answer([]);q=p.next()
        self.assertEqual(q['action'],'failed');self.assertIsNone(p.predict('a',[0]))

    def test_denied_measurement_releases_inquiry(self):
        m=General();scope=[[1],[2],[3],[4],[5]]
        call(m,'begin',key='a',scope=scope,train=scope[:2],audit=scope[2:4]);call(m,'next')
        self.assertTrue(call(m,'deny')['denied']);self.assertEqual(call(m,'next')['action'],'failed')
        self.assertIsNone(m.programs.predict('a',[1]))
        call(m,'begin',key='b',scope=scope,train=scope[:2],audit=scope[2:4])

    def test_failed_audit_blocks_promotion(self):
        m=General();r=learn(m,'bad',910000,lambda x:x[::-1],corrupt=True)
        self.assertEqual(r['terminal']['action'],'failed');self.assertNotIn('bad',m.programs.skills)
        self.assertEqual(r['wrong'],0)

    def test_scope_and_plan(self):
        m=General();learn(m,'inc',910000,lambda x:tuple(v+1 for v in x))
        self.assertEqual(m.ask('apply inc to [2]')['value'],(3,))
        self.assertEqual(m.ask('apply inc to [999]')['status'],'unknown')
        self.assertEqual(m.ask('apply inc to result of calculate 2*3')['value'],(7,))
        self.assertEqual(len(m.ask('plan from [0] to [4]')['steps']),4)
        self.assertEqual(m.ask('plan from [4] to [0]')['status'],'unknown')
        self.assertEqual(m.ask('plan from [0] to [0]')['steps'],[])

    def test_program_chain_matched_control(self):
        full=program_chain(910000,True);control=program_chain(910000,False)
        self.assertEqual(full['rows'][-1]['correct'],36)
        self.assertEqual(full['invalidation'],'invalid')
        self.assertEqual(control['rows'][-1]['abstained'],36)
        self.assertTrue(full['replay'] and control['replay'])

    def test_core_bridge_and_invalidation(self):
        r=integrated(910000);self.assertTrue(all(z['correct'] for z in r['rows']))
        self.assertTrue(r['replay']);self.assertEqual(r['after']['status'],'unknown')
        self.assertTrue(all(z['correct'] for z in r['plans']))

    def test_pending_checkpoint_and_tamper(self):
        m=General();call(m,'begin',key='k',scope=[[1],[2],[3],[4],[5]],train=[[1],[2]],audit=[[3],[4]])
        q=call(m,'next')
        with TemporaryDirectory() as d:
            p=Path(d)/'x.json';m.save(p);r=General.load(p)
            self.assertEqual(call(r,'next'),q)
            obj=json.loads(p.read_bytes());obj['payload']['fingerprint']='0'*64
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):General.load(p)
            obj['sha256']=hash_of(obj['payload']);p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):General.load(p)

    def test_rejected_core_commands(self):
        m=General()
        with self.assertRaises(ValueError):call(m,'core',request={'command':'new'})
        self.assertEqual(m.events,[])

    def test_constraint_search_and_budget(self):
        m=General();domains={'a':[1,2,3],'b':[1,2,3],'c':[1,2,3]}
        rules=[{'op':'distinct','vars':['a','b','c']},{'op':'lt','vars':['a','b']},{'op':'lt','vars':['b','c']}]
        r=call(m,'solve_constraints',domains=domains,constraints=rules)
        self.assertEqual(r['status'],'unique');self.assertEqual(r['solutions'],[{'a':1,'b':2,'c':3}])
        self.assertEqual(call(m,'solve_constraints',domains=domains,constraints=[],max_nodes=1)['status'],'budget')
        self.assertEqual(call(m,'solve_constraints',domains=domains,constraints=[])['status'],'ambiguous')
        rules.append({'op':'lt','vars':['c','a']})
        self.assertEqual(call(m,'solve_constraints',domains=domains,constraints=rules)['status'],'unsatisfiable')
        with self.assertRaises(ValueError):call(m,'solve_constraints',domains=domains,constraints=[{'op':'eval','vars':['a']}])

    def test_no_mutation_through_request(self):
        m=General();req={'command':'ingest','source':'s','text':'a is inside b.'};m.execute(req)
        req['text']='changed';self.assertEqual(m.events[0]['text'],'a is inside b.')


if __name__=='__main__':unittest.main()
