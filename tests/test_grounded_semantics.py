import json
from pathlib import Path
import tempfile
import unittest
from ix_zero.grounded_semantics import SemanticMemory, RecordedMeasurements, generate_probes, input_case, input_key, apply_model, fit
from ix_zero.semantic_experiment import Environment, LiveMeasurements, packet, plan_task, score
from ix_zero.semantic_transfer_experiment import ReversedEnvironment, ArityEnvironment
from ix_zero.research_workspace import ResearchWorkspace


def learn(m,e,k,**kw):
    tr,p,a,t=packet(e,k)
    return m.learn(tr,LiveMeasurements(e,p),LiveMeasurements(e,a),**kw),t


class GroundedTests(unittest.TestCase):
    def test_opaque_action_and_new_objects(self):
        e=Environment(351200);m=SemanticMemory();r,t=learn(m,e,0)
        self.assertTrue(r['promoted']);self.assertEqual(r['slot'],e.slots[0]);self.assertEqual(score(m,e,t)['correct'],32)
        self.assertEqual(r['model']['verb'],e.verbs[0]);self.assertTrue(r['audit'])
    def test_self_generated_probes_expose_hidden_condition(self):
        e=Environment(351201);tr,p,a,t=packet(e,2);g=e.fields[1]
        def equality(x):
            s,_,_,_,roles=input_case(x);return s[roles[0]][g]==s[roles[1]][g]
        self.assertFalse(any(equality(x) for x,_ in tr));self.assertTrue(any(equality(x) for x in p))
        m=SemanticMemory();r=m.learn(tr,LiveMeasurements(e,p),LiveMeasurements(e,a));self.assertTrue(r['promoted']);self.assertTrue(any(equality(row['input']) for row in r['queries']));self.assertEqual(score(m,e,t)['correct'],32)
        c=SemanticMemory();r=c.learn(tr,LiveMeasurements(e,p),LiveMeasurements(e,a),conditional=False);self.assertFalse(r['promoted'])
    def test_no_query_abstains_on_unresolved_ambiguity(self):
        e=Environment(351200);m=SemanticMemory();r,t=learn(m,e,2,query_budget=0)
        self.assertEqual(r['reason'],'ambiguous');self.assertEqual(score(m,e,t)['abstain'],32)
    def test_unsupported_arithmetic_abstains(self):
        e=Environment(351200);m=SemanticMemory();r,t=learn(m,e,4)
        self.assertFalse(r['promoted']);self.assertEqual(score(m,e,t)['abstain'],32)
    def test_actual_plan_and_empty_control(self):
        e=Environment(351200);m=SemanticMemory()
        for k in (0,1):self.assertTrue(learn(m,e,k)[0]['promoted'])
        result=plan_task(Environment(351200),m);self.assertTrue(result['success']);self.assertEqual(len(result['plan']['actions']),2)
        self.assertFalse(plan_task(Environment(351200),SemanticMemory())['success'])
        self.assertEqual(m.plan(result['start'],result['goal'],max_depth=0)['status'],'no_model_route')
    def test_audit_contradiction_rejects_candidate(self):
        e=Environment(351200);tr,p,a,_=packet(e,0);audit=[[x,e.observe(x)] for x in a];n=next(iter(audit[0][1]));f=e.fields[0];audit[0][1][n][f]+=1000
        r=fit(tr,LiveMeasurements(e,p),RecordedMeasurements(audit));self.assertFalse(r['promoted']);self.assertEqual(r['reason'],'audit_contradiction')
    def test_overlapping_partitions(self):
        e=Environment(351200);tr,p,a,_=packet(e,0)
        with self.assertRaises(ValueError):fit(tr,LiveMeasurements(e,p+[tr[0][0]]),LiveMeasurements(e,a))
    def test_nonlocal_effect_rejected(self):
        e=Environment(351200);tr,p,a,_=packet(e,0);n=list(tr[0][1])[-1];tr[0][1][n][e.fields[0]]+=7
        r=fit(tr,LiveMeasurements(e,p),LiveMeasurements(e,a));self.assertEqual(r['reason'],'nonlocal_or_schema_effect')
    def test_probe_generation_unique_unlabelled(self):
        e=Environment(351200);tr,p,a,_=packet(e,2);keys={input_key(input_case(x)) for x in p};self.assertEqual(len(keys),len(p));self.assertFalse(keys&{input_key(input_case(x)) for x,_ in tr});self.assertEqual(generate_probes(tr,limit=5),p[:5])
    def test_unknown_verb_and_schema_abstain(self):
        e=Environment(351200);m=SemanticMemory();r,t=learn(m,e,0);s,w=t[0];w=list(w);w[r['slot']]='unknown';self.assertIsNone(m.predict(s,w))
        s,w=t[1];s={n:{'different':0} for n in s};self.assertIsNone(m.predict(s,w))
    def test_invalid_typed_inputs(self):
        e=Environment(351200);s,w=e.case(0);s[next(iter(s))][e.fields[0]]=True
        with self.assertRaises(ValueError):input_case([s,w])
    def test_duplicate_lexeme_fails_before_observation(self):
        e=Environment(351200);m=SemanticMemory();learn(m,e,0);tr,p,a,_=packet(e,0);p=LiveMeasurements(e,p);a=LiveMeasurements(e,a)
        with self.assertRaises(ValueError):m.learn(tr,p,a)
        self.assertEqual(p.log,[]);self.assertEqual(a.log,[])


class TransferTests(unittest.TestCase):
    def test_analogical_mapping_and_withdrawal(self):
        e=Environment(361100);m=SemanticMemory();source,_=learn(m,e,0);self.assertTrue(source['promoted'])
        t=ReversedEnvironment(1361100);tr,p,a,h=packet(t,0);r=m.transfer(tr[:2],LiveMeasurements(t,p),LiveMeasurements(t,a));self.assertTrue(r['promoted']);self.assertEqual(r['source'],e.verbs[0]);self.assertEqual(score(m,t,h)['correct'],32)
        case=e.case(0);out=e.observe(case);out[next(iter(out))][e.fields[0]]+=1000
        rev=m.challenge(*case,out);self.assertEqual(rev['withdrawn'],sorted([e.verbs[0],t.verbs[0]]));self.assertIsNone(m.predict(*h[0]))
    def test_no_transfer_without_audited_source(self):
        e=ReversedEnvironment(1361100);tr,p,a,_=packet(e,0);m=SemanticMemory();r=m.transfer(tr[:2],LiveMeasurements(e,p),LiveMeasurements(e,a));self.assertEqual(r['reason'],'no_transfer_model')
    def test_variable_arity(self):
        e=ArityEnvironment(20361100);m=SemanticMemory()
        for k in range(2):
            inputs=[e.case(k) for _ in range(8)];tr=[[x,e.observe(x)] for x in inputs];p=generate_probes(tr);a=[e.case(k) for _ in range(12)];t=[e.case(k,size=6,spread=40) for _ in range(8)]
            r=m.learn(tr,LiveMeasurements(e,p),LiveMeasurements(e,a),query_budget=24);self.assertTrue(r['promoted']);self.assertEqual(score(m,e,t)['correct'],8)
    def test_workspace_dispatch_checkpoint_and_challenge(self):
        e=Environment(361100);tr,p,a,h=packet(e,0);w=ResearchWorkspace()
        rows=lambda xs:[[x,e.observe(x)] for x in xs]
        r=w.execute({'command':'task','train':tr,'pool':rows(p),'audit':rows(a)})
        self.assertEqual(r['chosen'],'semantic_inquire');self.assertTrue(r['result']['promoted'])
        t=ReversedEnvironment(1361100);tr,p,a,hh=packet(t,0)
        r=w.execute({'command':'semantic_transfer','train':tr[:2],'pool':[[x,t.observe(x)] for x in p],'audit':[[x,t.observe(x)] for x in a]});self.assertTrue(r['promoted'])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d,'checkpoint.json');w.save(f);m=ResearchWorkspace.load(f);self.assertEqual(m.execute({'command':'self_model'})['semantic_promotions'],2)
            s,words=hh[0];self.assertEqual(m.execute({'command':'semantic_predict','s':s,'words':words})['value'],t.observe(hh[0]));self.assertEqual(m.semantics.dependencies[t.verbs[0]],[e.verbs[0]])
            # Hash integrity and deterministic replay are distinct from authenticated provenance.
            o=json.loads(f.read_text());o['payload']['events']=[];f.write_text(json.dumps(o))
            with self.assertRaises(ValueError):ResearchWorkspace.load(f)

if __name__=='__main__':unittest.main()
