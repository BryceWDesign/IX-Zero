import json, tempfile, unittest
from pathlib import Path
from ix_zero.language import LanguageMind, LanguageEpisode, scene
from ix_zero.language_experiment import fixtures,run_case,auto_case,bridge_case
from ix_zero.cli import Session
from ix_zero.unified import UnifiedMind


def fitted(seed=99000,english=True,validate=True):
    f=fixtures(seed,english);m=LanguageMind();m.apply('new','lang')
    for x in f['train']:m.apply('observe','lang',*x)
    if validate:
        for x in f['audit']:assert m.apply('audit','lang',*x)
    return m,f

class GroundedLanguageTests(unittest.TestCase):
    def test_no_preloaded_lexicon(self):
        ep=LanguageEpisode();self.assertEqual(len(ep.models),720)
        self.assertIsNone(ep.predict('inspect red cube',[['o','cube','red']],False))
        self.assertEqual(ep.status()['lexicon'],{})

    def test_unseen_words_and_grammar(self):
        for seed in (99000,99001,99002):
            m,f=fitted(seed,False);ep=m.episodes['lang'];self.assertEqual(len(ep.models),1)
            self.assertEqual(ep.status()['lexicon'],{w:tuple(v) for w,v in f['vocab'].items()})
            self.assertEqual(ep.models[0].orders,tuple(tuple(f['order'][n]) for n in (3,5)))
            self.assertTrue(all(ep.predict(t,s)==tuple(y) for t,s,y in f['test']))

    def test_unseen_compositions_english_and_new_object_ids(self):
        m,f=fitted();ep=m.episodes['lang'];self.assertTrue(all(ep.predict(t,s)==tuple(y) for t,s,y in f['test']))
        self.assertEqual(ep.predict('inspect red cube',[['novel','cube','red']]),('inspect','novel'))
        self.assertEqual(ep.predict('compare red cube green sphere',[['x','sphere','green'],['y','cube','red']]),('compare','y','x'))

    def test_argument_order_matters(self):
        m,_=fitted();ep=m.episodes['lang'];s=[['a','cube','red'],['b','sphere','green']]
        self.assertEqual(ep.predict('compare red cube green sphere',s),('compare','a','b'))
        self.assertEqual(ep.predict('compare green sphere red cube',s),('compare','b','a'))

    def test_synonym_learned_from_grounded_demonstration(self):
        m,f=fitted(validate=False);ep=m.episodes['lang']
        m.apply('observe','lang','examine red cube',[['syn','cube','red']],['inspect','syn'])
        for x in f['audit']:m.apply('audit','lang',*x)
        self.assertEqual(ep.predict('examine blue sphere',[['new','sphere','blue']]),('inspect','new'))
        self.assertIsNone(ep.predict('observe blue sphere',[['new','sphere','blue']]))

    def test_ambiguity_and_missing_referent(self):
        m,_=fitted();ep=m.episodes['lang']
        self.assertIsNone(ep.predict('inspect red cube',[['a','cube','red'],['b','cube','red']]))
        self.assertIsNone(ep.predict('inspect red cube',[['a','cube','blue']]))
        self.assertIsNone(ep.predict('inspect yellow cube',[['a','cube','red']]))
        with self.assertRaises(ValueError):ep.predict('what is the weather today please',[['a','cube','red']])

    def test_case_punctuation_and_scene_order(self):
        m,_=fitted();ep=m.episodes['lang'];s=[['a','cube','red'],['b','sphere','green']]
        self.assertEqual(ep.predict('  COMPARE red cube green sphere? ',s),ep.predict('compare red cube green sphere',s[::-1]))

    def test_audit_per_arity_and_freeze(self):
        m,f=fitted(validate=False);ep=m.episodes['lang'];a=[r for r in f['audit'] if len(r[0].split())==3];b=[r for r in f['test'] if len(r[0].split())==5][0]
        before=ep.status()['lexicon'];self.assertIsNone(ep.predict(*a[0][:2]))
        for r in a:m.apply('audit','lang',*r)
        self.assertEqual(ep.status()['lexicon'],before);self.assertIsNone(ep.predict(*b[:2]))
        with self.assertRaises(ValueError):m.apply('observe','lang',*b)
        with self.assertRaises(ValueError):m.apply('audit','lang',a[0][0],a[0][1][::-1],a[0][2])

    def test_audits_not_label_fit(self):
        m,f=fitted(validate=False);ep=m.episodes['lang'];before=[(h.orders,dict(h.lexicon)) for h in ep.models]
        t,s,y=f['audit'][0];bad=['wrong',*y[1:]]
        self.assertFalse(m.apply('audit','lang',t,s,bad));self.assertTrue(ep.blocked)
        self.assertEqual(before,[(h.orders,h.lexicon) for h in ep.models]);self.assertIsNone(ep.predict(t,s,False))

    def test_training_contradiction(self):
        ep=LanguageEpisode();ep.observe('inspect red cube',[['a','cube','red']],['inspect','a'])
        ep.observe('inspect red cube',[['b','cone','green']],['compare','b']);self.assertTrue(ep.blocked)

    def test_generate_and_roundtrip(self):
        m,f=fitted();ep=m.episodes['lang'];t,s,y=f['test'][0]
        self.assertEqual(ep.predict(ep.express(s,y),s),tuple(y))
        self.assertIsNone(ep.express(s,['invent',*y[1:]]))

    def test_ix_inquiry_and_audits(self):
        a=auto_case(109000);self.assertEqual(a['correct'],a['total']);self.assertLess(a['training'],30)
        self.assertIn('inquire',[r['action'] for r in a['steps']]);self.assertIn('audit',[r['action'] for r in a['steps']]);self.assertTrue(all(r['trace'] for r in a['steps']))

    def test_learned_operator_and_relational_bridge(self):
        r=bridge_case(119000);self.assertTrue(r['success']);self.assertTrue(r['blocked'])
        self.assertTrue(all(p['value']==y for _,_,_,y,p,_ in r['predictions']))
        self.assertTrue(all(p['value']==y for _,_,_,y,p,_ in r['relational']))

    def test_symbolic_simulator_bridge(self):
        m,_=fitted();u=UnifiedMind();s=[['x','cube','red']]
        self.assertIsNone(m.simulate('lang','inspect red cube',s,[['ready','x']],None,u))
        for i in range(2):
            x=f't{i}';u.apply('observe',[['ready',x]],['inspect',x],[['done',x]])
        for i in range(3):
            x=f'a{i}';self.assertTrue(u.apply('audit',[['ready',x]],['inspect',x],[['done',x]]))
        r=m.simulate('lang','inspect red cube',s,[['ready','x']],None,u)
        self.assertEqual(r['outcome'],(frozenset({('done','x')}),None))
        self.assertFalse(u.apply('audit',[['ready','x']],['inspect','x'],[['ready','x']]))
        self.assertIsNone(m.simulate('lang','inspect red cube',s,[['ready','x']],None,u))

    def test_checkpoint_replay_and_corruption(self):
        m,f=fitted();t,s,y=f['test'][0]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'language.json';m.save(p);r=LanguageMind.load(p);self.assertEqual(r.events,m.events);self.assertEqual(r.episodes['lang'].predict(t,s),tuple(y))
            m.apply('retract','lang');m.save(p);self.assertTrue(LanguageMind.load(p).episodes['lang'].blocked)
            obj=json.loads(p.read_text());obj['data']['events'].pop();p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):LanguageMind.load(p)

    def test_invalid_inputs_and_no_mutation(self):
        m=LanguageMind();m.apply('new','lang')
        with self.assertRaises(ValueError):m.apply('observe','lang','inspect red cube',[['a','cube','red']],['inspect','missing'])
        self.assertEqual(len(m.events),1);self.assertFalse(m.episodes['lang'].training)
        with self.assertRaises(ValueError):scene([['a','cube','red'],['a','sphere','blue']])
        with self.assertRaises(ValueError):m.apply('new','lang')
        self.assertIsNone(m.compute('lang','rule red cube blue sphere',[['x','cube','red'],['y','sphere','blue']],{},__import__('ix_zero.operators',fromlist=['OperatorMind']).OperatorMind()))

    def test_console_context_save_reload(self):
        f=fixtures(99000,True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';s=Session(p)
            def cmd(c,a):return json.loads(s.handle(c+' '+json.dumps(a)))
            cmd('lnew',['lang'])
            for x in f['train']:cmd('lobserve',['lang',*x])
            for x in f['audit']:self.assertTrue(cmd('laudit',['lang',*x]))
            t,rows,y=f['test'][0];cmd('lcontext',['lang',rows]);self.assertEqual(json.loads(s.handle('say '+t))['intent'],y)
            s.handle('lsave');s=Session(p);self.assertEqual(cmd('linterpret',['lang',t,rows]),y)
            self.assertEqual(cmd('lexpress',['lang',rows,y]),t)

if __name__=='__main__':unittest.main()
