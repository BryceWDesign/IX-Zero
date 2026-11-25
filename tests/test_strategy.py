import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ix_zero.operators import OperatorMind,OperatorEpisode
from ix_zero.strategy import BASELINE,Strategy,StrategyMind,advance,compare,choose,query,proposals,validate_corpus
from ix_zero.strategy_experiment import corpus
from ix_zero.cli import Session

class StrategyTest(unittest.TestCase):
    def test_proposals_are_unique_and_bounded(self):
        self.assertEqual(len(set(proposals())),18)
        with self.assertRaises(ValueError):Strategy('exec','all','high')
    def test_baseline_matches_original(self):
        t=OperatorEpisode(3,3);pool=t.domain.inputs[::3]
        for _ in range(6):
            x=query(t,pool,BASELINE);self.assertEqual(x,t.query(pool))
            t.observe(x,x[0]|x[1])
    def test_frontier_counts_less_and_does_not_change_models(self):
        t=OperatorEpisode(4,5);pool=t.domain.inputs[:12];a=[0,0];b=[0,0];models=t.models
        query(t,pool,BASELINE,a);query(t,pool,Strategy('distinct','frontier','low'),b)
        self.assertLess(b[0],a[0]);self.assertIs(t.models,models)
    def test_gate_rejects_wrong_or_reduced_coverage(self):
        base=[dict(id='a',correct=10,incorrect=0,abstained=0,labels=20,query_work=100,redundant_validation_work=100)]
        for update in [dict(correct=9,incorrect=1),dict(correct=9,abstained=1),dict(labels=23)]:
            c=[dict(base[0],query_work=1,**update)];self.assertFalse(compare(c,base)['eligible'])
        self.assertTrue(compare([dict(base[0],query_work=49,labels=21)],base)['eligible'])
        self.assertFalse(compare(base,base)['eligible'])
    def test_partition_overlap_and_duplicate_families(self):
        c=corpus(1,1);validate_corpus(c)
        b=copy.deepcopy(c);b['confirmation'][0]['family']=b['selection'][0]['family']
        with self.assertRaises(ValueError):validate_corpus(b)
        b=copy.deepcopy(c);b['selection'][0]['audit'][0]=b['selection'][0]['train'][0]
        with self.assertRaises(ValueError):validate_corpus(b)
    def test_strategy_cannot_bypass_host(self):
        m=OperatorMind();m.apply('new','trial',3,3,False)
        with self.assertRaises(ValueError):m.advance('trial',[(0,0)],query_selector=lambda *_:(1,1))
        d=advance(m,'trial',[(0,0)],Strategy('minimax','frontier','low'))
        self.assertEqual(d['result'],(0,0));self.assertIsNone(m.episodes['trial'].predict((1,1)))
    def test_selection_confirms_only_one_challenger_and_refuses_regression(self):
        c=corpus(2,1,'heuristic');calls=[]
        def fake(t,p,details=False):
            calls.append((t['id'],p));return dict(id=t['id'],correct=1 if p==BASELINE or t['id'].startswith('selection') else 0,incorrect=0,abstained=0,labels=9,query_work=10 if p==BASELINE else 1,redundant_validation_work=0)
        with patch('ix_zero.strategy.episode',side_effect=fake):r=choose(c)
        self.assertEqual(r['decision'],'retain');self.assertEqual(r['selected'],dict(metric='distinct',weighting='all',tie='high',kernel='checked'))
        self.assertEqual(len(set(p for n,p in calls if n.startswith('confirmation'))),2)
    def test_policy_cannot_adopt_unconfirmed(self):
        c=corpus(2,1)
        def fake(t,p,details=False):return dict(id=t['id'],correct=1,incorrect=0,abstained=0,labels=9,query_work=10,redundant_validation_work=0)
        with patch('ix_zero.strategy.episode',side_effect=fake),patch('ix_zero.strategy.run_ix') as run:
            run.return_value.approvals_required=False;run.return_value.replies=['adopt']
            with self.assertRaises(ValueError):choose(c)
    def test_third_gate_rejects_regression(self):
        c=corpus(2,1);calls=[]
        def fake(t,p,details=False):
            calls.append((t['id'],p));return dict(id=t['id'],correct=1 if p==BASELINE or not t['id'].startswith('challenge') else 0,incorrect=0,abstained=0,labels=9,query_work=10,redundant_validation_work=10 if p==BASELINE else 0)
        with patch('ix_zero.strategy.episode',side_effect=fake):r=choose(c)
        self.assertTrue(r['confirmation_verdict']['eligible']);self.assertFalse(r['challenge_verdict']['eligible']);self.assertEqual(r['decision'],'retain')
    def test_packed_equivalence(self):
        for width in (2,3,4):
            t=OperatorEpisode(width,4);pool=t.domain.inputs
            for _ in range(6):
                a=[0,0];b=[0,0];x=query(t,pool,BASELINE,a)
                self.assertEqual(x,query(t,pool,Strategy(kernel='packed'),b));self.assertEqual(a[0],b[0]);self.assertEqual(b[1],0)
                t.observe(x,x[0]|x[1])
    def test_replay_rollback_and_cli(self):
        c=corpus(3,1)
        # Small replay fixture retains all actual strategy-selection code.
        for phase in ('selection','confirmation','challenge'):c[phase]=c[phase][:2]
        for phase in ('selection','confirmation','challenge'):
            for t in c[phase]:t['budget']=2
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';f=Path(d)/'corpus.json';f.write_text(json.dumps(c),encoding='utf-8')
            s=Session(p);s.handle('mfit '+json.dumps(str(f)));s.handle('msave');t=Session(p)
            self.assertEqual(s.strategy.current,t.strategy.current)
            with self.assertRaises(ValueError):t.strategy.fit(c)
            t.handle('onew ["trial",3,3,false]');self.assertEqual(json.loads(t.handle('madvance ["trial",[[0,0],[1,1]]]'))['action'],'inquire')
            t.handle('mrollback');t.handle('msave');self.assertTrue(Session(p).strategy.rolled_back)
            obj=json.loads(t.mpath.read_text());obj['payload']['rolled_back']=False;t.mpath.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):StrategyMind.load(t.mpath)

if __name__=='__main__':unittest.main()
