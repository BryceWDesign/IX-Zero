import json
from pathlib import Path
import tempfile
import unittest
from ix_zero.grid_relational import infer, shape, coordinate
from ix_zero.external_eval import evaluate
from ix_zero.research_workspace import ResearchWorkspace


def mask(g):
    h,w=len(g),len(g[0]);return [[g[i%h][j%w] if g[i//h][j//w] else 0 for j in range(w*w)] for i in range(h*h)]


class RelationalGridTests(unittest.TestCase):
    def test_two_coordinate_frames_learned(self):
        train=[[[1,0],[0,1]],[[0,2,2],[2,0,0]],[[3,0],[3,3]]];x=[[0,5],[5,0],[5,5]]
        r=infer([(g,mask(g)) for g in train],[x]);self.assertEqual(r['predictions'],[tuple(map(tuple,mask(x)))]);self.assertEqual(r['minimum_cost'],3)
    def test_learned_area_rank_recolor(self):
        # Different lengths/colors, different positions; not ARC task data.
        a=[[0,3,0,4,4],[0,3,0,0,0],[0,3,0,5,0]];b=[[8,8,0,7,0],[0,0,0,0,0],[6,6,6,0,0]]
        y=[[0,9,0,8,8],[0,9,0,0,0],[0,9,0,7,0]];z=[[8,8,0,7,0],[0,0,0,0,0],[9,9,9,0,0]]
        x=[[2,0,3,3,3],[0,0,0,0,0],[4,4,0,0,0]];t=((7,0,9,9,9),(0,0,0,0,0),(8,8,0,0,0))
        r=infer([(a,y),(b,z)],[x]);self.assertEqual(r['predictions'],[t])
    def test_ambiguity_abstains(self):
        r=infer([([[1]],[[1]])],[[[2,3],[4,5]]]);self.assertIsNone(r['predictions'][0]);self.assertGreater(r['distinct_predictions'],1)
    def test_budget_exhaustion_abstains(self):
        r=infer([([[1,2],[3,4]],[[1,2],[3,4]])],[[[2,3],[4,5]]],budget=1);self.assertTrue(r['budget_exhausted']);self.assertIsNone(r['predictions'][0])
    def test_invalid_and_geometry(self):
        with self.assertRaises(ValueError):infer([([[1,2],[3]],[[1]])],[[[1]]])
        self.assertIsNone(shape(('shrink',2,2),((1,2,3),)));self.assertEqual(coordinate('block',4,3,3,3,9,9),(1,1))
    def test_external_worker_and_workspace(self):
        train=[[[1,0],[0,1]],[[0,2,2],[2,0,0]],[[3,0],[3,3]]];x=[[0,5],[5,0],[5,5]]
        w=ResearchWorkspace();r=w.execute({'command':'grid_relational','train':[(g,mask(g)) for g in train],'inputs':[x]});self.assertEqual(r['predictions'][0],tuple(map(tuple,mask(x))))
        with tempfile.TemporaryDirectory() as d:
            Path(d,'task.json').write_text(json.dumps({'train':[{'input':g,'output':mask(g)} for g in train],'test':[{'input':x,'output':mask(x)}]}),encoding='utf-8')
            r=evaluate(d,budget=5000,modes=('relational',));self.assertEqual(r['summary']['relational']['exact_tasks'],1)

if __name__=='__main__':unittest.main()
