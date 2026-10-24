import unittest
from ecirpaper.fusion import reciprocal_rank_fusion

class FusionTests(unittest.TestCase):
    def test_agreement_and_symmetry(self):
        a=['a','b','c'];b=['b','c','a']
        self.assertEqual(reciprocal_rank_fusion(a,b),reciprocal_rank_fusion(b,a))
        result=reciprocal_rank_fusion(a,a)
        self.assertEqual([p for p,s in result],a)
        self.assertAlmostEqual(result[0][1],2/61)
    def test_ties_use_id(self):
        self.assertEqual([p for p,s in reciprocal_rank_fusion(['b','a'],['a','b'])],['a','b'])
    def test_reject_incomparable_lists(self):
        for a,b in [(['a'],['b']),(['a','a'],['a'])]:
            with self.assertRaises(ValueError):reciprocal_rank_fusion(a,b)
