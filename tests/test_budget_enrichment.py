import concurrent.futures
from decimal import Decimal
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('budget_enrichment',ROOT/'scripts/run_budget_enrichment.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class BudgetTests(unittest.TestCase):
    def test_parallel_reservations_never_exceed_limit(self):
        with tempfile.TemporaryDirectory() as d:
            budget=b.Budget(Path(d)/'ledger.sqlite',.001)
            def reserve(_):
                try:return budget.reserve(8,'test')
                except RuntimeError:return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p:results=list(p.map(reserve,range(20)))
            self.assertGreater(sum(r is not None for r in results),0)
            self.assertLessEqual(budget.used(),Decimal('.001'))
    def test_unknown_billing_reservation_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.sqlite'
            budget=b.Budget(path,1)
            identifier,reserved=budget.reserve(8,'test')
            reopened=b.Budget(path,1)
            self.assertEqual(reopened.used(),reserved)
            reopened.settle(identifier,Decimal('.00001'))
            self.assertEqual(budget.used(),Decimal('.00001'))
    def test_product_ids_prevent_batch_reassignment(self):
        with self.assertRaises(ValueError):b.parse('[{"id":1,"keywords":[],"situations":[],"synonyms":[]}]',[{'title':'steel cup'}])
    def test_grounded_keyword_filter(self):
        cleaned,dropped=b.parse('[{"id":0,"keywords":["steel","waterproof"],"situations":["drinking water"],"synonyms":["cup"]}]',[{'title':'Steel cup'}])
        self.assertEqual(cleaned[0]['keywords'],['steel'])
        self.assertEqual(dropped,[['waterproof']])
    def test_known_missing_key_separator_is_repaired_without_new_text(self):
        raw='[{"id":0,"keywords":["cup"],"situations["drinking water"],"synonyms":["drinking cup"]}]'
        cleaned,dropped=b.parse(raw,[{'title':'cup'}])
        self.assertEqual(cleaned[0]['situations'],['drinking water'])
        self.assertEqual(cleaned[0]['synonyms'],['drinking cup'])

    def test_long_metadata_is_token_capped(self):
        messages,inputs,estimate=b.pack([{'product_title':'long title '*1000,'product_description':'details '*10000}]*8)
        self.assertLessEqual(estimate,8*b.INPUT_LIMIT)
        self.assertEqual(len(inputs),8)

if __name__=='__main__':unittest.main()
