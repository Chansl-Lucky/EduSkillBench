import copy
import unittest
import rescore_glm305_comprehensive_evidence_20261009 as runner
from test_glm305_correctness_20261009 import CorrectnessTests

class EvidenceBindingTests(unittest.TestCase):
    def data(self):
        s=CorrectnessTests().scan();f=s['findings'][0]
        f.pop('claim_quote');f.pop('condition_quotes');f.update(claim_evidence=['E1'],condition_evidence=['Q1'])
        f['calculation'].pop('binding_quotes');f['calculation']['binding_evidence']=['E1']
        s['local_checks']=[copy.deepcopy(f)];s['local_checks'][0]['id']='LC1'
        s['coverage']={k:{'status':'checked','evidence_ids':['E1'],
            'verification':'核对完整实际材料','limitations':'局部'} for k in runner.CATEGORIES}
        return s
    def test_exact_source_binding(self):
        j=CorrectnessTests().job();r=runner.expand_evidence(self.data(),j)
        self.assertEqual(r['findings'][0]['claim_quote'],j['answer'])
        self.assertEqual(r['local_checks'][0]['id'],'F1')
        self.assertIn(r['findings'][0]['condition_quotes'][0],runner.base.question(j))
    def test_invented_evidence_rejected(self):
        s=self.data();s['findings'][0]['claim_evidence']=['E99']
        with self.assertRaises(ValueError):runner.expand_evidence(s,CorrectnessTests().job())
    def test_question_cannot_be_claim_answer(self):
        s=self.data();s['findings'][0]['claim_evidence']=['Q1']
        with self.assertRaises(ValueError):runner.expand_evidence(s,CorrectnessTests().job())
    def test_local_false_arithmetic_still_captured(self):
        s=self.data();s['findings']=[];r=runner.expand_evidence(s,CorrectnessTests().job())
        merged,_=runner.combined_proposals({'findings':[]},r)
        self.assertEqual(len(merged['findings']),1)
    def test_missing_category_not_silently_skipped(self):
        s=self.data();s['coverage'].pop('safety_ethics')
        with self.assertRaises(ValueError):runner.expand_evidence(s,CorrectnessTests().job())

if __name__=='__main__':unittest.main()
