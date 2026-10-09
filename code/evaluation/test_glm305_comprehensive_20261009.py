import copy
import unittest
import rescore_glm305_comprehensive_20261009 as runner
from test_glm305_correctness_20261009 import CorrectnessTests

class ComprehensiveTests(unittest.TestCase):
    def data(self):
        s=CorrectnessTests().scan()
        s['coverage']={k:{'status':'checked','evidence_quotes':['10+3=14'],
            'verification':'检查作者认可的算式','limitations':'仅本局部'} for k in runner.CATEGORIES}
        s['local_checks']=copy.deepcopy(s['findings']);s['findings'][0]['kind']='teaching_inference'
        return s
    def test_all_categories(self):
        runner.validate_expanded(self.data(),CorrectnessTests().job())
        s=self.data();s['coverage'].pop('evidence_integrity')
        with self.assertRaises(ValueError):runner.validate_expanded(s,CorrectnessTests().job())
    def test_no_quote_no_checked(self):
        s=self.data();s['coverage']['domain_truth']['evidence_quotes']=[]
        with self.assertRaises(ValueError):runner.validate_expanded(s,CorrectnessTests().job())
    def test_arithmetic_false_not_silently_passed(self):
        s=self.data();s['findings']=[]
        merged,channels=runner.combined_proposals({'findings':[]},s)
        self.assertEqual(len(merged['findings']),1)
        self.assertIn('exact_local_inequality_needs_context_confirmation',channels['F1'])
    def test_no_duplicate_punishment(self):
        s=self.data();p=CorrectnessTests().scan()
        merged,channels=runner.combined_proposals(p,s)
        self.assertEqual(len(merged['findings']),1);self.assertEqual(len(channels['F1']),3)
    def test_nonmath_can_be_confirmed(self):
        s=self.data();s['findings'][0].update(kind='fabricated_observation',calculation=None)
        merged,_=runner.combined_proposals({'findings':[]},s)
        r=CorrectnessTests().review();r['findings'][0]['calculator_binding_valid']=False
        runner.validate_final_review(r,merged,CorrectnessTests().job())
        p=runner.base.project(CorrectnessTests().job(),merged,r)
        self.assertEqual(p['report_score'],.5)

if __name__=='__main__':unittest.main()
