import json
import unittest
import rescore_glm305_comprehensive_grounded_20261009 as runner
from test_glm305_correctness_20261009 import CorrectnessTests

class GroundingTests(unittest.TestCase):
    def job(self):
        inp=runner.common.read(runner.ROOT/'artifacts/glm305_recovery_20261009_r1/cells/base/lesson-builder__cn24_09/input.json')
        old=runner.common.read(runner.ROOT/'artifacts/glm305_recovery_20261009_r1/cells/base/lesson-builder__cn24_09/result.json')
        return {'task_id':'lesson-builder__cn24_09','arm':'base','suite':'advisory',
            'case':inp['case'],'answer':inp['answer'],'baseline_verdict':old['raw_verdict'],'baseline_score':1.}
    def test_known_real_counterexample(self):
        j=self.job();r=runner.project_with_grounding(j,{'findings':[]},{'findings':[]})
        self.assertEqual(r['report_score'],.75);self.assertEqual(r['masked_dimensions'],['D4'])
        self.assertTrue(r['judge_program_disagreement'])
        self.assertTrue(r['unrelated_dimensions_unchanged'])
    def test_no_overgeneralization(self):
        j=self.job();j['task_id']='other-task'
        self.assertEqual(runner.verified_local_witness(j),[])
    def test_repaired_answer_not_forced_wrong(self):
        j=self.job();j['answer']=j['answer'].replace('答案 $P(2, 3)$','答案 $P(1, 1)$')
        self.assertEqual(runner.verified_local_witness(j),[])
    def test_noncomputable_check_not_zero_or_format_error(self):
        j=CorrectnessTests().job();s=CorrectnessTests().scan()
        s['local_checks']=s['findings'].copy();s['findings']=[];s['local_checks'][0]['calculation']=None
        s['coverage']={k:{'status':'checked','evidence_quotes':['10+3=14'],
            'verification':'仅原文核查，精确计算不适用','limitations':'不是计算证据'} for k in runner.CATEGORIES}
        runner.validate_expanded(s,j)
        merged,_=runner.combined_proposals({'findings':[]},s)
        self.assertEqual(merged['findings'],[])

if __name__=='__main__':unittest.main()
