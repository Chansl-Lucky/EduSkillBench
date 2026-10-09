import copy
import unittest
import rescore_glm305_correctness_20261009 as runner

class CorrectnessTests(unittest.TestCase):
    def job(self):
        return {'suite':'advisory','answer':'教师认可10+3=14。学生误答9+4=14，已纠正为13。',
            'baseline_score':1.,'baseline_verdict':{'items':[{'id':'D1','level':'优秀'},{'id':'D2','level':'优秀'}]},
            'case':{'context':'四则运算','user_prompt':'说明凑十。','applicable_ids':['D1','D2'],
                'criteria':[{'id':'D1','name':'内容准确性','levels':[{'label':'优秀'},{'label':'不合格'}]},
                            {'id':'D2','name':'学生参与','levels':[{'label':'优秀'},{'label':'不合格'}]}]}}
    def scan(self):
        return {'reviewed_units':['独立复算10+3'],'limitations':[],'findings':[{'id':'F1','kind':'math',
            'claim_quote':'10+3=14','condition_quotes':[],'suspected_error':'加法答案错误','recomputation':'10+3=13',
            'rubric_links':[{'id':'D1','criterion_quote':'内容准确性','dependency':'答案键的内容错误'}],
            'calculation':{'lhs':'10+3','rhs':'14','variables':{},'binding_quotes':['10+3=14']}}]}
    def review(self):
        return {'findings':[{'id':'F1','decision':'confirmed','stance':'endorsed','proof':'10+3=13不等于14',
            'affected_dimensions':['D1'],'mapping_reason':'仅内容准确性受损','calculator_binding_valid':True}]}
    def test_fraction(self):
        self.assertEqual(runner.exact_fraction('0.1+0.2',{}),runner.Fraction(3,10))
    def test_other_parameters(self):
        f=self.scan()['findings'][0]; f['calculation'].update(lhs='1/(n*(n+4))',rhs='(1/n-1/(n+4))/4',variables={'n':'3'})
        self.assertTrue(runner.calculator(f)['equal'])
        f['calculation']['rhs']='4*(1/n-1/(n+4))'
        self.assertFalse(runner.calculator(f)['equal'])
    def test_bounded_no_code(self):
        for expr in ('__import__("os").system("true")','[1][0]','2**10000','sqrt(2)','x.y','sum([1,2])'):
            with self.assertRaises((ValueError,SyntaxError)):runner.exact_fraction(expr,{})
    def test_undefined_not_error(self):
        f=self.scan()['findings'][0]; f['calculation']['lhs']='1/0'
        self.assertEqual(runner.calculator(f)['status'],'unsupported_or_undefined')
    def test_exact_quotes(self):
        runner.validate_scan(self.scan(),self.job())
        s=self.scan();s['findings'][0]['claim_quote']='10+3 =14'
        with self.assertRaises(ValueError):runner.validate_scan(s,self.job())
    def test_invented_rubric_rejected(self):
        s=self.scan();s['findings'][0]['rubric_links'][0]['criterion_quote']='全部不对'
        with self.assertRaises(ValueError):runner.validate_scan(s,self.job())
    def test_dimension_local(self):
        j=self.job(); before=copy.deepcopy(j)
        r=runner.project(j,self.scan(),self.review())
        self.assertEqual(r['report_score'],.5); self.assertEqual(r['corrected_verdict']['items'][1],j['baseline_verdict']['items'][1])
        self.assertEqual(j,before)
    def test_uncertain_no_deduction(self):
        r=self.review();r['findings'][0].update(decision='uncertain',stance='unknown',affected_dimensions=[])
        runner.validate_review(r,self.scan(),self.job())
        self.assertEqual(runner.project(self.job(),self.scan(),r)['report_score'],1.)
    def test_corrected_student_not_deducted(self):
        r=self.review();r['findings'][0].update(decision='dismissed',stance='corrected',affected_dimensions=[])
        runner.validate_review(r,self.scan(),self.job())
        self.assertEqual(runner.project(self.job(),self.scan(),r)['report_score'],1.)
    def test_unrelated_deduction_rejected(self):
        r=self.review();r['findings'][0]['affected_dimensions']=['D2']
        with self.assertRaises(ValueError):runner.validate_review(r,self.scan(),self.job())
    def test_contradiction_rejected(self):
        r=self.review();r['findings'][0].update(decision='dismissed',affected_dimensions=[])
        with self.assertRaises(ValueError):runner.validate_review(r,self.scan(),self.job())
    def test_gap_recorded_not_new_dimension(self):
        r=self.review();r['findings'][0]['affected_dimensions']=[]
        s=self.scan();s['findings'][0]['rubric_links']=[]
        runner.validate_review(r,s,self.job());p=runner.project(self.job(),s,r)
        self.assertEqual(p['report_score'],1.);self.assertEqual(p['coverage_gaps'],['F1'])

if __name__=='__main__':unittest.main()
