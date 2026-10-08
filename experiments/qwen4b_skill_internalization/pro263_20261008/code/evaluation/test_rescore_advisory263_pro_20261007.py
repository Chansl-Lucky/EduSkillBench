import unittest
from rescore_advisory263_pro_20261007 import extract_final, top_share, native

class RescoreTests(unittest.TestCase):
    def db(self):
        return {'session':[{'id':'root','parent_id':None},{'id':'child','parent_id':'root'}],
                'message':[{'id':'user','session_id':'root','time_created':1,'data':{'role':'user'}},
                           {'id':'plan','session_id':'root','time_created':2,'data':{'role':'assistant','finish':'tool-calls'}},
                           {'id':'childmsg','session_id':'child','time_created':5,'data':{'role':'assistant','finish':'stop'}},
                           {'id':'final','session_id':'root','time_created':4,'data':{'role':'assistant','finish':'stop'}}],
                'part':[{'id':'p1','message_id':'plan','time_created':2,'data':{'type':'text','text':'I will do it'}},
                        {'id':'p2','message_id':'final','time_created':4,'data':{'type':'text','text':'actual final'}},
                        {'id':'p3','message_id':'childmsg','time_created':5,'data':{'type':'text','text':'I refuse'}}]}

    def test_child_refusal_not_final(self):
        answer,meta=extract_final(self.db());self.assertEqual(answer,'actual final');self.assertTrue(meta['terminal'])

    def test_unfinished_latest_not_replaced_by_old_text(self):
        db=self.db();db['message'].append({'id':'later','session_id':'root','time_created':6,'data':{'role':'assistant','finish':'tool-calls'}})
        answer,meta=extract_final(db);self.assertEqual(answer,'');self.assertFalse(meta['terminal'])

    def test_multiple_parent_sessions_fail_closed(self):
        db=self.db();db['session'].append({'id':'other'});db['message'].append({'id':'u2','session_id':'other','data':{'role':'user'}})
        self.assertFalse(extract_final(db)[1]['terminal'])

    def test_task_dimension_top_fraction_not_weighted(self):
        case={'criteria':[{'id':'D1','levels':[{'label':'A'},{'label':'B'}],'weight':90},
                          {'id':'D2','levels':[],'weight':10}],'applicable_ids':['D1','D2']}
        self.assertEqual(top_share(case,{'items':[{'id':'D1','level':'A'},{'id':'D2','level':'部分满足'}]}),.5)

    def test_missing_dimension_rejected(self):
        case={'criteria':[{'id':'D1','levels':[]}],'applicable_ids':['D1']}
        with self.assertRaises(ValueError):top_share(case,{'items':[]})

    def test_native_top_requires_real_evidence(self):
        case={'criteria':[{'id':'D1','name':'one','description':'one','levels':[],'weight':None}],'applicable_ids':['D1']}
        item={'id':'D1','level':'满足','evidence':'','counterevidence':'','reason':'good'}
        with self.assertRaises(ValueError):native.validate({'items':[item]},case,'answer',require_counterevidence=True)

if __name__=='__main__':unittest.main()
