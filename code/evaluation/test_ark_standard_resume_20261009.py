import os
import unittest
import urllib.error
from unittest.mock import patch
import resume_glm305_ark_standard_20261009 as runner

class TransportTests(unittest.TestCase):
    def test_explicit_same_version(self):
        self.assertEqual(runner.request_model('deepseek-v4-pro'),'deepseek-v4-pro-ga-260813')
        self.assertEqual(runner.request_model(runner.EXPECTED),runner.EXPECTED)
    def test_no_other_model(self):
        with self.assertRaises(ValueError):runner.request_model('deepseek-v4-pro-260425')
    def test_endpoint_and_payload_otherwise_unchanged(self):
        calls=[]
        def original(prompt,model,**kw):calls.append((prompt,model,kw));return 'OK',{}
        with patch.object(runner,'ORIGINAL_SETUP',lambda:None),patch.object(runner.common.judge,'call_judge',original),patch.object(runner,'wait_network',return_value=False),patch.dict(os.environ,clear=False):
            runner.setup_worker()
            opts={'thinking_mode':'disabled','max_output_tokens':10000,'api_protocol':'chat_completions','output_schema':None}
            runner.common.judge.call_judge('unchanged prompt','deepseek-v4-pro',**opts)
            self.assertEqual(os.environ['ANTHROPIC_BASE_URL'],runner.ENDPOINT)
            self.assertEqual(calls,[('unchanged prompt',runner.EXPECTED,opts)])
    def test_unauthenticated_http_is_reachable(self):
        def opener(req,timeout):
            self.assertFalse(req.has_header('Authorization'))
            self.assertEqual(timeout,8)
            raise urllib.error.HTTPError(req.full_url,401,'unauthorized',{},None)
        with patch.object(runner.common.judge,'request_transport',return_value=(opener,'test')):
            self.assertEqual(runner.probe_network(),{'reachable':True,'http_status':401})
    def test_outage_is_not_scored(self):
        def opener(req,timeout):raise urllib.error.URLError('offline')
        with patch.object(runner.common.judge,'request_transport',return_value=(opener,'test')):
            self.assertFalse(runner.probe_network()['reachable'])

if __name__=='__main__':unittest.main()
