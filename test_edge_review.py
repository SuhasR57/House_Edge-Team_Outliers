"""Offline contract tests. Provider responses are explicit test fixtures, not live calls."""
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from pathlib import Path
from edge_review import ReviewError,call_gemini,validate_review,render,run,request_payload,deterministic_assessment,index_facts,read_env_key

ROOT=Path(__file__).resolve().parent
EVIDENCE_PATH=ROOT/'evidence.json'
if not EVIDENCE_PATH.exists():
    EVIDENCE_PATH=ROOT.parents[1]/'outputs/edge-shift-gemini/evidence.json'
EVIDENCE=json.loads(EVIDENCE_PATH.read_text())
FIXTURE={'summary':'Test fixture only. The measured candidate improves public validation.',
         'answerable':False,'abstention_reason':'Hidden-test ranking is not available.',
         'findings':[{'title':'Measured comparison','interpretation':'Compare the recorded validation points, with reused-validation caveats.','evidence_ids':['hierarchical.points','temporal.points','evaluation.caveat']}],
         'experiments':[{'hypothesis':'A sparse-history adjustment may help.','change':'Keep the change small and select on training episodes.','measurement':'Use held-out episode probability error.','success_criterion':'Retain only a consistent improvement against the frozen model.','evidence_ids':['hierarchical.short_history','evaluation.selection_rule']}],
         'limitations':['Public validation does not establish hidden-test performance.']}

class FakeOpener:
    def __init__(self,result=None,finish='STOP',error=None):
        self.result=FIXTURE if result is None else result
        self.finish=finish; self.error=error; self.request=None
    def open(self,request,timeout):
        self.request=request
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps({'status':'completed' if self.finish=='STOP' else 'incomplete','steps':[{'type':'model_output','content':[{'type':'text','text':json.dumps(self.result)}]}],
            'model':'fixture-model-not-live','id':'unit-test-fixture','usage':{'total_input_tokens':100,'total_output_tokens':50,'total_tokens':150}}).encode())


class ReviewTests(unittest.TestCase):
    def test_timeout_retries_then_succeeds(self):
        good=FakeOpener()
        class Recovering:
            calls=0
            def open(self,request,timeout):
                self.calls+=1
                self_timeout=timeout
                if self.calls==1: raise TimeoutError()
                return good.open(request,self_timeout)
        opener=Recovering()
        with patch('edge_review.time.sleep') as sleep:
            result,metadata=call_gemini('test','gemini-3.8-flash',{},opener)
        self.assertEqual(result,FIXTURE)
        self.assertEqual(metadata['attempts'],2)
        sleep.assert_called_once_with(5)
    def test_repeated_timeout_is_bounded(self):
        with patch('edge_review.time.sleep'), self.assertRaisesRegex(ReviewError,'after two attempts'):
            call_gemini('test','gemini-3.8-flash',{},FakeOpener(error=TimeoutError()))
    def test_service_unavailable_retries(self):
        error=urllib.error.HTTPError('https://example.invalid',503,'busy',{},io.BytesIO(b'{}'))
        with patch('edge_review.time.sleep') as sleep, self.assertRaisesRegex(ReviewError,'after two attempts'):
            call_gemini('test','gemini-3.8-flash',{},FakeOpener(error=error))
        sleep.assert_called_once_with(5)
    def test_interactions_schema_fields(self):
        payload=request_payload(EVIDENCE,'Review')
        self.assertFalse(payload['store'])
        self.assertEqual(payload['response_format']['mime_type'],'application/json')
        self.assertEqual(payload['response_format']['schema']['type'],'object')
        self.assertNotIn('generationConfig',payload)
    def test_provider_detail_redacts_actual_key(self):
        key='test-private-api-value'
        body=json.dumps({'error':{'message':'Invalid field responseFormat; key='+key}}).encode()
        error=urllib.error.HTTPError('https://example.invalid',400,'bad request',{},io.BytesIO(body))
        with self.assertRaises(ReviewError) as caught:
            call_gemini(key,'gemini-3.8-flash',{},FakeOpener(error=error))
        self.assertIn('responseFormat',str(caught.exception))
        self.assertNotIn(key,str(caught.exception))
    def test_env_quotes_comments_and_priority(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'.env'
            path.write_text('# local credentials\nGOOGLE_API_KEY=secondary\nexport GEMINI_API_KEY="test-value"\n')
            self.assertEqual(read_env_key(path),'test-value')
    def test_env_missing_and_blank(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'.env'
            self.assertEqual(read_env_key(path),'')
            path.write_text('GEMINI_API_KEY=\n')
            self.assertEqual(read_env_key(path),'')
    def test_env_is_never_executed_or_expanded(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'.env'
            path.write_text('GEMINI_API_KEY=$(not-a-command)\n')
            self.assertEqual(read_env_key(path),'$(not-a-command)')
    def test_env_invalid_quote_hides_value(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'.env'
            path.write_text('GEMINI_API_KEY="test-secret\n')
            with self.assertRaises(ReviewError) as caught: read_env_key(path)
            self.assertNotIn('test-secret',str(caught.exception))
    def test_real_metric_cards(self):
        measured=deterministic_assessment(EVIDENCE)
        self.assertAlmostEqual(measured['candidate_points'],47.458381385659806)
        self.assertEqual(measured['weakest_condition'],'short_history')
        self.assertTrue(measured['offline_artifact_checks_passed'])
    def test_valid_contract(self):
        self.assertEqual(validate_review(copy.deepcopy(FIXTURE),EVIDENCE),FIXTURE)
    def test_invented_reference_rejected(self):
        changed=copy.deepcopy(FIXTURE); changed['findings'][0]['evidence_ids']=['test.rank']
        with self.assertRaises(ReviewError): validate_review(changed,EVIDENCE)
    def test_missing_citations_rejected(self):
        changed=copy.deepcopy(FIXTURE); changed['experiments'][0]['evidence_ids']=[]
        with self.assertRaises(ReviewError): validate_review(changed,EVIDENCE)
    def test_unexplained_abstention_rejected(self):
        changed=copy.deepcopy(FIXTURE); changed['abstention_reason']=''
        with self.assertRaises(ReviewError): validate_review(changed,EVIDENCE)
    def test_extra_fields_rejected(self):
        changed=copy.deepcopy(FIXTURE); changed['guaranteed_prize']=True
        with self.assertRaises(ReviewError): validate_review(changed,EVIDENCE)
    def test_missing_limitations_rejected(self):
        changed=copy.deepcopy(FIXTURE); changed['limitations']=[]
        with self.assertRaises(ReviewError): validate_review(changed,EVIDENCE)
    def test_duplicate_evidence_rejected(self):
        changed=copy.deepcopy(EVIDENCE); changed['facts'].append(changed['facts'][0])
        with self.assertRaises(ReviewError): index_facts(changed)
    def test_html_escaping(self):
        changed=copy.deepcopy(FIXTURE); changed['summary']='<script>alert(1)</script>'
        output=render(EVIDENCE,changed,{})
        self.assertNotIn('<script>',output); self.assertIn('&lt;script&gt;',output)
    def test_key_is_header_only(self):
        fake=FakeOpener(); key='unit-test-only-secret'
        result,metadata=call_gemini(key,'gemini-3.8-flash',request_payload(EVIDENCE,'Review the model.'),fake)
        self.assertNotIn(key,fake.request.full_url)
        self.assertNotIn(key,fake.request.data.decode())
        self.assertEqual(fake.request.get_header('X-goog-api-key'),key)
        self.assertEqual(result,FIXTURE)
        self.assertEqual(metadata['usage']['total_tokens'],150)
    def test_quota_failure_has_no_key(self):
        error=urllib.error.HTTPError('https://example.invalid',429,'secret-in-error',{},io.BytesIO(b'secret-in-body'))
        with self.assertRaises(ReviewError) as caught:
            call_gemini('test','gemini-3.8-flash',{},FakeOpener(error=error))
        self.assertIn('429',str(caught.exception)); self.assertNotIn('secret',str(caught.exception))
    def test_incomplete_response_rejected(self):
        with self.assertRaises(ReviewError): call_gemini('test','gemini-3.8-flash',{},FakeOpener(finish='MAX_TOKENS'))
    def test_missing_key_fails_without_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            destination=Path(temp)/'report'
            with self.assertRaises(ReviewError): run(EVIDENCE,'Review','gemini-3.8-flash','',destination)
            self.assertFalse(destination.exists())
    def test_preview_cannot_claim_live(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            destination=Path(temp)/'preview'
            report=run(EVIDENCE,'Review','gemini-3.8-flash','',destination,preview=True)
            self.assertIn('Gemini has not run',report.read_text())
            self.assertEqual(json.loads((destination/'receipt.json').read_text())['status'],'not_run')
            self.assertFalse((destination/'review.json').exists())
    def test_success_fixture_receipt_and_secret_exclusion(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            destination=Path(temp)/'fixture'
            run(EVIDENCE,'Review','gemini-3.8-flash','unit-test-only-secret',destination,opener=FakeOpener())
            receipt=json.loads((destination/'receipt.json').read_text())
            self.assertEqual(receipt['response_id'],'unit-test-fixture')
            for file in destination.iterdir(): self.assertNotIn('unit-test-only-secret',file.read_text())
    def test_invalid_ai_does_not_create_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            destination=Path(temp)/'bad'
            with self.assertRaises(ReviewError): run(EVIDENCE,'Review','gemini-3.8-flash','test',destination,opener=FakeOpener(result={'bad':'response'}))
            self.assertFalse(destination.exists())
    def test_model_path_injection_rejected(self):
        with self.assertRaises(ReviewError): call_gemini('test','../../evil',{},FakeOpener())
    def test_evidence_has_no_raw_forecasts_or_histories(self):
        encoded=json.dumps(EVIDENCE)
        self.assertNotIn('main-test-e',encoded)
        self.assertNotIn('C:\\Users\\',encoded)

if __name__=='__main__':
    unittest.main(verbosity=2)
