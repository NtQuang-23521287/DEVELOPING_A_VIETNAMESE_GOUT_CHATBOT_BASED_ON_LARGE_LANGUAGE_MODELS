"""Kiểm tra kết nối các unit bằng backend thử; không giả vờ chạy LLM thật."""
import json
from pathlib import Path
import tempfile
import unittest

import unit02_model as step


class ProtocolBackend:
    calls = []
    info = {"backend": "test_only", "is_test_double": True}

    def __init__(self, config):
        self.config = config

    def generate(self, messages):
        self.calls.append(messages)
        return "TEST_ONLY_PROTOCOL_RESPONSE", {"output_tokens": None}


class ModelStepTests(unittest.TestCase):
    def run_in(self, out, **kwargs):
        return step.run_once(step.DEFAULT_QUESTIONS, "GOUT_ST_001__T1", step.DEFAULT_CONFIG,
                             step.DEFAULT_PROMPT, out, **kwargs)

    def test_preview_never_initializes_model(self):
        def forbidden_factory(config):
            raise AssertionError("Preview must not initialize a model")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview"
            self.run_in(out, preview=True, backend_factory=forbidden_factory)
            request = json.loads((out / "request.json").read_text())
            self.assertEqual([m['role'] for m in request['messages']], ['system', 'user'])
            self.assertEqual(request['messages'][1]['content'], request['sample']['user'])
            self.assertFalse((out / 'predictions.jsonl').exists())

    def test_success_records_single_call_raw_answer_and_provenance(self):
        ProtocolBackend.calls.clear()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            result = self.run_in(out, backend_factory=ProtocolBackend)
            manifest = json.loads((out / 'manifest.json').read_text())
            logged = json.loads((out / 'predictions.jsonl').read_text())
            self.assertEqual(len(ProtocolBackend.calls), 1)
            self.assertEqual(logged['answer'], 'TEST_ONLY_PROTOCOL_RESPONSE')
            self.assertEqual(logged['history'], [])
            self.assertEqual(logged['contexts'], [])
            self.assertTrue(result['is_test_double'])
            self.assertEqual(manifest['status'], 'completed_test_double')
            self.assertEqual(manifest['predictions_sha256'], step.sha(out/'predictions.jsonl'))

    def test_loading_failure_is_logged_without_fake_answer(self):
        def failing(config):
            raise RuntimeError('TEST_NO_WEIGHTS')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fail'
            result = self.run_in(out, backend_factory=failing)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['error_stage'], 'model_load')
            self.assertIsNone(result['answer'])
            self.assertFalse(manifest['generation_attempted'])

    def test_generation_failure_is_logged(self):
        class Failing(ProtocolBackend):
            def generate(self, messages):
                raise RuntimeError('TEST_OUT_OF_MEMORY')
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_in(Path(tmp)/'run', backend_factory=Failing)
            self.assertEqual(result['error_stage'], 'generation')
            self.assertIsNone(result['answer'])

    def test_empty_response_is_error(self):
        class Empty(ProtocolBackend):
            def generate(self, messages):
                return '  ', {}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_in(Path(tmp)/'run', backend_factory=Empty)['status'], 'error')

    def test_multi_turn_cannot_run_without_history(self):
        with self.assertRaisesRegex(ValueError, 'Buoc B chi chay single'):
            step.select_one_question(step.DEFAULT_QUESTIONS, 'GOUT_MT_001__T2')

    def test_answer_contamination_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [json.loads(line) for line in step.DEFAULT_QUESTIONS.read_text().splitlines()]
            rows[0]['ground_truth'] = 'SECRET_REFERENCE'
            bad = Path(tmp)/'bad.jsonl'
            bad.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'ground_truth'):
                step.select_one_question(bad, 'GOUT_ST_001__T1')

    def test_existing_output_is_not_overwritten_or_called_again(self):
        ProtocolBackend.calls.clear()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            self.run_in(out, backend_factory=ProtocolBackend)
            original = (out/'predictions.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):
                self.run_in(out, backend_factory=ProtocolBackend)
            self.assertEqual(len(ProtocolBackend.calls), 1)
            self.assertEqual((out/'predictions.jsonl').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
