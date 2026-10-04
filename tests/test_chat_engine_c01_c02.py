"""Kiểm tra C01-C02 bằng backend thử; không giả vờ đã chạy trọng số thật qua engine."""
from pathlib import Path
import inspect
import json
import tempfile
import unittest

import unit02_model as b
import unit03_chat_engine as c


class CountingBackend:
    init_calls = 0
    generate_calls = []
    info = {"backend": "test_only", "is_test_double": True, "model": "TEST_MODEL"}

    def __init__(self, config):
        type(self).init_calls += 1
        self.config = config

    def inspect_input(self, messages):
        return {
            "input_tokens": len(messages) * 10,
            "max_input_tokens": 2048,
            "max_new_tokens": 384,
            "model_context_window": 32768,
            "fits": True,
            "violations": [],
            "policy": "reject_no_truncation",
        }

    def generate(self, messages):
        type(self).generate_calls.append(messages)
        return "TEST_C02_ANSWER", {"output_tokens": 3, "finish_reason": "eos"}


class C01C02Tests(unittest.TestCase):
    def setUp(self):
        CountingBackend.init_calls = 0
        CountingBackend.generate_calls.clear()

    def test_c01_public_signature_has_no_reference(self):
        sig = inspect.signature(c.ChatEngine.chat)
        self.assertEqual(list(sig.parameters), ["self", "question", "history", "contexts"])
        self.assertNotIn("reference", sig.parameters)
        self.assertNotIn("ground_truth", sig.parameters)

    def test_c02_reuses_exact_b_message_builder(self):
        engine = c.create_chat_engine(backend_factory=CountingBackend)
        question = "Gút là gì?"
        result = engine.chat(question, [], [])
        expected = b.build_messages(question, b.read_text(b.DEFAULT_PROMPT))
        self.assertEqual(CountingBackend.generate_calls, [expected])
        self.assertEqual(result["call"]["messages"], expected)
        self.assertEqual(result["answer"], "TEST_C02_ANSWER")
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["call"]["rag_enabled"])
        self.assertFalse(result["call"]["reference_used"])

    def test_backend_is_loaded_once_and_reused(self):
        engine = c.create_chat_engine(backend_factory=CountingBackend)
        engine.chat("Câu 1", [], [])
        engine.chat("Câu 2", [], [])
        self.assertEqual(CountingBackend.init_calls, 1)
        self.assertEqual(len(CountingBackend.generate_calls), 2)

    def test_nonempty_broken_history_is_rejected(self):
        engine = c.create_chat_engine(backend_factory=CountingBackend)
        with self.assertRaises(ValueError):
            engine.chat("Câu hỏi", [{"role": "user", "content": "trước"}], [])
        self.assertEqual(CountingBackend.generate_calls, [])

    def test_nonempty_contexts_is_not_silently_ignored(self):
        engine = c.create_chat_engine(backend_factory=CountingBackend)
        with self.assertRaisesRegex(NotImplementedError, "D11"):
            engine.chat("Câu hỏi", [], [{"source_id": "S1", "text": "abc"}])
        self.assertEqual(CountingBackend.generate_calls, [])

    def test_run_one_writes_c02_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c02"
            record = c.run_one(
                c.DEFAULT_QUESTIONS,
                "GOUT_ST_001__T1",
                c.DEFAULT_CONFIG,
                c.DEFAULT_PROMPT,
                out,
                backend_factory=CountingBackend,
            )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            logged = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")
            self.assertEqual(logged["answer"], "TEST_C02_ANSWER")
            self.assertEqual(logged["sources"], [])
            self.assertEqual(logged["history"], [])
            self.assertEqual(logged["contexts"], [])
            self.assertEqual(manifest["status"], "completed_test_double")
            self.assertTrue(manifest["model_load_attempted"])
            self.assertTrue(manifest["generation_attempted"])
            self.assertFalse(manifest["rag_enabled"])
            self.assertFalse(manifest["reference_used"])

    def test_bad_backend_empty_answer_is_rejected(self):
        class EmptyBackend(CountingBackend):
            def generate(self, messages):
                return " ", {}
        engine = c.create_chat_engine(backend_factory=EmptyBackend)
        with self.assertRaisesRegex(ValueError, "rong"):
            engine.chat("Câu hỏi", [], [])


if __name__ == "__main__":
    unittest.main()
