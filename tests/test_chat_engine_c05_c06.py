"""Kiểm tra C05-C06: history thật vào lượt kế tiếp và policy reject-no-truncation."""
from pathlib import Path
import json
import tempfile
import unittest

import unit03_chat_engine as c


class SequenceBackend:
    init_calls = 0
    generate_calls = []
    inspect_calls = []
    info = {"backend": "test_only", "is_test_double": True, "model": "TEST_SEQUENCE"}

    def __init__(self, config):
        type(self).init_calls += 1
        self.config = config

    def inspect_input(self, messages):
        type(self).inspect_calls.append([dict(m) for m in messages])
        return {
            "input_tokens": len(messages) * 11,
            "max_input_tokens": 2048,
            "max_new_tokens": 384,
            "model_context_window": 32768,
            "fits": True,
            "violations": [],
            "policy": "reject_no_truncation",
        }

    def generate(self, messages):
        type(self).generate_calls.append([dict(m) for m in messages])
        n = len(type(self).generate_calls)
        return f"MODEL_T{n}", {
            "input_tokens": len(messages) * 11,
            "output_tokens": 3,
            "finish_reason": "eos",
        }


class OverflowSecondTurnBackend(SequenceBackend):
    generate_calls = []
    inspect_calls = []

    def inspect_input(self, messages):
        type(self).inspect_calls.append([dict(m) for m in messages])
        second = len(type(self).inspect_calls) >= 2
        return {
            "input_tokens": 3000 if second else 100,
            "max_input_tokens": 2048,
            "max_new_tokens": 384,
            "model_context_window": 32768,
            "fits": not second,
            "violations": ["max_input_tokens"] if second else [],
            "policy": "reject_no_truncation",
        }

    def generate(self, messages):
        type(self).generate_calls.append([dict(m) for m in messages])
        return "MODEL_T1", {"input_tokens": 100, "output_tokens": 3, "finish_reason": "eos"}


class C05C06Tests(unittest.TestCase):
    def setUp(self):
        SequenceBackend.init_calls = 0
        SequenceBackend.generate_calls.clear()
        SequenceBackend.inspect_calls.clear()
        OverflowSecondTurnBackend.generate_calls.clear()
        OverflowSecondTurnBackend.inspect_calls.clear()

    def test_c05_messages_are_system_history_then_current_user(self):
        engine = c.create_chat_engine(backend_factory=SequenceBackend)
        history = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "REAL_MODEL_A1"},
        ]
        result = engine.chat("Q2", history, [])
        messages = result["call"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "Q1")
        self.assertEqual(messages[2]["content"], "REAL_MODEL_A1")
        self.assertEqual(messages[-1]["content"], "Q2")
        self.assertEqual(result["call"]["history_turns"], 1)
        self.assertEqual(history[-1]["content"], "REAL_MODEL_A1")

    def test_c05_history_input_is_not_mutated(self):
        engine = c.create_chat_engine(backend_factory=SequenceBackend)
        history = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
        before = json.dumps(history, ensure_ascii=False, sort_keys=True)
        engine.chat("Q2", history, [])
        self.assertEqual(json.dumps(history, ensure_ascii=False, sort_keys=True), before)

    def test_c05_runner_uses_same_multi_case_and_real_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            result = c.run_two_turns(
                c.DEFAULT_QUESTIONS,
                "GOUT_MT_001",
                c.DEFAULT_CONFIG,
                c.DEFAULT_PROMPT,
                out,
                backend_factory=SequenceBackend,
            )
            rows = result["predictions"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["case_id"] == "GOUT_MT_001" for r in rows))
            self.assertEqual(rows[0]["answer"], "MODEL_T1")
            self.assertEqual(rows[1]["history_before"][-1]["content"], "MODEL_T1")
            self.assertIn(
                {"role": "assistant", "content": "MODEL_T1"},
                rows[1]["call"]["messages"],
            )
            self.assertTrue(result["manifest"]["c05_verified"])
            self.assertEqual(SequenceBackend.init_calls, 1)

    def test_c05_selector_rejects_single_case_as_history_source(self):
        with self.assertRaisesRegex(ValueError, "multi"):
            c.select_multi_case_turns(c.DEFAULT_QUESTIONS, "GOUT_ST_001", count=1)

    def test_c06_success_records_token_budget_and_policy(self):
        engine = c.create_chat_engine(backend_factory=SequenceBackend)
        result = engine.chat("Q", [], [])
        budget = result["call"]["input_budget"]
        self.assertTrue(budget["fits"])
        self.assertEqual(budget["policy"], "reject_no_truncation")
        self.assertIsInstance(budget["input_tokens"], int)
        self.assertEqual(len(SequenceBackend.inspect_calls), 1)
        self.assertEqual(len(SequenceBackend.generate_calls), 1)

    def test_c06_overflow_stops_before_generation(self):
        class AlwaysOverflow(SequenceBackend):
            generate_calls = []
            def inspect_input(self, messages):
                return {
                    "input_tokens": 3000,
                    "max_input_tokens": 2048,
                    "max_new_tokens": 384,
                    "model_context_window": 32768,
                    "fits": False,
                    "violations": ["max_input_tokens"],
                    "policy": "reject_no_truncation",
                }
            def generate(self, messages):
                type(self).generate_calls.append(messages)
                return "SHOULD_NOT_RUN", {}

        engine = c.create_chat_engine(backend_factory=AlwaysOverflow)
        with self.assertRaises(c.InputLimitError) as ctx:
            engine.chat("Q", [], [])
        self.assertEqual(ctx.exception.budget["input_tokens"], 3000)
        self.assertEqual(AlwaysOverflow.generate_calls, [])

    def test_c06_runner_records_second_turn_overflow_without_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            result = c.run_two_turns(
                c.DEFAULT_QUESTIONS,
                "GOUT_MT_001",
                c.DEFAULT_CONFIG,
                c.DEFAULT_PROMPT,
                out,
                backend_factory=OverflowSecondTurnBackend,
            )
            rows = result["predictions"]
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[1]["status"], "error")
            self.assertEqual(rows[1]["error_stage"], "input_budget")
            self.assertEqual(rows[1]["input_budget"]["input_tokens"], 3000)
            self.assertEqual(len(OverflowSecondTurnBackend.generate_calls), 1)
            self.assertEqual(len(result["history"]), 2)
            self.assertEqual(result["history"][-1]["content"], "MODEL_T1")

    def test_contexts_still_blocked_until_d11(self):
        engine = c.create_chat_engine(backend_factory=SequenceBackend)
        with self.assertRaisesRegex(NotImplementedError, "D11"):
            engine.chat("Q", [], [{"source_id": "S1", "text": "x"}])
        self.assertEqual(SequenceBackend.generate_calls, [])


if __name__ == "__main__":
    unittest.main()
