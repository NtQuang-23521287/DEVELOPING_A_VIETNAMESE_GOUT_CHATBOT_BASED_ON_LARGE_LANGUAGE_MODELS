from pathlib import Path
import tempfile
import unittest

import unit03_chat_engine as c


class ThreeTurnBackend:
    init_calls = 0
    inspect_calls = []
    generate_calls = []
    info = {"backend": "test_only", "is_test_double": True, "model": "TEST_3TURN"}

    def __init__(self, config):
        type(self).init_calls += 1
        self.config = config

    def inspect_input(self, messages):
        type(self).inspect_calls.append([dict(m) for m in messages])
        return {
            "input_tokens": 100 + 20 * len(messages),
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
        return f"MODEL_T{n}", {"input_tokens": 1, "output_tokens": 3, "finish_reason": "eos"}


class FailSecondBackend(ThreeTurnBackend):
    inspect_calls = []
    generate_calls = []

    def generate(self, messages):
        type(self).generate_calls.append([dict(m) for m in messages])
        if len(type(self).generate_calls) == 2:
            raise RuntimeError("boom-t2")
        return "MODEL_T1", {"input_tokens": 1, "output_tokens": 3, "finish_reason": "eos"}


class OverflowSecondBackend(ThreeTurnBackend):
    inspect_calls = []
    generate_calls = []

    def inspect_input(self, messages):
        type(self).inspect_calls.append([dict(m) for m in messages])
        second = len(type(self).inspect_calls) == 2
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
        return "MODEL_T1", {"input_tokens": 1, "output_tokens": 3, "finish_reason": "eos"}


class ModelLoadFailBackend:
    def __init__(self, config):
        raise RuntimeError("cannot-load")


class C07Tests(unittest.TestCase):
    def setUp(self):
        ThreeTurnBackend.init_calls = 0
        ThreeTurnBackend.inspect_calls.clear()
        ThreeTurnBackend.generate_calls.clear()
        FailSecondBackend.inspect_calls.clear()
        FailSecondBackend.generate_calls.clear()
        OverflowSecondBackend.inspect_calls.clear()
        OverflowSecondBackend.generate_calls.clear()

    def _run(self, backend):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "run"
        return c.run_three_turns(
            c.DEFAULT_QUESTIONS,
            "GOUT_MT_001",
            c.DEFAULT_CONFIG,
            c.DEFAULT_PROMPT,
            out,
            backend_factory=backend,
        )

    def test_c07_success_is_three_ordered_predictions(self):
        result = self._run(ThreeTurnBackend)
        rows = result["predictions"]
        self.assertEqual([r["turn_id"] for r in rows], [1, 2, 3])
        self.assertEqual([r["status"] for r in rows], ["ok", "ok", "ok"])
        self.assertTrue(result["manifest"]["c07_structure_verified"])
        self.assertTrue(result["manifest"]["c07_successful_three_turn_run"])
        self.assertEqual(result["manifest"]["successful_turns"], 3)
        self.assertEqual(len(result["history"]), 6)
        self.assertEqual(ThreeTurnBackend.init_calls, 1)

    def test_c07_each_turn_sees_all_previous_real_outputs(self):
        result = self._run(ThreeTurnBackend)
        rows = result["predictions"]
        self.assertEqual(rows[1]["history_before"][-1]["content"], "MODEL_T1")
        self.assertEqual(rows[2]["history_before"][-1]["content"], "MODEL_T2")
        self.assertEqual(rows[2]["call"]["history_turns"], 2)
        roles = [m["role"] for m in rows[2]["call"]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant", "user"])

    def test_c07_generation_error_marks_later_turn_skipped(self):
        result = self._run(FailSecondBackend)
        rows = result["predictions"]
        self.assertEqual([r["status"] for r in rows], ["ok", "error", "skipped"])
        self.assertEqual(rows[1]["error_stage"], "generation")
        self.assertEqual(rows[2]["skip_reason"]["depends_on_turn_id"], 2)
        self.assertEqual(len(FailSecondBackend.generate_calls), 2)
        self.assertEqual(result["manifest"]["successful_turns"], 1)
        self.assertEqual(result["manifest"]["skipped_turns"], 1)
        self.assertTrue(result["manifest"]["dependency_skip_verified"])
        self.assertFalse(result["manifest"]["c07_successful_three_turn_run"])

    def test_c07_input_overflow_skips_third_without_generation(self):
        result = self._run(OverflowSecondBackend)
        rows = result["predictions"]
        self.assertEqual([r["status"] for r in rows], ["ok", "error", "skipped"])
        self.assertEqual(rows[1]["error_stage"], "input_budget")
        self.assertEqual(len(OverflowSecondBackend.inspect_calls), 2)
        self.assertEqual(len(OverflowSecondBackend.generate_calls), 1)
        self.assertEqual(rows[2]["history_before"], rows[1]["history_after"])

    def test_c07_skipped_turn_never_enters_history(self):
        result = self._run(FailSecondBackend)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][-1]["content"], "MODEL_T1")
        self.assertNotIn("boom-t2", str(result["history"]))

    def test_c07_model_load_failure_still_records_three_ordered_rows(self):
        result = self._run(ModelLoadFailBackend)
        rows = result["predictions"]
        self.assertEqual([r["turn_id"] for r in rows], [1, 2, 3])
        self.assertEqual([r["status"] for r in rows], ["error", "skipped", "skipped"])
        self.assertEqual(rows[0]["error_stage"], "model_load")
        self.assertEqual(result["manifest"]["status"], "failed_model_load")
        self.assertTrue(result["manifest"]["dependency_skip_verified"])


if __name__ == "__main__":
    unittest.main()
