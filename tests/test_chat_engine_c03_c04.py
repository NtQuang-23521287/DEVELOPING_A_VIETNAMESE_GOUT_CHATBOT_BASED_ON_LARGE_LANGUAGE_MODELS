"""Kiểm tra C03-C04: reset history và append đúng output thành công."""
from pathlib import Path
import inspect
import json
import tempfile
import unittest

import unit03_chat_engine as c


class C03C04Tests(unittest.TestCase):
    def test_c03_new_history_is_empty(self):
        self.assertEqual(c.new_history(), [])

    def test_c03_each_case_gets_distinct_history_object(self):
        case_a = c.new_history()
        case_b = c.new_history()
        self.assertIsNot(case_a, case_b)
        case_a.append({"role": "user", "content": "A"})
        self.assertEqual(case_b, [])

    def test_c04_signature_has_no_reference_or_ground_truth(self):
        sig = inspect.signature(c.append_turn)
        self.assertEqual(
            list(sig.parameters), ["history", "question", "answer", "status"]
        )
        self.assertNotIn("reference", sig.parameters)
        self.assertNotIn("ground_truth", sig.parameters)

    def test_c04_appends_user_then_real_assistant_answer(self):
        before = c.new_history()
        after = c.append_turn(
            before,
            "Câu hỏi thật",
            "MODEL_OUTPUT_THAT_WAS_ACTUALLY_RETURNED",
            status="ok",
        )
        self.assertEqual(before, [])
        self.assertEqual(
            after,
            [
                {"role": "user", "content": "Câu hỏi thật"},
                {"role": "assistant", "content": "MODEL_OUTPUT_THAT_WAS_ACTUALLY_RETURNED"},
            ],
        )

    def test_c04_error_prediction_is_not_appended(self):
        before = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        after = c.append_turn(before, "Q2", None, status="error")
        self.assertEqual(after, before)
        self.assertIsNot(after, before)

    def test_c04_ok_requires_nonempty_model_answer(self):
        with self.assertRaisesRegex(ValueError, "answer"):
            c.append_turn([], "Q", None, status="ok")

    def test_c04_rejects_broken_existing_history(self):
        broken = [{"role": "assistant", "content": "orphan"}]
        with self.assertRaises(ValueError):
            c.append_turn(broken, "Q", "A", status="ok")

    def test_history_from_prediction_uses_prediction_answer(self):
        pred = {
            "question": "Q",
            "answer": "MODEL_A",
            "status": "ok",
            "ground_truth": "REFERENCE_MUST_NOT_BE_USED",
        }
        history = c.history_from_prediction(pred)
        self.assertEqual(history[1]["content"], "MODEL_A")
        self.assertNotIn("REFERENCE_MUST_NOT_BE_USED", json.dumps(history, ensure_ascii=False))

    def test_evidence_from_real_shape_does_not_call_model(self):
        pred = {
            "sample_id": "S1",
            "question": "Q thật",
            "answer": "A model thật",
            "status": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_path = base / "predictions.jsonl"
            pred_path.write_text(json.dumps(pred, ensure_ascii=False) + "\n", encoding="utf-8")
            out = base / "evidence"
            result = c.build_c03_c04_evidence(pred_path, out)
            history = json.loads((out / "history.json").read_text(encoding="utf-8"))
            self.assertFalse(result["model_called"])
            self.assertFalse(result["reference_used"])
            self.assertTrue(result["c03"]["passed"])
            self.assertTrue(result["c04"]["appended"])
            self.assertEqual(history[-1]["content"], "A model thật")


if __name__ == "__main__":
    unittest.main()
