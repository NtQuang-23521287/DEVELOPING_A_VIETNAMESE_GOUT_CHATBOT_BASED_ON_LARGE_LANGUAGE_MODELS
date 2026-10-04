import unittest

from app_service import ChatService


class FakeEngine:
    def __init__(self):
        self.calls = []

    def chat(self, question, history, contexts):
        self.calls.append((question, [dict(x) for x in history], list(contexts)))
        return {
            "answer": f"A:{question}",
            "sources": [],
            "call": {"history_turns": len(history) // 2, "rag_enabled": False},
        }


class FailingEngine:
    def chat(self, question, history, contexts):
        raise RuntimeError("boom")


class ChatServiceTests(unittest.TestCase):
    def test_engine_loaded_once_and_history_reused(self):
        engine = FakeEngine()
        factory_calls = []
        def factory():
            factory_calls.append(1)
            return engine

        service = ChatService(factory)
        r1 = service.chat("s1", "Q1")
        r2 = service.chat("s1", "Q2")
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(r1["history_turns"], 1)
        self.assertEqual(r2["history_turns"], 2)
        self.assertEqual(engine.calls[0][1], [])
        self.assertEqual(engine.calls[1][1], [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A:Q1"},
        ])
        self.assertEqual(engine.calls[1][2], [])

    def test_sessions_are_isolated(self):
        engine = FakeEngine()
        service = ChatService(lambda: engine)
        service.chat("A", "qa")
        service.chat("B", "qb")
        self.assertEqual(service.get_history("A")[0]["content"], "qa")
        self.assertEqual(service.get_history("B")[0]["content"], "qb")
        self.assertEqual(len(service.get_history("A")), 2)
        self.assertEqual(len(service.get_history("B")), 2)

    def test_reset_clears_only_one_session(self):
        service = ChatService(lambda: FakeEngine())
        service.chat("A", "qa")
        service.chat("B", "qb")
        service.reset("A")
        self.assertEqual(service.get_history("A"), [])
        self.assertEqual(len(service.get_history("B")), 2)

    def test_failed_generation_does_not_append_history(self):
        service = ChatService(lambda: FailingEngine())
        with self.assertRaises(RuntimeError):
            service.chat("A", "qa")
        self.assertEqual(service.get_history("A"), [])

    def test_rag_and_reference_are_not_exposed(self):
        service = ChatService(lambda: FakeEngine())
        out = service.chat("A", "qa")
        self.assertFalse(out["rag_enabled"])
        self.assertFalse(out["reference_used"])
        self.assertEqual(out["sources"], [])


if __name__ == "__main__":
    unittest.main()
