from gout_llmops.chat.engine import ChatEngine


class FakeRetriever:
    def retrieve(self, q):
        return [{"chunk_id": "c1", "text": "evidence", "metadata": {"source_id": "s1"}}]


class FakeBackend:
    def generate(self, messages, max_input_tokens):
        # Guard: no reference/ground truth argument exists in backend interface.
        assert messages[-1]["role"] == "user"
        return {"answer": "ok", "input_tokens": 10, "output_tokens": 1, "latency_seconds": 0.01}


def test_chat_engine_updates_history():
    e = ChatEngine(FakeBackend(), FakeRetriever(), "system", max_history_pairs=2)
    result, h = e.chat("question", [])
    assert result["answer"] == "ok"
    assert len(h) == 2
