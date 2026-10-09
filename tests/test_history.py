from gout_llmops.chat.history import append_turn, tail_history


def test_history_keeps_real_pairs_only():
    h = []
    h = append_turn(h, "q1", "a1")
    h = append_turn(h, "q2", "a2")
    assert tail_history(h, 1) == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
