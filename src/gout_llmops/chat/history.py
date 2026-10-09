from __future__ import annotations


def append_turn(history: list[dict[str, str]], user: str, assistant: str) -> list[dict[str, str]]:
    if not user.strip() or not assistant.strip():
        raise ValueError("Only successful non-empty user/assistant turns may enter history")
    return [*history, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def tail_history(history: list[dict[str, str]], max_pairs: int) -> list[dict[str, str]]:
    if max_pairs < 0:
        raise ValueError("max_pairs must be >= 0")
    return history[-2 * max_pairs :] if max_pairs else []
