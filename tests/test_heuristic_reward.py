from gout_llmops.evaluation.heuristic_reward import compute_reward


def test_reward_penalizes_risk_and_fatal():
    cfg = {
        "quality_weights": {"faithfulness": 0.5, "relevance": 0.5},
        "risk_penalties": {"safety_risk": 0.4},
        "fatal_penalty": 2.0,
    }
    safe = compute_reward({"faithfulness": 1, "relevance": 1, "safety_risk": 0}, cfg)
    risky = compute_reward({"faithfulness": 1, "relevance": 1, "safety_risk": 1}, cfg)
    fatal = compute_reward({"faithfulness": 1, "relevance": 1, "safety_risk": 1, "fatal_safety_violation": True}, cfg)
    assert safe["reward"] > risky["reward"] > fatal["reward"]
