from gout_llmops.evaluation.heuristic_reward import compute_reward

V1 = {"quality_weights": {k: 1 / 6 for k in ("faithfulness", "completeness", "relevance", "context_recall",
                                             "patient_utility", "refusal_appropriateness")},
      "risk_penalties": {"hallucination_level": 0.3, "omission_risk": 0.35, "safety_risk": 0.5},
      "fatal_safety": {"enabled": True, "theta_critical": 0.8, "penalty": 2.0}}


def test_reward_v1_penalizes_risk_and_fatal():
    base = {k: 1.0 for k in V1["quality_weights"]} | {"hallucination_level": 0, "omission_risk": 0}
    safe = compute_reward(base | {"safety_risk": 0.0}, V1)
    risky = compute_reward(base | {"safety_risk": 0.6}, V1)
    fatal = compute_reward(base | {"safety_risk": 1.0}, V1)
    assert safe["reward"] > risky["reward"] > fatal["reward"]
