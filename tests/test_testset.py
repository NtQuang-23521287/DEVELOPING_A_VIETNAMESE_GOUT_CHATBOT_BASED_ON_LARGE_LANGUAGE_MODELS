from gout_llmops.data.testset import load_cases, summarize_cases


def test_candidate_testset_structure():
    cases = load_cases("data/raw/testset/Group3_Testset_cases_v02.jsonl")
    s = summarize_cases(cases)
    assert s["cases"] == 76
    assert s["turns"] == 130
    assert s["by_scenario"]["single"] == 49
    assert s["by_scenario"]["multi"] == 27
    assert len({c.source_row_id for c in cases}) == 76
