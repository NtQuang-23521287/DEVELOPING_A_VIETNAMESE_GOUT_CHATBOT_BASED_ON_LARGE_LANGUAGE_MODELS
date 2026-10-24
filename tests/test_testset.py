from gout_llmops.data.testset import load_cases, summarize_cases


def test_candidate_testset_structure():
    cases = load_cases("data/data_filtered_final/Group3_Testset_cases_v02.jsonl")
    s = summarize_cases(cases)
    assert s["cases"] == 316
    assert s["turns"] == 316
    assert s["by_scenario"]["single"] == 316
    assert s["by_scenario"].get("multi", 0) == 0
    assert len({c.source_row_id for c in cases}) == 316
