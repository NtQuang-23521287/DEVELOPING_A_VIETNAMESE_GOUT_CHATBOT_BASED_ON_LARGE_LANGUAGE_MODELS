from gout_llmops.core.config import load_yaml


def test_model_configs_have_revision():
    for p in [
        "configs/models/qwen3_8b.yaml",
        "configs/models/seallms_v3_7b.yaml",
        "configs/models/gemma3_4b.yaml",
    ]:
        cfg = load_yaml(p)
        assert cfg["model_id"]
        assert len(cfg["revision"]) >= 8
