# Data & Artifact Versioning

## Git

Track source code, YAML config, prompts, rubrics, docs, tests và lightweight manifests.

## DVC

Track raw/processed datasets, FAISS indexes, SFT/DPO datasets và artifacts lớn. DVC hash phải được ghi hoặc liên kết với run.

## MLflow

Một run = một model checkpoint/config chạy trên một protocol đã khóa. Log params, operational metrics và artifact directory.

## Naming

- KB: `gout_kb_v1`, `gout_kb_v2`, ...
- Testset: `group3_candidate_v02`, sau review có thể thành `gout_eval_v1`.
- Prompt: `system_vi_v1`, `judge_vi_v1`.
- Protocol: `baseline_selection_v1`.

Không overwrite một version đã dùng trong báo cáo; tạo version mới.
