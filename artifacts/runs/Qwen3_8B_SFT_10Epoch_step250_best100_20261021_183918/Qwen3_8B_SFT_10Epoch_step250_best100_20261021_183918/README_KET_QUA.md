# SFT Qwen3-8B — Thí nghiệm 10 Epoch

- Tổng optimizer steps: **250**
- Best checkpoint: **step 100**
- Best eval_loss: **1.513800740**
- 197/198 train hợp lệ, 22/22 validation theo log thí nghiệm đã cung cấp; xem audit để xác nhận.
- Các đáp án y khoa là **DRAFT chưa được duyệt toàn bộ**.

## Validation history

| Epoch | Step | Eval loss |
|---:|---:|---:|
| 1 | 25 | 1.718910 |
| 2 | 50 | 1.594998 |
| 3 | 75 | 1.525193 |
| 4 | 100 | 1.513801 |
| 5 | 125 | 1.561631 |
| 6 | 150 | 1.704618 |
| 7 | 175 | 1.846391 |
| 8 | 200 | 1.981921 |
| 9 | 225 | 2.113046 |
| 10 | 250 | 2.155150 |

## Cách dùng

- Trọng số checkpoint tốt nhất: `run_artifacts/checkpoint-100/`
- Các adapter/checkpoint đều cần base model Qwen3-8B tương thích để suy luận.
- `experiment_snapshot/` chứa YAML, code và JSONL tại lúc sao lưu.
- `sha256_manifest.json` chứa thông tin xác minh file.
