# Evaluation Protocol v1

## Mục tiêu

Chọn mô hình nền trong cùng điều kiện RAG trước SFT. Protocol phải được khóa trước full run.

## Đơn vị đánh giá

- Single-turn: 1 case = 1 prediction.
- Multi-turn: 1 case = 3 prediction có thứ tự; T2/T3 dùng output thật của model trước đó.
- Nếu một turn multi lỗi, các turn phụ thuộc sau đó là `skipped_dependency_failure`, không biến thành điểm 0 giả.

## Fair comparison

Giữ cố định testset checksum, KB manifest checksum, system prompt checksum, retrieval config, token limits và evaluator version. Không tuning riêng một model rồi so với model khác bằng config cũ.

## Candidate testset hiện tại

Group3 v02 đã sạch cấu trúc nhưng `reference_status=pending_kb_alignment`. Có thể dùng để smoke-test generation/retrieval; chưa nên báo cáo các điểm faithfulness/completeness như ground-truth clinical benchmark cho đến khi reference/evidence được duyệt.

## 9 tiêu chí

6 quality: Faithfulness, Completeness, Relevance, Context Recall, Patient Utility, Refusal Appropriateness.

3 risk: Hallucination Level, Omission Risk, Safety Risk.

Risk có hướng lower-is-better. Fatal safety violation được ghi riêng, không che bằng average score.
