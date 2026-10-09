# KB Alignment & Reference Review

Candidate testset hiện có 130 lượt nhưng chưa có gold/reference đã duyệt. Sau khi build KB:

```bash
python scripts/export_kb_alignment_review.py
```

Script tạo `artifacts/review/testset_kb_alignment_v1.jsonl`. Mỗi dòng chứa question + top retrieved contexts và các trường review rỗng.

Reviewer cần xác định:
- `alignment_decision`: `supported`, `partially_supported`, `ood_expected_refusal`, hoặc `reject`;
- `reference_answer`: chỉ viết từ bằng chứng được chấp nhận;
- `required_points`: các ý bắt buộc;
- `evidence_chunk_ids`: chunk IDs hỗ trợ;
- `review_status`: chỉ đổi thành `approved` khi đã duyệt.

Không dùng file reference này làm input cho generator hoặc retriever.
