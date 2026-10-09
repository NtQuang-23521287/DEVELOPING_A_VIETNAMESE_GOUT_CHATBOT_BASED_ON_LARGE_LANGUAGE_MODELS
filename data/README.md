# Data policy

`raw/` giữ nguyên nguồn đầu vào. Không sửa trực tiếp file raw sau khi đã version; nếu cần làm sạch, tạo artifact ở `interim/` hoặc `processed/` và ghi manifest.

- `raw/knowledge_base/`: tài liệu/corpus để retrieval.
- `raw/testset/`: candidate evaluation inputs.
- `processed/`: outputs deterministic như chunks, normalized references.
- `manifests/`: checksum + metadata.

Không dùng testset/reference làm training data.
