# Copy vào repository cũ

Khuyến nghị không copy chồng từng file cũ. Tạo branch mới rồi đặt foundation này làm root mới, sau đó mang dữ liệu/code thật sự cần thiết vào theo module.

```bash
git checkout -b refactor/llmops-foundation
```

1. Backup repository hiện tại hoặc tạo tag.
2. Copy toàn bộ nội dung `Gout_LLMOps_v1/` vào root repository.
3. Giữ notebook/code cũ trong `legacy/` tạm thời nếu cần đối chiếu.
4. Không copy `__pycache__`, outputs, model weights hay secrets.
5. `pip install -e ".[dev]"` rồi chạy `pytest -q`.
6. Khi test pass, commit foundation trước khi bắt đầu benchmark.

Suggested first commits:

```text
chore: initialize llmops foundation
feat: register gout knowledge base v1
feat: validate group3 candidate evaluation set
feat: add reproducible baseline evaluation harness
```
