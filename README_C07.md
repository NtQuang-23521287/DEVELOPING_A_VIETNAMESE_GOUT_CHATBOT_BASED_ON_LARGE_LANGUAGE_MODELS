# C07 — chạy trọn hội thoại 3 lượt

Mục tiêu theo roadmap: một case MT được chạy theo thứ tự T1 → T2 → T3 bằng cùng `chat_engine` và history thật.

- Mỗi lượt `ok` mới được append vào history.
- Nếu một lượt `error`, mọi lượt phụ thuộc phía sau được ghi `skipped` và không gọi model.
- `skipped` không được tính là thành công.
- C07 vẫn chưa dùng RAG; `contexts=[]`, `reference_used=false`.

Hàm chính:

```python
run_three_turns(questions, case_id, config_path, prompt_path, out)
```

Bộ test toàn dự án: 46/46 pass. Bằng chứng C05-C06 inference thật nằm tại `evidence/c05_c06_real_20261004/`.

Để nghiệm thu C07 bằng model thật, mở `notebooks/Gout_C07_Colab.ipynb`, bật GPU, upload ZIP này và Run All. Notebook tạo `Gout_C07_real_result.zip`.
