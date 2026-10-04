# C05–C06 — history thật ở lượt kế tiếp + kiểm tra token budget

## Mục tiêu

- **C05:** chạy đúng một case multi-turn: lượt 1 sinh output thật, output đó được append vào `history`, rồi lượt 2 nhận `system -> user T1 -> assistant T1 -> user T2`. Không dùng output của case ST dù câu mở đầu giống nhau.
- **C06:** trước `model.generate()`, backend đếm token bằng đúng tokenizer/chat template. Nếu vượt `max_input_tokens` hoặc context window thì dừng với policy `reject_no_truncation`; không tự cắt system prompt, history hay câu hỏi cuối.

## Chưa làm ở bản này

- Chưa chạy đủ 3 lượt và chưa có logic `skipped` cho lượt phụ thuộc khi lỗi — đó là **C07**.
- `contexts` vẫn phải là `[]`; RAG chỉ bắt đầu ở D11.
- Chưa đánh giá chất lượng/y khoa của câu trả lời.

## API chính

```python
from unit03_chat_engine import create_chat_engine, new_history, append_turn

engine = create_chat_engine(
    config_path=Path("configs/unit03_hf_locked_from_b.json")
)
history = new_history()

r1 = engine.chat("Câu lượt 1", history, [])
history = append_turn(history, "Câu lượt 1", r1["answer"], status="ok")

r2 = engine.chat("Câu lượt 2", history, [])
```

`r2["call"]["messages"]` phải chứa đủ history. `r2["call"]["input_budget"]` ghi `input_tokens`, giới hạn, `fits` và policy.

## Chạy kiểm thử kỹ thuật

```bash
python -m unittest discover -s tests -v
```

Bản đóng gói này đạt **40/40 tests** trước khi tạo ZIP.

## Chạy thật trên Colab

Mở `notebooks/Gout_C05_C06_Colab.ipynb`, bật GPU, upload ZIP này và chạy từ trên xuống. Notebook dùng model/revision đã khóa từ bước B và chạy **GOUT_MT_001 T1 -> T2**.

Kết quả tải về: `Gout_C05_C06_real_result.zip`. Gửi file này lại trước khi sang **C07**.
