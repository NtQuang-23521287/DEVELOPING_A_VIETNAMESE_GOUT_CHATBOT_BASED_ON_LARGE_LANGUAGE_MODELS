# C03-C04 — tạo và cập nhật history

Mục tiêu đúng theo lộ trình:

- **C03:** một case/hội thoại mới bắt đầu bằng `history=[]`; mỗi lần tạo là một list độc lập.
- **C04:** sau một lượt **thành công**, append đúng hai message theo thứ tự `user -> assistant`, trong đó assistant là **output thật của model/prediction**, không phải reference.
- Nếu lượt lỗi (`status != "ok"`), history được giữ nguyên.
- Chưa làm **C05**: `ChatEngine.chat(...)` vẫn chưa nhận history không rỗng để sinh lượt kế tiếp.
- Chưa làm RAG: `contexts` vẫn để `[]` đến D11.

## API mới

```python
from unit03_chat_engine import new_history, append_turn

history = new_history()
history = append_turn(
    history,
    question="Câu hỏi lượt 1",
    answer="Output thật từ model",
    status="ok",
)
```

Sau C04:

```python
[
  {"role": "user", "content": "Câu hỏi lượt 1"},
  {"role": "assistant", "content": "Output thật từ model"}
]
```

Lưu ý: C04 chỉ **xây history**. Việc đưa history này trở lại `ChatEngine.chat()` để model nhìn thấy ở lượt 2 là **C05**.
