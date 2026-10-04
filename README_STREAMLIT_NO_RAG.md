# Streamlit + FastAPI prototype — chưa tích hợp RAG

## Mục tiêu

Tạo hệ thống chạy được theo luồng:

`Streamlit -> HTTP POST /api/chat -> FastAPI -> ChatService -> ChatEngine C01-C07 -> Qwen -> answer -> Streamlit`

Knowledge Base vẫn được giữ trong project nhưng **không được đọc, truy xuất hay đưa vào prompt** ở bản này.

## Thành phần

- `app_service.py`: quản lý session/history và dùng một `ChatEngine`/model instance duy nhất.
- `backend_api.py`: API FastAPI.
- `run_backend.py`: chạy backend ở `127.0.0.1:8000`.
- `streamlit_app.py`: giao diện chat Streamlit ở `127.0.0.1:8501`.
- `run_system.sh`: chạy cả backend + Streamlit trên Linux/macOS/WSL.
- `tests/test_app_service.py`: test session/history/no-RAG không cần tải model.

## API

### `GET /health`
Kiểm tra backend và xem model đã nạp chưa.

### `POST /api/chat`

```json
{
  "session_id": "abc123",
  "question": "Bệnh gút là gì?"
}
```

Backend luôn gọi:

```python
engine.chat(question, history, [])
```

`contexts=[]`, vì RAG chưa được tích hợp.

### `POST /api/reset`
Xóa history của một session.

## Chạy trên máy có Python

Từ thư mục `gout_units_v01`:

```bash
pip install -r requirements-app.txt
```

Nếu máy chưa có PyTorch, cài bản PyTorch phù hợp CPU/CUDA trước.

Cách nhanh nhất — một lệnh, dùng được Windows/Linux/macOS:

```bash
python run_system.py
```

Sau đó mở `http://127.0.0.1:8501`.

Cách 2 — hai terminal:

```bash
python run_backend.py
```

Terminal thứ hai:

```bash
streamlit run streamlit_app.py
```

Mở trình duyệt tại `http://127.0.0.1:8501`.

Cách 3 — Linux/macOS/WSL:

```bash
./run_system.sh
```

## Hành vi hội thoại

- Model chỉ nạp một lần ở backend và được tái sử dụng.
- Mỗi trình duyệt Streamlit có một `session_id` riêng.
- Lượt sau dùng output model của lượt trước làm history.
- Nếu generation lỗi, backend trả lỗi HTTP; không tạo câu trả lời giả và không append history lỗi.
- Nút **Xóa hội thoại** xóa history backend và tạo session mới ở UI.

## Lưu ý về roadmap

Đây là **prototype UI/API sớm theo yêu cầu**, không đánh dấu H04-H06 là hoàn tất nghiệm thu vì roadmap gốc đặt H sau khi RAG/model/evaluation/release đã được khóa. Khi tới H, có thể tái sử dụng code này và kiểm tra lại với cấu hình phát hành cuối.

## Model config

Backend mặc định dùng `configs/unit03_hf_locked_from_b.json`, tức Qwen2.5-1.5B-Instruct tại commit đã dùng trong C02-C07. Có thể override bằng biến môi trường `GOUT_MODEL_CONFIG`, nhưng không nên đổi khi đang kiểm chứng prototype.
