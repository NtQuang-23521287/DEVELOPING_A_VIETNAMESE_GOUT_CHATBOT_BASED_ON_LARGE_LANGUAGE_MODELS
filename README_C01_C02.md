# C01–C02 — tạo `chat_engine` dùng lại luồng bước B

Bạn đã chạy xong bước B bằng model thật. Kết quả B cho thấy `GOUT_ST_001__T1` đã đi qua Qwen/Qwen2.5-1.5B-Instruct và có prediction `status=ok`. Bản C01–C02 này **không đổi model, prompt hay generation config**; mục tiêu chỉ là tạo một API chung để các bước hội thoại, RAG và đánh giá gọi về sau. Để lần chạy C02 thật dùng đúng trọng số đã thấy ở B, gói có `configs/unit03_hf_locked_from_b.json`, trong đó `revision` được khóa về commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` lấy từ manifest B.

## C01 làm gì?

C01 chốt contract:

```python
chat(question, history, contexts)
```

Đầu ra:

```python
{
    "answer": "...",
    "sources": [],
    "call": {...}
}
```

Không có tham số `reference`, `ground_truth` hay đáp án chuẩn. Reference chỉ được ghép ở bước đánh giá E, không đi vào chat engine.

Ở đúng mốc C01–C02:

- `history` phải là `[]`; lịch sử thật làm từ C03–C05.
- `contexts` phải là `[]`; RAG/context thật làm từ D11.
- `sources=[]` vì chưa có RAG.
- Nếu truyền history/context không rỗng, engine báo lỗi rõ thay vì âm thầm bỏ qua.

## C02 làm gì?

`unit03_chat_engine.py` không viết lại luồng inference. Nó tái sử dụng trực tiếp:

- `unit02_model.load_config()`
- `unit02_model.build_messages()` — đúng logic B03/B04
- `unit02_model.HFBackend` — đúng backend B05
- `backend.generate(messages)` — đúng logic B06

Như vậy, câu đơn lượt trước đây chạy qua bước B nay có thể gọi qua một hàm chung từ Python.

## Kiểm tra nhanh, chưa tải model

Trong thư mục `gout_units_v01`:

```powershell
py -m unittest discover -s tests -v
```

Các test C01–C02 dùng backend thử có đánh dấu rõ `test_only`; chúng kiểm tra contract và orchestration, không giả vờ là inference LLM thật.

## Chạy C02 bằng model thật trên Colab

Mở `notebooks/Gout_C01_C02_Colab.ipynb`, bật GPU rồi chạy từ trên xuống. Notebook sẽ:

1. upload ZIP bản C01–C02 này;
2. cài dependency giống bước B;
3. chạy unit test;
4. nạp đúng model/config B;
5. gọi `ChatEngine.chat(question, [], [])` cho `GOUT_ST_001__T1`;
6. đóng gói `runs/c02_real` để bạn gửi lại.

Hoặc trên máy đã cài môi trường bước B:

```powershell
py unit03_chat_engine.py --config configs/unit03_hf_locked_from_b.json --out runs/c02_real
```

Kết quả gồm:

- `predictions.jsonl`: question, answer, `sources=[]`, call metadata;
- `manifest.json`: checksum code/prompt/data, backend/revision và trạng thái.

## Nghiệm thu C01–C02

C01 đạt khi:

- chữ ký đúng `chat(question, history, contexts)`;
- không có reference trong public API;
- output có `answer`, `sources`, `call`.

C02 đạt khi:

- engine gọi đúng `build_messages()` của B;
- dùng cùng backend B, không tạo logic inference thứ hai;
- backend được nạp một lần và có thể gọi lại;
- một câu chạy qua engine cho answer thật và metadata;
- chưa đưa history hay RAG vào trước các unit tương ứng.

## Bước sau

Sau khi có `c02_real` `status=ok`, làm C03–C04: tạo `history=[]` cho case mới và thêm **câu trả lời thật của model** vào lịch sử sau lượt đầu. Chưa chạy luôn C05–C07 nếu chưa kiểm tra C03–C04.


> Trạng thái mới: C02 đã được chạy thật; C03-C04 đã hoàn tất trong README_C03_C04.md.
