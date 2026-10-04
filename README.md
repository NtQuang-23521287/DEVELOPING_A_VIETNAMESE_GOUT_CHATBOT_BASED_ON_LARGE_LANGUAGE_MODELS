# Gout LLMOps — xây từng phần nhỏ

## Trạng thái mới nhất: C01–C06

C05–C06 đã được triển khai trong `unit03_chat_engine.py`: lượt kế tiếp dùng history thật của cùng multi-turn case và input được preflight theo policy `reject_no_truncation`. Xem `README_C05_C06.md` và `notebooks/Gout_C05_C06_Colab.ipynb`. C07 chưa làm.


> Cập nhật C03–C04: C02 đã được người dùng chạy inference thật qua `chat_engine`. C03 tạo history mới độc lập; C04 append đúng output thật của C02. Xem **README_C03_C04.md**.

Bắt đầu ở đây. Bạn chưa cần biết cách SFT/DPO, chưa cần GPU hay cài thư viện Python ngoài.

**Mục tiêu duy nhất của lần này:** đọc dữ liệu đã có và lấy đúng một nhóm gồm một câu đơn lượt và một hội thoại ba lượt. Kết quả là bốn lượt câu hỏi để sau này thử chatbot.

## Chạy trên Windows

1. Giải nén gói.
2. Mở Terminal trong thư mục `gout_units_v01` có file `unit01_data.py`.
3. Kiểm tra Python:

```powershell
py --version
```

Cần Python 3.10 trở lên. Nếu máy dùng lệnh `python` thay cho `py`, thay toàn bộ `py` bằng `python`; trên Linux có thể dùng `python3`.

4. Xem kết quả ngay, chưa ghi file:

```powershell
py unit01_data.py
```

5. Lưu kết quả:

```powershell
py unit01_data.py --out runs/unit01
```

Kết quả cần nhìn thấy:

```text
status: DATA_ONLY_NO_MODEL
full_dataset: 116 cases, 58 groups, 232 turns
selected: 2 cases, 1 group, 4 turns
selected_group_id: GOUT_ST_001
model_called: false
```

Đầu ra thật được in dưới dạng JSON, sau đó là bốn câu hỏi. Ví dụ đã kiểm tra nằm trong `examples/unit01/`.

## Hiểu bốn dòng đầu ra

| Dòng | Nội dung | Khi chạy model sau này |
|---|---|---|
| GOUT_ST_001__T1 | Câu hỏi đơn lượt | Lịch sử rỗng |
| GOUT_MT_001__T1 | Câu mở đầu hội thoại | Lịch sử rỗng của hội thoại mới |
| GOUT_MT_001__T2 | Câu hỏi tiếp theo | Nhìn thấy câu trả lời thật của MT lượt 1 |
| GOUT_MT_001__T3 | Câu hỏi thứ ba | Nhìn thấy hai lượt MT trước |

Câu mở đầu ST/MT trùng nhau là có chủ đích. Không lấy câu trả lời ST đưa vào lịch sử MT. Script hiện chỉ đọc câu hỏi, chưa tạo history và chưa gọi model.

## Mở file nào?

| File | Để làm gì? |
|---|---|
| `docs/LO_TRINH_UNIT.md` | Xem toàn bộ các khối, unit, phụ thuộc và cách nghiệm thu |
| `unit01_data.py` | Mã nguồn khối A; comment gắn A01–A12 |
| `data/eval/inputs.jsonl` | 116 case lấy nguyên từ bản dữ liệu v0.2 |
| `runs/unit01/questions.jsonl` | Bốn lượt câu hỏi sau khi bạn chạy lệnh lưu |
| `runs/unit01/manifest.json` | Số lượng và checksum đầu vào |
| `tasks.json` | Danh sách công việc cho chương trình/AI khác đọc |
| `progress.json` | Đã làm gì và bước nào tiếp theo |

## Chạy lại hoặc chọn nhóm khác

Để xem lại, chỉ cần `py unit01_data.py`. Nếu muốn lưu một lần chạy mới, chọn thư mục mới:

```powershell
py unit01_data.py --group-id GOUT_ST_002 --out runs/unit02
```

Thư mục output đã tồn tại sẽ được báo lỗi để tránh ghi đè. Không cần xóa dữ liệu cũ.

## Các lỗi thường gặp

| Thông báo/tình huống | Cách xử lý |
|---|---|
| Không tìm thấy lệnh py | Thử `python --version`; cần cài Python nếu cả hai lệnh đều không có |
| Không tìm thấy unit01_data.py | Mở Terminal trong thư mục chứa file này |
| Output đã tồn tại / FileExistsError | Đổi `--out` thành `runs/unit01_lan2` |
| Báo ground_truth/reference lẫn trong input | Dùng file inputs.jsonl đã tách reference, không trỏ vào test gốc có đáp án |
| Thiếu single hoặc multi trong nhóm | Kiểm tra nguồn dữ liệu; bản này dành cho bộ ST/MT ghép cặp của bạn |

## Kiểm tra kỹ thuật

```powershell
py -m unittest discover -s tests -v
```

Bản C03–C04 hiện có **32 kiểm thử kỹ thuật** cho A, B và C01–C04. Chúng kiểm tra cả reset history giữa case, thứ tự `user -> assistant`, không append lượt lỗi và không lấy `ground_truth/reference`. Không có kiểm thử chất lượng y khoa.

## Việc tiếp theo

Bước B và C01–C04 đã hoàn tất kỹ thuật. `unit03_chat_engine.py` hiện có `new_history()` và `append_turn(...)`; bằng chứng C04 được tạo từ chính output model thật của C02. Bước tiếp theo là **C05**: đưa history này vào lượt kế tiếp để model thực sự nhìn thấy câu trả lời lượt 1; sau đó **C06** kiểm tra giới hạn độ dài đầu vào.

Nguồn dữ liệu: gói kế thừa v0.2, từ repository `NtQuang-23521287/Large_Language_Models_in_the_Vietnamese_Gout_Domain` tại commit đã kiểm kê `a5a01e5dfb9691a700d69853065765e7fb280e4b` và hai file test 58 mẫu bạn đã cung cấp. File trong gói này không chứa reference hay đáp án. Kiểm tra schema không chứng minh hết mọi khả năng rò rỉ nội dung hoặc trùng ngữ nghĩa.


## Trạng thái cập nhật 2026-10-04

- A01–A12: hoàn tất.
- B01–B08: inference thật đã có bằng chứng.
- C01–C07: hoàn tất; C07 đã chạy đủ hội thoại 3 lượt thật.
- D01–D02: hoàn tất kỹ thuật về source registry/provenance; **clinical review vẫn pending**.
- Tiếp theo: D03–D04.

Xem `README_D01_D02.md`.
