# Bước B — Một câu hỏi → một model → một câu trả lời

Bạn đã hiểu cách lấy câu hỏi ở A. Lần này chỉ làm thêm một việc: **gửi một câu đơn lượt tới model và lưu câu trả lời thật**.

Luồng: `questions.jsonl` → chọn `GOUT_ST_001__T1` → tạo `messages` → nạp model → sinh câu trả lời → `predictions.jsonl`.

## Những gì đã chuẩn bị

- `unit02_model.py`: chọn câu hỏi, tạo messages, nạp model, sinh câu trả lời và lưu kết quả/lỗi.
- `configs/unit02_hf.json`: cấu hình model dùng thử.
- `prompts/system_vi_v1.txt`: chỉ dẫn vai trò chatbot.
- `notebooks/Gout_Buoc_B_Colab.ipynb`: chạy từng ô trên Colab.

Model mẫu là `Qwen/Qwen2.5-1.5B-Instruct`, có hướng dẫn chính thức dùng với Transformers và hỗ trợ tiếng Việt theo model card. Đây là lựa chọn để thử nối luồng; chưa có kết quả benchmark chứng minh phù hợp nhất với Gout. Chỉ thử model này trước, chưa mở rộng nhiều backend.

## 1. Hiểu từng unit đang làm

| Unit | Việc thực hiện | Chỗ trong code |
|---|---|---|
| B01 | Dùng backend Hugging Face để thử | `HFBackend` |
| B02 | Ghi model và xác định commit trọng số | `ModelConfig`, `resolved_revision` |
| B03 | Đọc system prompt | `prompts/system_vi_v1.txt` |
| B04 | Ghép system + user thành messages | `build_messages()` |
| B05 | Tải tokenizer và model | `HFBackend.__init__()` |
| B06 | Sinh phần câu trả lời mới | `HFBackend.generate()` |
| B07 | Ghi kết quả vào JSONL | `run_once()` |
| B08 | Bạn đọc output và nhận xét | Sau khi chạy thật |

Các unit B01–B07 đã có code. Phần model thật chưa chạy trong môi trường tạo gói vì nơi này không có PyTorch/Transformers. Kiểm tra bằng backend thử chỉ xác nhận luồng dữ liệu và ghi lỗi, không xác nhận khả năng nạp trọng số hoặc chất lượng model. B08 vẫn chờ output thật của bạn.

## 2. Xem prompt ngay trên máy, chưa cần cài model

Từ thư mục `gout_units_v01`:

```powershell
py unit02_model.py --preview --out runs/b_preview
```

Mở `runs/b_preview/request.json`. Phần `messages` chỉ gồm:

```json
[
  {"role": "system", "content": "Chỉ dẫn vai trò chatbot..."},
  {"role": "user", "content": "Giai đoạn 1 của bệnh gút trên lâm sàng được gọi là gì và đặc điểm của nó?"}
]
```

Đoạn trên rút gọn system prompt để giải thích. File request thật chứa đầy đủ nội dung đã gửi. Trường `sample` của request là metadata để bạn kiểm tra; chỉ `messages` đi vào model. Không đưa category, group_id hoặc đáp án chuẩn vào prompt.

Preview không tải model, không gọi model, không tạo predictions giả.

## 3. Chạy trên Colab

1. Tải notebook `Gout_Buoc_B_Colab.ipynb` và gói ZIP bản đã cập nhật.
2. Mở https://colab.research.google.com/ và chọn **Upload notebook** để mở file `.ipynb`.
3. Chọn **Runtime → Change runtime type → GPU**; dùng GPU khả dụng, chẳng hạn T4 nếu được cấp.
4. Chạy từng ô từ trên xuống. Ô tải dữ liệu sẽ yêu cầu chọn `Gout_Build_tung_unit_v01.zip` **bản mới có unit02_model.py**.
5. Xem câu hỏi/prompt trước, sau đó chạy ô nạp model và sinh câu trả lời.
6. Chạy ô cuối để tải gói kết quả về máy.

Colab có thể chưa cấp được GPU hoặc ngắt phiên. Nếu chưa có GPU, dừng ở ô kiểm tra tài nguyên và báo lại; notebook không âm thầm chuyển sang chạy CPU lâu. Kết quả trên máy Colab là tạm thời nên tải về sau lần chạy. Không cần mở API server, tài khoản API trả phí hay thiết lập Drive cho bài thử này.

Lần tải trọng số đầu tiên cần mạng và thời gian. Dung lượng tải không nằm trong ZIP code. Notebook cài Transformers 4.57.1 và dùng PyTorch đã có trên Colab; phiên bản thực tế được ghi lại trong manifest. Đây chưa phải môi trường đã được nghiệm thu trên mọi máy.

## 4. Nếu chạy trực tiếp trên máy đã có PyTorch

Máy cần đủ RAM/GPU cho model. Với máy riêng, cài PyTorch phù hợp hệ thống theo https://pytorch.org/get-started/locally/ trước; gói không tự chọn CUDA cho máy bạn.

```powershell
py -m pip install -r requirements-unit02.txt
py unit02_model.py --out runs/b_first
```

Mặc định dùng bốn câu mẫu đúng đầu ra A đã kèm gói. Nếu bạn đã có kết quả chạy A tại `runs/unit01/questions.jsonl`:

```powershell
py unit02_model.py --questions runs/unit01/questions.jsonl --sample-id GOUT_ST_001__T1 --out runs/b_first_from_my_data
```

`device=auto` chọn CUDA nếu có, nếu không sẽ dùng CPU. CPU có thể chậm; không coi thời gian đó như tốc độ GPU. Với cấu hình mẫu này, CPU dùng float32 và CUDA dùng float16; so sánh khoa học sau này phải cố định chế độ suy luận.

## 5. Kết quả có gì?

| File | Nội dung |
|---|---|
| `request.json` | Câu được chọn, messages đầy đủ, cấu hình sinh |
| `predictions.jsonl` | Một câu trả lời thật hoặc bản ghi lỗi; không có điểm chất lượng |
| `manifest.json` | Model/revision thực tế, code/prompt/data checksum, môi trường, trạng thái |

Các trường cần xem đầu tiên trong prediction:

- `sample_id`: đúng câu hỏi nào.
- `question`: câu người dùng gửi.
- `answer`: câu trả lời nguyên văn của model.
- `status`: `ok` hoặc `error`.
- `model_meta.finish_reason`: `eos`, `length` hoặc `other`. Nếu `length`, câu có thể bị cắt do giới hạn token.
- `generation_ms`: thời gian bước sinh tính ở phía chương trình, gồm định dạng/tokenize/decode; tách khỏi thời gian tải model.

`revision=main` trong cấu hình được giải thành commit trước khi tải. Tokenizer và model nạp cùng commit; `resolved_revision` lưu trong manifest. Khi cần chạy lại đúng trọng số, thay `revision` bằng commit đã lưu. Seed và greedy decoding không bảo đảm mọi máy cho kết quả giống từng byte.

Không kỳ vọng câu trả lời giống đáp án chuẩn từng chữ. Ở bước này, chỉ kiểm tra đúng câu hỏi được gửi và có đầu ra thật; việc chấm đúng/sai theo bằng chứng thuộc khối E.

## 6. Đọc lỗi

| Lỗi | Cách xử lý |
|---|---|
| ZIP không có unit02_model.py | Tải lại ZIP đã cập nhật ở tin nhắn bước B |
| GPU chưa sẵn sàng | Chọn GPU trong runtime; nếu không được cấp thì báo lại |
| Tải Hugging Face thất bại | Kiểm tra mạng/quyền truy cập; giữ manifest và thông báo lỗi |
| CUDA out of memory | Khởi động lại phiên sạch, chỉ nạp một model; giảm token nếu phù hợp rồi ghi cấu hình mới |
| Import thư viện thất bại | Xem traceback/phiên bản; restart runtime sau thay thư viện khi cần |
| Output đã tồn tại | Chọn thư mục `--out` mới; không xóa run cũ |
| Chọn câu đa lượt | B chỉ nhận single lượt 1; các lượt MT cần history sẽ làm ở C |

Khi model tải/sinh lỗi, `answer=null`, `status=error`, có `error_stage`. Đừng thay lỗi bằng câu viết sẵn. Notebook vẫn cho tải gói log ở ô cuối sau khi lỗi để kiểm tra.

## 7. Khi nào chuyển sang C?

Khi bạn nhìn thấy một prediction thật với `status=ok`, câu hỏi khớp đầu vào, câu trả lời không rỗng, manifest có revision và file kết quả đã tải về. Sau đó gửi lại câu trả lời hoặc gói kết quả; mình sẽ hướng dẫn B08 và ghép các hàm thành `chat_engine`, rồi thêm history cho ba lượt.

## Tài liệu đối chiếu

- Model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Chat template: https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating
- Colab: https://research.google.com/colaboratory/faq.html

Gói chỉ có prompt giới hạn vai trò, chưa có bộ kiểm soát y khoa đã kiểm chứng. Chưa RAG, judge, SFT, DPO hoặc giao diện chatbot.
