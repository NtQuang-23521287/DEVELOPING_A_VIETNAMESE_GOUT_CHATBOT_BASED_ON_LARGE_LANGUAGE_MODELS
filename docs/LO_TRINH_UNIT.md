# Lộ trình xây khóa luận LLMOps Gout từ các unit nhỏ

## 1. Đích cuối và trạng thái hiện tại

Đích cuối: một quy trình tái lập được để đánh giá và so sánh LLM/chatbot Gout tiếng Việt, thực nghiệm RAG, SFT và DPO theo đề cương, rồi demo cấu hình được chọn. Chatbot là sản phẩm để thử nghiệm; câu hỏi nghiên cứu, dữ liệu và phương pháp đánh giá mới quyết định ý nghĩa của khóa luận.

Bản này kế thừa dữ liệu đồ án chuyên ngành của bạn. Đã đối chiếu inputs.jsonl với gói Gout_LLMOps_Ke_thua_du_lieu_v02.zip: 116 case, 58 nhóm, 232 lượt. ST = single-turn, MT = multi-turn. Mỗi nhóm hiện có một ST một lượt và một MT ba lượt bắt đầu cùng câu hỏi. Bốn lượt trong một nhóm không phải bốn quan sát độc lập.

246 cặp SFT và 143 bài RAG là ứng viên kế thừa đã kiểm kê, vẫn cần rà soát. Group3 có 316 truy vấn cần làm sạch nên chưa dùng ở luồng tối thiểu. Các reference và benchmark cũ không tự trở thành test cuối chưa từng được dùng.

Đã kiểm tra khối A đọc/chọn câu hỏi. B01–B07 đã có code và notebook; chưa nghiệm thu bằng model thật. Chưa triển khai chat_engine, chưa xây index mới, chưa huấn luyện hay đánh giá y khoa. Script đánh giá nháp trong workspace có thể được tái sử dụng sau, nhưng không xem là toàn bộ khóa luận đã hoàn thành.

## 2. Hiểu bức tranh bằng ba luồng

1. **Trả lời:** câu hỏi + lịch sử + nguồn được truy xuất → chat_engine → câu trả lời và nguồn.
2. **Đánh giá:** bộ câu hỏi → nhiều lần gọi cùng chat_engine → lưu prediction → ghép reference ở bước chấm → báo cáo.
3. **Cải thiện:** baseline → xem lỗi → sửa dữ liệu/RAG/prompt hoặc SFT/DPO khi phù hợp → chạy lại trên dev.

Một chat_engine phục vụ cả chatbot và bộ đánh giá. Reference đi vào bộ chấm, không đi vào chat_engine. Model SFT/DPO là các ứng viên thay cho model nền; không phải các chatbot riêng phải viết lại từ đầu.

## 3. Một unit được coi là đủ nhỏ khi nào?

- Chỉ có một trách nhiệm quan sát được.
- Biết đầu vào nào cần có và tạo ra đầu ra gì.
- Có một cách kiểm tra hoàn thành rõ ràng.
- Có thể sửa/thay unit mà không viết lại toàn hệ thống.

Không cần chia đến từng dấu ngoặc hoặc câu lệnh Python. Với việc lặp lại, đơn vị là một mẫu: duyệt một đáp án, tạo vector một đoạn, chấm một câu. Sau khi làm đúng một mẫu, dùng vòng lặp để mở rộng. Đơn vị trong bảng là loại công việc; hoàn thành toàn bộ dữ liệu còn phải chạy đủ các mẫu và kiểm tra tỷ lệ lỗi.

## 4. Các mốc thực hiện

| Mốc | Khối | Thấy được gì? | Chưa cần làm |
|---|---|---|---|
| 0 | S | Một trang phạm vi/giả thuyết và thông tin máy | Chưa chọn cả bộ công cụ |
| 1 | A | Đọc ra đúng 4 lượt của một nhóm | Model, GPU |
| 2 | B | Một câu hỏi có một câu trả lời thật và file kết quả | RAG, SFT, DPO |
| 3 | C | Hội thoại ba lượt dùng output thực ở lượt trước | UI |
| 4 | D | Câu trả lời đi kèm bằng chứng truy xuất | Huấn luyện |
| 5 | E | Một run baseline có báo cáo; sau đó so sánh model | Test cuối |
| 6 | F | SFT nạp lại được và so sánh với baseline | Tự động chọn SFT làm bản tốt nhất |
| 7 | G | DPO nạp lại được và so sánh với SFT | Tự động chọn DPO làm bản tốt nhất |
| 8 | H | Nghiệm thu và demo đúng cấu hình đã đánh giá | Dịch vụ phục vụ quy mô lớn |
| Song song | R | Code, môi trường, phương pháp và lỗi có bằng chứng | Tự động hóa trước khi lệnh chạy đúng |

Khối S và việc chuẩn bị nhãn E01–E03 có thể làm song song với A–D. Bản kỹ thuật nhỏ có thể chạy khi nhãn còn chờ duyệt, nhưng điểm đó chưa đủ để kết luận y khoa.

## 5. Ánh xạ về năm phase trong đề cương

| Phase | Khối/unit chính | Điều kiện đầu ra |
|---|---|---|
| 1. Dữ liệu và khung đánh giá | S, A, D; E01–E03; thiết kế giao diện C | Có dữ liệu có nguồn, schema, rubric và luồng thử nhỏ |
| 2. Tuyển chọn model | B, C, E | Baseline và báo cáo so sánh trên dữ liệu phát triển đã chốt |
| 3. SFT | F + dùng lại E | Adapter/checkpoint và bằng chứng thay đổi so với baseline |
| 4. DPO | G + dùng lại E | Preference đã duyệt và báo cáo DPO vs SFT |
| 5. Nghiệm thu, triển khai | H | Cấu hình đã khóa, đánh giá cuối, demo có thể khởi động lại |

R dùng xuyên suốt. Thứ tự code A→B→C→D giúp sớm thấy chương trình chạy; không làm thay đổi cấu trúc năm phase của báo cáo.

## 6. Danh sách unit đầy đủ

Đây là bảng thiết kế, không phải lời khẳng định tất cả code đã tồn tại. Chỉ A01–A12 mang trạng thái đã kiểm tra kỹ thuật. Các mã phụ thuộc chỉ ra việc cần có trước; không bắt buộc làm mọi bảng theo thứ tự từ trên xuống.

Tổng cộng **92 unit**, gồm **12 unit khối A đã hiện thực**.

### Khối S — Chốt phạm vi nghiên cứu

Đích: Một trang mô tả đúng điều cần chứng minh.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| S01 | Viết một câu mô tả người dùng mục tiêu | Không có | Đề cương → Một câu về đối tượng dùng | Không trộn chatbot giáo dục người bệnh với công cụ kê đơn cho bác sĩ. |
| S02 | Viết danh sách loại yêu cầu được trả lời | S01 | Phạm vi đề tài → Danh sách phạm vi | Mỗi loại có một câu hỏi minh họa. |
| S03 | Viết danh sách loại yêu cầu cần hỏi lại hoặc giới hạn trả lời | S01 | Phạm vi đề tài → Danh sách tình huống giới hạn | Mỗi loại có hành vi mong đợi; chưa tự gán nhãn đúng y khoa. |
| S04 | Viết một giả thuyết so sánh | S02, S03 | Mục tiêu khóa luận → Một giả thuyết | Nêu rõ hai cấu hình cần so sánh và biến thay đổi. |
| S05 | Chép nguyên tên chín tiêu chí từ đề cương | S04 | Bản đề cương đã chốt → Danh mục tiêu chí | Không tự thay bằng chín tiêu chí khác hoặc tự đặt trọng số. |
| S06 | Ghi môi trường tính toán thực tế | Không có | Máy dự định dùng → machine_profile.json | Có OS, RAM, GPU/VRAM hoặc ghi CPU; có nơi dự định chạy model. |

### Khối A — Đọc một nhóm câu hỏi

Đích: Một nhóm → hai case → bốn lượt; chưa gọi model.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| A01 ✓ | Đọc file UTF-8 | Không có | inputs.jsonl → Chuỗi văn bản | Đọc được tiếng Việt và BOM; file không tồn tại phải báo lỗi. |
| A02 ✓ | Giải mã từng dòng JSON | A01 | Chuỗi văn bản → Danh sách object | JSON lỗi phải chỉ đúng số dòng; file rỗng bị từ chối. |
| A03 ✓ | Kiểm tra không có trường đáp án | A02 | Danh sách object → Dữ liệu qua kiểm tra | Phát hiện cả ground_truth nằm bên trong turns; không đọc file reference. |
| A04 ✓ | Kiểm tra định danh bản ghi | A03 | Một bản ghi → case_id, group_id hợp lệ | ID không rỗng, không chứa khoảng trắng; category có giá trị. |
| A05 ✓ | Kiểm tra và chuẩn hóa các lượt | A04 | Một case → Case chỉ chứa trường được phép | Giữ nguyên câu hỏi; turn_id liên tục; single có 1 lượt, multi có ≥2. |
| A06 ✓ | Phát hiện case_id trùng | A05 | Danh sách case → Danh sách không trùng ID | Thêm một case trùng phải làm kiểm tra thất bại. |
| A07 ✓ | Gom case theo group_id | A06 | Danh sách case → Bảng nhóm → case | Không mất hoặc nhân đôi case khi gom. |
| A08 ✓ | Kiểm tra liên kết ST/MT | A07 | Một nhóm kế thừa → Nhóm hợp lệ | Có đúng một single và một multi; cùng câu mở đầu và category. |
| A09 ✓ | Chọn một nhóm theo ID | A08 | Bảng nhóm và group_id → Hai case của một nhóm | Không lấy lẻ ST hoặc MT; ID không tồn tại phải báo lỗi. |
| A10 ✓ | Xuất từng lượt câu hỏi | A09 | Hai case → Bốn bản ghi lượt | Mỗi lượt giữ sample_id/case_id/group_id; không thêm đáp án hay history giả. |
| A11 ✓ | Đếm case, nhóm và lượt | A08 | Tập đầy đủ hoặc nhóm chọn → Bảng số lượng | Tập hiện tại: 116/58/232; nhóm đầu: 2/1/4. |
| A12 ✓ | Lưu kết quả vào thư mục mới | A10, A11 | Câu hỏi và bản tóm tắt → questions.jsonl + manifest.json | Có checksum đầu vào; không ghi đè output cũ; model_called=false. |

### Khối B — Một câu hỏi → một model → một câu trả lời

Đích: Mốc chạy xuyên suốt nhỏ nhất, chưa có RAG.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| B01 | Chọn một backend chạy thử | S06 | Thông tin máy và model có thể dùng → Một lựa chọn backend | Chốt một đường chạy trước: local Transformers hoặc endpoint đang có. |
| B02 | Ghi định danh model | B01 | Model đã chọn → Model ID + revision/checkpoint | Truy ra được trọng số nào đã dùng; không dùng tên mẫu REPLACE. |
| B03 | Viết system prompt bản đầu | S02, S03 | Phạm vi chatbot → system_vi_v1.txt | Nêu vai trò, giới hạn, cách xử lý thiếu thông tin; không chứa đáp án test. |
| B04 | Tạo messages cho một câu | A10, B03 | Một user question → Một system + một user message | Nội dung câu hỏi giữ nguyên; không nhận reference hoặc nhãn category làm gợi ý. |
| B05 | Khởi tạo một kết nối hoặc nạp một model | B02 | Cấu hình backend → Đối tượng model dùng lại | Lỗi cấu hình được báo rõ; model local không nạp lại mỗi câu. |
| B06 | Sinh một câu trả lời | B04, B05 | Messages → Văn bản model trả về | Không rỗng; đầu ra thật; API lỗi không thay bằng câu giả lập. |
| B07 | Ghi một bản kết quả | B06 | Câu hỏi, câu trả lời và thời gian → Một dòng predictions.jsonl | Có sample_id, cấu hình sinh, trạng thái; giữ nguyên đầu ra model. |
| B08 | Đọc và rà soát thủ công một đầu ra | B07 | Một prediction → Ghi chú kiểm tra | Xác nhận đúng câu hỏi; chưa kết luận chất lượng toàn mô hình. |

### Khối C — Ghép chat_engine và hội thoại

Đích: Một hàm chung dùng cho chatbot và đánh giá.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| C01 | Định nghĩa đầu vào/đầu ra của chat_engine | B07 | Luồng một câu đã chạy → Chữ ký chat(question, history, contexts) | Không có tham số reference; trả answer, nguồn và thông tin lần gọi. |
| C02 | Đóng gói luồng một câu vào chat_engine | C01 | Hàm đã có ở B → Hàm gọi được từ Python | Cùng đầu vào đi qua cùng logic tạo prompt và sinh câu trả lời. |
| C03 | Tạo lịch sử rỗng cho hội thoại mới | C02 | Một case → history=[] | Case mới không mang câu trả lời từ case trước. |
| C04 | Thêm một cặp user/assistant vào lịch sử | C03 | Câu hỏi và câu trả lời thực → Lịch sử sau một lượt | Không dùng đáp án chuẩn thay câu trả lời model; lỗi không được thêm thành lượt thành công. |
| C05 | Gọi lượt kế tiếp với lịch sử | C04 | History và câu hỏi tiếp theo → Câu trả lời lượt tiếp theo | Lượt 2 thực sự nhìn thấy output lượt 1; không nối dữ liệu của case khác. |
| C06 | Kiểm tra giới hạn độ dài đầu vào | C05 | Messages và tokenizer/backend limit → Số token hoặc lỗi có ghi nhận | Không âm thầm bỏ câu hỏi cuối hay system prompt; chính sách vượt giới hạn phải được ghi. |
| C07 | Chạy một hội thoại ba lượt | C06, A09 | Case MT đã chọn → Ba predictions có thứ tự | Nếu một lượt lỗi, các lượt phụ thuộc được đánh dấu skipped; không tính là thành công. |

### Khối D — Thêm RAG từng phần

Đích: Một câu hỏi → bằng chứng có nguồn → câu trả lời.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| D01 | Lập bản ghi nguồn cho một tài liệu | Không có | PDF/TXT/bài RAG kế thừa → source_registry entry | Có tên, phiên bản nếu biết, nguồn, đường dẫn, checksum và trạng thái duyệt. |
| D02 | Duyệt một nguồn cho phạm vi đề tài | D01, S02 | Tài liệu và xuất xứ → Quyết định nguồn kèm người duyệt | Không tự coi tài liệu cũ là hướng dẫn y khoa hiện hành; giữ nguồn chưa duyệt riêng. |
| D03 | Trích văn bản của một tài liệu | D02 | File gốc → Văn bản có trang/mục khi có | Đối chiếu một đoạn với gốc; không bịa số trang khi nguồn không có. |
| D04 | Làm sạch một loại nhiễu | D03 | Văn bản trích xuất → Văn bản sạch + quy tắc | Ví dụ bỏ header lặp; không làm mất số, đơn vị hoặc phủ định. |
| D05 | Chia một mục tài liệu thành các đoạn | D04 | Một mục văn bản → Danh sách chunk | Không mất đoạn; mỗi chunk còn source_id và vị trí gốc. |
| D06 | Gán ID ổn định cho một đoạn | D05 | Nội dung và phiên bản nguồn → chunk_id | Cùng dữ liệu/quy tắc cho cùng ID; nguồn đổi phải truy vết được. |
| D07 | Tạo vector cho một đoạn | D06 | Một chunk + embedding config → Một vector | Ghi đúng embedding/revision, tiền tố, chuẩn hóa và số chiều. |
| D08 | Xây index và bảng metadata | D07 | Các vector và chunk → index.faiss + metadata | Số vector khớp số ID; thử đọc lại và kiểm tra ánh xạ. |
| D09 | Tạo vector cho một query | D07, A10 | Câu hỏi → Query vector | Cùng embedding và quy ước với index; không trộn hai model embedding. |
| D10 | Lấy top-k đoạn cho một query | D08, D09 | Query vector + index → Danh sách chunk có thứ tự | Không có ID âm/ngoài bảng; xem thủ công đoạn có liên quan không. |
| D11 | Đưa bằng chứng vào chat_engine | D10, C02 | Câu hỏi + các chunk → Messages chứa bằng chứng | Có ranh giới tài liệu/câu hỏi, trích ID; tài liệu được xử lý như dữ liệu. |
| D12 | Kiểm tra ID nguồn trong một câu trả lời | D11 | Output + ID đã cung cấp → Danh sách ID hợp lệ/không hợp lệ | ID tồn tại chỉ là kiểm tra cấu trúc, không phải kết luận nguồn hỗ trợ phát biểu. |

### Khối E — Đánh giá baseline và tuyển chọn model

Đích: Một run có dữ liệu, câu trả lời, điểm và lỗi truy vết được.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| E01 | Chuẩn hóa một đáp án tham chiếu | S05 | Reference kế thừa → Một nhãn chờ duyệt | Giữ sample_id và nguồn; tách hoàn toàn khỏi file đưa cho model. |
| E02 | Duyệt một đáp án tham chiếu | E01 | Nhãn + tài liệu chứng cứ → Quyết định và bằng chứng | Có người duyệt/trạng thái; điểm chưa duyệt không được trình bày như chuẩn y khoa. |
| E03 | Viết rubric cho một tiêu chí | S05 | Tên một tiêu chí → Thang điểm và ví dụ | Có điều kiện áp dụng, giá trị N/A và cách xử lý thiếu dữ liệu; lặp cho các tiêu chí. |
| E04 | Khóa tập câu hỏi của một run | A08, S04 | Nhóm được chọn → Danh sách case + checksum | Giữ ST/MT cùng nhóm; ghi rõ benchmark kế thừa và lịch sử sử dụng. |
| E05 | Khóa chính sách truy xuất của một run | D10, E04 | Query policy và KB → Context snapshot/config có phiên bản | Không dùng reference để truy xuất; so sánh model phải giữ điều kiện truy xuất tương đương. |
| E06 | Chạy một case qua engine và ghi kết quả | C07, D11, E05 | Case đã khóa → Prediction từng lượt | Reset history giữa case; chụp cấu hình/code/prompt/KB của lần chạy. |
| E07 | Ghi nhận một lỗi thực thi | E06 | Lỗi backend hoặc retrieval → Error/skipped record | Không xóa khỏi mẫu số và không biến lỗi thành điểm 0 hoặc câu trả lời hợp lệ. |
| E08 | Chấm một đầu ra theo một tiêu chí | E02, E03, E06 | Prediction + nhãn + rubric → Điểm + lý do + nguồn chấm | Chấm sau sinh; người chấm hoặc judge có cấu hình/phiên bản. |
| E09 | Kiểm tra cấu trúc một kết quả chấm | E08 | Score record → Điểm hợp lệ hoặc lỗi | Loại điểm ngoài thang, JSON sai, thiếu lý do; giữ N/A riêng. |
| E10 | Đối chiếu judge với người chấm trên mẫu đã chọn | E09 | Điểm judge và điểm người → Báo cáo bất đồng | Hiệu chuẩn trên dev; không mặc định judge đúng; ghi mẫu và phương pháp đối chiếu. |
| E11 | Tổng hợp một chỉ số | E09 | Scores và predictions → Một thống kê kèm mẫu số | Tách single/multi; nêu số thiếu; giữ đơn vị nhóm khi tính thống kê. |
| E12 | Tạo một báo cáo run | E07, E10, E11 | Kết quả chi tiết → report.md + summary.json | Có lỗi, độ phủ chấm và cấu hình; không chỉ một con số trung bình. |
| E13 | Ghi một run vào MLflow | E12 | Config, metrics, artifact → Run có ID | Mở lại truy ra prediction tương ứng; Colab có thể nhập gói kết quả sau. |
| E14 | So sánh hai cấu hình theo một giả thuyết | E12, S04 | Hai run tương đương → Bảng so sánh theo cặp | Giữ KB/test/rubric ổn định; dùng 58 nhóm, không giả định 232 lượt độc lập. |

### Khối F — SFT sau baseline

Đích: Một adapter nạp lại được và có báo cáo trên dev.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| F01 | Chọn model nền từ baseline | E14 | Kết quả chất lượng/tài nguyên → Model/revision được chọn | Có lý do chọn theo tiêu chí nghiên cứu, không chỉ vì model mới. |
| F02 | Kiểm tra một cặp SFT kế thừa | F01 | Một trong 246 cặp ứng viên → Cặp được duyệt hoặc trả lại | Không tự dùng đáp án chép toàn bài RAG; có nguồn/quyết định duyệt. |
| F03 | Gán vai trò train/dev theo nhóm gốc | F02, E04 | Nhóm dữ liệu → Split manifest | Nhóm liên quan không sang cả train và eval; ghi giới hạn kiểm tra trùng ngữ nghĩa. |
| F04 | Tokenize một cặp bằng chat template | F03 | Cặp SFT + tokenizer model nền → Một training example | Xem lại phần user/assistant; không dùng template sai model. |
| F05 | Kiểm tra mask loss của một mẫu | F04 | Tokenized example → Mask/labels đã kiểm tra | Đúng vùng cần học; không tạo loss trên toàn bộ câu hỏi do lỗi cấu hình. |
| F06 | Chạy vài bước huấn luyện thử | F05, S06 | Mẫu train + config QLoRA → Loss và bộ nhớ thử | Không OOM; loss hữu hạn; xác định tham số nào được cập nhật. |
| F07 | Lưu một checkpoint SFT | F06 | Trạng thái huấn luyện → Adapter/checkpoint + config | Checkpoint là bản thử hoặc bản đầy đủ phải ghi rõ; có model nền/tokenizer. |
| F08 | Nạp lại checkpoint trong tiến trình mới | F07 | Adapter và model nền → Một câu trả lời | Không phụ thuộc trạng thái RAM phiên train; checkpoint tương ứng đúng model nền. |
| F09 | Đánh giá SFT bằng luồng baseline | F08, E12 | Ứng viên SFT + dev → Báo cáo SFT vs baseline | Không dùng test cuối để chọn checkpoint; bản kém hơn không tự thay baseline. |

### Khối G — Preference và DPO

Đích: Một ứng viên DPO có thể so sánh lại với SFT.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| G01 | Chọn một prompt thuộc tập train | F09 | Train case + context → Prompt có lịch sử/KB cố định | Không lấy prompt từ tập test cuối; giữ đúng context sẽ đưa vào trainer. |
| G02 | Sinh một ứng viên cho prompt | G01 | Mô hình SFT + seed/config → Một câu trả lời ứng viên | Lưu prompt/context/seed; lặp để có nhiều ứng viên khác nhau. |
| G03 | Chấm một ứng viên theo rubric | G02, E03 | Ứng viên + rubric → Điểm có lý do | Phát hiện nhãn không chắc; không tự coi heuristic là chuẩn y khoa. |
| G04 | Tạo một cặp chosen/rejected | G03 | Hai ứng viên có phân biệt → Một preference pair | Chosen và rejected khác nhau; giữ nguyên prompt và lý do chọn. |
| G05 | Duyệt một cặp preference | G04 | Preference pair → Cặp duyệt hoặc loại | Bỏ cặp hòa điểm/nhãn không đáng tin; lưu trạng thái người duyệt. |
| G06 | Xác minh reference policy của DPO | G05, F08 | Checkpoint SFT + trainer config → Bản ghi reference đúng phiên bản | Reference phải tương ứng SFT dự định; phân biệt model nền với adapter SFT. |
| G07 | Chạy vài bước DPO thử | G06 | Cặp preference đã duyệt → Loss/log/bộ nhớ thử | Chạy được và kiểm tra train/reference đúng; ghi rõ đây là smoke training. |
| G08 | Lưu một checkpoint DPO | G07 | Trạng thái train → Checkpoint DPO + cấu hình | Ghi model nền và cấu hình; không ghi đè bản SFT. |
| G09 | Nạp lại checkpoint DPO trong tiến trình mới | G08 | Checkpoint và model nền → Một câu trả lời từ DPO | Không phụ thuộc RAM phiên train; xác nhận đủ trọng số/adapter cần thiết. |
| G10 | Đánh giá DPO trên dev | G09, E12 | Ứng viên DPO → Bảng DPO vs SFT vs baseline | Đánh giá cùng engine; giữ bản tốt nhất theo tiêu chí, DPO có thể không cải thiện. |

### Khối H — Nghiệm thu và demo

Đích: Một chatbot dùng đúng cấu hình đã được đánh giá.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| H01 | Lập bộ test cuối được quản lý riêng | E02, F03 | Thiết kế nghiên cứu và tình huống mới → Test manifest độc lập phù hợp | Không tái dùng benchmark cũ dưới tên unseen; không dùng test cuối để chọn tham số. |
| H02 | Khóa một ứng viên phát hành | G10, E14 | Kết quả các ứng viên → Release manifest | Khóa model/adapter, prompt, KB, retriever và chính sách; chọn baseline/SFT/DPO theo kết quả. |
| H03 | Chạy nghiệm thu cấu hình đã khóa | H01, H02, E12 | Ứng viên + test cuối → Báo cáo nghiệm thu | Chạy sau khi ngừng chọn tham số; thất bại thì ghi nhận, quy định lại vòng nghiên cứu. |
| H04 | Viết một endpoint gọi chat_engine | H03, C02 | Request câu hỏi + session → Response cấu trúc | Dùng đúng engine/config đã đánh giá; lỗi trả trạng thái rõ. |
| H05 | Viết một ô chat gọi endpoint | H04 | Câu hỏi người dùng → Hiển thị answer + nguồn | Một câu đi từ UI đến engine và hiện đúng kết quả. |
| H06 | Tách history giữa hai phiên người dùng | H05, C03 | Hai session → Hai lịch sử riêng | Câu từ phiên A không xuất hiện trong phiên B. |
| H07 | Đóng gói bản demo có thể khởi động lại | H06 | Code và cấu hình đã khóa → Hướng dẫn chạy + artifact | Khởi động lại vẫn nạp đúng model/KB; không coi Colab là dịch vụ luôn chạy. |
| H08 | Ghi một phản hồi và liên kết với run | H07 | Phản hồi thử nghiệm → Bản ghi đã giảm dữ liệu cá nhân | Truy ra phiên bản; phản hồi không tự trở thành dữ liệu train đã duyệt. |
| H09 | Thử quay về phiên bản trước | H07 | Hai release có manifest → Lần rollback thử | Phục hồi đồng bộ model, prompt và KB; không chỉ đổi trọng số. |

### Khối R — Ghi chép và tái lập song song

Đích: Bằng chứng đủ để viết báo cáo khóa luận.

| ID | Một việc cần làm | Cần có trước | Đầu vào → Đầu ra | Hoàn thành khi |
|---|---|---|---|---|
| R01 | Lưu một phiên bản code bằng Git | A12 | Code đã kiểm tra → Commit | Tái mở được đúng phiên bản; không ghi API key vào repo. |
| R02 | Ghi dependency của một môi trường đã chạy | B08 | Môi trường thực tế → File khóa/phiên bản | Không tự chốt CUDA khi chưa biết GPU; phân biệt data/train/serve khi cần. |
| R03 | Khai báo một stage DVC sau khi lệnh chạy ổn | E12, R01 | Script + input/output/config → Stage chạy lại được | Sửa đầu vào làm stage cần chạy lại; DVC không tự cấp GPU. |
| R04 | Viết một đoạn phương pháp từ lần chạy thực | E12 | Config và log → Đoạn báo cáo + liên kết artifact | Mô tả việc đã làm; không viết thiết kế dự kiến như kết quả hoàn thành. |
| R05 | Viết một phân tích lỗi từ mẫu cụ thể | E12 | Một case lỗi → Lỗi + bằng chứng + hướng sửa | Phân biệt lỗi dữ liệu, truy xuất, model, history hoặc bộ chấm. |

## 7. Dùng công cụ lúc nào?

| Khi đã đến | Công cụ bổ sung nếu cần | Mục đích |
|---|---|---|
| A | Python chuẩn | Đọc và kiểm tra dữ liệu |
| B | Một backend model | Trả lời được một câu |
| D | Một embedding và FAISS | Tìm đoạn liên quan |
| E | Bộ chấm; MLflow sau khi có run | Chấm, so sánh và lưu thực nghiệm |
| F/G | Bộ công cụ train phù hợp máy | SFT và DPO |
| H | API và UI đơn giản | Demo hệ thống đã đánh giá |
| R | Git trước; DVC sau khi script ổn | Giữ phiên bản và chạy lại |

Chưa cần đưa vào mốc 1: reranker, RAGAS, Docker, nhiều model, giao diện đẹp, tự động gửi job GPU. Chúng chỉ được thêm khi có nhu cầu cụ thể và phép đo chứng minh lợi ích.

## 8. Quy tắc đi từng bước

1. Chọn đúng một unit tiếp theo có đủ phụ thuộc.
2. Giải thích bằng một ví dụ trước khi code.
3. Chạy một mẫu, xem trực tiếp đầu ra.
4. Kiểm tra lỗi có thể làm sai kết quả của unit đó.
5. Ghi trạng thái và file bằng chứng rồi mới mở rộng.

Mỗi lần trao đổi sau chỉ cần nói “làm tiếp B01” hoặc “giải thích A08”. Không cần nắm tất cả SFT/DPO ngay để hoàn thành A.

## 9. Việc tiếp theo sau gói này

Chạy lệnh trong README và kiểm tra xuất hiện 4 câu đúng nhóm. Sau đó đi tới S06/B01: ghi máy chạy và chọn một backend. Khi biết backend, viết B04–B07 để một câu hỏi có một câu trả lời thật. Chỉ sau mốc này mới ghép thành chat_engine ở C.

Ghi chú nghiệm thu: hoàn thành A chỉ chứng minh cấu trúc và liên kết dữ liệu đúng theo hợp đồng này. Nó không chứng minh nhãn y khoa đúng, không kiểm tra đầy đủ trùng ngữ nghĩa và không cho biết model nào tốt hơn.

## Cập nhật bước B — 04/10/2026

Người dùng đã hiểu A. B01–B07 hiện có code và notebook hướng dẫn, chưa nghiệm thu bằng model thật trong môi trường tạo gói. Các trạng thái mới nằm trong tasks.json/progress.json. B08 chờ đọc output thật; chưa chuyển sang C. Mở README_BUOC_B.md để chạy một câu.
