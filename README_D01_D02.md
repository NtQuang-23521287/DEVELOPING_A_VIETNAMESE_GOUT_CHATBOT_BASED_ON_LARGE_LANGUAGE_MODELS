# D01–D02 — Chốt Knowledge Base thật của project

## Knowledge Base chính thức của project

Từ bản này, nguồn chính của khối D là **file PDF người dùng cung cấp**:

```text
data/kb/raw_docs/gout_guideline_1.pdf
```

SHA-256 được khóa:

```text
763fd9a729ab5beccf81eb78d083fcf292ff9cbc1bf620d099529ac7a5c478ca
```

File có 6 trang. Metadata PDF ghi `361/QĐ-BYT`. Nội dung là phần **BỆNH GÚT (gout)** thuộc tài liệu *Hướng dẫn chẩn đoán và điều trị các bệnh cơ xương khớp*, Bộ Y tế Việt Nam, Quyết định 361/QĐ-BYT ngày 25/01/2014.

## D01

`unit04_sources.py` tính checksum trực tiếp từ chính PDF nằm trong project và tạo `source_registry` có:

- source ID ổn định
- số quyết định/ngày ban hành
- cơ quan ban hành
- đường dẫn local tái lập được
- SHA-256 + byte size
- page count
- trang nguồn chính thức dùng để kiểm chứng provenance

Không còn dùng `Guideline_ACR_2020.txt` làm KB của luồng chính.

## D02

Nguồn được duyệt cho **phạm vi project**:

```text
project_knowledge_base = approved
rag_retrieval_source   = approved
source_allowed_for_rag = true
```

Nhưng giữ riêng hai khái niệm:

- **nguồn chính thức/versioned**: đã xác minh
- **khuyến cáo mới nhất năm 2026**: KHÔNG tự tuyên bố

Do đó:

```text
currentness_claimed = false
clinical_currentness_review_status = pending
```

Điều này không chặn RAG của khóa luận; nó chỉ ngăn việc báo cáo sai rằng tài liệu 2014 chắc chắn là guideline lâm sàng mới nhất hiện nay.

## Chạy D01–D02

Không cần GPU:

```bash
python unit04_sources.py --out runs/d01_d02_project_kb
```

Đầu ra:

```text
runs/d01_d02_project_kb/
├── source_registry.jsonl
├── review_decision.json
└── manifest.json
```

## Bước sau

- D03: trích text **theo từng trang của chính PDF này**
- D04: làm sạch lỗi xuống dòng/khoảng trắng nhưng không đổi nội dung y khoa
- D05–D06: chunk + stable chunk ID
- D07–D10: embedding/index/retrieval
- D11: đưa context retrieval vào `chat_engine`
