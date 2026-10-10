# Nguồn và các mục cần rà soát

## Nguồn dữ liệu trực tiếp

- Repository và commit: https://github.com/NtQuang-23521287/Large_Language_Models_in_the_Vietnamese_Gout_Domain/tree/a5a01e5dfb9691a700d69853065765e7fb280e4b
- Hai file người dùng cung cấp: gout_test_cases.jsonl và gout_multi_turn_test_cases.jsonl.
- Hash từng file, đường dẫn và nguồn lấy nằm trong configs/inputs.lock.json. Số lượng là kết quả đọc trực tiếp snapshot, không lấy từ README.

## Cơ sở phương pháp

Gallifant et al. (2025), The TRIPOD-LLM reporting guideline for studies using large language models. Nature Medicine 31, 60–69.
https://doi.org/10.1038/s41591-024-03425-5

Dùng làm tham chiếu cho minh bạch báo cáo dữ liệu, con người giám sát và hiệu năng theo tác vụ. Các schema, group_id và vai trò legacy_eval trong gói là đề xuất triển khai cho khóa luận, không phải định dạng bắt buộc từ bài báo.

Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
https://arxiv.org/abs/2005.11401

Dùng làm cơ sở phân biệt kho bằng chứng truy xuất với dữ liệu hỏi–đáp huấn luyện/đánh giá.

## Kiểm tra có mục tiêu theo nguồn y khoa

FitzGerald et al. (2020), 2020 American College of Rheumatology Guideline for the Management of Gout. Arthritis Care & Research 72, 744–760. DOI: 10.1002/acr.24180.

Trang chính thức: https://rheumatology.org/gout-guideline

PDF chính thức: https://assets.contentstack.io/v3/assets/bltee37abb6b278ab2c/blt04d52e3b6ff5112f/632cab5b258fb55f6b2186af/gout-guideline-2020.pdf

| ID | Nội dung cần duyệt | Vị trí đối chiếu |
|---|---|---|
| GOUT_ST_011 và các lượt MT liên quan | Đáp án về thời điểm bắt đầu ULT; ACR có khuyến nghị có điều kiện bắt đầu trong đợt cấp khi đã có chỉ định | PDF trang 6, mục Timing of ULT initiation |
| GOUT_ST_044 | Khuyến nghị bổ sung vitamin C cần đối chiếu lại | PDF trang 11, Table 7 |
| GOUT_ST_050 | Khẳng định xét nghiệm uric niệu trước thuốc tăng thải là bắt buộc cần đối chiếu lại | PDF trang 7–8, Table 4 và Uricosurics |
| GOUT_ST_052 | Phân biệt vitamin C trong thực phẩm với dùng thực phẩm bổ sung | PDF trang 11–12, Management of lifestyle factors |

Đây là rà soát có mục tiêu, chưa duyệt toàn bộ nhãn. Giữ đáp án cũ và cờ rà soát; chưa tự thay bằng đáp án điều trị mới. Khác biệt giữa các hướng dẫn cần được người có chuyên môn giải quyết theo phiên bản và phạm vi áp dụng.

Hai PDF cục bộ đều có 6 trang, metadata cho thấy được xuất qua Google Docs. Tên tài liệu và nội dung trích chưa đủ xác nhận đó là toàn bộ bản ban hành chính thức. Bản ACR tiếng Việt trong repo cũng cần liên kết với bản gốc và kiểm tra bản dịch.
