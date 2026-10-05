"""
BƯỚC 2 — Sinh ngược (Reverse Question Generation).

VẤN ĐỀ: Group1_RAG_Corpus.csv (bài viết vinmec) chỉ có cột `content` — đoạn văn dài,
không có câu hỏi đi kèm — không dùng trực tiếp để tinh chỉnh SFT được (SFT cần format
hỏi-đáp). Trong khi đó Group2_SFT_Seed_QA.csv (QA lấy từ HuggingFace) có thể không đủ
500-800 mẫu, hoặc không phủ hết các khía cạnh có trong bài viết y khoa.

GIẢI PHÁP — Sinh ngược: cho một LLM ĐỌC từng đoạn `content` trong Group 1 (coi như
"câu trả lời" có sẵn), rồi yêu cầu nó ĐẶT CÂU HỎI mà đoạn văn đó trả lời được. Ngược
hoàn toàn với cách tạo dữ liệu thông thường (có câu hỏi → đi tìm câu trả lời).

  content trong Group 1 (answer có sẵn) --> LLM đặt câu hỏi ngược --> (question, content) mới
                                                                        cùng schema Group 2

ĐIỂM QUAN TRỌNG để giảm rủi ro ảo giác khi sinh dữ liệu:
  - content của cặp mới LUÔN LÀ NGUYÊN VĂN đoạn trong Group 1 — model KHÔNG được
    paraphrase lại câu trả lời (chỉ được sinh câu hỏi). Nhờ vậy answer chắc chắn đúng
    với nguồn đã lọc, rủi ro sai lệch chỉ nằm ở phần câu hỏi.
  - Có bước kiểm tra chéo tự động (sanity check) + gắn cờ để rà tay, đúng tinh thần
    "kết hợp kiểm tra chéo rủi ro" đã ghi trong đề cương.

Input:  data_filtered_final/Group1_RAG_Corpus.csv (cột mặc định: content)
Output: data_filtered_final/Group2b_SFT_SinhNguoc.csv               (đạt kiểm tra tự động)
        data_filtered_final/Group2b_SFT_SinhNguoc_can_kiem_tra.csv  (cần rà tay)

Sau bước này: gộp Group2_SFT_Seed_QA.csv + Group2b_SFT_SinhNguoc.csv (+ phần đã duyệt
từ file can_kiem_tra) thành 1 file SFT tổng, dùng làm input cho 03_preprocess.py.

Chạy trên Colab:
    python 02_sinh_nguoc.py
    python 02_sinh_nguoc.py --input data_filtered_final/Group1_RAG_Corpus.csv --text-col content
"""
import argparse
import re

import pandas as pd
from tqdm import tqdm

# ============================================================================
# PHẦN GỌI LLM — thay bằng backend bạn có sẵn.
#
# Sinh dữ liệu là việc làm 1 LẦN, OFFLINE, không phải mô hình sẽ deploy cho người
# dùng cuối — nên KHÔNG bị ràng buộc bởi giới hạn "GPU T4 miễn phí" như mô hình chatbot
# thật. Có thể dùng model mạnh hơn ở bước này để câu hỏi sinh ra tự nhiên, đa dạng hơn.
#
# Option A (mặc định, chạy local trên Colab, miễn phí):
#   dùng 1 trong các model bạn đang benchmark làm nền (SeaLLMs 3 / Qwen3 / Gemma3)
# Option B (nếu có API key, chất lượng câu hỏi thường tự nhiên hơn):
#   gọi qua API (Anthropic/OpenAI/...) — xem hàm call_llm_via_api bị comment bên dưới
# ============================================================================

_pipe = None

def call_llm_local(prompt: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct") -> str:
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        import torch
        _pipe = pipeline("text-generation", model=model_name,
                          torch_dtype=torch.bfloat16, device_map="auto")
    out = _pipe([{"role": "user", "content": prompt}], max_new_tokens=80, do_sample=True, temperature=0.7)
    return out[0]["generated_text"][-1]["content"].strip()


# def call_llm_via_api(prompt: str) -> str:
#     """Ví dụ dùng Anthropic API thay cho model local — bỏ comment nếu bạn có API key."""
#     import anthropic
#     client = anthropic.Anthropic(api_key="...")
#     msg = client.messages.create(
#         model="claude-sonnet-4-6", max_tokens=100,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return msg.content[0].text.strip()


PROMPT_TEMPLATE = """Bạn là trợ lý tạo dữ liệu huấn luyện cho chatbot tư vấn bệnh Gút.
Đọc đoạn văn bản y khoa dưới đây, sau đó đặt DUY NHẤT MỘT câu hỏi tự nhiên mà một
bệnh nhân thực sự có thể hỏi bác sĩ, sao cho đoạn văn bản này CHÍNH LÀ câu trả lời
phù hợp cho câu hỏi đó.

QUY TẮC BẮT BUỘC:
- Chỉ trả về DUY NHẤT câu hỏi, không thêm lời dẫn, không đánh số, không giải thích.
- Câu hỏi phải là điều bệnh nhân thật sự quan tâm — không hỏi kiểu học thuật ("đoạn văn nói về gì").
- Không được thêm thông tin y khoa mới không có trong đoạn văn.

Đoạn văn bản:
\"\"\"{passage}\"\"\"

Câu hỏi:"""


def is_valid_question(question: str, passage: str) -> tuple[bool, str]:
    """Kiểm tra chéo tự động — lọc nhanh các câu hỏi rõ ràng có vấn đề trước khi rà tay."""
    q = question.strip()
    if not q or len(q) < 8:
        return False, "câu hỏi quá ngắn / rỗng"
    if not q.endswith("?"):
        return False, "không kết thúc bằng dấu hỏi (khả năng model trả lời lạc đề)"
    if len(q) > 300:
        return False, "quá dài bất thường (khả năng model sinh thêm nội dung ngoài câu hỏi)"
    refusal_markers = ["tôi không thể", "xin lỗi", "không có thông tin", "as an ai"]
    if any(m in q.lower() for m in refusal_markers):
        return False, "model từ chối / lạc hướng thay vì đặt câu hỏi"

    # Kiểm tra liên quan nội dung: ít nhất 1 "từ nội dung" trong câu hỏi phải xuất hiện
    # trong đoạn văn gốc (word overlap thô — không hoàn hảo nhưng bắt được lỗi lạc đề rõ)
    q_words = set(re.findall(r"\w{4,}", q.lower()))
    p_words = set(re.findall(r"\w{4,}", passage.lower()))
    overlap = q_words & p_words
    if len(overlap) < 1:
        return False, "câu hỏi có vẻ không liên quan nội dung đoạn văn (0 từ trùng)"

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_filtered_final/Group1_RAG_Corpus.csv")
    parser.add_argument("--text-col", default="content",
                         help="Tên cột chứa đoạn văn (Group1_RAG_Corpus.csv dùng cột 'content')")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Giới hạn số đoạn xử lý thử trước, để kiểm tra chất lượng trước khi chạy full")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Không thấy cột '{args.text_col}' trong {args.input}. "
                          f"Các cột hiện có: {list(df.columns)} — chỉnh lại --text-col cho khớp.")

    passages = df[args.text_col].dropna().astype(str).tolist()
    if args.max_samples:
        passages = passages[:args.max_samples]

    ok_out, review_out = [], []
    for i, passage in enumerate(tqdm(passages, desc="Sinh ngược câu hỏi")):
        passage = passage.strip()
        if len(passage.split()) < 15:
            continue  # đoạn quá ngắn, không đáng để sinh câu hỏi riêng

        prompt = PROMPT_TEMPLATE.format(passage=passage[:1500])  # cắt bớt nếu quá dài
        try:
            question = call_llm_local(prompt)
        except Exception as e:
            print(f"  !! Lỗi sinh câu hỏi cho đoạn #{i}: {e}")
            continue

        # Giữ đúng schema (question, content) như Group2_SFT_Seed_QA.csv để gộp trực tiếp
        item = {"question": question, "content": passage}
        valid, reason = is_valid_question(question, passage)
        if valid:
            ok_out.append(item)
        else:
            item["ly_do_can_kiem_tra"] = reason
            review_out.append(item)

    out_dir = "data_filtered_final"
    pd.DataFrame(ok_out).to_csv(f"{out_dir}/Group2b_SFT_SinhNguoc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(review_out).to_csv(f"{out_dir}/Group2b_SFT_SinhNguoc_can_kiem_tra.csv", index=False, encoding="utf-8-sig")

    print(f"\nHOÀN TẤT sinh ngược.")
    print(f"  - Đạt kiểm tra tự động, sẵn sàng gộp vào Group 2: {len(ok_out)} cặp")
    print(f"    -> {out_dir}/Group2b_SFT_SinhNguoc.csv")
    print(f"  - Cần rà tay thêm (đưa Claude/người đọc lại): {len(review_out)} cặp")
    print(f"    -> {out_dir}/Group2b_SFT_SinhNguoc_can_kiem_tra.csv")
    print(f"\nBước tiếp theo:")
    print(f"  1. Rà file Group2b_SFT_SinhNguoc_can_kiem_tra.csv, giữ lại cặp hợp lệ.")
    print(f"  2. Gộp Group2_SFT_Seed_QA.csv + Group2b_SFT_SinhNguoc.csv (+ phần đã duyệt)")
    print(f"     thành 1 file SFT tổng (cùng 2 cột question, content) trước khi chạy 03_preprocess.py.")
    print(f"  3. LƯU Ý: content trong các cặp này là NGUYÊN VĂN đoạn tài liệu gốc — không")
    print(f"     paraphrase — nên rủi ro sai thông tin y khoa thấp hơn nhiều so với để LLM")
    print(f"     tự sinh cả câu trả lời. Vẫn nên có 1 vòng đọc lướt trước khi dùng để huấn")
    print(f"     luyện, đúng tinh thần 'kiểm tra chéo rủi ro' trong đề cương.")


if __name__ == "__main__":
    main()

