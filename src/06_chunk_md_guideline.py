"""
06_chunk_md_guideline.py
-----------------------------
Chunk 1 file .txt dạng MARKDOWN (vd: bản dịch ACR2020 hướng dẫn gout, có
heading #, ##, ###) — không phải văn bản pháp lý dạng Điều/Khoản, cũng không
phải CSV Group1_RAG_Corpus.

Khác với 04_chunking.py (input CSV, tách theo Điều/Khoản/Chương hoặc cắt cố
định) và 05_chunk_txt_transcript.py (input .txt, cắt cố định theo KÝ TỰ,
không quan tâm heading), script này:
    - Tách văn bản theo heading cấp ### (giữ "breadcrumb" heading cha ## / #
      làm ngữ cảnh, vd "Kết quả/Khuyến nghị > Allopurinol").
    - Trong mỗi section, nếu quá dài thì cắt tiếp theo TOKEN (giống 04),
      có prefix breadcrumb ở đầu mỗi sub-chunk để không mất ngữ cảnh khi
      embedding.
    - Ghi ra ĐỒNG THỜI 2 nơi để tương thích cả 2 schema đang dùng trong dự án:
        (A) data_filtered_final/rag_chunks.jsonl  (APPEND, schema giống
            Group1: chunk_id, text, heading, source)
        (B) data/kb/chunks/doc00X_chunks.jsonl    (file MỚI, schema giống
            script 05: chunk_id, source, text, start_char, end_char, heading)

Cách dùng:
    1. Đặt file .txt vào: data/kb/raw_docs/
    2. Sửa TXT_FILENAME và SOURCE_NAME bên dưới cho khớp
    3. Chạy: python 06_chunk_md_guideline.py
"""

import hashlib
import json
import re
from pathlib import Path

# ==== SỬA CÁC ĐƯỜNG DẪN NÀY CHO KHỚP MÁY BẠN (giống 04/05) ====
BASE_DIR = Path(r"C:\Users\Admin\KLTN\Large_Language_Models_in_the_Vietnamese_Gout_Domain\data")
RAW_DOCS_DIR = BASE_DIR / "kb" / "raw_docs"
CHUNKS_DIR = BASE_DIR / "kb" / "chunks"

DATA_FILTERED_DIR = BASE_DIR.parent / "data_filtered_final"
RAG_CHUNKS_PATH = DATA_FILTERED_DIR / "rag_chunks.jsonl"

TXT_FILENAME = "acr2020_bs_khoa_transcript_FULL.txt"   # đổi tên nếu khác
SOURCE_NAME = "ACR2020_Gout_Guideline_VN"              # nhãn source cho cả 2 output

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 30

# Regex nhận diện heading Markdown: #, ##, ### ... ở đầu dòng
MD_HEADING_RE = re.compile(r"(?m)^(#{1,3})\s+(.*)$")


def _tokenize(text: str) -> list[str]:
    return text.split()


def _make_hash_chunk_id(source_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source_id}-{idx}-{text[:100]}".encode("utf-8")).hexdigest()[:12]
    return f"{source_id}-{idx}-{h}"


def chunk_fixed_length(text: str, prefix: str = "") -> list[str]:
    """Sliding window theo số từ, có overlap (giống 04_chunking.chunk_fixed_length)."""
    tokens = _tokenize(text)
    if len(tokens) <= CHUNK_SIZE_TOKENS:
        return [text] if len(tokens) >= MIN_CHUNK_TOKENS else ([text] if text.strip() else [])

    chunks = []
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    for start in range(0, len(tokens), step):
        window = tokens[start:start + CHUNK_SIZE_TOKENS]
        if len(window) < MIN_CHUNK_TOKENS:
            break
        piece = " ".join(window)
        chunks.append(f"{prefix}{piece}" if prefix else piece)
        if start + CHUNK_SIZE_TOKENS >= len(tokens):
            break
    return chunks


def parse_markdown_sections(text: str):
    """
    Tách văn bản theo heading cấp ### (level 3), giữ breadcrumb heading cha
    (# và ##) làm ngữ cảnh.

    Trả về list các dict: {"breadcrumb": str, "text": str, "start": int, "end": int}
    trong đó "text" là toàn bộ nội dung của section đó (kể cả dòng heading ###),
    "start"/"end" là vị trí ký tự trong văn bản gốc.
    """
    matches = list(MD_HEADING_RE.finditer(text))
    if not matches:
        # Không có heading nào -> coi cả file là 1 section không breadcrumb
        return [{"breadcrumb": None, "text": text, "start": 0, "end": len(text)}]

    sections = []
    current_h1 = None
    current_h2 = None

    # Nhóm các heading thành các "block" bắt đầu từ heading ### (hoặc từ đầu
    # văn bản nếu có nội dung trước heading ### đầu tiên dưới 1 heading ##).
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end].rstrip()

        # Nội dung thực sự (bỏ dòng heading đầu tiên) -> dùng để lọc các
        # section "rỗng", chỉ có mỗi dòng heading vì có heading con ngay sau
        # (vd "## Tóm tắt" với "### Mục tiêu" theo ngay sau, không có đoạn
        # văn nào nằm giữa hai heading này).
        remainder = block_text.split("\n", 1)[1].strip() if "\n" in block_text else ""
        has_real_content = len(remainder) > 0

        if level == 1:
            current_h1 = title
            current_h2 = None
            if has_real_content:
                sections.append({
                    "breadcrumb": current_h1,
                    "text": block_text,
                    "start": start,
                    "end": end,
                })
        elif level == 2:
            current_h2 = title
            if has_real_content:
                breadcrumb = " > ".join([p for p in [current_h1, current_h2] if p])
                sections.append({
                    "breadcrumb": breadcrumb,
                    "text": block_text,
                    "start": start,
                    "end": end,
                })
        else:  # level == 3 -> đơn vị chunk chính theo yêu cầu
            # Với ### thì giữ luôn kể cả khi ngắn (đây là granularity chính
            # người dùng muốn), chỉ bỏ khi hoàn toàn không có nội dung.
            if has_real_content:
                breadcrumb = " > ".join([p for p in [current_h1, current_h2, title] if p])
                sections.append({
                    "breadcrumb": breadcrumb,
                    "text": block_text,
                    "start": start,
                    "end": end,
                })

    return sections


def build_chunks(sections):
    """
    Với mỗi section (level ### là chính), nếu nội dung ngắn thì giữ nguyên
    1 chunk; nếu dài thì cắt tiếp theo token, có prefix breadcrumb.
    Trả về list dict: {"heading": breadcrumb, "text": chunk_text,
                        "start_char": int, "end_char": int}
    Lưu ý: với các section bị cắt nhỏ thành nhiều sub-chunk, start_char/end_char
    được tính xấp xỉ theo vị trí bắt đầu của section gốc (vì việc re-locate
    chính xác từng sub-chunk theo ký tự sau khi tokenize là không cần thiết
    cho mục đích RAG).
    """
    out = []
    for sec in sections:
        sec_text = sec["text"]
        if not sec_text.strip():
            continue
        n_tokens = len(_tokenize(sec_text))
        breadcrumb = sec["breadcrumb"]
        prefix = f"[{breadcrumb}] " if breadcrumb else ""

        if n_tokens <= CHUNK_SIZE_TOKENS * 1.5:
            out.append({
                "heading": breadcrumb,
                "text": sec_text,
                "start_char": sec["start"],
                "end_char": sec["end"],
            })
        else:
            for piece in chunk_fixed_length(sec_text, prefix=prefix):
                out.append({
                    "heading": breadcrumb,
                    "text": piece,
                    "start_char": sec["start"],
                    "end_char": sec["end"],
                })
    return out


def next_doc_index(chunks_dir: Path) -> int:
    max_idx = -1
    if not chunks_dir.exists():
        return 0
    for f in chunks_dir.glob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                m = re.match(r"doc(\d+)_chunk", obj.get("chunk_id", ""))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def main():
    txt_path = RAW_DOCS_DIR / TXT_FILENAME
    if not txt_path.exists():
        print(f"Không tìm thấy: {txt_path}")
        return

    text = txt_path.read_text(encoding="utf-8")
    sections = parse_markdown_sections(text)
    chunks = build_chunks(sections)

    # ---- (A) APPEND vào rag_chunks.jsonl (schema Group1) ----
    DATA_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAG_CHUNKS_PATH, "a", encoding="utf-8") as fout:
        for i, c in enumerate(chunks):
            chunk_id = _make_hash_chunk_id(SOURCE_NAME, i, c["text"])
            record = {
                "chunk_id": chunk_id,
                "text": c["text"],
                "heading": c["heading"],
                "source": SOURCE_NAME,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"(A) Đã append {len(chunks)} chunk vào: {RAG_CHUNKS_PATH}")

    # ---- (B) Tạo file MỚI trong data/kb/chunks (schema script 05 + heading) ----
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    doc_idx = next_doc_index(CHUNKS_DIR)
    out_path = CHUNKS_DIR / f"doc{doc_idx:03d}_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, c in enumerate(chunks):
            record = {
                "chunk_id": f"doc{doc_idx:03d}_chunk{i:04d}",
                "source": SOURCE_NAME,
                "text": c["text"],
                "heading": c["heading"],
                "start_char": c["start_char"],
                "end_char": c["end_char"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"(B) Đã tạo {len(chunks)} chunk (doc{doc_idx:03d}), lưu tại: {out_path}")


if __name__ == "__main__":
    main()
