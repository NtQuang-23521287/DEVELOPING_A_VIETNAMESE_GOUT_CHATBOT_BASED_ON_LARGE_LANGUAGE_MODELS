"""
BƯỚC 3 — Gộp các nguồn SFT & tiền xử lý.

Gộp Group2_SFT_Seed_QA.csv (QA gốc từ HuggingFace) + Group2b_SFT_SinhNguoc.csv (QA sinh
ngược từ Group 1, đã qua kiểm tra tự động ở bước 2) thành 1 bộ SFT duy nhất, rồi làm sạch:
  1. Chuẩn hoá Unicode tiếng Việt (NFC) — tránh lỗi 1 ký tự có nhiều cách encode
  2. Bỏ thẻ HTML còn sót lại khi crawl
  3. Bỏ câu chào hỏi / cảm ơn rập khuôn ở đầu-cuối câu hỏi & câu trả lời (rác cho retrieval)
  4. Chuẩn hoá khoảng trắng, xuống dòng thừa
  5. Loại trùng lặp theo câu hỏi đã chuẩn hoá
  6. Lọc bản ghi quá ngắn (không đủ thông tin) hoặc quá dài bất thường (nghi lỗi crawl)

Input:  data_filtered_final/Group2_SFT_Seed_QA.csv   (cột: question, content)
        data_filtered_final/Group2b_SFT_SinhNguoc.csv (cột: question, content — từ bước 2)
Output: data_filtered_final/Group2_SFT_Processed.csv
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data_filtered_final")
SEED_QA_PATH = DATA_DIR / "Group2_SFT_Seed_QA.csv"
SINH_NGUOC_PATH = DATA_DIR / "Group2b_SFT_SinhNguoc.csv"
OUT_PATH = DATA_DIR / "Group2_SFT_Processed.csv"

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Các cụm mở đầu/kết thúc rập khuôn thường gặp trong dữ liệu hỏi-đáp y tế crawl từ web
GREETING_PATTERNS = [
    r"^(chào|xin chào|kính chào)\s+(bác sĩ|bs|các bác sĩ)[,.\s]*",
    r"^(bác sĩ|bs)\s+(ơi|cho (em|cháu|tôi) hỏi)[,.\s]*",
]
CLOSING_PATTERNS = [
    r"(em|cháu|tôi)\s+(xin\s+)?cảm ơn\s*(bác sĩ)?[!.]*\s*$",
    r"(mong\s+)?(bác sĩ|bs)\s+(giải đáp|tư vấn|trả lời)\s+(giúp|sớm)?[!.]*\s*$",
]

GREETING_RE = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)
CLOSING_RE = re.compile("|".join(CLOSING_PATTERNS), re.IGNORECASE)

MIN_CHARS = 20        # bản ghi ngắn hơn mức này -> gần như không có thông tin, loại bỏ
MAX_CHARS = 20_000    # dài bất thường -> khả năng lỗi crawl (dính nhiều bài viết), loại bỏ


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_html(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    return text


def strip_boilerplate(text: str) -> str:
    text = GREETING_RE.sub("", text.strip())
    text = CLOSING_RE.sub("", text.strip())
    return text.strip()


def collapse_whitespace(text: str) -> str:
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text)
    text = normalize_unicode(text)
    text = strip_html(text)
    text = strip_boilerplate(text)
    text = collapse_whitespace(text)
    return text


def load_and_merge() -> pd.DataFrame:
    frames = []
    if SEED_QA_PATH.exists():
        df_seed = pd.read_csv(SEED_QA_PATH)
        df_seed["source"] = "seed_qa"
        frames.append(df_seed[["question", "content", "source"]])
        print(f"Nạp {len(df_seed)} cặp từ {SEED_QA_PATH}")
    else:
        print(f"!! Không thấy {SEED_QA_PATH} — bỏ qua (chạy 01_phan_loai.py trước)")

    if SINH_NGUOC_PATH.exists():
        df_sn = pd.read_csv(SINH_NGUOC_PATH)
        df_sn["source"] = "sinh_nguoc"
        frames.append(df_sn[["question", "content", "source"]])
        print(f"Nạp {len(df_sn)} cặp từ {SINH_NGUOC_PATH}")
    else:
        print(f"!! Không thấy {SINH_NGUOC_PATH} — bỏ qua (chạy 02_sinh_nguoc.py trước nếu cần)")

    if not frames:
        raise SystemExit("Không có nguồn SFT nào để xử lý. Chạy 01_phan_loai.py (và 02_sinh_nguoc.py) trước.")

    return pd.concat(frames, ignore_index=True)


def main():
    df = load_and_merge()

    df["question"] = df["question"].apply(clean_text)
    df["content"] = df["content"].apply(clean_text)

    # Lọc theo độ dài tổng thể
    full_len = (df["question"] + " " + df["content"]).str.len()
    before = len(df)
    df = df[(full_len >= MIN_CHARS) & (full_len <= MAX_CHARS)]
    dropped_len = before - len(df)

    # Loại trùng theo câu hỏi đã chuẩn hoá (bắt trùng giữa seed_qa và sinh_nguoc nếu có)
    df["_dedup_key"] = df["question"].str.strip().str.lower().str[:200]
    before = len(df)
    df = df.drop_duplicates(subset="_dedup_key").drop(columns="_dedup_key")
    dropped_dup = before - len(df)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nTiền xử lý xong:")
    print(f"  - Loại do quá ngắn/quá dài bất thường: {dropped_len}")
    print(f"  - Loại do trùng câu hỏi: {dropped_dup}")
    print(f"  - Giữ lại: {len(df)}")
    print(f"Đã lưu: {OUT_PATH}")


if __name__ == "__main__":
    main()
