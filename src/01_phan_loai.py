import pandas as pd
import re
import os
import ast
import json
import glob

print("BẮT ĐẦU PHÂN LOẠI VÀ LỌC DỮ LIỆU BỆNH GÚT...\n")

output_dir = "data_filtered_final"
os.makedirs(output_dir, exist_ok=True)

GOUT_KEYWORDS = ['gút', 'gout', 'axit uric', 'acid uric', 'tophi', 'colchicine', 'allopurinol']
KEYWORD_PATTERN = re.compile('|'.join(GOUT_KEYWORDS), re.IGNORECASE)

def count_keywords(text):
    if pd.isna(text):
        return 0
    return len(KEYWORD_PATTERN.findall(str(text)))

def read_any(path):
    """Đọc được cả .csv lẫn .parquet, tự nhận diện theo đuôi file."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def find_file(keyword):
    files = glob.glob(f"*{keyword}*.csv") + glob.glob(f"*{keyword}*.parquet")
    return files[0] if files else None

# Đã vá lỗi và thiết kế lại hàm bóc tách an toàn tuyệt đối
def extract_first_user_query(messages_str):
    if pd.isna(messages_str): return ""
    text = str(messages_str)
    messages = []

    try:
        messages = json.loads(text)
    except:
        try:
            messages = ast.literal_eval(text)
        except:
            pass

    query = ""

    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', ''))
                if role in ['user', 'human']:
                    query = str(msg.get('content', msg.get('value', '')))
                    break

    if not query:
        blocks = re.findall(r'\{[^{}]+\}', text)
        for block in blocks:
            if any(k in block.lower() for k in ["'user'", '"user"', "'human'", '"human"']):
                match = re.search(r"['\"](?:content|value)['\"]\s*:\s*(['\"])(.*?)\1(?:\s*[,}])", block, re.DOTALL | re.IGNORECASE)
                if match:
                    query = match.group(2)
                    break

    if query:
        # Xóa sạch suy nghĩ của AI nếu bị dính vào
        query = re.sub(r"<think>.*?</think>", "", query, flags=re.DOTALL)
        query = query.strip()

    return query

# =====================================================================
# NHÓM 1: NGỮ LIỆU RAG
# =====================================================================
print("1. Đang xử lý Nhóm 1 (Ngữ liệu RAG)...")
file_vinmec = find_file("vinmec_article_content")

if file_vinmec:
    print(f" -> Đã tìm thấy: {file_vinmec}")
    df_vinmec = read_any(file_vinmec)
    title_mask = df_vinmec['title'].str.contains('gout|gút', case=False, na=False)
    content_mask = df_vinmec['content'].apply(count_keywords) >= 5
    df_rag = df_vinmec[title_mask | content_mask].copy()

    output_rag = f"{output_dir}/Group1_RAG_Corpus.csv"
    df_rag.to_csv(output_rag, index=False, encoding='utf-8-sig')
    print(f" -> Đã lọc thành công {len(df_rag)} bài viết (Lưu tại: {output_rag})")
else:
    print(" [!] KHÔNG TÌM THẤY FILE VINMEC NÀO. Vui lòng kiểm tra lại thư mục!")

# =====================================================================
# NHÓM 2: HẠT GIỐNG SFT
# =====================================================================
print("\n2. Đang xử lý Nhóm 2 (Hạt giống Hỏi-Đáp SFT)...")
file_qa1 = find_file("medical_qa")
file_qa2 = find_file("train-00000")
qa_dfs = []

if file_qa1:
    print(f" -> Đã tìm thấy: {file_qa1}")
    df_qa1 = read_any(file_qa1)
    if 'title' in df_qa1.columns: df_qa1 = df_qa1.rename(columns={'title': 'question'})
    qa_dfs.append(df_qa1[['question', 'content']])

if file_qa2:
    print(f" -> Đã tìm thấy: {file_qa2}")
    df_qa2 = read_any(file_qa2)
    if 'answer' in df_qa2.columns: df_qa2 = df_qa2.rename(columns={'answer': 'content'})
    qa_dfs.append(df_qa2[['question', 'content']])

if qa_dfs:
    df_sft_seed = pd.concat(qa_dfs, ignore_index=True)
    df_sft_seed['hit_count'] = df_sft_seed['question'].apply(count_keywords) + df_sft_seed['content'].apply(count_keywords)
    df_sft_seed = df_sft_seed[df_sft_seed['hit_count'] > 0].drop_duplicates(subset=['question'])

    output_sft = f"{output_dir}/Group2_SFT_Seed_QA.csv"
    df_sft_seed[['question', 'content']].to_csv(output_sft, index=False, encoding='utf-8-sig')
    print(f" -> Đã gom được {len(df_sft_seed)} cặp Hỏi-Đáp (Lưu tại: {output_sft})")
else:
    print(" [!] Không tìm thấy file medical_qa hoặc train-00000.")

# =====================================================================
# NHÓM 3: TEST CASES (Nâng cấp bộ bóc tách đa chuẩn)
# =====================================================================
print("\n3. Đang xử lý Nhóm 3 (Test Cases để kiểm thử rủi ro)...")
file_synthetic = find_file("train_2")

if file_synthetic:
    print(f" -> Đã tìm thấy: {file_synthetic}")
    df_syn = read_any(file_synthetic)

    df_syn['target_disease'] = df_syn['target_disease'].fillna('')
    gout_mask = df_syn['target_disease'].astype(str).str.contains('gout|gút|acid uric|axit uric', case=False, na=False)
    df_syn_gout = df_syn[gout_mask].copy()

    if 'patient_persona' in df_syn_gout.columns:
        df_syn_gout = df_syn_gout.drop(columns=['patient_persona'])

    df_syn_gout['test_query'] = df_syn_gout['messages'].apply(extract_first_user_query)
    df_test_cases = df_syn_gout[df_syn_gout['test_query'].str.strip().str.len() > 0][['target_disease', 'test_query']]

    output_test = f"{output_dir}/Group3_Test_Queries.csv"
    df_test_cases.to_csv(output_test, index=False, encoding='utf-8-sig')
    print(f" -> Đã trích xuất {len(df_test_cases)} câu hỏi tình huống (Lưu tại: {output_test})")
else:
    print(" [!] Chưa tìm thấy file hội thoại tổng hợp (train_2) — Nhóm 3 sẽ bổ sung sau.")

print("\nHOÀN TẤT QUÁ TRÌNH PHÂN LOẠI!")