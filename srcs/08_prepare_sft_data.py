"""
src/08_prepare_sft_data.py
Gộp các file CSV chứa dữ liệu Hỏi - Đáp, chuyển sang định dạng ChatML (JSONL)
và chia thành tập Train / Validation cho QLoRA.
"""
import pandas as pd
import json
import random
from pathlib import Path

# Đường dẫn tĩnh dựa trên cấu trúc repo của Luan
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "data_filtered_final"
OUTPUT_DIR = BASE_DIR / "data" / "sft"

# Đảm bảo thư mục đầu ra tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_format_csv(file_path):
    if not file_path.exists():
        print(f"[!] Bỏ qua: Không tìm thấy {file_path}")
        return []
    
    df = pd.read_csv(file_path)
    # Nhận diện cột tự động (dựa trên các file Luan có)
    q_col = 'question' if 'question' in df.columns else df.columns[0]
    a_col = 'content' if 'content' in df.columns else ('answer' if 'answer' in df.columns else df.columns[1])
    
    formatted_data = []
    for _, row in df.iterrows():
        if pd.isna(row[q_col]) or pd.isna(row[a_col]):
            continue
        
        # Định dạng chuẩn ChatML
        formatted_data.append({
            "messages": [
                {"role": "user", "content": str(row[q_col]).strip()},
                {"role": "assistant", "content": str(row[a_col]).strip()}
            ]
        })
    return formatted_data

def main():
    # Các file cần gộp (bạn có thể thêm file khác vào danh sách này nếu có)
    files_to_merge = [
        DATA_DIR / "Group2_SFT_Seed_QA.csv",
        DATA_DIR / "Group2b_SFT_SinhNguoc.csv"
    ]
    
    all_data = []
    for f in files_to_merge:
        print(f"Đang xử lý: {f.name}")
        all_data.extend(load_and_format_csv(f))
        
    print(f"\nTổng số cặp QA gộp được: {len(all_data)}")
    if len(all_data) == 0:
        print("Không có dữ liệu, vui lòng kiểm tra lại đường dẫn file CSV.")
        return

    # Trộn ngẫu nhiên và chia Train (90%) / Val (10%)
    random.seed(42)  # Cố định seed để dễ track lại kết quả
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Ghi file JSONL
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n[Thành công] Đã lưu {len(train_data)} mẫu vào data/sft/train.jsonl")
    print(f"[Thành công] Đã lưu {len(val_data)} mẫu vào data/sft/val.jsonl")

if __name__ == "__main__":
    main()
