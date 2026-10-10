"""
BƯỚC 5 — Build Vector DB (ChromaDB) + pipeline ingest có thể chạy lại khi cập nhật tài liệu.

TẠI SAO CHỌN CHROMADB (thay vì FAISS thuần) cho đề tài này:
  - ChromaDB lưu kèm text + metadata cùng vector trong 1 nơi (persistent client) ->
    không cần tự quản lý thêm 1 file ánh xạ "id vector -> text gốc" như khi dùng FAISS thuần.
  - Hỗ trợ upsert theo id sẵn có -> rất hợp với yêu cầu "chạy lại khi cập nhật tài liệu"
    (không tạo bản ghi trùng, tự động ghi đè nếu nội dung đổi).
  - Hỗ trợ filter theo metadata (vd chỉ lấy chunk từ "source_dataset X") -> hữu ích khi
    sau này thêm Quyết định 361/QĐ-BYT làm nguồn riêng, ưu tiên cao hơn nguồn QA cộng đồng.
  - Chạy tốt trên Google Colab, không cần biên dịch thêm (FAISS đôi khi kén phiên bản CUDA/CPU).

  Nếu sau này cần tối ưu tốc độ truy vấn ở quy mô hàng triệu vector (không phải trường hợp
  của đề tài này — chỉ vài nghìn chunk), có thể cân nhắc chuyển sang FAISS với chỉ mục
  IVF/HNSW. Với quy mô kho tri thức Gút (~ vài nghìn chunk), ChromaDB là đủ nhanh.

IDEMPOTENT INGEST: mỗi chunk đã có `chunk_id` ổn định (sinh từ hash nội dung ở bước 3).
Khi chạy lại script này sau khi cập nhật tài liệu, Chroma sẽ UPSERT theo đúng id đó:
  - chunk cũ không đổi nội dung -> giữ nguyên vector (không tính toán lại, nhờ so sánh hash)
  - chunk nội dung đổi -> id mới sinh ra khác -> chunk cũ (id cũ) không còn được ingest lại,
    nên cần chạy thêm bước dọn rác (xem hàm `prune_stale_chunks`) để xoá các id không còn
    xuất hiện trong lần ingest mới nhất.

Chạy trên Colab:
    python 05_build_vector_db.py --model bkai-foundation-models/vietnamese-bi-encoder
"""
import argparse
import json

import chromadb
from chromadb.utils import embedding_functions

from pathlib import Path
from config import VECTOR_DB_DIR, COLLECTION_NAME

CHUNKS_DIR = Path("data_filtered_final")


def load_chunks() -> list[dict]:
    path = CHUNKS_DIR / "rag_chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_collection(model_name: str):
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def ingest(collection, chunks: list[dict], batch_size: int = 128) -> set[str]:
    all_ids = set()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [{
            "heading": c.get("heading") or "",
            "source_dataset": c.get("source_dataset") or "",
            "type": c.get("type") or "",
        } for c in batch]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        all_ids.update(ids)
        print(f"  ingested {min(i + batch_size, len(chunks))}/{len(chunks)}")
    return all_ids


def prune_stale_chunks(collection, current_ids: set[str]):
    """Xoá khỏi Chroma những chunk_id KHÔNG còn xuất hiện trong lần ingest mới nhất
    (vd: do tài liệu gốc bị sửa/xoá nên hash chunk_id thay đổi -> id cũ trở thành rác)."""
    existing = collection.get(include=[])  # chỉ lấy ids, không lấy documents/embeddings cho nhẹ
    stale_ids = [eid for eid in existing["ids"] if eid not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
        print(f"  đã xoá {len(stale_ids)} chunk cũ không còn khớp với dữ liệu hiện tại")
    else:
        print("  không có chunk cũ nào cần dọn")


def sanity_check(collection, query: str = "người bị gout nên kiêng ăn gì", k: int = 3):
    res = collection.query(query_texts=[query], n_results=k)
    print(f"\n[Sanity check] Query: '{query}'")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"  (dist={dist:.3f}, nguồn={meta.get('source_dataset')}) {doc[:120]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bkai-foundation-models/vietnamese-bi-encoder",
                         help="Tên embedding model (nên lấy từ kết quả 04_benchmark_embeddings.py)")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Nạp {len(chunks)} chunk từ {CHUNKS_DIR / 'rag_chunks.jsonl'}")

    client, collection = get_collection(args.model)
    print(f"Collection '{COLLECTION_NAME}' hiện có {collection.count()} vector trước khi ingest.")

    current_ids = ingest(collection, chunks)
    prune_stale_chunks(collection, current_ids)

    print(f"\nHoàn tất. Collection '{COLLECTION_NAME}' hiện có {collection.count()} vector.")
    print(f"Dữ liệu được lưu persistent tại: {VECTOR_DB_DIR}")

    sanity_check(collection)


if __name__ == "__main__":
    main()
