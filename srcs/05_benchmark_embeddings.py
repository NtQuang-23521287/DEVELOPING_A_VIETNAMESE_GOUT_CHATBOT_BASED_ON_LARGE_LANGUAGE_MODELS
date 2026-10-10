"""
BƯỚC 5 — Chọn embedding model tiếng Việt & benchmark thử retrieval.

Cách benchmark ở đây là kiểu "proxy nhanh" (không cần bộ nhãn thủ công đầy đủ):
với mỗi câu query mẫu, ta có sẵn 1 danh sách "từ khóa kỳ vọng" (expected_keywords).
Nếu trong top-k chunk truy xuất được có ít nhất 1 chunk chứa 1 trong các từ khóa đó
=> coi là "hit". Tính Hit@k trung bình cho từng model.

Đây KHÔNG thay thế cho đánh giá RAG đầy đủ (RAGAS ở Giai đoạn 4), mà chỉ để nhanh
chóng loại bớt các model embedding tiếng Việt hoạt động kém, trước khi đưa vào pipeline.
"""
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np

# Giữ lại import ROOT từ config, bỏ import EMBEDDING_CANDIDATES
from config import ROOT

CHUNKS_DIR = Path("data_filtered_final")
TOP_K = 5

# Danh sách 3 model bạn muốn đưa vào vòng thi đấu
MY_MODELS = [
    "bkai-foundation-models/vietnamese-bi-encoder",
    "BAAI/bge-m3",
    "NbAiLab/nb-sbert-v2-base"
]


def load_corpus() -> list[dict]:
    path = CHUNKS_DIR / "rag_chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_queries() -> list[dict]:
    with open(ROOT / "sample_queries.json", encoding="utf-8") as f:
        return json.load(f)


def cosine_top_k(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int) -> list[int]:
    # corpus_vecs và query_vec đã được normalize -> cosine sim = dot product
    sims = corpus_vecs @ query_vec
    return np.argsort(-sims)[:k].tolist()


def is_hit(retrieved_texts: list[str], expected_keywords: list[str]) -> bool:
    joined = " ".join(retrieved_texts).lower()
    return any(kw.lower() in joined for kw in expected_keywords)


def benchmark_model(model_name: str, corpus: list[dict], queries: list[dict]) -> dict:
    print(f"\n[+] Đang tải & benchmark: {model_name}")
    t0 = time.time()
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"    !! Không tải được model: {e}")
        return {"model": model_name, "error": str(e)}
    load_time = time.time() - t0

    corpus_texts = [c["text"] for c in corpus]
    t0 = time.time()
    corpus_vecs = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)
    corpus_vecs = np.asarray(corpus_vecs)
    encode_time = time.time() - t0

    hits = 0
    per_query_latency = []
    for q in queries:
        t0 = time.time()
        q_vec = model.encode(q["query"], normalize_embeddings=True)
        top_idx = cosine_top_k(np.asarray(q_vec), corpus_vecs, TOP_K)
        per_query_latency.append(time.time() - t0)

        retrieved_texts = [corpus_texts[i] for i in top_idx]
        if is_hit(retrieved_texts, q["expected_keywords"]):
            hits += 1

    return {
        "model": model_name,
        "hit_at_k": hits / len(queries),
        "k": TOP_K,
        "n_queries": len(queries),
        "embedding_dim": corpus_vecs.shape[1],
        "load_time_sec": round(load_time, 1),
        "encode_corpus_time_sec": round(encode_time, 1),
        "avg_query_latency_ms": round(sum(per_query_latency) / len(per_query_latency) * 1000, 1),
    }


def main():
    corpus = load_corpus()
    queries = load_queries()
    print(f"Corpus: {len(corpus)} chunk | Queries mẫu: {len(queries)}")

    # Chạy vòng lặp qua danh sách 3 model đã định nghĩa ở trên
    results = [benchmark_model(m, corpus, queries) for m in MY_MODELS]

    print("\n" + "=" * 90)
    print(f"{'Model':45s} {'Hit@' + str(TOP_K):8s} {'Dim':6s} {'Latency/query(ms)':18s}")
    print("=" * 90)
    for r in sorted(results, key=lambda x: x.get("hit_at_k", -1), reverse=True):
        if "error" in r:
            print(f"{r['model']:45s} LỖI: {r['error'][:50]}")
            continue
        print(f"{r['model']:45s} {r['hit_at_k']:<8.2f} {r['embedding_dim']:<6d} {r['avg_query_latency_ms']:<18.1f}")

    print("\nGợi ý đọc kết quả:")
    print("- Hit@k cao hơn = model 'hiểu' đúng ngữ nghĩa câu hỏi y khoa tiếng Việt tốt hơn.")
    print("- Với bộ câu query mẫu, đây chỉ là chỉ báo NHANH.")
    print("- Cân nhắc thêm cả tốc độ (latency) nếu chatbot cần phản hồi realtime trên GPU T4.")


if __name__ == "__main__":
    main()