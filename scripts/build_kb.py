import argparse
from gout_llmops.core.config import load_yaml
from gout_llmops.rag.documents import load_sources
from gout_llmops.rag.chunking import chunk_documents
from gout_llmops.rag.index import build_faiss_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rag/rag_v1.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    docs = load_sources(cfg)
    ccfg = cfg["chunking"]
    chunks = chunk_documents(docs, int(ccfg["chunk_chars"]), int(ccfg["overlap"]))
    manifest = build_faiss_index(chunks, cfg)
    print(f"Built {len(chunks)} chunks -> {cfg['output_dir']}")
    print(manifest)


if __name__ == "__main__":
    main()
