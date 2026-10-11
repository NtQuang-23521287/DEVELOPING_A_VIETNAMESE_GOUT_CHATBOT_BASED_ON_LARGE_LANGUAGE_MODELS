from __future__ import annotations

import argparse
import json
from pathlib import Path

from gout_llmops.core.config import load_yaml
from gout_llmops.rag.chunking import Chunk, chunk_documents
from gout_llmops.rag.documents import load_sources
from gout_llmops.rag.index import build_faiss_index


def load_prebuilt_chunks(cfg: dict) -> list[Chunk]:
    chunks: list[Chunk] = []

    for source_cfg in cfg["sources"]:
        if source_cfg.get("kind") != "prechunked_jsonl":
            raise ValueError(
                "When chunking.method=prebuilt_jsonl_v1, "
                "all sources must have kind=prechunked_jsonl"
            )

        path = Path(source_cfg["path"])

        if not path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                row = json.loads(line)

                chunk_id = str(
                    row.get("chunk_id", "")
                ).strip()

                text = str(
                    row.get("text", "")
                ).strip()

                if not chunk_id:
                    raise ValueError(
                        f"{path}:{line_no} missing chunk_id"
                    )

                if not text:
                    raise ValueError(
                        f"{path}:{line_no} missing text"
                    )

                metadata = {
                    "source_id": (
                        row.get("source")
                        or source_cfg["source_id"]
                    ),
                    "source_path": path.as_posix(),
                    "tier": source_cfg.get(
                        "tier",
                        "unknown",
                    ),
                    "review_status": source_cfg.get(
                        "review_status",
                        "unknown",
                    ),
                }

                # Giữ metadata đã có trong file chunks
                for key in (
                    "source",
                    "heading",
                    "start_char",
                    "end_char",
                    "page",
                    "title",
                    "url",
                ):
                    if row.get(key) is not None:
                        metadata[key] = row[key]

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=metadata,
                    )
                )

    # Không cho trùng chunk_id
    ids = [c.chunk_id for c in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk_id detected in KB"
        )

    expected = cfg.get(
        "chunking",
        {},
    ).get("expected_chunks")

    if (
        expected is not None
        and len(chunks) != int(expected)
    ):
        raise ValueError(
            f"Expected {expected} chunks, "
            f"but loaded {len(chunks)}"
        )

    return chunks


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/rag/rag_v1.yaml",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    chunk_cfg = cfg.get("chunking", {})
    method = chunk_cfg.get("method")

    if method == "prebuilt_jsonl_v1":
        chunks = load_prebuilt_chunks(cfg)

    else:
        # Giữ backward compatibility nếu sau này muốn
        # build lại từ raw documents.
        docs = load_sources(cfg)

        chunks = chunk_documents(
            docs,
            int(chunk_cfg["chunk_chars"]),
            int(chunk_cfg["overlap"]),
        )

    print(f"Loaded {len(chunks)} chunks")

    manifest = build_faiss_index(
        chunks,
        cfg,
    )

    print(
        f"Built {len(chunks)} chunks "
        f"-> {cfg['output_dir']}"
    )

    print(json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
    