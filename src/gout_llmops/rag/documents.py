from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any]


def load_pdf(path: str | Path, source_id: str, tier: str, review_status: str) -> list[Document]:
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                doc_id=f"{source_id}:page:{i}",
                text=text,
                metadata={
                    "source_id": source_id,
                    "source_path": p.as_posix(),
                    "page": i,
                    "tier": tier,
                    "review_status": review_status,
                },
            )
        )
    return docs


def load_csv_articles(path: str | Path, source_id: str, tier: str, review_status: str, cfg: dict) -> list[Document]:
    p = Path(path)
    title_col = cfg.get("title_column", "title")
    content_col = cfg.get("content_column", "content")
    url_col = cfg.get("url_column", "url")
    docs = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            text = (row.get(content_col) or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    doc_id=f"{source_id}:row:{idx}",
                    text=text,
                    metadata={
                        "source_id": source_id,
                        "source_path": p.as_posix(),
                        "source_row": idx,
                        "title": (row.get(title_col) or "").strip(),
                        "url": (row.get(url_col) or "").strip(),
                        "tier": tier,
                        "review_status": review_status,
                    },
                )
            )
    return docs


def load_sources(rag_config: dict) -> list[Document]:
    docs: list[Document] = []
    for source in rag_config["sources"]:
        kind = source["kind"]
        common = {
            "path": source["path"],
            "source_id": source["source_id"],
            "tier": source.get("tier", "unknown"),
            "review_status": source.get("review_status", "unknown"),
        }
        if kind == "pdf":
            docs.extend(load_pdf(**common))
        elif kind == "csv_articles":
            docs.extend(load_csv_articles(**common, cfg=source.get("csv", {})))
        else:
            raise ValueError(f"Unsupported source kind: {kind}")
    return docs
