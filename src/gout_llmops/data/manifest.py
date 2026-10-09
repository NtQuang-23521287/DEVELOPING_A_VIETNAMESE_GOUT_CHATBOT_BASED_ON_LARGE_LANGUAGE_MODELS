from __future__ import annotations

from pathlib import Path
from typing import Iterable
from gout_llmops.core.hashing import sha256_file


def build_directory_manifest(root: str | Path, extensions: Iterable[str] | None = None) -> dict:
    root = Path(root)
    allowed = {e.lower() for e in extensions} if extensions else None
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if allowed and p.suffix.lower() not in allowed:
            continue
        files.append(
            {
                "path": p.relative_to(root).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    return {"root": root.as_posix(), "file_count": len(files), "files": files}
