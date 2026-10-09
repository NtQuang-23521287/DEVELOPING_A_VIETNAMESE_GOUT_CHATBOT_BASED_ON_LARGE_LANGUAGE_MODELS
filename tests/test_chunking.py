from gout_llmops.rag.chunking import char_window_chunks
from gout_llmops.rag.documents import Document


def test_chunking_is_deterministic():
    doc = Document("d1", "a" * 1500, {"source_id": "x"})
    a = char_window_chunks(doc, chunk_chars=700, overlap=120)
    b = char_window_chunks(doc, chunk_chars=700, overlap=120)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert len(a) == 3


def test_invalid_overlap():
    import pytest
    doc = Document("d1", "abc", {})
    with pytest.raises(ValueError):
        char_window_chunks(doc, chunk_chars=100, overlap=100)
