"""Real PDF parsing with mocked memory extraction and indexing."""

from io import BytesIO
from unittest.mock import Mock

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from backend.app.models.memory import MemoryExtractionBatch, ExtractedMemory, DatePrecision
from backend.app.models.vector import MemoryIndexResult
from backend.app.services.ingestion import TranscriptIngestionService, InvalidUploadError
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository


def _pdf_bytes(with_text: bool) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if with_text:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 250 Td (I met a friend.) Tj ET")
        page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("with_text", [True, False])
def test_pdf_upload_extracts_text_or_rejects_empty_pdf(tmp_path, with_text):
    database = SQLiteDatabase(tmp_path / "test.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    model = Mock()
    model.invoke.return_value = MemoryExtractionBatch(memories=[ExtractedMemory(
        title="Meeting", summary="I met a friend.", people=["friend"],
        location=None, event_date=None, date_precision=DatePrecision.UNKNOWN,
        emotion=None, confidence=0.9, uncertainty_notes=None,
        evidence_start_offset=0, evidence_end_offset=15,
    )])
    index = Mock()
    index.index_memories.side_effect = lambda ids: [
        MemoryIndexResult(memory_id=memory_id, indexed=True) for memory_id in ids
    ]
    raw_root = tmp_path / "raw"
    service = TranscriptIngestionService(raw_root, repository, model, index)
    content = _pdf_bytes(with_text)
    try:
        if not with_text:
            with pytest.raises(InvalidUploadError):
                service.ingest(filename="memory.pdf", content=content)
            model.invoke.assert_not_called()
            index.index_memories.assert_not_called()
            assert repository.list_memories() == []
            return
        result = service.ingest(filename="memory.pdf", content=content)
        transcript = repository.get_transcript(result.transcript_id)
        assert transcript is not None
        assert "I met a friend." in transcript.normalized_content
        assert result.memory_count == result.indexed_memory_count == 1
        assert (raw_root / "memory.pdf").read_bytes() == content
        assert repository.list_memory_sources(result.memory_ids[0])
    finally:
        database.close()
