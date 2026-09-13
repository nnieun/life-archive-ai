from datetime import UTC, datetime

from backend.app.models.memory import DatePrecision, MemoryExtractionProposal, MemoryExtractionProposalBatch
from backend.app.storage.models import TranscriptSegmentRecord
from backend.app.services.memory_extraction import _resolve_candidates


def test_untraceable_proposal_is_skipped_without_discarding_grounded_proposal():
    segment = TranscriptSegmentRecord(
        segment_id="seg", transcript_id="tr", chunk_index=0,
        start_offset=0, end_offset=12, content="친구와 공원에 갔다.",
        created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
    )
    batch = MemoryExtractionProposalBatch(memories=[
        MemoryExtractionProposal(
            title="없는 기억", summary="원문에 없는 내용", people=[], location=None,
            event_date=None, date_precision=DatePrecision.UNKNOWN, emotion=None,
            confidence=0.5, uncertainty_notes=None, evidence_text="없는 문장",
        ),
        MemoryExtractionProposal(
            title="공원 방문", summary="친구와 공원에 갔다.", people=["친구"],
            location="공원", event_date=None, date_precision=DatePrecision.UNKNOWN,
            emotion=None, confidence=0.9, uncertainty_notes=None,
            evidence_text="친구와 공원에 갔다.",
        ),
    ])
    candidates = _resolve_candidates(batch, segment)
    assert [candidate.title for candidate in candidates] == ["공원 방문"]
