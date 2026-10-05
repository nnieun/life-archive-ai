"""Shared retrieval representation of structured memory fields."""

from backend.app.storage.models import MemoryRecord


def memory_search_text(memory: MemoryRecord) -> str:
    fields = [memory.title.strip(), memory.summary.strip()]
    if memory.people:
        fields.append("인물: " + ", ".join(memory.people))
    for label, value in (("장소", memory.location), ("날짜", memory.event_date),
                         ("감정", memory.emotion)):
        if value:
            fields.append(f"{label}: {value}")
    return "\n".join(fields)
