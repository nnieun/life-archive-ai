"""Map normalized text lines back to the immutable stored source text."""

from backend.app.services.transcript_loader import normalize_transcript


def original_line_number(raw: str, normalized: str, offset: int) -> int:
    raw_lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    target = normalized.count("\n", 0, min(max(offset, 0), len(normalized)))
    cursor = 0
    for index, line in enumerate(normalized.split("\n")):
        while cursor < len(raw_lines) and normalize_transcript(raw_lines[cursor]) != line:
            cursor += 1
        if cursor == len(raw_lines):
            raise ValueError("Normalized transcript no longer matches its source")
        if index == target:
            return cursor + 1
        cursor += 1
    raise ValueError("Source line could not be located")
