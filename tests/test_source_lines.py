from backend.app.services.source_lines import original_line_number
from backend.app.services.transcript_loader import normalize_transcript


def test_original_lines_survive_blank_line_collapse_and_crlf():
    raw = "\r\n\r\n  첫 문장  \r\n\r\n\r\n  다음 문장\r\n"
    text = normalize_transcript(raw)
    assert original_line_number(raw, text, 0) == 3
    assert original_line_number(raw, text, text.index("다음")) == 6


def test_duplicate_lines_keep_source_order():
    raw = "\n동일\n\n\n동일"
    text = normalize_transcript(raw)
    assert original_line_number(raw, text, text.rindex("동일")) == 5
