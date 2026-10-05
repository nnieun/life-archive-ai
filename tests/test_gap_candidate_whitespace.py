from backend.app.services.gap_reconstruction import _contains_text


def test_gap_evidence_accepts_line_break_as_space_only():
    assert _contains_text("대구 중리동\n달서초등학교 근처", "중리동 달서초등학교")
    assert not _contains_text("대구 중리동 떡볶이", "대구 서문시장 떡볶이")
