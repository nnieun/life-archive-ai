from backend.app.services.memory_extraction import _fold_whitespace


def test_evidence_whitespace_can_be_compared_without_changing_source():
    folded, mapping = _fold_whitespace("첫  문장\n둘째 문장")
    assert folded == "첫 문장 둘째 문장"
    assert "문장" == folded[2:4]
    assert mapping[0] == 0
