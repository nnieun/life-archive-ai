from zipfile import ZipFile
from io import BytesIO

from frontend.autobiography_export import build_docx


def test_docx_export_contains_korean_content():
    data = build_docx("나의 기억", [("1장. 시작", "친구를 만났다.\n\n즐거운 하루였다.")])
    with ZipFile(BytesIO(data)) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
    assert "나의 기억" in document
    assert "친구를 만났다." in document
    assert "[mem_" not in document
