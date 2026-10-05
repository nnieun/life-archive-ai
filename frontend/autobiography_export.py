"""Generate a dependency-light Korean DOCX download."""

from html import escape
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile


def build_docx(title: str, chapters: list[tuple[str, str]]) -> bytes:
    body = [f"<w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>{escape(title)}</w:t></w:r></w:p>"]
    for chapter_title, content in chapters:
        body.append(f"<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{escape(chapter_title)}</w:t></w:r></w:p>")
        for paragraph in content.split("\n\n"):
            body.append(f"<w:p><w:r><w:t xml:space=\"preserve\">{escape(paragraph.strip())}</w:t></w:r></w:p>")
    document = f"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>{''.join(body)}<w:sectPr/></w:body></w:document>"
    files = {
        "[Content_Types].xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/></Types>",
        "_rels/.rels": "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>",
        "word/document.xml": document,
    }
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content.encode("utf-8"))
    return output.getvalue()
