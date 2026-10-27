import re
import zlib

from app.schemas import ClusterInfo
from app.services.export import build_pdf
from tests.test_export import paper, review
from tests.test_export_markdown import full_review


def pdf_text(pdf: bytes) -> str:
    """All text drawn in the PDF (reportlab emits Flate streams of '(text) Tj' operators)."""
    out = []
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        try:
            data = zlib.decompress(stream)
        except zlib.error:
            try:  # reportlab wraps streams in ASCII85 before Flate by default
                import base64

                data = zlib.decompress(base64.a85decode(stream.strip(), adobe=True))
            except Exception:
                continue
        for m in re.findall(rb"\(((?:\\.|[^\\)])*)\)\s*Tj", data):
            s = m.decode("latin-1")
            s = re.sub(r"\\([0-7]{3})", lambda o: bytes([int(o.group(1), 8)]).decode("cp1252"), s)
            out.append(re.sub(r"\\(.)", r"\1", s))
    return " ".join(out)


def test_pdf_contains_labels_themes_and_claims():
    text = pdf_text(build_pdf(full_review()))
    assert "Cluster 1: sparse, attention" in text
    assert "Misc / unclustered" in text
    assert "Both study memory [1][2]" in text
    assert "Memory is the bottleneck." in text
    assert "memory is the bottleneck" in text and "[verified]" in text
    assert "not stated in the abstract" in text
    assert "1 claim(s) dropped" in text
    assert "Warning: semantic_scholar: HTTP 429" in text


def test_pdf_is_valid_for_markup_and_non_latin_text():
    nasty = "<b>R&D</b> α 北京 “quoted”"
    p = paper(title=nasty, authors=("A & B",))
    pdf = build_pdf(review({"0": [p]}, info={"0": ClusterInfo(label=nasty, theme_summary=nasty)}))
    assert pdf.startswith(b"%PDF")
    text = pdf_text(pdf)
    flat = text.replace(" ", "")
    assert "<b>R&D</b>" in flat  # shown literally: escaped, not interpreted as markup
    assert "?" in flat and "“quoted”" in flat  # non-cp1252 -> '?', curly quotes kept


def test_pdf_of_plain_summary_paper_and_empty_review():
    assert build_pdf(review({"0": [paper()]})).startswith(b"%PDF")
    assert build_pdf(review({}, message="No papers found.")).startswith(b"%PDF")
