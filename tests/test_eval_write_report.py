import pytest

from eval import write_report as wr

RESULTS = """# Title

## Environment

- Date: today

## Clustering (Part A: proxy)

| a | b |

### Part B

not run

## Summarization

table here
"""


def test_section_returns_body_up_to_next_same_level_heading_and_demotes_subheadings():
    body = wr.section(RESULTS, "Clustering (Part A")
    assert body.startswith("| a | b |")
    assert "#### Part B" in body and "not run" in body
    assert "table here" not in body and "Environment" not in body
    assert wr.section(RESULTS, "Environment") == "- Date: today"
    assert wr.section(RESULTS, "Summarization") == "table here"


def test_missing_heading_is_an_error():
    with pytest.raises(KeyError, match="Nope"):
        wr.section(RESULTS, "Nope")


def test_render_replaces_includes_and_leaves_other_text(tmp_path):
    (tmp_path / "results.md").write_text(RESULTS)
    out = wr.render("before\n{{include: results.md # Environment}}\nafter", tmp_path)
    assert out == "before\n- Date: today\nafter"


def test_render_fails_loudly_when_results_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="make eval"):
        wr.render("{{include: results.md # Environment}}", tmp_path)


def test_pdf_conversion_is_skipped_without_pandoc(tmp_path, monkeypatch):
    monkeypatch.setattr(wr.shutil, "which", lambda name: None)
    assert "pandoc is not installed" in wr.to_pdf(tmp_path / "r.md")


def test_committed_report_matches_the_template_and_committed_results():
    """docs/report.md must be regenerated whenever the results or the template change."""
    assert wr.OUT.read_text(encoding="utf-8") == wr.render(wr.TEMPLATE.read_text(encoding="utf-8"))
    text = wr.OUT.read_text(encoding="utf-8")
    assert "{{include" not in text and "TODO(rusheil)" in text
