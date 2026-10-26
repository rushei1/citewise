import re
import shutil
import subprocess
from collections import Counter

import pytest

from app.schemas import Claim, ClusterInfo, PaperSummary
from app.services.export import _tex_theme, build_latex
from tests.test_export import paper, review
from tests.test_export_markdown import full_review


def envs(tex: str) -> tuple[Counter, Counter]:
    return Counter(re.findall(r"\\begin\{(\w+)\}", tex)), Counter(
        re.findall(r"\\end\{(\w+)\}", tex)
    )


def test_environments_and_braces_are_balanced():
    tex = build_latex(full_review())
    begins, ends = envs(tex)
    assert begins == ends and begins["document"] == 1
    bare = tex.replace(r"\{", "").replace(r"\}", "")
    assert bare.count("{") == bare.count("}")
    assert tex.rstrip().endswith(r"\end{document}")


def test_theme_citations_become_cite_commands():
    tex = build_latex(full_review())
    cites = r"\cite{lee2023sparse}\cite{anonnddense}"
    assert rf"Both study memory {cites} and *speed* \cite{{lee2023sparse}}." in tex


def test_every_cite_has_a_bibitem_and_vice_versa():
    tex = build_latex(full_review())
    cited = set(re.findall(r"\\cite\{([^}]*)\}", tex))
    items = re.findall(r"\\bibitem\{([^}]*)\}", tex)
    assert len(items) == len(set(items)) == 3
    assert {k for c in cited for k in c.split(",")} <= set(items)


def test_citation_to_unknown_paper_is_removed():
    assert _tex_theme("A [1] B [5].", ["k1", "k2"]) == r"A \cite{k1} B ."
    assert _tex_theme("A [1, 2] [7].", ["k1", "k2"]) == r"A \cite{k1,k2} ."


def test_special_characters_are_escaped_everywhere():
    nasty = r"R&D 100% {x} a_b #1 $5 ~ ^ \tt"
    p = paper(
        title=nasty,
        authors=("A & B",),
        venue="J_1",
        structured=PaperSummary(problem=[Claim(text=nasty, evidence=nasty)]),
    )
    r = review(
        {"0": [p, p]},
        info={"0": ClusterInfo(label=nasty, theme_summary=f"{nasty} [1].")},
        topic=nasty,
        warnings=[nasty],
    )
    tex = build_latex(r)
    assert nasty not in tex
    assert (
        r"R\&D 100\% \{x\} a\_b \#1 \$5 \textasciitilde{} \textasciicircum{} \textbackslash{}tt"
        in tex
    )
    begins, ends = envs(tex)
    assert begins == ends
    assert r"\emph{J\_1}" in tex and r"A \& B" in tex


def test_unicode_is_made_safe_for_pdflatex():
    p = paper(title="Learning α with ‘quotes’ – dashes, Núñez", authors=("José Núñez",))
    tex = build_latex(review({"0": [p]}))
    assert "Learning ? with `quotes' -- dashes, Núñez" in tex  # accents kept, Greek replaced
    assert "α" not in tex and "\u2019" not in tex


def test_url_with_special_characters_stays_inside_url():
    p = paper().model_copy(update={"link": "http://x.org/a_b?c=1&d=2%20#frag"})
    assert r"\url{http://x.org/a_b?c=1&d=2%20#frag}" in build_latex(review({"0": [p]}))


def test_unclustered_group_is_last_and_labelled():
    tex = build_latex(full_review())
    assert tex.index(r"\section{Cluster 1: sparse, attention}") < tex.index(
        r"\section{Misc / unclustered}"
    )


def test_empty_review_has_no_bibliography_and_still_closes_the_document():
    tex = build_latex(review({}, message="No papers found for this topic."))
    assert "thebibliography" not in tex and "No papers found" in tex
    begins, ends = envs(tex)
    assert begins == ends


def test_latex_route(client):
    rid = client.get("/review", params={"topic": "t"}).json()["review_id"]
    r = client.get(f"/reviews/{rid}/latex")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-tex")
    assert f"review_{rid}.tex" in r.headers["content-disposition"]
    assert r.text.startswith(r"\documentclass")
    assert client.get("/reviews/999/latex").status_code == 404


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_output_compiles_with_pdflatex(tmp_path):
    (tmp_path / "r.tex").write_text(build_latex(full_review()), encoding="utf-8")
    done = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "r.tex"],
        cwd=tmp_path,
        capture_output=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout.decode(errors="replace")[-2000:]
