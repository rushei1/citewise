from datetime import UTC, datetime

import pytest

from app.schemas import Paper, ReviewResponse
from app.services.export import build_bibtex, citation_keys, tex_escape


def paper(title="Vision Transformers", authors=("John Smith",), year=2023, **kw):
    return Paper(
        title=title,
        summary="s",
        link="http://arxiv.org/abs/1",
        authors=list(authors),
        year=year,
        **kw,
    )


def review(clusters, info=None, topic="topic", message=None, warnings=()):
    return ReviewResponse(
        review_id=7,
        topic=topic,
        created_at=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        clusters=clusters,
        cluster_info=info or {},
        message=message,
        warnings=list(warnings),
    )


# ---- citation keys ----


def test_key_format():
    assert citation_keys([paper()]) == ["smith2023vision"]


def test_duplicate_keys_get_suffixes_in_order():
    keys = citation_keys([paper(), paper(), paper(), paper(title="Other Idea")])
    assert keys == ["smith2023vision", "smith2023visiona", "smith2023visionb", "smith2023other"]
    assert len(set(keys)) == 4


def test_many_duplicates_stay_unique():
    keys = citation_keys([paper() for _ in range(40)])
    assert len(set(keys)) == 40


def test_suffixed_key_cannot_collide_with_a_real_one():
    keys = citation_keys([paper(), paper(title="Visiona"), paper()])
    assert len(set(keys)) == 3


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"authors": ()}, "anon2023vision"),
        ({"year": None}, "smithndvision"),
        ({"authors": ("José Núñez",)}, "nunez2023vision"),
        ({"authors": ("Lee, Ann",)}, "lee2023vision"),
        ({"authors": ("Ann van der Berg",)}, "berg2023vision"),
        ({"title": "The Role of Attention"}, "smith2023role"),
        ({"title": "On the"}, "smith2023untitled"),
        ({"title": "###"}, "smith2023untitled"),
        ({"authors": ("O'Brien-Smith",)}, "obriensmith2023vision"),
    ],
)
def test_key_edge_cases(kw, expected):
    assert citation_keys([paper(**kw)]) == [expected]


# ---- escaping ----


def test_tex_escape_handles_every_special_char_once():
    assert tex_escape(r"a & b % c $ d # e _ f { g } h ~ i ^ j \ k") == (
        r"a \& b \% c \$ d \# e \_ f \{ g \} h \textasciitilde{} i \textasciicircum{} j "
        r"\textbackslash{} k"
    )
    assert tex_escape("plain text, 100") == "plain text, 100"


# ---- BibTeX ----


def test_bibtex_entry_fields():
    p = paper(
        authors=("John Smith", "Ann Lee"),
        venue="CVPR 2023",
        doi="10.1/x_y",
        arxiv_id="2303.15105",
    )
    bib = build_bibtex(review({"0": [p]}))
    assert bib.startswith("@article{smith2023vision,")
    assert "  title = {{Vision Transformers}}," in bib
    assert "  author = {John Smith and Ann Lee}," in bib
    assert "  journal = {CVPR 2023}," in bib
    assert "  doi = {10.1/x_y}," in bib  # raw: no \_ inside a DOI
    assert "  eprint = {2303.15105},\n  archivePrefix = {arXiv}," in bib
    assert bib.endswith("  url = {http://arxiv.org/abs/1}\n}\n")


def test_bibtex_without_venue_is_misc():
    assert build_bibtex(review({"0": [paper()]})).startswith("@misc{smith2023vision,")


def test_bibtex_escapes_titles_and_authors_and_keeps_braces_balanced():
    p = paper(title="R&D at 100% {fast} $5_a", authors=("A & B",))
    bib = build_bibtex(review({"0": [p]}))
    assert r"title = {{R\&D at 100\% \{fast\} \$5\_a}}" in bib
    assert r"author = {A \& B}" in bib
    assert bib.count("{") - bib.count(r"\{") == bib.count("}") - bib.count(r"\}")


def test_bibtex_strips_braces_and_whitespace_from_urls_and_ids():
    p = paper(doi=" 10.1/{x} ", arxiv_id="12 34")
    bib = build_bibtex(review({"0": [p]}))
    assert "doi = {10.1/x}" in bib and "eprint = {1234}" in bib


def test_bibtex_multiline_title_is_collapsed():
    assert "{{A B}}" in build_bibtex(review({"0": [paper(title="A\n  B")]}))


def test_bibtex_order_follows_clusters_with_unclustered_last():
    r = review(
        {"-1": [paper(title="Noise")], "1": [paper(title="Second")], "0": [paper(title="First")]}
    )
    keys = [line for line in build_bibtex(r).splitlines() if line.startswith("@")]
    assert [k.split("{")[1].rstrip(",")[len("smith2023") :] for k in keys] == [
        "first",
        "second",
        "noise",
    ]


def test_bibtex_keys_are_unique_across_clusters():
    bib = build_bibtex(review({"0": [paper()], "1": [paper()]}))
    assert "@misc{smith2023vision," in bib and "@misc{smith2023visiona," in bib


def test_bibtex_of_empty_review_is_empty():
    assert build_bibtex(review({})) == ""


# ---- route ----


def test_bibtex_route(client, fetcher):
    rid = client.get("/review", params={"topic": "t"}).json()["review_id"]
    r = client.get(f"/reviews/{rid}/bibtex")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-bibtex")
    assert f"review_{rid}.bib" in r.headers["content-disposition"]
    assert r.text.count("@misc{") == 6
    assert client.get("/reviews/999/bibtex").status_code == 404
