import re
import unicodedata
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from app.schemas import CLAIM_FIELDS, Paper, ReviewResponse, sorted_cluster_ids

TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
TITLE_STOPWORDS = {
    "a", "an", "the", "on", "of", "for", "in", "to", "and", "with", "from", "by", "at",
    "is", "are", "towards", "toward",
}  # fmt: skip


def tex_escape(s: str) -> str:
    """Escape LaTeX/BibTeX special characters in one pass, so nothing is escaped twice."""
    return "".join(TEX_SPECIAL.get(c, c) for c in s)


def _one_line(s: str) -> str:
    return " ".join(s.split())


def ordered_papers(review: ReviewResponse) -> list[Paper]:
    """Papers in display order: clusters numerically, the unclustered group last."""
    return [p for cid in sorted_cluster_ids(review.clusters) for p in review.clusters[cid]]


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def citation_keys(papers: list[Paper]) -> list[str]:
    """One unique key per paper, like smith2023vision: first author's family name, year, first
    significant title word. Collisions get a, b, c ... suffixes in order of appearance."""
    used: set[str] = set()
    keys = []
    for p in papers:
        name = p.authors[0] if p.authors else ""
        family = name.split(",")[0].split()[-1] if "," in name else (name.split() or [""])[-1]
        family = re.sub(r"[^a-z0-9]", "", _ascii(family).lower()) or "anon"
        words = re.findall(r"[a-z0-9]+", _ascii(p.title).lower())
        word = next((w for w in words if w not in TITLE_STOPWORDS), "untitled")
        base = f"{family}{p.year or 'nd'}{word}"
        key, n = base, 0
        while key in used:
            n += 1
            key = base + (chr(96 + n) if n <= 26 else f"_{n}")
        used.add(key)
        keys.append(key)
    return keys


def _bib_value(s: str) -> str:
    """URLs, DOIs and ids are kept verbatim; only characters that unbalance braces are removed."""
    return re.sub(r"[{}\s]", "", s)


def _bib_entry(key: str, p: Paper) -> str:
    fields = [("title", "{" + tex_escape(_one_line(p.title)) + "}")]  # inner braces keep case
    if p.authors:
        fields.append(("author", " and ".join(tex_escape(_one_line(a)) for a in p.authors)))
    if p.year:
        fields.append(("year", str(p.year)))
    if p.venue:
        fields.append(("journal", tex_escape(_one_line(p.venue))))
    if p.doi:
        fields.append(("doi", _bib_value(p.doi)))
    if p.arxiv_id:
        fields += [("eprint", _bib_value(p.arxiv_id)), ("archivePrefix", "arXiv")]
    fields.append(("url", _bib_value(p.link)))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@{'article' if p.venue else 'misc'}{{{key},\n{body}\n}}"


def build_bibtex(review: ReviewResponse) -> str:
    """BibTeX for every paper in the review, from the stored metadata only."""
    papers = ordered_papers(review)
    entries = [_bib_entry(k, p) for k, p in zip(citation_keys(papers), papers, strict=True)]
    return "\n\n".join(entries) + "\n" if entries else ""


def cluster_heading(review: ReviewResponse, cid: str) -> str:
    if cid == "-1":
        return "Misc / unclustered"
    info = review.cluster_info.get(cid)
    return (
        f"Cluster {int(cid) + 1}: {info.label}"
        if info and info.label
        else f"Cluster {int(cid) + 1}"
    )


def _authors_short(p: Paper) -> str:
    names = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
    return f"{names}, {p.year}" if names and p.year else names or (str(p.year) if p.year else "")


def _md(s: str, brackets: bool = True) -> str:
    """Escape Markdown syntax in text taken from papers and models."""
    return re.sub(r"([\\`*_<>" + (r"\[\]" if brackets else "") + r"])", r"\\\1", _one_line(s))


def _md_url(url: str) -> str:
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _md_paper(n: int, p: Paper) -> list[str]:
    meta = f" ({_md(_authors_short(p))})" if _authors_short(p) else ""
    lines = [f"{n}. **{_md(p.title)}**{meta} — [link]({_md_url(p.link)})"]
    s = p.structured
    if s is None:
        return lines + [f"   {_md(p.summary)}"]
    for name in CLAIM_FIELDS:
        claims = getattr(s, name)
        if name in s.not_stated:
            lines.append(f"   - **{name.capitalize()}:** _not stated in the abstract_")
        for c in claims:
            lines += [
                f"   - **{name.capitalize()}:** {_md(c.text)}",
                f"     > “{_md(c.evidence)}” — verified in abstract",
            ]
    if s.unverified_claims:
        lines.append(f"   - _{s.unverified_claims} claim(s) dropped: quote not found in abstract_")
    return lines


def build_markdown(review: ReviewResponse) -> str:
    """The whole review as Markdown: theme summaries, then per-paper structured claims.

    [n] in a theme refers to the n-th paper in that cluster's numbered list."""
    out = [f"# Literature review: {_md(review.topic)}", ""]
    out.append(f"Generated {review.created_at:%Y-%m-%d %H:%M} UTC.")
    if any(p.structured for p in ordered_papers(review)):
        out.append("Claims are grounded in each paper's abstract only; full texts were not read.")
    if review.warnings:
        out += ["", "> **Warnings**"] + [f"> - {_md(w)}" for w in review.warnings]
    if not review.clusters:
        out += ["", _md(review.message or "No papers found.")]
    for cid in sorted_cluster_ids(review.clusters):
        info = review.cluster_info.get(cid)
        out += ["", f"## {_md(cluster_heading(review, cid))}", ""]
        if info and info.theme_summary:
            out += [_md(info.theme_summary, brackets=False), ""]
        if info and info.invalid_citations:
            bad = ", ".join(f"[{n}]" for n in info.invalid_citations)
            out += [f"_Removed citations that matched no paper in this cluster: {bad}_", ""]
        for n, p in enumerate(review.clusters[cid], 1):
            out += _md_paper(n, p)
    return "\n".join(out) + "\n"


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(review: ReviewResponse) -> bytes:
    """Render a review as a PDF. All user/arXiv text is XML-escaped for Paragraph markup."""
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Lazy Researcher Review: {escape(review.topic)}", styles["Title"]),
        Paragraph(f"Generated {review.created_at:%Y-%m-%d %H:%M} UTC", styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]
    if not review.clusters:
        story.append(Paragraph(escape(review.message or "No papers found."), styles["Normal"]))

    for i, cluster_id in enumerate(sorted_cluster_ids(review.clusters)):
        if i:
            story.append(PageBreak())
        heading = "Misc / unclustered" if cluster_id == "-1" else f"Cluster {int(cluster_id) + 1}"
        story.append(Paragraph(heading, styles["Heading2"]))
        for paper in review.clusters[cluster_id]:
            link = escape(paper.link, {'"': "&quot;"})
            story += [
                Paragraph(escape(paper.title), styles["Heading4"]),
                Paragraph(
                    f'<link href="{link}" color="blue">{escape(paper.link)}</link>',
                    styles["Normal"],
                ),
                Spacer(1, 0.15 * cm),
                Paragraph(escape(paper.summary), styles["BodyText"]),
                Spacer(1, 0.4 * cm),
            ]

    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=f"Review: {review.topic}").build(
        story, onFirstPage=_footer, onLaterPages=_footer
    )
    return buf.getvalue()
