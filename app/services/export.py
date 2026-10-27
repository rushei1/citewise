import re
import unicodedata
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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
    return re.sub(r"[{}\\\s]", "", s)


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


TEX_TYPOGRAPHY = {
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "–": "--",
    "—": "---",
    "…": r"\ldots{}",
    " ": "~",
    "−": "-",
}


def _tex(s: str) -> str:
    """Text safe for pdflatex with T1 + utf8: specials escaped, typographic punctuation mapped,
    and anything outside Latin-1 (Greek, CJK, ...) replaced by '?' rather than failing the build."""
    out = []
    for c in _one_line(s):
        if c in TEX_TYPOGRAPHY:
            out.append(TEX_TYPOGRAPHY[c])
        elif c in TEX_SPECIAL:
            out.append(TEX_SPECIAL[c])
        else:
            out.append(c if ord(c) <= 0xFF else "?")
    return "".join(out)


def _tex_theme(text: str, keys: list[str]) -> str:
    """Escape a theme summary and turn its [n] citations into \\cite{key}."""

    def cite(m: re.Match) -> str:
        picked = [keys[int(n) - 1] for n in m.group(1).split(",") if 1 <= int(n) <= len(keys)]
        return f"\\cite{{{','.join(picked)}}}" if picked else ""

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", cite, _tex(text))


def _tex_authors(p: Paper) -> str:
    return _tex(", ".join(p.authors)) if p.authors else ""


def _tex_item(p: Paper, key: str) -> list[str]:
    meta = ", ".join(x for x in (_tex_authors(p), str(p.year) if p.year else "") if x)
    lines = [f"\\item \\textbf{{{_tex(p.title)}}}~\\cite{{{key}}}" + (f" ({meta})" if meta else "")]
    s = p.structured
    if s is None:
        return lines + [f"  {_tex(p.summary)}"]
    lines.append("  \\begin{itemize}")
    for name in CLAIM_FIELDS:
        if name in s.not_stated:
            lines.append(
                f"    \\item \\textbf{{{name.capitalize()}.}} \\emph{{not stated in abstract}}"
            )
        for c in getattr(s, name):
            lines.append(
                f"    \\item \\textbf{{{name.capitalize()}.}} {_tex(c.text)} "
                f"(\\emph{{``{_tex(c.evidence)}''}})"
            )
    if s.unverified_claims:
        lines.append(
            f"    \\item \\emph{{{s.unverified_claims} claim(s) dropped: quote not found}}"
        )
    return lines + ["  \\end{itemize}"]


def _tex_bibitem(key: str, p: Paper) -> str:
    parts = [f"{_tex_authors(p)}." if p.authors else "", f"{_tex(p.title)}."]
    if p.venue:
        parts.append(f"\\emph{{{_tex(p.venue)}}}.")
    if p.year:
        parts.append(f"{p.year}.")
    parts.append(f"\\url{{{_bib_value(p.link)}}}")
    return f"\\bibitem{{{key}}} " + " ".join(x for x in parts if x)


def build_latex(review: ReviewResponse) -> str:
    """A self-contained .tex: sections per cluster, \\cite for theme citations and a
    thebibliography block, so a single pdflatex run suffices (no bibtex step)."""
    papers = ordered_papers(review)
    keys = citation_keys(papers)
    out = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[a4paper,margin=2.5cm]{geometry}",
        r"\usepackage[hidelinks]{hyperref}",
        f"\\title{{Literature review: {_tex(review.topic)}}}",
        f"\\date{{{review.created_at:%Y-%m-%d}}}",
        r"\begin{document}",
        r"\maketitle",
    ]
    if any(p.structured for p in papers):
        out += ["", "Claims are grounded in each paper's abstract only; full texts were not read."]
    if review.warnings:
        out += ["", r"\begin{itemize}"] + [f"\\item {_tex(w)}" for w in review.warnings]
        out.append(r"\end{itemize}")
    if not review.clusters:
        out += ["", _tex(review.message or "No papers found.")]
    offset = 0
    for cid in sorted_cluster_ids(review.clusters):
        members = review.clusters[cid]
        member_keys = keys[offset : offset + len(members)]
        offset += len(members)
        info = review.cluster_info.get(cid)
        out += ["", f"\\section{{{_tex(cluster_heading(review, cid))}}}", ""]
        if info and info.theme_summary:
            out += [_tex_theme(info.theme_summary, member_keys), ""]
        out.append(r"\begin{enumerate}")
        for p, key in zip(members, member_keys, strict=True):
            out += _tex_item(p, key)
        out.append(r"\end{enumerate}")
    if papers:
        out += ["", f"\\begin{{thebibliography}}{{{len(papers)}}}"]
        out += [_tex_bibitem(k, p) for k, p in zip(keys, papers, strict=True)]
        out.append(r"\end{thebibliography}")
    return "\n".join(out + [r"\end{document}", ""])


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _pdf(s: str) -> str:
    """Escape for reportlab Paragraph markup. Base-14 Helvetica covers Windows-1252 only, so
    anything outside it (Greek, CJK, ...) becomes '?' instead of a black box."""
    safe = "".join(c if _in_cp1252(c) else "?" for c in _one_line(s))
    return escape(safe)


def _in_cp1252(c: str) -> bool:
    try:
        c.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


def _pdf_paper(n: int, p: Paper, styles) -> list:
    link = escape(p.link, {'"': "&quot;"})
    meta = _authors_short(p)
    story = [
        Paragraph(f"[{n}] {_pdf(p.title)}", styles["Heading4"]),
        Paragraph(
            (f"{_pdf(meta)}<br/>" if meta else "")
            + f'<link href="{link}" color="blue">{escape(p.link)}</link>',
            styles["Normal"],
        ),
        Spacer(1, 0.15 * cm),
    ]
    s = p.structured
    if s is None:
        story.append(Paragraph(_pdf(p.summary), styles["BodyText"]))
    else:
        item = ParagraphStyle("claim", parent=styles["BodyText"], leftIndent=0.6 * cm)
        quote = ParagraphStyle("quote", parent=item, leftIndent=1.2 * cm, textColor="#444444")
        for name in CLAIM_FIELDS:
            label = f"<b>{name.capitalize()}:</b>"
            if name in s.not_stated:
                story.append(Paragraph(f"{label} <i>not stated in the abstract</i>", item))
            for c in getattr(s, name):
                story.append(Paragraph(f"{label} {_pdf(c.text)}", item))
                story.append(Paragraph(f"&quot;{_pdf(c.evidence)}&quot; [verified]", quote))
        if s.unverified_claims:
            note = f"{s.unverified_claims} claim(s) dropped: quote not found in abstract"
            story.append(Paragraph(f"<i>{note}</i>", item))
    return story + [Spacer(1, 0.4 * cm)]


def build_pdf(review: ReviewResponse) -> bytes:
    """Render a review as a PDF: cluster labels, theme summaries ([n] = n-th paper of that
    cluster) and each paper's structured claims with their evidence quotes."""
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Citewise Review:{_pdf(review.topic)}", styles["Title"]),
        Paragraph(f"Generated {review.created_at:%Y-%m-%d %H:%M} UTC", styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]
    for w in review.warnings:
        story.append(Paragraph(f"<i>Warning: {_pdf(w)}</i>", styles["Normal"]))
    if not review.clusters:
        story.append(Paragraph(_pdf(review.message or "No papers found."), styles["Normal"]))

    for i, cluster_id in enumerate(sorted_cluster_ids(review.clusters)):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(_pdf(cluster_heading(review, cluster_id)), styles["Heading2"]))
        info = review.cluster_info.get(cluster_id)
        if info and info.theme_summary:
            story += [Paragraph(_pdf(info.theme_summary), styles["BodyText"]), Spacer(1, 0.3 * cm)]
        for n, paper in enumerate(review.clusters[cluster_id], 1):
            story += _pdf_paper(n, paper, styles)

    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=f"Review: {_pdf(review.topic)}").build(
        story, onFirstPage=_footer, onLaterPages=_footer
    )
    return buf.getvalue()
