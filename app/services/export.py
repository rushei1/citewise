from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from app.schemas import ReviewResponse


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

    for i, cluster_id in enumerate(sorted(review.clusters, key=int)):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(f"Cluster {int(cluster_id) + 1}", styles["Heading2"]))
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
