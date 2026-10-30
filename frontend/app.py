import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REVIEW_TIMEOUT = 900  # first run downloads models; the LLM backend makes one call per paper

SOURCES = ["arxiv", "semantic_scholar"]
SUMMARIZERS = ["hf", "llm"]
CLUSTER_METHODS = ["kmeans", "agglomerative", "hdbscan"]
CLAIM_FIELDS = ["problem", "method", "results", "limitations"]
# label -> (API path suffix, mime type, file extension)
EXPORTS = {
    "PDF": ("pdf", "application/pdf", "pdf"),
    "BibTeX": ("bibtex", "application/x-bibtex", "bib"),
    "Markdown": ("markdown", "text/markdown", "md"),
    "LaTeX": ("latex", "application/x-tex", "tex"),
}

st.set_page_config(page_title="Citewise", layout="wide")
st.title("🧠 Citewise")
st.subheader("Enter a research topic to get auto-generated literature summaries")


def api_get(path: str, timeout: int = 30, **params) -> requests.Response | None:
    """GET from the API; show the API's error message and return None on any failure."""
    try:
        res = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
    except requests.RequestException as e:
        st.error(f"❌ Could not reach the API at {API_URL}: {e}")
        return None
    if res.status_code != 200:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        st.error(f"❌ API error {res.status_code}: {detail}")
        return None
    return res


@st.cache_data(show_spinner=False)
def fetch_export(review_id: int, fmt: str) -> bytes | None:
    res = api_get(f"/reviews/{review_id}/{fmt}")
    return res.content if res else None


def load_selected() -> None:
    review_id = st.session_state.selected_review
    if review_id is None:
        return
    res = api_get(f"/reviews/{review_id}")
    if res:
        st.session_state.review = res.json()


def cluster_ids(review: dict) -> list[str]:
    """Numeric order, with the unclustered group (-1) last."""
    return sorted(review["clusters"], key=lambda i: (int(i) < 0, int(i)))


def render_structured(summary: dict) -> None:
    """Problem/method/results/limitations. Every claim shown here had its quote found in the
    abstract; the quote is the caption underneath."""
    for field in CLAIM_FIELDS:
        if field in summary["not_stated"]:
            st.markdown(f"**{field.capitalize()}:** _not stated in the abstract_")
        for claim in summary[field]:
            st.markdown(
                f"**{field.capitalize()}:** {claim['text']} ✅",
                help="Verified: the quote below appears in the abstract.",
            )
            st.caption(f"“{claim['evidence']}”")
    if summary["unverified_claims"]:
        st.caption(
            f"⚠️ {summary['unverified_claims']} claim(s) were dropped because their supporting "
            "quote was not found in the abstract."
        )


def render_review(review: dict) -> None:
    st.success(f"✅ Review for topic: {review['topic']}")
    if review["cached"]:
        st.info("⚡ From cache")
    if review.get("message"):
        st.warning(review["message"])
    for warning in review.get("warnings", []):
        st.warning(f"⚠️ {warning}")

    cols = st.columns(len(EXPORTS))
    for col, (label, (fmt, mime, ext)) in zip(cols, EXPORTS.items(), strict=True):
        data = fetch_export(review["review_id"], fmt)
        if data:
            col.download_button(
                f"📥 {label}",
                data=data,
                file_name=f"literature_review_{review['review_id']}.{ext}",
                mime=mime,
                key=f"export_{review['review_id']}_{fmt}",
            )

    info = review.get("cluster_info", {})
    for cluster_id in cluster_ids(review):
        label = info.get(cluster_id, {}).get("label") or f"Cluster {int(cluster_id) + 1}"
        st.markdown(f"### 🔗 {label}")
        theme = info.get(cluster_id, {}).get("theme_summary")
        if theme:
            st.markdown(theme)
            st.caption("Written only from the verified claims below; [n] is the n-th paper listed.")
        for n, paper in enumerate(review["clusters"][cluster_id], 1):
            with st.expander(f"📄 [{n}] {paper['title']}"):
                st.markdown(f"🔗 [View Paper]({paper['link']})")
                authors = ", ".join(paper.get("authors", [])[:3])
                byline = ", ".join(str(x) for x in (authors, paper.get("year")) if x)
                if byline:
                    st.caption(byline)
                if paper.get("structured"):
                    render_structured(paper["structured"])
                else:
                    st.markdown(paper["summary"])


# Previous reviews (survive reruns via the API, not the filesystem)
history_res = api_get("/reviews")
history = history_res.json() if history_res else []
st.selectbox(
    "📂 Load previous review",
    [None] + [h["id"] for h in history],
    format_func=lambda i: (
        "None"
        if i is None
        else next(f"{h['topic']} ({h['created_at'][:10]})" for h in history if h["id"] == i)
    ),
    key="selected_review",
    on_change=load_selected,
)

topic = st.text_input(
    "🔍 Research Topic", placeholder="e.g. Vision Transformers for Medical Imaging"
)

with st.expander("⚙️ Options"):
    c1, c2, c3 = st.columns(3)
    sources = c1.multiselect("Sources", SOURCES, default=["arxiv"])
    summarizer = c2.selectbox(
        "Summarizer",
        SUMMARIZERS,
        help="hf: BART free-text summary. llm: structured claims, each backed by a quote from "
        "the abstract (needs an LLM configured on the server).",
    )
    method = c3.selectbox(
        "Cluster method",
        CLUSTER_METHODS,
        help="hdbscan finds its own number of clusters and may leave outliers unclustered.",
    )
    c1, c2 = st.columns(2)
    auto_k = c1.checkbox("Choose the number of clusters automatically")
    num_clusters = c1.slider("Clusters", 1, 8, 3, disabled=auto_k or method == "hdbscan")
    max_results = c2.slider("Papers to fetch", 1, 30, 10)
    c1, c2, c3, c4 = st.columns(4)
    year_from = c1.number_input("From year (0 = any)", 0, 2100, 0)
    year_to = c2.number_input("To year (0 = any)", 0, 2100, 0)
    min_citations = c3.number_input("Min. citations", 0, value=0)
    venue = c4.text_input("Venue contains")
    st.caption("Citation and venue filters need Semantic Scholar; arXiv papers lack that data.")

if st.button("Generate Review"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    elif not sources:
        st.warning("Pick at least one source.")
    else:
        params = {
            "topic": topic,
            "num_clusters": "auto" if auto_k else num_clusters,
            "max_results": max_results,
            "cluster_method": method,
            "summarizer": summarizer,
            "sources": ",".join(sources),
        }
        optional = {
            "year_from": year_from,
            "year_to": year_to,
            "min_citations": min_citations,
            "venue_contains": venue.strip(),
        }
        params |= {k: v for k, v in optional.items() if v}
        with st.spinner("Fetching and summarizing papers..."):
            res = api_get("/review", timeout=REVIEW_TIMEOUT, **params)
        if res:
            st.session_state.review = res.json()

review = st.session_state.get("review")
if review:
    render_review(review)
