import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REVIEW_TIMEOUT = 600  # first run downloads and loads the models

st.set_page_config(page_title="Lazy Researcher Toolkit", layout="wide")
st.title("🧠 Lazy Researcher Toolkit")
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
def fetch_pdf(review_id: int) -> bytes | None:
    res = api_get(f"/reviews/{review_id}/pdf")
    return res.content if res else None


def load_selected() -> None:
    review_id = st.session_state.selected_review
    if review_id is None:
        return
    res = api_get(f"/reviews/{review_id}")
    if res:
        st.session_state.review = res.json()


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
col1, col2 = st.columns(2)
num_clusters = col1.slider("Clusters", 1, 8, 3)
max_results = col2.slider("Papers to fetch", 1, 30, 10)

if st.button("Generate Review"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Fetching and summarizing papers..."):
            res = api_get(
                "/review",
                timeout=REVIEW_TIMEOUT,
                topic=topic,
                num_clusters=num_clusters,
                max_results=max_results,
            )
        if res:
            st.session_state.review = res.json()

review = st.session_state.get("review")
if review:
    st.success(f"✅ Review for topic: {review['topic']}")
    if review["cached"]:
        st.info("⚡ From cache")
    if review.get("message"):
        st.warning(review["message"])

    pdf = fetch_pdf(review["review_id"])
    if pdf:
        st.download_button(
            "📥 Export to PDF",
            data=pdf,
            file_name=f"literature_review_{review['review_id']}.pdf",
            mime="application/pdf",
        )

    for cluster_id in sorted(review["clusters"], key=int):
        st.markdown(f"### 🔗 Cluster {int(cluster_id) + 1}")
        for paper in review["clusters"][cluster_id]:
            with st.expander(f"📄 {paper['title']}"):
                st.markdown(f"🔗 [View Paper]({paper['link']})")
                st.markdown(paper["summary"])
