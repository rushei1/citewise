import streamlit as st
import requests
import os
import json
import tempfile
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# Setup
st.set_page_config(page_title="Lazy Researcher Toolkit", layout="wide")
st.title("🧠 Lazy Researcher Toolkit")
st.subheader("Enter a research topic to get auto-generated literature summaries")

# Storage path
history_dir = Path("history")
history_dir.mkdir(exist_ok=True)

# Input
topic = st.text_input("🔍 Research Topic", placeholder="e.g. Vision Transformers for Medical Imaging")

# PDF Export
def export_pdf(clusters, topic):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Lazy Researcher Review: {topic}")
    y -= 30

    c.setFont("Helvetica", 11)
    for cluster_id in sorted(clusters.keys(), key=int):
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, f"🔗 Cluster {int(cluster_id) + 1}")
        y -= 20

        for paper in clusters[cluster_id]:
            title, summary, link = paper['title'], paper['summary'], paper['link']
            text = f"📄 {title}\n🔗 {link}\n{summary}"
            for line in text.split('\n'):
                wrapped = [line[i:i + 110] for i in range(0, len(line), 110)]
                for w in wrapped:
                    if y < 40:
                        c.showPage()
                        y = height - 50
                    c.drawString(50, y, w)
                    y -= 15
            y -= 20

    c.save()
    return temp_file.name

# Load a saved review
saved_files = list(history_dir.glob("*.json"))
if saved_files:
    options = [f.stem for f in saved_files]
    selected_file = st.selectbox("📂 Load previous review", ["None"] + options)

    if selected_file != "None":
        with open(history_dir / f"{selected_file}.json", "r") as f:
            data = json.load(f)
        st.success(f"Loaded previous topic: {data['topic']}")
        clusters = data["clusters"]

        for cluster_id in sorted(clusters.keys(), key=int):
            st.markdown(f"### 🔗 Cluster {int(cluster_id) + 1}")
            for paper in clusters[cluster_id]:
                with st.expander(f"📄 {paper['title']}"):
                    st.markdown(f"🔗 [View Paper]({paper['link']})", unsafe_allow_html=True)
                    st.markdown(paper['summary'])

        if st.button("📥 Export to PDF"):
            pdf_path = export_pdf(clusters, data['topic'])
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", data=f, file_name="literature_review.pdf", mime="application/pdf")

# Generate new review
if st.button("Generate Review") and topic:
    with st.spinner("Fetching and summarizing papers..."):
        try:
            res = requests.get("http://127.0.0.1:8000/review", params={"topic": topic})
            data = res.json()

            # Save for later
            file_name = topic.lower().replace(" ", "_")
            with open(history_dir / f"{file_name}.json", "w") as f:
                json.dump(data, f, indent=2)

            st.success(f"✅ Review generated for topic: {data['topic']}")
            clusters = data["clusters"]

            for cluster_id in sorted(clusters.keys(), key=int):
                st.markdown(f"### 🔗 Cluster {int(cluster_id) + 1}")
                for paper in clusters[cluster_id]:
                    with st.expander(f"📄 {paper['title']}"):
                        st.markdown(f"🔗 [View Paper]({paper['link']})", unsafe_allow_html=True)
                        st.markdown(paper['summary'])

            if st.button("📥 Export to PDF"):
                pdf_path = export_pdf(clusters, topic)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", data=f, file_name="literature_review.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"❌ Something went wrong: {str(e)}")
