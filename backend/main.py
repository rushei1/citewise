from fastapi import FastAPI, Query
import arxiv
from summarize import summarize_paper
from cluster import cluster_summaries

app = FastAPI()

@app.get("/review")
def generate_review(topic: str = Query(..., description="Enter research topic")):
    search = arxiv.Search(query=topic, max_results=10, sort_by=arxiv.SortCriterion.Relevance)
    papers = list(search.results())

    structured_data = []
    summaries_for_clustering = []

    for paper in papers:
        title = paper.title.strip()
        abstract = paper.summary.strip()
        summary = summarize_paper(title, abstract)
        link = paper.entry_id

        entry = {
            "title": title,
            "summary": summary,
            "link": link
        }
        structured_data.append(entry)

        summaries_for_clustering.append(f"{title}\n{summary}")

    clustered = cluster_summaries(summaries_for_clustering)

    final_clusters = {}
    for idx, label in enumerate(clustered):
        label = str(label)  # JSON key must be string
        final_clusters.setdefault(label, []).append(structured_data[idx])

    return {
        "topic": topic,
        "clusters": final_clusters
    }