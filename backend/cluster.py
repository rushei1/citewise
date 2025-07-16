from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer("all-MiniLM-L6-v2")

def cluster_summaries(summaries, num_clusters=3):
    embeddings = model.encode(summaries)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)

    return [int(label) for label in kmeans.labels_]
