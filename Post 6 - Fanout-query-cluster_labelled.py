import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans

# Dummy data again
queries = [
    "how to optimize ai content",
    "what is llm optimization",
    "best ai optimization tools",
    "chatgpt vs gemini",
    "python script for gsc data",
    "gsc api export tutorial",
    "ai seo automation guide",
    "measure ai search volatility",
    "how chatgpt picks sources",
    "top ai content optimizer"
]
df = pd.DataFrame({"query": queries})

# TF-IDF + clustering
vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.6)
X = vec.fit_transform(df["query"])
model = KMeans(n_clusters=3, random_state=42).fit(X)

# Extract top terms per cluster for auto-labels
terms = np.array(vec.get_feature_names_out())
centroids = model.cluster_centers_
labels = []
for i in range(model.n_clusters):
    top_idx = centroids[i].argsort()[::-1][:5]
    cluster_terms = ", ".join(terms[top_idx])
    labels.append(cluster_terms)
    print(f"Cluster {i}: {cluster_terms}")

# Assign human-friendly labels (simplified)
auto_labels = {
    0: "How-To / Informational",
    1: "Comparative Queries",
    2: "Technical / Tool-related"
}
df["cluster"] = model.labels_
df["label"] = df["cluster"].map(auto_labels)

# Visualization (PCA to 2D)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X.toarray())
plt.figure(figsize=(7,5))
scatter = plt.scatter(coords[:,0], coords[:,1], c=df["cluster"], cmap="tab10", s=90, alpha=0.7)
for i, txt in enumerate(df["query"]):
    plt.text(coords[i,0]+0.02, coords[i,1]+0.02, txt, fontsize=8)
plt.title("Auto-Labeled Query Clusters")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(handles=scatter.legend_elements()[0], labels=auto_labels.values(), title="Cluster Label")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
