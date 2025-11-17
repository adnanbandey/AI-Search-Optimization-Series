import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===== Dummy GSC-like data =====
data = {
    "query": [
        # informational queries
        "how to optimize ai content",
        "what is llm optimization",
        "how chatgpt picks sources",
        "why ai search is volatile",
        "how to measure ai traffic",
        # comparative queries
        "best ai optimization tools",
        "gemini vs chatgpt",
        "chatgpt vs perplexity",
        "best ai content optimizer",
        "top ai seo platforms",
        # navigational queries
        "surfer login",
        "chatgpt api login",
        "google search console login",
        "download gsc data",
        "access ai seo dashboard",
        # random extras to diversify
        "python script for ai traffic",
        "analyze ai queries in gsc",
        "build clustering model in python",
        "fanout queries example",
        "ai overview data analysis"
    ]
}
df = pd.DataFrame(data)

# ===== Text cleaning =====
q = (df["query"].astype(str)
     .str.lower()
     .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
     .str.replace(r"\s+", " ", regex=True)
     .str.strip())

# ===== Custom stopwords =====
custom_stops = {
    'ai', 'llm', 'chatgpt', 'gemini', 'perplexity', 'google',
    'search', 'console', 'optimizer', 'login', 'tools', 'api'
}

def stop_tokenizer(s):
    toks = s.split()
    return [t for t in toks if t not in custom_stops and len(t) > 1]

# ===== TF-IDF + LSA =====
vec = TfidfVectorizer(tokenizer=stop_tokenizer, ngram_range=(1,2), min_df=1, max_df=0.6)
X = vec.fit_transform(q)

svd = TruncatedSVD(n_components=min(50, X.shape[1]-1), random_state=42)
Z = svd.fit_transform(X)

# ===== Optimal cluster selection =====
best_k, best_s, best_model = None, -1, None
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Z)
    if len(set(labels)) > 1:
        s = silhouette_score(Z, labels)
        if s > best_s:
            best_k, best_s, best_model = k, s, km

df["cluster"] = best_model.labels_
print("Cluster sizes:\n", df["cluster"].value_counts().sort_index())
print(f"\nBest k = {best_k}, Silhouette = {best_s:.3f}")

# ===== Top terms per cluster =====
def top_terms_per_cluster(model, Z, vec, topn=8):
    centroids = model.cluster_centers_
    terms = np.array(vec.get_feature_names_out())
    centroid_tfidf = centroids @ svd.components_
    for i in range(model.n_clusters):
        top_idx = centroid_tfidf[i].argsort()[::-1][:topn]
        print(f"\nCluster {i} top terms: {', '.join(terms[top_idx])}")

top_terms_per_cluster(best_model, Z, vec)

# ===== Sample queries per cluster =====
for i in range(best_k):
    print(f"\n=== Cluster {i} Sample Queries ===")
    print(df.loc[df["cluster"]==i, "query"].head(5).to_string(index=False))
