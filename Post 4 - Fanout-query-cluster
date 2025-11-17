import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv("data/gsc_queries.csv")
queries = df["Query"].astype(str).tolist()

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X = vectorizer.fit_transform(queries)
model = KMeans(n_clusters=3, random_state=42).fit(X)

for i, label in enumerate(model.labels_):
    print(f"Cluster {label}: {queries[i]}")


#script i used to create viz using dummy data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from adjustText import adjust_text

queries = [
    "ai optimization", "chatgpt optimization tips", "llm optimization guide",
    "ai search volatility", "gemini search overview",
    "python script for seo", "ga4 api tutorial", "google search console export",
    "data visualization python", "seo automation with python"
]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(queries)
model = KMeans(n_clusters=3, random_state=42).fit(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X.toarray())

plt.figure(figsize=(7,5))
plt.scatter(coords[:,0], coords[:,1], c=model.labels_, cmap='tab10', s=80, alpha=0.7)

texts = []
for i, txt in enumerate(queries):
    texts.append(plt.text(coords[i,0], coords[i,1], txt, fontsize=8))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.title("Dummy Example: Clustering GSC Queries")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

