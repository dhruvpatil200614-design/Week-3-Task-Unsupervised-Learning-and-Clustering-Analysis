# Week 3 Task: Unsupervised Learning and Clustering Analysis
# Dataset: UCI Wine Recognition Dataset
# Algorithm: K-Means Clustering
# Optional comparison: Agglomerative (Hierarchical) Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 1. Load the public Wine dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# IMPORTANT:
# The target/class column is NOT used for clustering.
# Clustering is performed only on the 13 chemical measurements.
X = df.drop(columns=["target"])

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find a suitable number of clusters
k_values = range(2, 9)
inertias = []
silhouette_scores = []
calinski_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(alpha=0.25)
plt.show()

# Silhouette plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(alpha=0.25)
plt.show()

# 4. Choose k=3
# k=3 gives the highest silhouette score among k=2..8 in this analysis.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print("\nSilhouette score for K-Means:", silhouette_score(X_scaled, labels))
print("Calinski-Harabasz score:", calinski_harabasz_score(X_scaled, labels))

# 5. Add cluster labels
clustered = X.copy()
clustered["Cluster"] = labels

print("\nCluster counts:")
print(clustered["Cluster"].value_counts().sort_index())

# 6. Cluster profile
cluster_profile = clustered.groupby("Cluster").mean()
print("\nCluster mean profile:")
print(cluster_profile)

# 7. Standardized centroid profile
centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns,
    index=["Cluster 0", "Cluster 1", "Cluster 2"]
)
print("\nStandardized cluster centroids:")
print(centroids)

# 8. PCA for 2-D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster in range(3):
    mask = labels == cluster
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster}",
        alpha=0.75
    )

centroid_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroid_pca[:, 0],
    centroid_pca[:, 1],
    marker="X",
    s=150,
    label="Centroids"
)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("K-Means Clusters using PCA")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# 9. Optional: Hierarchical clustering comparison
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
print(
    "\nAgglomerative clustering silhouette score:",
    silhouette_score(X_scaled, agg_labels)
)

# 10. Save final dataset
clustered.to_csv("wine_clustered_data.csv", index=False)
print("\nSaved: wine_clustered_data.csv")
