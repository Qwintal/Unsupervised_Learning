import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mpl_toolkits.mplot3d import Axes3D


# Importing dataset

df = pd.read_csv("CC GENERAL.csv")
df = df.drop(columns=["CUST_ID"])
df = df.fillna(df.mean())


# Preprocessing & PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)


# Hierarchical Clustering
linked = linkage(X_pca, method='ward') # Variance ( single, complete, average, ward )

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=20)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# Choose number of clusters
k = 4
clusters = fcluster(linked, k, criterion='maxclust')
df['cluster'] = clusters

print("Cluster sizes:\n", df['cluster'].value_counts())
print("\nCluster means:\n", df.groupby('cluster').mean())


# Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=clusters, cmap='viridis', s=30)
ax.set_title("Hierarchical Clustering (3D PCA projection)")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()


# Evaluation Metrics
sil_score = silhouette_score(X_scaled, clusters)
db_score = davies_bouldin_score(X_scaled, clusters)
ch_score = calinski_harabasz_score(X_scaled, clusters)

# Dunn Index function
def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    inter_cluster = []
    intra_cluster = []

    for i in unique_clusters:
        cluster_i = X[labels == i]
        intra_dist = np.max(cdist(cluster_i, cluster_i))  
        intra_cluster.append(intra_dist)
        for j in unique_clusters:
            if i < j:
                cluster_j = X[labels == j]
                inter_dist = np.min(cdist(cluster_i, cluster_j)) 
                inter_cluster.append(inter_dist)
    return np.min(inter_cluster) / np.max(intra_cluster)

dunn_val = dunn_index(X_scaled, clusters)

print("\n Evaluation Metrics (Hierarchical)")
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Index: {db_score:.3f}")
print(f"Calinski-Harabasz Index: {ch_score:.3f}")
print(f"Dunn Index: {dunn_val:.3f}")
