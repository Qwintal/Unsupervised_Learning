# Importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import seaborn as sns
import warnings
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from mpl_toolkits.mplot3d import Axes3D




# Importing dataset
df = pd.read_csv("CC GENERAL.csv")
df.head(5)
df.info()
df.shape
df.describe()

# Cleaning Dataset
df = df.dropna(subset=['CREDIT_LIMIT']) # Removing null values row of credit limit column
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True) # Filling meadian value in minimum payments because distribution is not symmetric (skewed)
df.shape
df.drop(columns=['CUST_ID'], inplace=True)
df.duplicated().sum()
df1 = df.copy()
df1.shape

# Scaling Data
scaler = StandardScaler()   # Scales all columns equally 
X_scaled = scaler.fit_transform(df)

# we tried using robustscaling for outlier handinling

# PCA 
pca = PCA() # Reducese dimenstions ( columns into 2D or 3D )
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_ # for scree plot 
cumulative_variance = explained_variance.cumsum() # accumulation of all all pca (variance)


plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1), cumulative_variance, marker='o', linestyle='--') # + 1 for range
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs Number of Components")
plt.grid(True)
plt.show()



pca = PCA(n_components=8) 
X_pca = pca.fit_transform(X_scaled)

# K means 

# Elbow Method to find optimal k
inertia1 = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
    kmeans.fit(X_scaled)
    inertia1.append(kmeans.inertia_)

# Plot the graph
plt.plot(range(2, 11), inertia1, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(X_pca)


plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="Set2", s=60)
plt.title(f"KMeans Clusters Visualization (k={6})", fontsize=14)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.show()
df1.head()


# 3D visualization
# Fit PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Run KMeans on 3D PCA space
kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(X_pca)

# 3D visualization with clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:,0], X_pca[:,1], X_pca[:,2],
    c=labels, cmap="Set2", s=60, alpha=0.8
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D KMeans Clusters (PCA Projection)")

# Legend for clusters
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)

plt.show()

# Evaluation Metrics
inertia_val = kmeans.inertia_
sil_score = silhouette_score(X_scaled, labels)
db_score = davies_bouldin_score(X_scaled, labels)

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

dunn_val = dunn_index(X_scaled, labels)

print(f"Inertia: {inertia_val:.2f}")
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Index: {db_score:.3f}")
print(f"Dunn Index: {dunn_val:.3f}")

from sklearn.neighbors import LocalOutlierFactor

# ---- Anomaly Detection ----
# Using LocalOutlierFactor (LOF) is a good idea — it’s more consistent than just treating faraway points from centroids as anomalies.
# K-means, DBSCAN follow hard clustering thats why so many anomlay 
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  # 5% anomalies
outlier_labels = lof.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal

# Separate normal points and anomalies in PCA 2D
pca_2d = PCA(n_components=2).fit_transform(X_scaled)
normal = pca_2d[outlier_labels == 1]
anomalies = pca_2d[outlier_labels == -1]

# ---- Plot anomalies on PCA scatter ----
plt.figure(figsize=(8,6))
plt.scatter(normal[:, 0], normal[:, 1], c='lightblue', s=50, alpha=0.6, label="Normal Points")
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', s=60, edgecolors='k', label="Anomalies")
plt.title("Anomaly Detection with LOF (2D PCA Projection)", fontsize=14)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()