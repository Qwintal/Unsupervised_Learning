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


# DBSCAN

# Drop non-numeric columns (like CUST_ID)
df_numeric = df.drop(columns=["CUST_ID"])

# Handle missing values (if any)
df_numeric = df_numeric.fillna(df_numeric.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(6,4))
plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()

n_components = 8
X_reduced = PCA(n_components=n_components).fit_transform(X_scaled)

neighbors = 10  
nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X_reduced)
distances, indices = nbrs.kneighbors(X_reduced)

distances = np.sort(distances[:, -1])  
plt.figure(figsize=(6,4))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{neighbors}-NN distance")
plt.title("K-distance Graph (use elbow for eps)")
plt.show()



eps_val = 2.5
min_samples_val = 10

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
labels = dbscan.fit_predict(X_reduced)



n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Estimated clusters: {n_clusters}")
print(f"Noise points: {list(labels).count(-1)}")

if n_clusters > 1:
    sil_score = silhouette_score(X_reduced, labels)
    print(f"Silhouette Score: {sil_score:.3f}")
else:
    print("Silhouette Score not defined (only one cluster).")


plt.figure(figsize=(6,4))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap="plasma", s=30)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clusters (2D PCA projection)")
plt.show()


''' 
DBSCAN produces only 1 cluster + noise, then silhouette/Davies Bouldin/ other evalution matric
will be undefined.
'''

"""
eps is too large, DBSCAN connects almost every point together into 1 giant cluster.
eps is too small, almost everything is marked as noise ).
"""