# DBSCAN (outlier removal) + KMeans clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

df = pd.read_csv("CC GENERAL.csv")
df.drop(columns=['CUST_ID'], inplace=True)
df = df.dropna(subset=['CREDIT_LIMIT'])
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# Step 1: DBSCAN for outlier removal

dbscan = DBSCAN(eps=2.5, min_samples=10)  
db_labels = dbscan.fit_predict(X_scaled)

# Keep only non-noise points
mask = db_labels != -1
X_clean = X_scaled[mask]
df_clean = df[mask].reset_index(drop=True)

print(f"Original data: {df.shape[0]} rows")
print(f"After DBSCAN outlier removal: {df_clean.shape[0]} rows")
print(f"Outliers removed: {df.shape[0] - df_clean.shape[0]}")


# Step 2: PCA for dimensionality reduction

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clean)


# Step 3: KMeans 

# Choose k
k_opt = 6
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_clean)


# Step 4: Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", s=60)
plt.title(f"KMeans Clusters (k={k_opt}) after Outlier Removal")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()


# Step 5: Cluster evaluation
sil = silhouette_score(X_clean, labels)
db = davies_bouldin_score(X_clean, labels)
print("\n=== KMeans Evaluation (after outlier removal) ===")
print(f"Silhouette Score: {sil:.3f}")
print(f"Davies-Bouldin Index: {db:.3f}")


# Step 6: Business insights (cluster means)
df_clean['cluster'] = labels
cluster_summary = df_clean.groupby('cluster').mean().round(2)
cluster_sizes = df_clean['cluster'].value_counts().sort_index()

print("\nCluster sizes:")
print(cluster_sizes)
print("\nCluster means (for interpretation):")
print(cluster_summary)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# === Load your cluster summary file ===
cluster_summary = pd.read_csv("cluster_summary.csv", index_col=0)

# === Normalize values for radar chart (so all features fit on same scale) ===
scaler = MinMaxScaler()
cluster_normalized = pd.DataFrame(
    scaler.fit_transform(cluster_summary),
    columns=cluster_summary.columns,
    index=cluster_summary.index
)

# === Radar Chart Function ===
def make_radar_chart(df, title, save_path=None):
    labels = df.columns
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, row in df.iterrows():
        values = row.tolist()
        values += values[:1]  # repeat first value to close circle
        ax.plot(angles, values, label=f"Cluster {idx}")
        ax.fill(angles, values, alpha=0.1)

    # Formatting
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save chart if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# === Create and save radar chart ===
make_radar_chart(cluster_normalized, "Customer Cluster Profiles (Radar Chart)", save_path="cluster_radar_chart.png")
