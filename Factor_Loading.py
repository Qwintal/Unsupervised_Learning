import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CC GENERAL.csv")
df.head(5)
df.info()
df.shape
df.describe()

# Cleaning Dataset
df = df.dropna(subset=['CREDIT_LIMIT']) # Removing null values row of credit limit column
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True) # Filling meadian value in minimum payments
df.shape
df.drop(columns=['CUST_ID'], inplace=True)
df.duplicated().sum()
df1 = df.copy()
df1.shape


# 1. Scale the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 2. Fit PCA
pca = PCA(n_components=5)   # choose number of PCs you want
pca.fit(X_scaled)

# 3. Compute factor loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loading_df = pd.DataFrame(
    loadings,
    index=df.columns,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print("\nPCA Factor Loadings:")
print(loading_df.round(3))

# 4. Optional: visualize as heatmap
plt.figure(figsize=(10,6))
sns.heatmap(loading_df, annot=True, cmap="coolwarm", center=0)
plt.title("PCA Factor Loadings Heatmap")
plt.show()

