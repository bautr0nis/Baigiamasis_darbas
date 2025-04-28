# cluster_categories_3d_with_connections.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import StandardScaler

# 1. Įkeliame duomenis
data = pd.read_csv("data/generated/weekly_env_data_filled.csv")

# 2. Pasiruošiam klasterizacijai
features = data.groupby('translated_name').agg({
    'avg_price': 'mean',
    'price_elasticity': 'mean',
    'base_demand': 'mean'
}).reset_index()

# 3. Standartizacija
scaler = StandardScaler()
X = scaler.fit_transform(features[['avg_price', 'price_elasticity', 'base_demand']])

# 4. KMeans klasterizacija
kmeans = KMeans(n_clusters=4, random_state=42)
features['cluster'] = kmeans.fit_predict(X)

# 5. Grafinė 3D vizualizacija
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    features['avg_price'],
    features['price_elasticity'],
    features['base_demand'],
    c=features['cluster'],
    cmap='tab10',
    s=60,
    label=features['translated_name']
)

# 6. Pavaizduojame klasterių centrus
centers = kmeans.cluster_centers_
centers_unscaled = scaler.inverse_transform(centers)

for center in centers_unscaled:
    ax.scatter(center[0], center[1], center[2], c='black', s=100, marker='X')

# 7. Linijos nuo taško iki klasterio centro
for i, row in features.iterrows():
    cluster_id = row['cluster']
    center = centers_unscaled[cluster_id]
    ax.plot(
        [row['avg_price'], center[0]],
        [row['price_elasticity'], center[1]],
        [row['base_demand'], center[2]],
        'k--', alpha=0.4
    )

ax.set_xlabel('Vidutinė Kaina')
ax.set_ylabel('Elastingumas')
ax.set_zlabel('Bazinė Paklausa')
ax.set_title('Klasteriai su jungtimis iki centrų')
ax.view_init(elev=20, azim=120)

os.makedirs("analysis/real/Clusters", exist_ok=True)
plt.savefig("analysis/Clusters/category_clusters_3d_connections.png")
plt.close()

# 8. Išsaugom klasterizuotus duomenis
features.to_csv("data/category_clusters.csv", index=False)

print("✅ Sugeneruotas 3D grafikas su jungtimis ir išsaugoti klasteriai!")