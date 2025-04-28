import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import os

# === 1. Nuskaitome duomenis ===
data = pd.read_csv("data/olist_data_augmented.csv")

# === 2. Agreguojame kliento lygiu ===
customer_data = data.groupby("customer_id").agg({
    "order_id": "nunique",
    "payment_value": "sum",
    "review_score": "mean",
    "delivery_delay_days": "mean"
}).reset_index()

customer_data.rename(columns={
    "order_id": "order_count",
    "payment_value": "total_spent",
    "review_score": "avg_review_score",
    "delivery_delay_days": "avg_delay_days"
}, inplace=True)

# === 3. Pašalinam NaN jei yra ===
customer_data.fillna(0, inplace=True)

# === 4. Normalizuojame duomenis ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data.drop("customer_id", axis=1))

# === 5. Randame optimalų klasterių skaičių (naudojam Elbow metodą) ===
os.makedirs("analysis/Clusters", exist_ok=True)
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Klasterių skaičius')
plt.ylabel('Inercija')
plt.title('Elbow metodas klasterių skaičiui pasirinkti')
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/Clusters/elbow_method.png")
plt.close()

# === 6. Klasterizacija su KMeans (k=4) ===
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
customer_data['cluster'] = kmeans.fit_predict(X_scaled)

# === 7. UMAP vizualizacija ===
umap_model = umap.UMAP(random_state=42)
umap_components = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=umap_components[:, 0], y=umap_components[:, 1], hue=customer_data['cluster'], palette='tab10', s=60)
plt.title('Klientų klasteriai (UMAP + KMeans)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Klasteris')
plt.tight_layout()
plt.savefig("analysis/Clusters/customer_clusters_umap.png")
plt.close()

# === 8. Klasterių charakteristikos ===
summary = customer_data.groupby('cluster').agg({
    'order_count': 'mean',
    'total_spent': 'mean',
    'avg_review_score': 'mean',
    'avg_delay_days': 'mean'
}).round(2)

summary.to_csv("analysis/Clusters/cluster_summary.csv")
print("\u2705 Klientų klasterizacija ir analizė atlikta ir išsaugota!")