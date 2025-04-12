# cluster_category_elasticity.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Įkeliam duomenis
df = pd.read_csv("data/merged_olist_data.csv")

# 2. Paimam tik unikalią elastingumo informaciją
elasticity_df = df[['product_category_name', 'price_demand_corr']].drop_duplicates()

# 3. Pašalinam kategorijas be koreliacijos (NaN)
elasticity_df = elasticity_df.dropna(subset=['price_demand_corr'])

# 4. Normalizuojam koreliaciją
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(elasticity_df[['price_demand_corr']])

# 5. KMeans klasteriai
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
elasticity_df['category_cluster'] = kmeans.fit_predict(X_scaled)

# 6. Vizualizacija (viena dimensija + klasteris)
plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=elasticity_df['price_demand_corr'],
    y=[0] * len(elasticity_df),  # viena dimensija
    hue=elasticity_df['category_cluster'],
    palette='Set1',
    s=100
)
plt.xlabel("Price-Demand Correlation")
plt.title("📦 Kategorijų klasteriai pagal kainų elastingumą")
plt.yticks([])
plt.tight_layout()
plt.savefig("data/category_elasticity_clusters.png")
plt.show()

# 7. Sujungiam atgal
df = df.drop(columns=['category_cluster'], errors='ignore')
df = df.merge(elasticity_df[['product_category_name', 'category_cluster']], on='product_category_name', how='left')

# 8. Išsaugom
df.to_csv("data/merged_olist_data.csv", index=False)
print("✅ 'category_cluster' pagal elastingumą pridėtas ir išsaugotas!")