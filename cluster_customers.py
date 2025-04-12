# cluster_customers.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed dataset
df = pd.read_csv('data/merged_olist_data.csv')

# 2. Aggregate customer-level data
customer_df = df.groupby('customer_id').agg({
    'payment_value': 'mean',
    'review_score': 'mean',
    'delivery_delay_days': 'mean',
    'order_id': 'count',
    'customer_state': 'first',
    'category_elasticity_group': lambda x: x.mode()[0] if not x.mode().empty else 'neutral'
}).reset_index()

customer_df.columns = ['customer_id', 'avg_payment_value', 'avg_review_score', 'avg_delivery_delay_days',
                       'customer_order_count', 'customer_state', 'dominant_elasticity_group']

# 3. One-hot encode categorical features
encoder_state = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
state_encoded = encoder_state.fit_transform(customer_df[['customer_state']])
state_df = pd.DataFrame(state_encoded, columns=encoder_state.get_feature_names_out(['customer_state']))

encoder_elasticity = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
elasticity_encoded = encoder_elasticity.fit_transform(customer_df[['dominant_elasticity_group']])
elasticity_df = pd.DataFrame(elasticity_encoded, columns=encoder_elasticity.get_feature_names_out(['dominant_elasticity_group']))

# 4. Combine all features
X = pd.concat([
    customer_df[['avg_payment_value', 'avg_review_score', 'avg_delivery_delay_days', 'customer_order_count']],
    state_df,
    elasticity_df
], axis=1)

# 5. Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 6. UMAP dimensionality reduction
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)
customer_df['component_1'] = embedding[:, 0]
customer_df['component_2'] = embedding[:, 1]

# 7. KMeans clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
customer_df['customer_cluster'] = kmeans.fit_predict(X_scaled)

# 8. Visualize
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=customer_df,
    x='component_1',
    y='component_2',
    hue='customer_cluster',
    palette='Set2',
    s=60,
    alpha=0.85
)
plt.title("📊 Klientų klasteriai (UMAP + KMeans)", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title='Klasteris')
plt.tight_layout()
plt.savefig("data/customer_clusters_umap.png")
plt.show()

# 9. Pašalinam dublikatus iš df prieš merge
drop_cols = [
    'customer_order_count', 'customer_cluster',
    'avg_payment_value', 'avg_review_score',
    'avg_delivery_delay_days', 'dominant_elasticity_group'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

if 'customer_cluster' in df.columns:
    df.drop(columns=['customer_cluster'], inplace=True)

# 10. Merge su naujais laukais

df = df.merge(
    customer_df[['customer_id', 'customer_cluster',
                 'avg_payment_value', 'avg_review_score',
                 'avg_delivery_delay_days', 'customer_order_count',
                 'dominant_elasticity_group']],
    on='customer_id',
    how='left'
)
df.to_csv('data/merged_olist_data.csv', index=False)
print("✅ Klientų klasteriai pridėti ir išsaugoti su vietovės informacija!")