import pandas as pd

# Load enriched dataset
df = pd.read_csv("data/merged_olist_data.csv")

# Group by cluster and get mean of relevant features
summary = df.groupby('customer_cluster').agg({
    'avg_payment_value': 'mean',
    'avg_review_score': 'mean',
    'avg_delivery_delay_days': 'mean',
    'customer_order_count': 'mean',
    'customer_state': lambda x: x.mode()[0],
    'dominant_elasticity_group': lambda x: x.mode()[0]
}).reset_index()

summary.columns = ['Klasteris', 'Vid. mokėjimas', 'Vid. įvertinimas', 'Vid. vėlavimas (d)', 'Vid. užsakymų sk.', 'Dažniausia būsena', 'Dažniausia elastingumo grupė']

print("📊 Klientų klasterių apibūdinimas:\n")
print(summary.to_string(index=False))

# Optional: export to markdown or csv
summary.to_csv("data/customer_cluster_summary.csv", index=False)