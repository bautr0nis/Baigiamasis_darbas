# preprocess_data(old).py
import pandas as pd

# === 1. Load datasets ===
orders = pd.read_csv('data/olist_orders_dataset.csv')
items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')

# === 2. Merge all data ===
df = orders.merge(items, on='order_id') \
           .merge(products, on='product_id') \
           .merge(payments, on='order_id') \
           .merge(reviews[['order_id', 'review_score']], on='order_id', how='left') \
           .merge(customers[['customer_id', 'customer_city', 'customer_state', 'customer_zip_code_prefix']], on='customer_id', how='left')

# === 3. Convert dates ===
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# === 4. Delivery delay ===
df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
df['delivery_delay_days'] = df['delivery_delay_days'].fillna(0).astype(int)

# === 5. Time features ===
df['order_day'] = df['order_purchase_timestamp'].dt.day
df['order_month'] = df['order_purchase_timestamp'].dt.month
df['order_year'] = df['order_purchase_timestamp'].dt.year

# === 6. Cost estimation ===
df['cost'] = df['payment_value'] - df['freight_value']

# === 7. Customer order count (loyalty proxy) ===
order_counts = orders.groupby('customer_id').size().reset_index(name='customer_order_count')
df = df.merge(order_counts, on='customer_id', how='left')

# === 8. Drop incomplete rows ===
df = df.dropna(subset=['price', 'freight_value', 'cost', 'review_score'])

# === 9. Category elasticity group assignment ===
elasticities = []
for cat, group in df.groupby('product_category_name'):
    if group['price'].nunique() > 1 and len(group) > 30:
        qty_by_price = group.groupby('price').size()
        corr = group['price'].corr(qty_by_price.reindex(group['price']).fillna(0))
        elasticities.append((cat, corr))
    else:
        elasticities.append((cat, 0))  # No variation or too little data

elasticity_df = pd.DataFrame(elasticities, columns=['product_category_name', 'price_demand_corr'])

# Grouping into quantile bins
elasticity_df['category_elasticity_group'] = pd.qcut(
    elasticity_df['price_demand_corr'],
    q=3,
    labels=['inelastic', 'neutral', 'elastic']
)

df = df.merge(elasticity_df[['product_category_name', 'price_demand_corr', 'category_elasticity_group']],
              on='product_category_name', how='left')

# === 10. Save merged dataset ===
df.to_csv('data/merged_olist_data.csv', index=False)
print("✅ Visi duomenys paruošti ir išsaugoti į 'merged_olist_data.csv'!")