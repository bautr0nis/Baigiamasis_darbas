# preprocess_data.py
import pandas as pd

# === Load datasets ===
orders = pd.read_csv('data/olist_orders_dataset.csv')
items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')

# === Merge all ===
df = orders.merge(items, on='order_id') \
           .merge(products, on='product_id') \
           .merge(payments, on='order_id') \
           .merge(reviews[['order_id', 'review_score']], on='order_id', how='left') \
           .merge(customers[['customer_id', 'customer_state', 'customer_city', 'customer_zip_code_prefix']], on='customer_id', how='left')

# === Convert dates ===
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# === Delivery delay ===
df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
df['delivery_delay_days'] = df['delivery_delay_days'].fillna(0).astype(int)

# === Order time features ===
df['order_day'] = df['order_purchase_timestamp'].dt.day
df['order_month'] = df['order_purchase_timestamp'].dt.month
df['order_year'] = df['order_purchase_timestamp'].dt.year

# === Cost estimate ===
df['cost'] = df['payment_value'] - df['freight_value']

# === Customer loyalty ===
order_counts = orders.groupby('customer_id').size().reset_index(name='customer_order_count')
df = df.merge(order_counts, on='customer_id', how='left')

# === Drop rows with critical NAs ===
df = df.dropna(subset=['price', 'freight_value', 'cost', 'review_score'])

# === Save to CSV ===
df.to_csv('data/merged_olist_data.csv', index=False)
print("✅ Enriched dataset saved with customer features!")