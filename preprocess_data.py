# preprocess_data.py
import pandas as pd

# Load datasets
orders = pd.read_csv('data/olist_orders_dataset.csv')
items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')

# Merge
df = orders.merge(items, on='order_id') \
           .merge(products, on='product_id') \
           .merge(payments, on='order_id') \
           .merge(reviews[['order_id', 'review_score']], on='order_id', how='left')

# Convert dates
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# Add delay in delivery
df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
df['delivery_delay_days'] = df['delivery_delay_days'].fillna(0).astype(int)

# Add day/month
df['order_day'] = df['order_purchase_timestamp'].dt.day
df['order_month'] = df['order_purchase_timestamp'].dt.month
df['order_year'] = df['order_purchase_timestamp'].dt.year

# Approximate cost: assume payment_value = price + freight + margin
df['cost'] = df['payment_value'] - df['freight_value']  # rough estimate

# Drop NAs
df = df.dropna(subset=['price', 'freight_value', 'cost', 'review_score'])

# Save
df.to_csv('data/merged_olist_data.csv', index=False)
print("✅ Enriched dataset saved!")