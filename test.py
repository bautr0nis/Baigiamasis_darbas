import pandas as pd

df = pd.read_csv("data/merged_olist_data.csv")
df["quantity"] = df.groupby(["order_id", "product_id"])["order_item_id"].transform("count")

correlation = df["price"].corr(df["quantity"])
print(f"📊 Correlation between price and quantity: {correlation:.3f}")