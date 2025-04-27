import pandas as pd

# 1. Įkeliam duomenis
df = pd.read_csv("data/merged_olist_data.csv")

# 2. Konvertuojam datą
df["order_date"] = pd.to_datetime(df["order_approved_at"])

# 3. Pridedam savaitės numerį ir metus (kad skirtingi metai nesusimaišytų)
df["year_week"] = df["order_date"].dt.strftime('%Y-%U')  # %U – week number (Sun-start)

# 4. Grupavimas pagal kategoriją ir savaitę
weekly_category_demand = df.groupby(["product_category_name", "year_week"]).size().reset_index(name="quantity_sold")

# 5. Peržiūra
print(weekly_category_demand.head())