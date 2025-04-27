import pandas as pd
import numpy as np
import os

# === 1. Įkeliam originalų failą ===
data = pd.read_csv("data/merged_olist_data.csv")
data["order_approved_at"] = pd.to_datetime(data["order_approved_at"])

# === 2. Sukuriam week identifikatorių
data["year_week"] = data["order_approved_at"].dt.strftime('%Y-%U')

# === 3. Suskaičiuojam kiek buvo parduota prekių kiekvieną savaitę pagal kategoriją
weekly_demand = (
    data.groupby(["product_category_name", "year_week"])
    .size()
    .reset_index(name="base_demand")
)

# === 4. Įkeliam kategorijų vertimus su elastingumu ===
cat_map = pd.read_csv("data/unique_categories_translated.csv")

# === 5. Sujungiam viską į pagrindinį failą
data = data.merge(cat_map, on="product_category_name", how="left")
data = data.merge(weekly_demand, on=["product_category_name", "year_week"], how="left")

# === 6. Jei trūksta base_demand – priskiriam minimalų 2
data["base_demand"].fillna(2, inplace=True)
data["base_demand"] = data["base_demand"].astype(int)

# === 7. Tvarkom cost: jei cost < 0, prilyginam 80% payment_value
data.loc[data["cost"] < 0, "cost"] = data.loc[data["cost"] < 0, "payment_value"] * 0.8

# === 8. Pridedam base_price (original price)
data["base_price"] = data["price"]

# === 9. Sugeneruojam stock kaip [1, base_demand + 3]
data["stock"] = data["base_demand"].apply(lambda x: np.random.randint(1, x + 3))

# === 10. Išsaugom naudoti su Advanced env
os.makedirs("data", exist_ok=True)
data.to_csv("data/olist_data_augmented.csv", index=False)
print("✅ Duomenys sujungti ir išsaugoti: data/olist_data_augmented.csv")