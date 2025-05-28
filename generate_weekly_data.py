# generate_weekly_env_data_updated.py

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# === 1. Įkeliame sujungtą failą ===
data = pd.read_csv("data/olist_data_augmented.csv")
data['order_approved_at'] = pd.to_datetime(data['order_approved_at'], errors='coerce')
data['year_week'] = data['order_approved_at'].dt.strftime('%Y-%U')

# === 2. Grupavimas pagal savaitę ir kategoriją ===
weekly = data.groupby(['year_week', 'translated_name']).agg({
    'price': 'mean',
    'base_price': 'mean',
    'price_elasticity': 'mean'
}).reset_index()

# === 3. Generuojame base_demand (vidutinę paklausą) ===
weekly['base_demand'] = np.random.randint(50, 200, size=len(weekly))

# === 4. Pridedame kainos triukšmą (±10%) ===
weekly['avg_price'] = weekly['price'] * np.random.uniform(0.9, 1.1, size=len(weekly))

# === 5. Pridedame logiškai apskaičiuotą avg_cost (~70-90% avg_price) ===
weekly['avg_cost'] = weekly['avg_price'] * np.random.uniform(0.7, 0.9, size=len(weekly))

# === 6. Skaičiuojame paklausą su elastingumu + triukšmu ===
weekly['demand'] = weekly.apply(
    lambda row: row['base_demand'] * (row['avg_price'] / row['base_price'])**(row['price_elasticity']),
    axis=1
)
weekly['demand'] *= np.random.uniform(0.85, 1.15, size=len(weekly))
weekly['demand'] = weekly['demand'].clip(lower=1)

# === 7. Skaičiuojame sandėlį (papildymas kas 4 savaitę) ===
stock_levels = []
stock = 0
prev_cat = None
for i, row in weekly.iterrows():
    cat = row['translated_name']
    demand = row['demand']
    if cat != prev_cat:
        stock = np.random.randint(int(demand) + 10, int(demand) + 30)
    else:
        if int(row['year_week'].split("-")[1]) % 4 == 0:
            stock = demand + np.random.randint(10, 25)
        else:
            stock = max(0, stock - quantity)
    stock_levels.append(stock)
    quantity = min(demand, stock)
    prev_cat = cat

weekly['stock'] = stock_levels

# === 8. Pardavimai = min(demand, stock) ===
weekly['quantity_sold'] = weekly[['demand', 'stock']].min(axis=1).astype(int)

# === 9. Logikos patikrinimai ===
weekly['base_demand'] = weekly['base_demand'].astype(int)
weekly['price_elasticity'] = weekly['price_elasticity'].fillna(-1.5)

# === 10. Galutiniai stulpeliai ===
weekly = weekly[[
    'year_week', 'translated_name', 'avg_price', 'avg_cost',
    'base_price', 'base_demand', 'demand', 'quantity_sold', 'stock', 'price_elasticity'
]]

# === 11. Išsaugojimas ===
os.makedirs("data", exist_ok=True)
output_path = "data/weekly_env_data.csv"
weekly.to_csv(output_path, index=False)

print(f"✅ Sugeneruotas atnaujintas savaitinis failas: {output_path}")