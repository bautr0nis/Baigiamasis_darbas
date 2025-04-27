import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# === 1. Product categories ===
categories = [
    "electronics", "books", "clothing", "beauty", "home",
    "sports", "toys", "groceries", "automotive", "unknown"
]

n_samples = 5000

# === 2. Generate base data ===
data = {
    "price": np.round(np.random.uniform(20, 500, n_samples), 2),
    "cost": np.round(np.random.uniform(10, 300, n_samples), 2),
    "freight_value": np.round(np.random.uniform(5, 50, n_samples), 2),
    "review_score": np.round(np.random.normal(4.0, 0.8, n_samples).clip(1, 5), 1),
    "delivery_delay_days": np.round(np.random.normal(2, 1.5, n_samples).clip(0, 10), 1),
    "product_category_name": np.random.choice(categories, n_samples),
    "order_month": np.random.randint(1, 13, n_samples),
    "order_day": np.random.randint(1, 29, n_samples),
    "customer_order_count": np.random.poisson(2, n_samples),
}

# === 3. DataFrame setup ===
data = pd.DataFrame(data)

# === 4. Base price ===
data["base_price"] = data["price"] * np.random.uniform(0.9, 1.1, n_samples)
data["base_price"] = data["base_price"].round(2)

# === 5. Adjusted base demand and elasticity ===
data["base_demand"] = np.random.randint(2, 10, size=n_samples)  # 👈 smaller demand
data["price_elasticity"] = np.round(np.random.uniform(-1.5, -3.0, n_samples), 2)
data["stock"] = np.random.randint(1, 15, size=n_samples)

# === 6. Simulated demand using elasticity formula ===
demand = data["base_demand"] * (data["price"] / data["base_price"]).pow(data["price_elasticity"])
data["simulated_quantity"] = demand

# === 7. Add realistic noise + clip ===
data["simulated_quantity_fixed"] = np.round(
    data["simulated_quantity"] * np.random.uniform(0.9, 1.1, n_samples)
).clip(lower=1, upper=10)

# === 8. Save to CSV ===
os.makedirs("data", exist_ok=True)
data.to_csv("data/synthetic_olist_data2.csv", index=False)
print("✅ Synthetic data saved to data/synthetic_olist_data2.csv")