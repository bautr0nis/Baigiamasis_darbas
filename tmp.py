import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Load original dataset
df = pd.read_csv("data/generated/weekly_env_data_augmented.csv")

# 2. Create full grid of all (translated_name, year_week)
all_categories = df['translated_name'].unique()
all_weeks = df['year_week'].unique()

full_index = pd.MultiIndex.from_product(
    [all_categories, all_weeks],
    names=['translated_name', 'year_week']
)

# 3. Reindex your data
df = df.set_index(['translated_name', 'year_week'])
df_full = df.reindex(full_index).reset_index()

# 4. Fill missing values using moving averages
columns_to_fill = ['avg_price', 'avg_cost', 'base_price', 'base_demand', 'demand', 'price_elasticity']

for col in columns_to_fill:
    df_full[col] = (df_full.groupby('translated_name')[col]
                    .transform(lambda x: x.fillna(x.rolling(window=3, min_periods=1, center=True).mean()))
                    .fillna(method='ffill')
                    .fillna(method='bfill'))

# Stock and quantity_sold — assume 0 if missing
df_full['stock'] = df_full['stock'].fillna(0)
df_full['quantity_sold'] = df_full['quantity_sold'].fillna(0)

# Fill missing 'translated_name' if any (should not be)
df_full['translated_name'] = df_full['translated_name'].fillna('unknown')

# 5. Save the fixed dataset
os.makedirs("data/generated", exist_ok=True)
output_path = "data/generated/weekly_env_data_filled.csv"
df_full.to_csv(output_path, index=False)

print(f"✅ Moving-average filled data saved to {output_path}")

# === 6. Validation chart: missing weeks check ===

# Mark if avg_price is still missing
df_full['missing'] = df_full['avg_price'].isna()

missing_stats = (df_full.groupby('translated_name')['missing']
                 .mean()
                 .sort_values(ascending=False) * 100)

# Plot missing data
plt.figure(figsize=(16, 7))
missing_stats.plot(kind='bar')
plt.title("✅ Percentage of Missing Weeks per Category After Filling")
plt.ylabel("Missing Weeks (%)")
plt.xlabel("Category")
plt.tight_layout()
plt.show()