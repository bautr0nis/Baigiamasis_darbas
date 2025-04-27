import pandas as pd
import numpy as np
import os

np.random.seed(42)

# === 1. Load original data ===
weekly = pd.read_csv("data/weekly_env_data.csv")

# Ensure correct types
weekly['year_week'] = weekly['year_week'].astype(str)

# === 2. Find the last year and week ===
last_year = int(weekly['year_week'].str[:4].max())

# Parameters
years_to_generate = 5
min_weeks_per_category = 100
annual_inflation_rate = 0.02  # +2% per year

# === 3. Helper functions ===
def generate_new_weeks(existing_weeks, years_to_add):
    """Generate new year_week labels"""
    all_weeks = []
    last_year = int(max([int(y.split('-')[0]) for y in existing_weeks]))
    for year in range(last_year + 1, last_year + 1 + years_to_add):
        for week in range(1, 53):
            all_weeks.append(f"{year}-{week:02d}")
    return all_weeks

# === 4. Generate future data ===
future_data = []

new_year_weeks = generate_new_weeks(weekly['year_week'], years_to_generate)
unique_categories = weekly['translated_name'].unique()

for category in unique_categories:
    cat_data = weekly[weekly['translated_name'] == category].copy()
    if cat_data.empty:
        continue

    base_row = cat_data.sample(1, random_state=42).iloc[0]

    for i, yw in enumerate(new_year_weeks):
        inflation_multiplier = (1 + annual_inflation_rate) ** (i // 52)

        # Seasonality effect
        week_num = int(yw.split('-')[1])
        seasonality = 1.0
        if 47 <= week_num <= 52:
            seasonality += np.random.uniform(0.15, 0.30)  # holiday boost
        if 25 <= week_num <= 35:
            seasonality -= np.random.uniform(0.05, 0.10)  # summer dip

        new_entry = base_row.copy()
        new_entry['year_week'] = yw
        new_entry['avg_price'] *= inflation_multiplier * np.random.uniform(0.98, 1.02)
        new_entry['avg_cost'] *= inflation_multiplier * np.random.uniform(0.98, 1.02)
        new_entry['base_price'] *= inflation_multiplier * np.random.uniform(0.98, 1.02)
        new_entry['base_demand'] = max(1, int(new_entry['base_demand'] * seasonality * np.random.uniform(0.95, 1.05)))
        new_entry['demand'] = max(1, int(new_entry['demand'] * seasonality * np.random.uniform(0.95, 1.05)))
        new_entry['stock'] = max(new_entry['demand'], new_entry['stock'])
        new_entry['quantity_sold'] = min(new_entry['stock'], new_entry['demand'])
        future_data.append(new_entry)

# === 5. Create future dataframe ===
future_df = pd.DataFrame(future_data)

# === 6. Combine original + future ===
augmented = pd.concat([weekly, future_df], ignore_index=True)

# === 7. Check for low-data categories and duplicate them ===
cat_counts = augmented['translated_name'].value_counts()
low_cats = cat_counts[cat_counts < min_weeks_per_category].index

for cat in low_cats:
    need_more = min_weeks_per_category - cat_counts[cat]
    sample_cat = augmented[augmented['translated_name'] == cat].sample(need_more, replace=True, random_state=42)

    # Slight variation
    sample_cat['avg_price'] *= np.random.uniform(0.97, 1.03, size=len(sample_cat))
    sample_cat['avg_cost'] *= np.random.uniform(0.97, 1.03, size=len(sample_cat))
    sample_cat['base_demand'] = (sample_cat['base_demand'] * np.random.uniform(0.95, 1.05, size=len(sample_cat))).astype(int)
    sample_cat['demand'] = (sample_cat['demand'] * np.random.uniform(0.95, 1.05, size=len(sample_cat))).astype(int)
    sample_cat['quantity_sold'] = (sample_cat['quantity_sold'] * np.random.uniform(0.95, 1.05, size=len(sample_cat))).astype(int)
    augmented = pd.concat([augmented, sample_cat], ignore_index=True)

# === 8. Save augmented data ===
os.makedirs("data/generated", exist_ok=True)
augmented_output = "data/generated/weekly_env_data_augmented.csv"
augmented.to_csv(augmented_output, index=False)

print(f"✅ Augmented data saved to: {augmented_output}")
